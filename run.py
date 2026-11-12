"""Train and evaluate the RopStitch PixelStitch residual adapter."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from model import (InvalidPair, PixelResidual, RopTPS, fixed_overlap,
                   guarded_render, objective, photo)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LOSS_WEIGHTS = dict(cycle=0.01, smooth=0.015, nonoverlap=0.04, coverage=0.5, fold=0.15)
FORMAT = "ropstitch-pixel-residual-v1"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(root, manifest=None):
    views = []
    for side in ("input1", "input2"):
        files = {}
        for path in (root / side).iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                if path.stem in files:
                    raise ValueError(f"Duplicate image ID in {side}: {path.stem}")
                files[path.stem] = path.resolve()
        views.append(files)
    if manifest is None:
        if views[0].keys() != views[1].keys():
            raise ValueError(f"Image IDs differ between input1 and input2 in {root}")
        names = sorted(views[0])
    else:
        names = [Path(line.strip()).stem for line in manifest.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Empty or duplicate pair list: {manifest or root}")
    try:
        return [(name, views[0][name], views[1][name]) for name in names]
    except KeyError as exc:
        raise ValueError(f"Pair {exc} is missing in {root}") from exc


def load_pair(pair):
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in pair[1:]]
    if any(im is None for im in images):
        raise InvalidPair("Image failed to decode")
    if images[0].shape != images[1].shape:
        images = [cv2.resize(im, (512, 512)) for im in images]
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).cuda().float() / 255


def disjoint_train_val(train, val):
    training = {file_hash(path) for _, a, b in train for path in (a, b)}
    for name, a, b in val:
        if file_hash(a) in training or file_hash(b) in training:
            raise ValueError(f"Validation pair {name} shares an exact image with training")


def save_checkpoint(path, model, optimizer, epoch, best, cfg, rng=True):
    state = dict(format=FORMAT, epoch=epoch, best=best, config=cfg, model=model.state_dict(),
                 optimizer=optimizer.state_dict())
    if rng:
        state["torch_rng"] = torch.get_rng_state()
        state["cuda_rng"] = torch.cuda.get_rng_state()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError(f"Wrong checkpoint format: {path}")
    return state


def options(args):
    return dict(max_shift=args.max_shift, iterations=args.iters, edge=args.max_edge,
                rop_alignment_sha256=file_hash(args.alignment),
                rop_coefficient_sha256=file_hash(args.coefficient))


def log_line(path, item):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


@torch.no_grad()
def inference_metrics(pair, geom, model, retention, guard):
    canvas = geom(load_pair(pair))
    residual, _, base = model(canvas)
    images, support, scale = guarded_render(canvas, base, residual, retention, guard)
    mask = fixed_overlap(canvas)
    visible = support > 0.99
    prior = canvas.support > 0.99
    valid = int(mask.sum())
    metrics = dict(
        id=pair[0], scale=scale, fixed_overlap_pixels=valid,
        baseline_photo=float(photo(canvas.images, mask)) if valid else None,
        refined_photo=float(photo(images, mask)) if valid else None,
        view_retention=((visible & prior).sum((1, 2, 3)) /
                        prior.sum((1, 2, 3)).clamp_min(1)).tolist(),
        union_pixels=int(visible.any(0).sum()), overlap_pixels=int(visible.all(0).sum()),
        baseline_union_pixels=int(prior.any(0).sum()), baseline_overlap_pixels=int(prior.all(0).sum()),
    )
    return images, visible, metrics


def train(args, geom, cfg):
    pairs = discover(args.data, args.list)
    val = discover(args.val_data, args.val_list) if args.val_data else []
    if val:
        disjoint_train_val(pairs, val)
    if args.resume is None:
        cfg["pixel_init_sha256"] = file_hash(args.pixel_weights)
    cfg.update(seed=args.seed, loss_weights=LOSS_WEIGHTS, pair_ids=[p[0] for p in pairs])
    model = PixelResidual(args.pixel, args.pixel_weights if args.resume is None else None,
                          cfg["max_shift"], cfg["iterations"]).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    start, best = 1, None
    if args.resume:
        state = checkpoint(args.resume)
        cfg["pixel_init_sha256"] = state["config"]["pixel_init_sha256"]
        if state["config"] != cfg:
            raise ValueError("Resume configuration differs from the saved training run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["epoch"] + 1, state["best"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
    if args.epochs < start:
        raise ValueError("--epochs is the final epoch number, not the number of additional epochs")
    args.out.mkdir(parents=True)
    log_line(args.out / "plan.jsonl", dict(
        config=cfg, pair_count=len(pairs), validation_count=len(val), initial_lr=args.lr,
        data=str(args.data), validation=str(args.val_data) if val else None,
        resume=str(args.resume) if args.resume else None,
    ))
    print(f"Trainable flow parameters: {sum(p.numel() for p in parameters):,}; "
          f"{len(pairs)} train / {len(val)} validation pairs", flush=True)
    for epoch in range(start, args.epochs + 1):
        order = list(pairs)
        random.Random(args.seed + epoch).shuffle(order)
        totals = []
        for index, pair in enumerate(order, 1):
            try:
                canvas = geom(load_pair(pair))
                if int(fixed_overlap(canvas).sum()) < 64:
                    raise InvalidPair("Fewer than 64 fixed overlap pixels")
            except InvalidPair as exc:
                log_line(args.out / "skipped.jsonl", dict(epoch=epoch, id=pair[0], reason=str(exc)))
                continue
            optimizer.zero_grad(set_to_none=True)
            residual, cycle, base = model(canvas)
            total, losses = objective(canvas, residual, cycle, base, LOSS_WEIGHTS)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite training loss: {pair[0]}")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            metrics = {key: float(value.detach()) for key, value in losses.items()}
            metrics.update(epoch=epoch, step=index, id=pair[0], loss=float(total.detach()),
                           grad_norm=float(norm))
            totals.append(metrics["loss"])
            log_line(args.out / "steps.jsonl", metrics)
            if index == 1 or index % 25 == 0 or index == len(order):
                print(f"epoch {epoch} pair {index}/{len(order)} loss={metrics['loss']:.6f} "
                      f"grad={metrics['grad_norm']:.6f}", flush=True)
        if not totals:
            raise ValueError(f"Epoch {epoch} had no usable training pairs")
        val_scores = []
        for pair in val:
            try:
                _, _, row = inference_metrics(pair, geom, model, args.min_retention, False)
            except InvalidPair as exc:
                row = dict(id=pair[0], error=str(exc))
            if row.get("refined_photo") is not None:
                val_scores.append(row["refined_photo"] - row["baseline_photo"])
            log_line(args.out / "validation.jsonl", dict(epoch=epoch, **row))
        if val and not val_scores:
            raise ValueError("No valid validation pairs")
        score = float(np.mean(val_scores)) if val else None
        is_best = score is not None and (best is None or score < best)
        if is_best:
            best = score
        save_checkpoint(args.out / "last.pt", model, optimizer, epoch, best, cfg)
        if is_best:
            save_checkpoint(args.out / "best.pt", model, optimizer, epoch, best, cfg)
        row = dict(epoch=epoch, pairs=len(totals), skipped=len(order)-len(totals),
                   train_loss=float(np.mean(totals)), val_photo_change=score)
        log_line(args.out / "epochs.jsonl", row)
        print(json.dumps(row), flush=True)


def image_file(path, image):
    if not cv2.imwrite(str(path), np.rint(np.clip(image, 0, 1)*255).astype(np.uint8)):
        raise OSError(f"Could not save {path}")


def test(args, geom, cfg, state):
    if state:
        saved = state["config"]
        if any(cfg[k] != saved[k] for k in ("rop_alignment_sha256", "rop_coefficient_sha256")):
            raise ValueError("RopStitch checkpoints differ from training")
    model = PixelResidual(args.pixel, args.pixel_weights if not state else None,
                          cfg["max_shift"], cfg["iterations"]).cuda()
    if state:
        model.load_state_dict(state["model"])
    pairs = discover(args.data, args.list)
    args.out.mkdir(parents=True)
    log_line(args.out / "plan.jsonl", dict(config=cfg, guard=not args.no_guard,
             source=str(args.checkpoint or args.pixel_weights), pairs=len(pairs)))
    valid = []
    for i, pair in enumerate(pairs, 1):
        try:
            images, masks, row = inference_metrics(pair, geom, model, args.min_retention,
                                                   not args.no_guard)
            if not args.metrics_only:
                directory = args.out / pair[0]
                directory.mkdir()
                colored = (images * masks).permute(0, 2, 3, 1).cpu().numpy()
                mask = masks[:, 0].cpu().numpy()
                fused = colored.sum(axis=0) / np.maximum(mask.sum(axis=0)[..., None], 1)
                image_file(directory / "blend.png", fused)
                for view in (0, 1):
                    image_file(directory / f"warp{view+1}.png", colored[view])
                    image_file(directory / f"mask{view+1}.png", mask[view])
            valid.append(row)
        except InvalidPair as exc:
            row = dict(id=pair[0], error=str(exc))
        log_line(args.out / "pairs.jsonl", row)
        if i == 1 or i % 25 == 0 or i == len(pairs):
            print(f"tested {i}/{len(pairs)}; valid {len(valid)}", flush=True)
    changes = [row["refined_photo"] - row["baseline_photo"] for row in valid
               if row["baseline_photo"] is not None]
    result = dict(requested=len(pairs), valid=len(valid), failed=len(pairs)-len(valid),
                  fallback=sum(row["scale"] == 0 for row in valid),
                  mean_fixed_overlap_photo_change=float(np.mean(changes)) if changes else None)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not valid:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "test"):
        p = subcommands.add_parser(name)
        for flag in ("rop", "pixel", "data", "out"):
            p.add_argument(f"--{flag}", type=Path, required=True)
        for flag in ("alignment", "coefficient", "pixel-weights", "list"):
            p.add_argument(f"--{flag}", type=Path)
        p.add_argument("--gpu", type=int, choices=(0, 1), default=1)
        p.add_argument("--max-edge", type=int)
        p.add_argument("--max-shift", type=float)
        p.add_argument("--iters", type=int)
        p.add_argument("--min-retention", type=float, default=0.995)
        if name == "train":
            p.add_argument("--epochs", type=int, default=10)
            p.add_argument("--lr", type=float, default=1e-5)
            p.add_argument("--seed", type=int, default=2026)
            p.add_argument("--val-data", type=Path)
            p.add_argument("--val-list", type=Path)
            p.add_argument("--resume", type=Path)
        else:
            p.add_argument("--checkpoint", type=Path)
            p.add_argument("--no-guard", action="store_true")
            p.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.out.exists():
        parser.error("--out must be a new directory")
    if args.command == "train" and args.val_list and not args.val_data:
        parser.error("--val-list requires --val-data")
    previous = checkpoint(args.resume) if args.command == "train" and args.resume else (
        checkpoint(args.checkpoint) if args.command == "test" and args.checkpoint else None
    )
    default = previous["config"] if previous else dict(edge=512, max_shift=16.0, iterations=4)
    args.max_edge = default["edge"] if args.max_edge is None else args.max_edge
    args.max_shift = default["max_shift"] if args.max_shift is None else args.max_shift
    args.iters = default["iterations"] if args.iters is None else args.iters
    if not (64 <= args.max_edge <= 1200 and 0 < args.max_shift <= 64 and 1 <= args.iters <= 12
            and 0 < args.min_retention <= 1):
        parser.error("Invalid max edge, shift, iteration count or retention")
    if args.command == "train" and (args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("Invalid training duration or learning rate")
    args.alignment = args.alignment or args.rop / "woCoefNet/model_homo/epoch100_model.pth"
    args.coefficient = args.coefficient or args.rop / "wCoefNet/model_coef/epoch050_coefmodel.pth"
    args.pixel_weights = args.pixel_weights or args.pixel / "checkpoints/ckpt.pth"
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable on requested physical GPU {args.gpu}")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(getattr(args, "seed", 2026))
    torch.backends.cudnn.benchmark = False
    print(f"GPU {args.gpu}: {torch.cuda.get_device_name(0)}", flush=True)
    start = time.monotonic()
    cfg = options(args)
    geom = RopTPS(args.rop, args.alignment, args.coefficient, "cuda:0", args.max_edge)
    if args.command == "train":
        train(args, geom, cfg)
    else:
        test(args, geom, cfg, previous)
    print(f"Finished in {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
