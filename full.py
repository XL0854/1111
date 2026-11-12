"""Staged training, joint fine-tuning and testing from paired original images."""

import argparse
import json
import math
import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from full_model import FullModel, PHASES, synthetic_pair, training_objective
from model import InvalidPair
from run import (discover, disjoint_train_val, file_hash, image_file, inference_metrics,
                 load_pair, log_line)


FORMAT = "ropstitch-pixel-full-v2"
LEGACY_FORMAT = "ropstitch-pixel-full-v1"
MODEL_KEYS = ("max_edge", "max_shift", "iters", "homo_bound", "tps_bound", "plane_mode",
              "alignment_adapter", "fusion_mode", "fusion_width")
DEFAULTS = dict(max_edge=512, max_shift=16.0, iters=4, homo_bound=160.0, tps_bound=96.0,
                plane_mode="learned", alignment_adapter="msda", fusion_mode="seam",
                fusion_width=24)
PHASE_LR = dict(geometry=1e-4, plane=1e-4, residual=1e-5, fusion=1e-4, joint=3e-6)


def distributed():
    return dist.is_available() and dist.is_initialized()


def primary():
    return not distributed() or dist.get_rank() == 0


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def schedule(value):
    try:
        stages = [(name, int(count)) for name, count in (x.split(":") for x in value.split(","))]
        indices = [PHASES.index(name) for name, _ in stages]
        if sorted(set(indices)) != indices or any(count < 1 for _, count in stages):
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use ordered positive phase counts: geometry:10,plane:5,residual:5,joint:5"
        ) from exc
    return [name for name, count in stages for _ in range(count)]


def pair_fingerprint(pairs):
    digest = []
    for name, a, b in pairs:
        digest.append([name, file_hash(a), file_hash(b)])
    return digest


def save(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") not in (FORMAT, LEGACY_FORMAT):
        raise ValueError("Expected a full-pipeline checkpoint created by full.py")
    state["config"]["model"].setdefault("plane_mode", "learned")
    if state["format"] == LEGACY_FORMAT:
        state["config"]["model"].setdefault("alignment_adapter", "none")
        state["config"]["model"].setdefault("fusion_mode", "mean")
        state["config"]["model"].setdefault("fusion_width", 24)
    return state


def phase_optimizer(model, phase, multiplier):
    unwrap(model).set_phase(phase)
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(parameters, lr=PHASE_LR[phase]*multiplier, weight_decay=1e-4), parameters


def gradient_groups(model):
    model = unwrap(model)
    result = {}
    for name in ("geometry", "alignment_adapter", "coef", "flow", "fusion"):
        module = getattr(model, name)
        grads = [p.grad.detach().norm().square() for p in (() if module is None else module.parameters())
                 if p.grad is not None]
        result[name] = float(torch.stack(grads).sum().sqrt()) if grads else 0.0
    return result


def train(args, previous, base_state=None):
    rank = dist.get_rank() if distributed() else 0
    world = dist.get_world_size() if distributed() else 1
    pairs = discover(args.data, args.list)
    if len(pairs) < world:
        raise ValueError(f"Distributed training needs at least {world} image pairs")
    validation = discover(args.val_data, args.val_list) if args.val_data else []
    if validation:
        disjoint_train_val(pairs, validation)
    model_cfg = {key: getattr(args, key) for key in MODEL_KEYS}
    base_hash = (previous["config"].get("base_checkpoint_sha256") if previous else
                 file_hash(args.base_checkpoint) if args.base_checkpoint else None)
    configuration = dict(model=model_cfg, seed=args.seed, lr_multiplier=args.lr_multiplier,
                         train=pair_fingerprint(pairs), val=pair_fingerprint(validation),
                         base_checkpoint_sha256=base_hash)
    start, best = 0, None
    if previous:
        if configuration != previous["config"]:
            raise ValueError("Resume needs identical dataset, model options, seed and LR multiplier")
        start = previous["epoch"]
        if args.schedule[:start] != previous["schedule"][:start]:
            raise ValueError("Completed schedule prefix differs from the checkpoint")
        best = previous["best"]
    if len(args.schedule) <= start:
        raise ValueError("Training schedule has no epochs after the checkpoint")
    init = previous["init"] if previous else ("base" if base_state is not None else args.init)
    if init == "scratch" and not previous and args.schedule[0] != "geometry":
        raise ValueError("Scratch training must start with the geometry phase")
    model = FullModel(args, "checkpoint" if previous or base_state is not None else init).cuda()
    if previous:
        model.load_state_dict(previous["model"])
        torch.set_rng_state(previous["torch_rng"])
        torch.cuda.set_rng_state(previous["cuda_rng"])
    elif base_state is not None:
        incompatible = model.load_state_dict(base_state["model"], strict=False)
        allowed = ("alignment_adapter.", "fusion.")
        if incompatible.unexpected_keys or any(
                not key.startswith(allowed) for key in incompatible.missing_keys):
            raise ValueError(f"Incompatible base checkpoint: {incompatible}")
    if distributed():
        local_rank = int(os.environ["LOCAL_RANK"])
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                        broadcast_buffers=False, find_unused_parameters=True)
    init_keys = ("alignment", "pixel_weights") if args.plane_mode == "fixed-half" else (
        "alignment", "coefficient", "pixel_weights"
    )
    init_hashes = previous["init_hashes"] if previous else (
        {k: file_hash(getattr(args, k)) for k in init_keys}
        if init == "pretrained" else {}
    )
    if primary():
        args.out.mkdir(parents=True)
        log_line(args.out / "plan.jsonl", dict(config=configuration, schedule=args.schedule,
                 init=init, init_hashes=init_hashes, train_pairs=len(pairs),
                 validation_pairs=len(validation), world_size=world))
    if distributed():
        dist.barrier()
    old_phase = None
    for offset in range(start, len(args.schedule)):
        epoch, phase = offset + 1, args.schedule[offset]
        if old_phase != phase:
            model.zero_grad(set_to_none=True)
            optimizer, parameters = phase_optimizer(model, phase, args.lr_multiplier)
            if previous and offset == start and phase == previous["phase"]:
                optimizer.load_state_dict(previous["optimizer"])
            old_phase = phase
        order = list(pairs)
        random.Random(args.seed + epoch).shuffle(order)
        local_order = order[rank::world]
        steps = math.ceil(len(order) / world)
        while len(local_order) < steps:
            local_order.append(order[(rank + len(local_order) * world) % len(order)])
        values = []
        local_skipped = 0
        step_log = args.out / ("steps.jsonl" if world == 1 else f"steps_rank{rank}.jsonl")
        skip_log = args.out / ("skipped.jsonl" if world == 1 else f"skipped_rank{rank}.jsonl")
        for index, pair in enumerate(local_order, 1):
            optimizer.zero_grad(set_to_none=True)
            target = None
            failure = None
            try:
                images = load_pair(pair)
                if phase == "geometry" and init == "scratch" and (index + epoch + rank) % 2 == 0:
                    synthetic_seed = args.seed + epoch*100_003 + rank*1_000_003 + index
                    images, target = synthetic_pair(images, synthetic_seed)
                loss, terms = model(images, phase, target)
            except (InvalidPair, torch.linalg.LinAlgError) as exc:
                failure = str(exc)
            if distributed():
                usable = torch.tensor([failure is None], device="cuda", dtype=torch.int32)
                dist.all_reduce(usable, op=dist.ReduceOp.MIN)
                if not bool(usable.item()):
                    log_line(skip_log, dict(epoch=epoch, phase=phase, id=pair[0],
                                            reason=failure or "peer rank had an invalid pair"))
                    local_skipped += 1
                    continue
            elif failure is not None:
                log_line(skip_log, dict(epoch=epoch, phase=phase, id=pair[0], reason=failure))
                local_skipped += 1
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss during {phase}: {pair[0]}")
            loss.backward()
            groups = gradient_groups(model) if not values else None
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            result = dict(epoch=epoch, phase=phase, id=pair[0], step=index, loss=float(loss.detach()),
                          synthetic=target is not None, grad_norm=float(norm))
            if groups is not None:
                result["gradient_groups"] = groups
            result.update({key: float(value.detach()) for key, value in terms.items()})
            values.append(result["loss"])
            log_line(step_log, result)
            if primary() and (index == 1 or index % 25 == 0 or index == len(local_order)):
                print(f"{phase} epoch={epoch} local_pair={index}/{len(local_order)} "
                      f"world={world} loss={result['loss']:.6f} "
                      f"grad={float(norm):.6f} groups={groups}", flush=True)
        totals = torch.tensor([sum(values), len(values), local_skipped], device="cuda",
                              dtype=torch.float64)
        if distributed():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, used, skipped = totals.tolist()
        if not used:
            raise ValueError(f"No valid training pairs in epoch {epoch} ({phase})")

        val_values = []
        if primary() and validation and phase in ("residual", "fusion", "joint"):
            evaluation_model = unwrap(model)
            for pair in validation:
                try:
                    _, _, row = inference_metrics(
                        pair, lambda images: evaluation_model.build(images, "test")[0],
                        evaluation_model.flow, args.min_retention, False,
                    )
                    if row["refined_photo"] is not None:
                        val_values.append(row["refined_photo"])
                except InvalidPair as exc:
                    row = dict(id=pair[0], error=str(exc))
                log_line(args.out / "validation.jsonl", dict(epoch=epoch, **row))
            if not val_values:
                raise ValueError("No usable validation pairs")
        if primary():
            score = float(np.mean(val_values)) if val_values else None
            improved = score is not None and (best is None or score < best)
            if improved:
                best = score
            state = dict(format=FORMAT, model=unwrap(model).state_dict(), config=configuration,
                         epoch=epoch, phase=phase, schedule=args.schedule, best=best, init=init,
                         init_hashes=init_hashes, optimizer=optimizer.state_dict(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
            save(args.out / "last.pt", state)
            if improved:
                save(args.out / "best.pt", state)
            row = dict(epoch=epoch, phase=phase, used=int(used), skipped=int(skipped),
                       train_loss=float(total_loss / used), validation_photo=score,
                       world_size=world)
            log_line(args.out / "epochs.jsonl", row)
            print(json.dumps(row), flush=True)
        if distributed():
            dist.barrier()


@torch.no_grad()
def test(args, state):
    model = FullModel(args, "checkpoint").cuda().eval()
    model.load_state_dict(state["model"])
    pairs = discover(args.data, args.list)
    args.out.mkdir(parents=True)
    log_line(args.out / "plan.jsonl", dict(checkpoint=str(args.checkpoint),
             checkpoint_sha256=file_hash(args.checkpoint), guard=not args.no_guard,
             options={key: getattr(args, key) for key in MODEL_KEYS}))
    passed = []
    for i, pair in enumerate(pairs, 1):
        try:
            images, masks, row = inference_metrics(
                pair, lambda im: model.build(im, "test")[0], model.flow,
                args.min_retention, not args.no_guard,
            )
            if not args.metrics_only:
                path = args.out / pair[0]
                path.mkdir()
                warps = (images * masks).permute(0, 2, 3, 1).cpu().numpy()
                mask = masks[:, 0].cpu().numpy()
                fused, probability, _ = model.fuse(images, masks.float())
                image_file(path / "blend.png", fused[0].permute(1, 2, 0).cpu().numpy())
                if probability is not None:
                    image_file(path / "seam_probability.png",
                               probability[0, 0].cpu().numpy())
                for view in (0, 1):
                    image_file(path / f"warp{view+1}.png", warps[view])
                    image_file(path / f"mask{view+1}.png", mask[view])
            passed.append(row)
        except (InvalidPair, torch.linalg.LinAlgError) as exc:
            row = dict(id=pair[0], error=str(exc))
        log_line(args.out / "pairs.jsonl", row)
        if i == 1 or i % 25 == 0 or i == len(pairs):
            print(f"Tested {i}/{len(pairs)}; valid {len(passed)}", flush=True)
    changes = [r["refined_photo"]-r["baseline_photo"] for r in passed if r["baseline_photo"] is not None]
    summary = dict(requested=len(pairs), valid=len(passed), failed=len(pairs)-len(passed),
                   fallback=sum(r["scale"] == 0 for r in passed),
                   mean_fixed_overlap_photo_change=float(np.mean(changes)) if changes else None)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "test"):
        p = commands.add_parser(name)
        for flag in ("rop", "pixel", "data", "out"):
            p.add_argument(f"--{flag}", type=Path, required=True)
        p.add_argument("--list", type=Path)
        p.add_argument("--gpu", type=int, choices=(0, 1), default=1)
        for key in ("max_edge", "max_shift", "iters", "homo_bound", "tps_bound", "fusion_width"):
            p.add_argument("--"+key.replace("_", "-"), type=type(DEFAULTS[key]))
        p.add_argument("--plane-mode", choices=("learned", "fixed-half"))
        p.add_argument("--alignment-adapter", choices=("none", "msda"))
        p.add_argument("--fusion-mode", choices=("mean", "seam"))
        p.add_argument("--min-retention", type=float, default=0.995)
        if name == "train":
            p.add_argument("--init", choices=("scratch", "pretrained"), default="scratch")
            p.add_argument("--schedule", type=schedule)
            p.add_argument("--seed", type=int, default=2026)
            p.add_argument("--lr-multiplier", type=float, default=1.0)
            for key in ("alignment", "coefficient", "pixel_weights", "resume", "base_checkpoint",
                        "val_data", "val_list"):
                p.add_argument("--"+key.replace("_", "-"), type=Path)
        else:
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--no-guard", action="store_true")
            p.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if args.command != "train":
            parser.error("Multi-GPU mode is supported for training only")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.out.exists():
        parser.error("--out must be a new directory")
    if args.command == "train":
        if args.val_list and not args.val_data:
            parser.error("--val-list requires --val-data")
        if not math.isfinite(args.lr_multiplier) or args.lr_multiplier <= 0:
            parser.error("Invalid learning-rate multiplier")
        if args.resume and args.base_checkpoint:
            parser.error("Use either --resume or --base-checkpoint, not both")
        previous = read_checkpoint(args.resume) if args.resume else None
        base_state = read_checkpoint(args.base_checkpoint) if args.base_checkpoint else None
        args.alignment = args.alignment or args.rop / "woCoefNet/model_homo/epoch100_model.pth"
        args.coefficient = args.coefficient or args.rop / "wCoefNet/model_coef/epoch050_coefmodel.pth"
        args.pixel_weights = args.pixel_weights or args.pixel / "checkpoints/ckpt.pth"
    else:
        previous = read_checkpoint(args.checkpoint)
        base_state = None
    for key in MODEL_KEYS:
        if getattr(args, key) is None:
            setattr(args, key, previous["config"]["model"][key] if previous else DEFAULTS[key])
    if previous and args.plane_mode != previous["config"]["model"]["plane_mode"]:
        parser.error("The plane mode must match the checkpoint")
    if args.command == "train":
        if args.schedule is None:
            if args.base_checkpoint:
                args.schedule = schedule("geometry:10,residual:10,fusion:20")
            else:
                args.schedule = schedule("geometry:100,residual:50,fusion:20"
                                         if args.plane_mode == "fixed-half"
                                         else "geometry:10,plane:5,residual:5,fusion:5,joint:5")
        if args.plane_mode == "fixed-half" and "plane" in args.schedule:
            parser.error("Fixed-half mode has no learnable plane phase")
        if "fusion" in args.schedule and args.fusion_mode != "seam":
            parser.error("The fusion phase requires --fusion-mode seam")
    if not (64 <= args.max_edge <= 1200 and 0 < args.max_shift <= 64 and 1 <= args.iters <= 12
            and 0 < args.homo_bound <= 224 and 0 < args.tps_bound <= 160
            and 0 < args.min_retention <= 1 and 8 <= args.fusion_width <= 64
            and args.fusion_width % 4 == 0):
        parser.error("Invalid model bounds or retention")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable on requested GPU{args.gpu}")
    torch.manual_seed(getattr(args, "seed", 2026))
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    try:
        if args.command == "train":
            train(args, previous, base_state)
        else:
            test(args, previous)
    finally:
        if distributed():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
