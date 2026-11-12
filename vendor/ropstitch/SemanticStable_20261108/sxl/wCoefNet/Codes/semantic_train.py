"""Offline, from-scratch training and evaluation of SemanticStitchModel."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random


FIELDS = ("image1", "image2", "sam1", "sam2")


def load_manifest(path):
    samples = []
    pair_ids, pairs = set(), set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            pair_id = row["id"]
            if not isinstance(pair_id, str) or not pair_id:
                raise ValueError("id must be a nonempty string")
            files = tuple((path.parent / row[k]).resolve(strict=True) for k in FIELDS)
            pair = frozenset(files[:2])
            if (pair_id in pair_ids or pair in pairs or len(pair) != 2
                    or files[2] == files[3] or not all(f.is_file() for f in files)):
                raise ValueError("duplicate id/pair, repeated view or non-file input")
            pair_ids.add(pair_id)
            pairs.add(pair)
            samples.append((pair_id, files))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if not samples:
        raise ValueError(f"Empty manifest: {path}")
    return samples


def image_paths(rows):
    return sorted({str(path) for _, paths in rows for path in paths[:2]})


def make_manifest(args):
    def index(folder, extensions):
        files = {}
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                if path.stem in files:
                    raise ValueError(f"Duplicate stem {path.stem} in {folder}")
                files[path.stem] = str(path.resolve())
        return files

    extensions = {".jpg", ".jpeg", ".png"}
    sources = (
        index(args.image_root / "input1", extensions),
        index(args.image_root / "input2", extensions),
        index(args.sam_root / "sam_mask1", {".npz"}),
        index(args.sam_root / "sam_mask2", {".npz"}),
    )
    all_stems = set.union(*(set(source) for source in sources))
    if not all_stems or any(set(source) != all_stems for source in sources):
        missing = {name: len(all_stems - set(source)) for name, source in zip(FIELDS, sources)}
        raise ValueError(f"Four directories must have the same nonempty stem set. Missing counts: {missing}")
    with args.output.open("x", encoding="utf-8") as stream:
        for stem in sorted(all_stems):
            row = {name: source[stem] for name, source in zip(FIELDS, sources)}
            stream.write(json.dumps(dict(id=stem, **row), ensure_ascii=False) + "\n")
    print(f"{len(all_stems)} pairs: {args.output}")


def load_pair(row, device):
    import torch.nn.functional as F
    from semantic_stitch import read_image, read_sam

    _, (image1, image2, sam1, sam2) = row
    images = tuple(read_image(path, device) for path in (image1, image2))
    if images[0].shape != images[1].shape:
        raise ValueError(f"Pair must have equal image dimensions: {row[0]}")
    ids, foreground = zip(*(read_sam(path, device) for path in (sam1, sam2)))
    for label in ids:
        if label.shape[-2:] not in (images[0].shape[-2:], (512, 512)):
            raise ValueError(f"SAM dimensions do not match: {row[0]}")
    canonical = tuple(F.interpolate(im, (512, 512), mode="bilinear", align_corners=False, antialias=True)
                      for im in images)
    labels = tuple(F.interpolate(m.float(), (512, 512), mode="nearest-exact").long() for m in ids)
    masks = tuple(F.interpolate(m, (512, 512), mode="nearest-exact") for m in foreground)
    return canonical, labels, masks, images, foreground


def synthetic_translation(images, labels, masks, rng):
    import torch.nn.functional as F

    width = 384
    x1, y1, x2, y2 = (rng.randrange(113) for _ in range(4))
    positions = ((x1, y1), (x2, y2))

    def crops(tensor, mode):
        return tuple(F.interpolate(tensor[..., y:y + width, x:x + width],
                                   (512, 512), mode=mode) for x, y in positions)

    shift = images[0].new_tensor([(x1 - x2) / width, (y1 - y2) / width])
    return (crops(images[0], "bilinear"), tuple(x.long() for x in crops(labels[0].float(), "nearest-exact")),
            crops(masks[0], "nearest-exact"), shift)


def synthetic_projective(images, labels, masks, rng):
    import torch
    from semantic_model import corners_like
    from semantic_refine import plane_grid, project, sample
    from utils.torch_DLT import Tensor_ACA_rect

    image = images[0]
    corners = corners_like(image)
    first = image.new_tensor([rng.uniform(48, 112), rng.uniform(48, 112)])
    second = image.new_tensor([rng.uniform(48, 112), rng.uniform(48, 112)])
    jitter = image.new_tensor([[[rng.uniform(-16, 16), rng.uniform(-16, 16)] for _ in range(4)]])
    h1 = Tensor_ACA_rect(corners, corners * (352 / 512) + first)
    h2 = Tensor_ACA_rect(corners, corners * (352 / 512) + second + jitter)
    grid = plane_grid((image.new_zeros(2), image.new_ones(2)), (512, 512))
    maps = [project(h, grid) for h in (h1, h2)]
    return (tuple(sample(image, xy) for xy in maps),
            tuple(sample(labels[0].float(), xy, "nearest").long() for xy in maps),
            tuple(sample(masks[0], xy, "nearest") for xy in maps),
            torch.linalg.solve(h2, h1))


def phase_at(epoch, config):
    if epoch < config["alignment_epochs"]:
        return "alignment"
    if epoch < config["alignment_epochs"] + config["plane_epochs"]:
        return "plane"
    return "joint"


def metric_values(metrics):
    values = {key: float(value.detach()) for key, value in metrics.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite loss or geometry metric")
    return values


def append_json(path, item):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")


def evaluate(model, rows, device, config, phase, output=None, save_images=0):
    import torch
    from semantic_refine import (composition_weight_on_canvas, image_and_support, inverse_tps,
                                 output_size, plane_grid, sample, seam_blend)
    from semantic_stitch import write_png

    model.eval()
    totals, failures = {}, 0
    saved = 0
    with torch.no_grad():
        for pair_id, files in rows:
            record = dict(id=pair_id)
            try:
                images, ids, masks, native, native_masks = load_pair((pair_id, files), device)
                meshes, source, coef, metrics = model(
                    images, ids, masks, phase=phase, alpha=0.5, resolution=config["resolution"],
                    stable=config.get("stable", False),
                )
                record.update(metrics=metric_values(metrics), plane_coefficients=coef.flatten().tolist())
                if min(record["metrics"]["jacobian"], record["metrics"]["inverse_jacobian"]) <= 0:
                    raise ValueError("Sampled mesh or inverse TPS fold")
                if record["metrics"]["overlap"] < 0.01:
                    raise ValueError("Sampled overlap is below 1%")
                if output is not None and saved < save_images:
                    controls = torch.cat([m.flatten(1, 2) for m in meshes], 1)
                    bounds = (controls.amin((0, 1)), controls.amax((0, 1)))
                    size = output_size(bounds, native[0].shape[-2:], 2_000_000)
                    grid = plane_grid(bounds, size)
                    inverse = [inverse_tps(mesh, source, grid) for mesh in meshes]
                    warped, valid = zip(*(image_and_support(im, xy) for im, xy in zip(native, inverse)))
                    fg = [sample(mask, xy) for mask, xy in zip(native_masks, inverse)]
                    weight = None
                    if config.get("stable", False):
                        weight = composition_weight_on_canvas(
                            images, masks, meshes, source, grid, model.composition, config["resolution"],
                        )
                    stitched, _ = seam_blend(warped, valid, fg, inverse, model.composition, weight)
                    if not torch.isfinite(stitched).all():
                        raise ValueError("Non-finite render")
                    name = f"preview_{saved + 1:04d}.png"
                    write_png(output / name, stitched, rgb_values=True)
                    record["preview"] = name
                    saved += 1
                for key, value in record["metrics"].items():
                    totals[key] = totals.get(key, 0.0) + value
            except (OSError, ValueError, KeyError, torch.linalg.LinAlgError) as exc:
                record["failure"] = str(exc)
                failures += 1
            if output is not None:
                append_json(output / "pairs.jsonl", record)
    valid = len(rows) - failures
    return dict(total=len(rows), valid=valid, failures=failures,
                means={key: value / valid for key, value in totals.items()} if valid else {})


def save_checkpoint(path, state):
    import torch

    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train(args, device, train_rows, val_rows):
    import torch
    from semantic_model import SemanticStitchModel

    config = {key: getattr(args, key) for key in (
        "alignment_epochs", "plane_epochs", "joint_epochs", "resolution", "seed", "lr",
    )}
    if args.stable:
        config["stable"] = True
    fingerprint = tuple(hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.train_list, args.val_list))
    used = image_paths(train_rows + val_rows)
    torch.manual_seed(config["seed"])
    model = SemanticStitchModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    start_epoch, best = 0, None
    if args.resume:
        state = torch.load(args.output / "last.pth", map_location="cpu", weights_only=True)
        if state["config"] != config or state["manifest_sha256"] != fingerprint or state["images"] != used:
            raise ValueError("Resume requires the same data paths, manifests and training options.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = state["epoch"], state["best_score"]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    epochs = config["alignment_epochs"] + config["plane_epochs"] + config["joint_epochs"]
    for epoch in range(start_epoch, epochs):
        phase = phase_at(epoch, config)
        torch.manual_seed(config["seed"] + epoch)
        rng = random.Random(config["seed"] + epoch)
        rows = train_rows.copy()
        rng.shuffle(rows)
        model.configure_training(phase)
        totals = {}
        print(f"starting epoch {epoch + 1}/{epochs} {phase}: {len(rows)} training pairs", flush=True)
        for index, row in enumerate(rows):
            try:
                images, ids, masks, _, _ = load_pair(row, device)
                shift, target_h = None, None
                if phase == "alignment" and index % 2 == 0:
                    if args.stable and index % 4 == 0:
                        images, ids, masks, target_h = synthetic_projective(images, ids, masks, rng)
                    else:
                        images, ids, masks, shift = synthetic_translation(images, ids, masks, rng)
                alpha = rng.uniform(0.15, 0.85)
                optimizer.zero_grad(set_to_none=True)
                _, _, _, metrics = model(images, ids, masks, phase=phase, alpha=alpha,
                                         resolution=config["resolution"], shift=shift,
                                         target_h=target_h, stable=args.stable)
                for key, value in metric_values(metrics).items():
                    totals[key] = totals.get(key, 0.0) + value
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3, error_if_nonfinite=True)
                optimizer.step()
                completed = index + 1
                if args.log_every and (completed == 1 or completed % args.log_every == 0):
                    print(f"epoch {epoch + 1}/{epochs} {phase} pair {completed}/{len(rows)} "
                          f"mean_loss={totals['loss'] / completed:.5f}", flush=True)
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                append_json(args.output / "failures.jsonl", dict(epoch=epoch + 1, id=row[0], reason=str(exc)))
                raise RuntimeError(f"Training failed at epoch {epoch + 1}, pair {row[0]}") from exc
        validation = evaluate(model, val_rows, device, config, phase)
        means = validation["means"]
        score = (means["alignment"] + 0.1 * means["shape"] + validation["failures"] / len(val_rows)
                 + 0.1 * means["ghost"] + 0.5 * means["seam"]
                 + 0.5 * max(0, 0.12 - means["overlap"])
                 + 0.5 * max(0, 0.29 - min(means["coverage_left"], means["coverage_right"]))
                 if means else None)
        improved = phase == "joint" and score is not None and (best is None or score < best)
        if improved:
            best = score
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                     epoch=epoch + 1, phase=phase, best_score=best, manifest_sha256=fingerprint, images=used)
        if improved:
            save_checkpoint(args.output / "best.pth", state)
        save_checkpoint(args.output / "last.pth", state)
        result = dict(epoch=epoch + 1, phase=phase, train={k: v / len(rows) for k, v in totals.items()},
                      validation=validation, selection_score=score)
        append_json(args.output / "epochs.jsonl", result)
        print(f"epoch {epoch + 1}/{epochs} {phase} loss={result['train']['loss']:.5f} "
              f"validation={score} failures={validation['failures']}", flush=True)


def test(args, device, rows):
    import torch
    from semantic_model import SemanticStitchModel

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["phase"] != "joint":
        raise ValueError("Test requires a checkpoint from the joint phase.")
    if set(state["images"]) & set(image_paths(rows)):
        raise ValueError("Test manifest shares image paths with training or validation.")
    model = SemanticStitchModel().to(device)
    model.load_state_dict(state["model"])
    args.output.mkdir(parents=True, exist_ok=False)
    results = evaluate(model, rows, device, state["config"], "joint", args.output, args.save_images)
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    if results["valid"] == 0:
        raise SystemExit("All test pairs failed. See pairs.jsonl.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--image-root", type=Path, required=True)
    manifest.add_argument("--sam-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("train")
    testing = commands.add_parser("test")
    for command in (training, testing):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1"), default="cpu")
        command.add_argument("--cpu-threads", type=int, default=4)
    training.add_argument("--train-list", type=Path, required=True)
    training.add_argument("--val-list", type=Path, required=True)
    training.add_argument("--alignment-epochs", type=int, default=10)
    training.add_argument("--plane-epochs", type=int, default=5)
    training.add_argument("--joint-epochs", type=int, default=10)
    training.add_argument("--resolution", type=int, default=128)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--log-every", type=int, default=256,
                          help="Log the first and every N optimizer steps; 0 disables step logging.")
    training.add_argument("--stable", action="store_true",
                          help="Projective warmup, edge alignment and separate geometry/seam gradients.")
    training.add_argument("--resume", action="store_true")
    testing.add_argument("--test-list", type=Path, required=True)
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--save-images", type=int, default=0)
    args = parser.parse_args()
    if args.command == "manifest":
        make_manifest(args)
        return
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive.")
    if args.command == "train":
        if args.log_every < 0:
            parser.error("--log-every must be nonnegative.")
        if (min(args.alignment_epochs, args.plane_epochs, args.joint_epochs) < 1
                or not 64 <= args.resolution <= 256 or not 0 < args.lr <= 0.01):
            parser.error("Each phase needs at least one epoch, resolution 64–256 and LR in (0, 0.01].")
        train_rows, val_rows = load_manifest(args.train_list), load_manifest(args.val_list)
        if set(image_paths(train_rows)) & set(image_paths(val_rows)):
            parser.error("Training and validation share image paths.")
    else:
        if args.save_images < 0:
            parser.error("--save-images must be nonnegative.")
        test_rows = load_manifest(args.test_list)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else args.device[-1]
    import torch

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu" if args.device == "cpu" else "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA device {args.device} is not available.")
    if args.command == "train":
        train(args, device, train_rows, val_rows)
    else:
        test(args, device, test_rows)


if __name__ == "__main__":
    main()
