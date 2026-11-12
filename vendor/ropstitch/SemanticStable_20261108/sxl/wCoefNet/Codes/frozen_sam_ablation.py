"""Paired SAM ablation on one manifest using frozen official RopStitch."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

MODES = ("ropstitch", "neutral_equal", "sam_equal", "sam_adaptive")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def run(args):
    import numpy as np
    import torch
    from evaluate_stitch_quality import rop_overlap_metrics
    from network import CoefNetwork, Network, get_rigid_mesh
    from semantic_model import corners_like
    from semantic_refine import (image_and_support, inverse_tps, mesh_geometry, output_size,
                                 plane_grid, refine, sample)
    from semantic_stitch import load_checkpoint
    from semantic_train import load_manifest, load_pair
    from utils.torch_DLT import Tensor_ACA_rect

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise ValueError(f"Physical GPU {args.gpu} is unavailable.")
    torch.cuda.set_per_process_memory_fraction(0.18)
    device = torch.device("cuda:0")
    align = load_checkpoint(Network(pretrained=False), args.alignment, device, "RopStitch")
    coef = load_checkpoint(CoefNetwork(), args.coef, device, "RopStitch CoefNet")
    rows = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    configuration = dict(
        manifest_sha256=digest(args.manifest), alignment_sha256=digest(args.alignment),
        coef_sha256=digest(args.coef), alpha=0.5, steps=args.steps, resolution=args.resolution,
        max_shift_px=args.max_shift, paired_overlap="minimum support across all four variants",
        comparison_order=list(MODES), bootstrap_seed=42, bootstrap_draws=5000,
    )
    (args.output / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    print(f"Ready: frozen checkpoints, {len(rows)} validation pairs", flush=True)

    def evaluate_case(row):
        with torch.no_grad():
            images, ids, foreground, native, _ = load_pair(row, device)
            left, right, _, mesh1, mesh2 = align(*images, coef, alpha=0.5)
            corners = corners_like(mesh1)
            homographies = tuple(Tensor_ACA_rect(corners, x[-1]) for x in (left, right))
            source = get_rigid_mesh(1, 512, 512) / 512
            meshes = (mesh1 / 512, mesh2 / 512)
            if any(not torch.isfinite(m).all() for m in meshes):
                raise ValueError("Non-finite official mesh.")
            nodes = torch.cat([m.flatten(1, 2) for m in meshes], 1)
            padding = args.max_shift / 512
            bounds = (nodes.amin((0, 1)) - padding, nodes.amax((0, 1)) + padding)
            grid = plane_grid(bounds, output_size(bounds, native[0].shape[-2:], 2_000_000))
        results = {"ropstitch": meshes}
        diagnostics = {"ropstitch": {"selected_step": 0, "stop_reason": "fixed"}}
        for name, mode in (("neutral_equal", "equal"), ("sam_equal", "equal"), ("sam_adaptive", "semantic")):
            labels = tuple(torch.zeros_like(x) for x in ids) if name == "neutral_equal" else ids
            masks = tuple(torch.zeros_like(x) for x in foreground) if name == "neutral_equal" else foreground
            try:
                selected, _, _, matches, _, _, info = refine(
                    images, labels, masks, homographies, meshes, source,
                    mode=mode, steps=args.steps, max_shift_px=args.max_shift, resolution=args.resolution,
                )
                results[name], diagnostics[name] = selected, {**info, "matches": len(matches)}
            except (ValueError, torch.linalg.LinAlgError) as exc:
                results[name] = meshes
                diagnostics[name] = {"selected_step": 0, "stop_reason": "fallback", "reason": str(exc)}
        with torch.no_grad():
            renders = {}
            for name, selected in results.items():
                coordinates = [inverse_tps(m, source, grid) for m in selected]
                warped, supports = zip(*(image_and_support(im, xy) for im, xy in zip(native, coordinates)))
                overlap = supports[0] * supports[1]
                union = supports[0] + supports[1] - overlap
                raw = tuple(((im + 1) * mask * 127.5)[0].permute(1, 2, 0).cpu().numpy()
                            for im, mask in zip(warped, supports))
                whole = torch.ones_like(source[:, :-1, :-1, 0])
                protected = tuple(1 + 2 * sample(fg, source, "nearest")[:, 0, :-1, :-1]
                                  for fg in foreground)
                shapes, jac = zip(*(mesh_geometry(mesh, source, whole) for mesh in selected))
                foreground_shape = sum(mesh_geometry(mesh, source, w)[0]
                                       for mesh, w in zip(selected, protected)) / 2
                renders[name] = (raw, overlap, dict(
                    own_overlap_pixels=float(overlap.sum()), coverage=float(union.mean()),
                    shape=float(sum(shapes) / 2), foreground_shape=float(foreground_shape),
                    jacobian_min=float(torch.stack(jac).amin()),
                ))
            shared = torch.stack([overlap for _, overlap, _ in renders.values()]).amin(0)
            shared_mask = shared[0].permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy().astype(np.float64)
            scores = {}
            for name, (raw, _, geometry) in renders.items():
                values = rop_overlap_metrics(*(x.astype(np.float64) for x in raw), shared_mask)
                if not all(math.isfinite(v) for v in (*values.values(), *geometry.values())):
                    raise ValueError("Non-finite score on the common overlap.")
                scores[name] = {**values, **geometry, **diagnostics[name]}
            return {"id": row[0], "modes": scores}

    started = time.monotonic()
    valid = []
    with (args.output / "pairs.jsonl").open("x", encoding="utf-8") as stream:
        for i, row in enumerate(rows, 1):
            try:
                record = evaluate_case(row)
                valid.append(record["modes"])
            except (OSError, ValueError, torch.linalg.LinAlgError) as exc:
                record = {"id": row[0], "failure": str(exc)}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            if i in (1, 8, 16) or i % 32 == 0 or i == len(rows):
                stream.flush()
                print(f"Scored {i}/{len(rows)}; complete paired rows={len(valid)}; "
                      f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    if not valid:
        raise ValueError("No valid paired cases. See pairs.jsonl.")
    metrics = ("mPSNR", "mSSIM", "own_overlap_pixels", "coverage", "shape", "foreground_shape")
    resample = np.random.default_rng(42).integers(0, len(valid), size=(5000, len(valid)))
    means = {}
    for mode in MODES:
        means[mode] = {key: float(np.mean([row[mode][key] for row in valid])) for key in metrics}
        means[mode]["changed"] = sum(row[mode]["selected_step"] > 0 for row in valid)
        means[mode]["fallback"] = sum(row[mode]["stop_reason"] == "fallback" for row in valid)
        means[mode]["nonpositive_grid"] = sum(row[mode]["jacobian_min"] <= 0 for row in valid)
    differences = {}
    for mode, control in (("neutral_equal", "ropstitch"), ("sam_adaptive", "ropstitch"),
                          ("sam_adaptive", "neutral_equal"), ("sam_equal", "neutral_equal"),
                          ("sam_adaptive", "sam_equal")):
        delta = {}
        for key in metrics:
            diff = np.array([row[mode][key] - row[control][key] for row in valid])
            ci = np.quantile(diff[resample].mean(1), [0.025, 0.975])
            delta[key] = {"mean": float(diff.mean()), "ci95": ci.tolist(), "higher": int((diff > 0).sum())}
        differences[f"{mode}_vs_{control}"] = delta
    report = dict(total=len(rows), valid=len(valid), invalid=len(rows)-len(valid),
                  configuration=configuration, means=means, paired_deltas=differences)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Completed: {args.output / 'summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "alignment", "coef", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max-shift", type=float, default=12)
    args = parser.parse_args()
    if not (1 <= args.steps <= 250 and 64 <= args.resolution <= 256 and 0 < args.max_shift <= 32):
        parser.error("Invalid refinement budget.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)


if __name__ == "__main__":
    main()
