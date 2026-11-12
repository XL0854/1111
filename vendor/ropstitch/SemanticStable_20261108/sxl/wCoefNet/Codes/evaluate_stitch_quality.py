"""Overlap metrics and geometry/composition diagnostics on a fixed manifest."""

import argparse
import json
import math
import os
from pathlib import Path


def rop_overlap_metrics(left, right, mask):
    """RopStitch's maskPSNR/maskSSIM convention, with explicit invalid cases."""
    import numpy as np
    from skimage.metrics import structural_similarity

    weight = float(mask.sum())
    if weight < 3 or min(left.shape[:2]) < 7:
        raise ValueError("Insufficient overlap or image size for mPSNR/mSSIM.")
    left, right = left * mask, right * mask
    _, ssim_map = structural_similarity(left, right, data_range=255, channel_axis=2, full=True)
    mse = float(np.square((left - right) / 255).sum()) / weight
    return dict(mPSNR=math.inf if mse == 0 else -10 * math.log10(mse),
                mSSIM=float((ssim_map * mask).sum()) / (weight + 1e-6),
                overlap_pixels=weight / 3)


def run(args):
    import numpy as np
    import torch
    from network import CoefNetwork, Network
    from semantic_model import SemanticStitchModel
    from semantic_refine import (composition_weight_on_canvas, image_and_support, inverse_tps, mesh_geometry, output_size,
                                 plane_grid, project, sample, seam_blend)
    from semantic_stitch import load_checkpoint, write_png
    from semantic_train import load_manifest, load_pair

    torch.set_num_threads(2)
    device = torch.device("cuda:0" if args.gpu is not None else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"Physical GPU {args.gpu} is unavailable.")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["phase"] != "joint":
        raise ValueError("Expected a joint-phase checkpoint.")
    model = SemanticStitchModel().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    original = load_checkpoint(Network(pretrained=False), args.rop_alignment, device, "RopStitch")
    original_coef = load_checkpoint(CoefNetwork(), args.rop_coef, device, "RopStitch CoefNet")
    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    modes = ("global_h", "base_tps", "semantic_tps", "ropstitch_fixed")
    collected = {mode: [] for mode in modes}
    args.output.mkdir(parents=True, exist_ok=False)

    def as_array(tensor):
        return tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)

    def render(mode, meshes, source, native, native_masks, homographies, preview, images, masks):
        if any(not torch.isfinite(m).all() for m in meshes):
            raise ValueError("Non-finite mesh.")
        controls = torch.cat([m.flatten(1, 2) for m in meshes], dim=1)
        bounds = (controls.amin((0, 1)), controls.amax((0, 1)))
        size = output_size(bounds, native[0].shape[-2:], 2_000_000)
        grid = plane_grid(bounds, size)
        inverse = ([project(h, grid) for h in homographies] if homographies is not None else
                   [inverse_tps(m, source, grid) for m in meshes])
        warped, support = zip(*(image_and_support(im, xy) for im, xy in zip(native, inverse)))
        if any(not torch.isfinite(x).all() for x in (*warped, *support)):
            raise ValueError("Non-finite render.")
        overlap = support[0] * support[1]
        # Original RopStitch supplies zero-padded warps prior to masking.
        # Our resampler first normalizes boundary color by support, so restore
        # the zero-padded representation before applying the same metrics.
        raw = [as_array((im + 1) * mask * 127.5) for im, mask in zip(warped, support)]
        mask = as_array(overlap.expand(-1, 3, -1, -1))
        score = rop_overlap_metrics(*raw, mask)
        weights = torch.ones_like(source[:, :-1, :-1, 0])
        shape, jac = zip(*(mesh_geometry(m, source, weights) for m in meshes))
        score.update(coverage=float((support[0] + support[1] - overlap).mean()),
                     overlap=float(overlap.mean()), shape=float(sum(shape) / 2),
                     jacobian_min=float(torch.stack(jac).amin()))
        if not all(math.isfinite(v) for v in score.values()):
            raise ValueError("Non-finite overlap or geometry metric.")
        if preview is not None:
            foreground = [sample(m, xy) for m, xy in zip(native_masks, inverse)]
            blend, _ = seam_blend(warped, support, foreground, inverse)
            write_png(preview / f"{mode}_feather.png", blend, rgb_values=True)
            if mode == "semantic_tps":
                weight = None
                if state["config"].get("stable", False):
                    weight = composition_weight_on_canvas(
                        images, masks, meshes, source, grid, model.composition, state["config"]["resolution"],
                    )
                learned, soft = seam_blend(warped, support, foreground, inverse, model.composition, weight)
                write_png(preview / "semantic_tps_learned.png", learned, rgb_values=True)
                write_png(preview / "semantic_tps_weight.png", soft, rgb_values=False)
                write_png(preview / "semantic_tps_overlap.png", overlap, rgb_values=False)
                for i, (im, mask) in enumerate(zip(warped, support), 1):
                    write_png(preview / f"semantic_tps_warp{i}.png", (im + 1) * mask - 1, rgb_values=True)
        return score

    with torch.inference_mode(), (args.output / "pairs.jsonl").open("x", encoding="utf-8") as log:
        for n, row in enumerate(rows, 1):
            images, ids, masks, native, native_masks = load_pair(row, device)
            geometry = {}
            meshes, source, _, _ = model(images, ids, masks, phase="joint", alpha=0.5,
                                         resolution=state["config"]["resolution"], geometry=geometry)
            homographies = geometry["homographies"]
            global_meshes = tuple(project(torch.linalg.inv(h), source) for h in homographies)
            *_, baseline1, baseline2 = original(*images, original_coef, alpha=0.5)
            candidates = (global_meshes, geometry["base_meshes"], meshes, (baseline1 / 512, baseline2 / 512))
            preview = None
            if n <= args.previews:
                preview = args.output / f"pair_{n:04d}"
                preview.mkdir()
                for i, im in enumerate(native, 1):
                    write_png(preview / f"input{i}.png", im, rgb_values=True)
            result = {"id": row[0], "modes": {}}
            for mode, candidate in zip(modes, candidates):
                try:
                    values = render(mode, candidate, source, native, native_masks,
                                    homographies if mode == "global_h" else None, preview, images, masks)
                    collected[mode].append(values)
                except (ValueError, torch.linalg.LinAlgError) as exc:
                    values = {"failure": str(exc)}
                result["modes"][mode] = values
            log.write(json.dumps(result, allow_nan=False) + "\n")
            if n <= args.previews or n % 64 == 0 or n == len(rows):
                log.flush()
                print(f"Evaluated {n}/{len(rows)} pairs", flush=True)
    summary = {}
    for mode, values in collected.items():
        summary[mode] = dict(total=len(rows), valid=len(values), failures=len(rows) - len(values),
                             means={k: float(np.mean([x[k] for x in values]))
                                    for k in values[0]} if values else {},
                             negative_jacobian=sum(x["jacobian_min"] <= 0 for x in values))
    report = dict(alpha=0.5, checkpoint_epoch=state["epoch"], modes=summary)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "checkpoint", "rop-alignment", "rop-coef", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--previews", type=int, default=8)
    args = parser.parse_args()
    if args.limit < 0 or args.previews < 0:
        parser.error("Limits must be nonnegative.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu is None else str(args.gpu)
    run(args)


if __name__ == "__main__":
    main()
