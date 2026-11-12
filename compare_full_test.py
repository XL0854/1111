"""Evaluate the original RopStitch and the trained fixed-half residual model."""

import argparse
from contextlib import contextmanager
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

from full import MODEL_KEYS, read_checkpoint
from full_model import FullModel
from model import (Canvas, InvalidPair, grid_with_offset, guarded_render, mesh_canvas,
                   pixel_grid, warp)
from run import discover, file_hash, image_file, load_pair, log_line


ARMS = ("original_ropstitch", "fixed_half_pixel")
METRICS = ("mpsnr_db", "mssim", "valid_canvas", "overlap_canvas",
           "source_coverage", "nonoverlap_shape", "nonoverlap_shape_tail10")


@contextmanager
def original_imports(directory):
    def conflicts(name):
        return name in ("network", "grid_res", "dataset", "utils") or name.startswith("utils.")

    saved = {name: sys.modules.pop(name) for name in tuple(sys.modules) if conflicts(name)}
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path.pop(0)
        for name in tuple(sys.modules):
            if conflicts(name):
                del sys.modules[name]
        sys.modules.update(saved)


class Original:
    def __init__(self, directory, alignment, coefficient):
        with original_imports(directory):
            spec = importlib.util.spec_from_file_location("rop_original_eval", directory / "test.py")
            self.runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.runner)
            import network
            if Path(network.__file__).resolve() != directory / "network.py":
                raise ImportError("Original RopStitch network resolved to a different directory")
            self.net = self.runner.Network().cuda().eval()
            self.coef = self.runner.CoefNetwork().cuda().eval()
            for model, path in ((self.net, alignment), (self.coef, coefficient)):
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"],
                                      strict=True)
            self.rigid = network.get_rigid_mesh(1, 512, 512) / 512

    def render(self, images, adapter):
        left, right = (images * 2 - 1).split(1)
        alpha, _, _, _, info = self.runner.ternary_search(
            self.net, self.coef, left, right, low=-1.0, high=2.0, max_iter=5,
            max_out_height=4000,
        )
        selected = info["best_details"]
        if selected is None:
            raise InvalidPair("Original search produced no scored candidate")
        fallback = selected["warp_fallback"]
        if fallback:
            h, w = images.shape[-2:]
            factor = min(1.0, adapter.edge / max(h, w))
            h, w = max(32, round(h * factor)), max(32, round(w * factor))
            raster = F.interpolate(images, (h, w), mode="bilinear", align_corners=False)
            base = pixel_grid(h, w, images)
            grid = grid_with_offset(base, torch.zeros_like(base)).expand(2, -1, -1, -1)
            canvas = Canvas(images, grid, raster, torch.ones_like(raster[:, :1]))
        else:
            scale = images.new_tensor([images.shape[-1], images.shape[-2]])
            meshes = tuple(selected[name] / scale for name in ("mesh_ref", "mesh_tgt"))
            canvas = mesh_canvas(images, meshes, self.rigid, adapter.inverse_tps,
                                 adapter.plane_grid, adapter.edge)
        return canvas.images, canvas.support, canvas.grid, dict(
            alpha=float(alpha), fallback=bool(fallback), search_fallbacks=info["fallback_count"]
        )


def refined(images, model):
    canvas = model.build(images, "test")[0]
    residual, _, base = model.flow(canvas)
    result, support, scale = guarded_render(canvas, base, residual, 0.995)
    grid = canvas.grid if scale == 0 else warp(
        canvas.grid.permute(0, 3, 1, 2), grid_with_offset(base, residual * scale),
        padding="border",
    ).permute(0, 2, 3, 1)
    return result, support, grid, dict(residual_scale=scale, fallback=scale == 0)


def geometry_metrics(grid, valid, source_size):
    source_h, source_w = source_size
    xy = (grid + 1) * grid.new_tensor([source_w / 2, source_h / 2])
    own_only = valid & ~valid.flip(0)
    interior = own_only[:, :-1, :-1] & own_only[:, 1:, :-1] & own_only[:, :-1, 1:]
    dx = xy[:, :-1, 1:] - xy[:, :-1, :-1]
    dy = xy[:, 1:, :-1] - xy[:, :-1, :-1]
    trace = dx.square().sum(-1) + dy.square().sum(-1)
    determinant = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    largest_squared = (trace + (trace.square() - 4 * determinant.square()).clamp_min(0).sqrt()) / 2
    condition = (largest_squared / determinant.abs().clamp_min(1e-8)).clamp_min(1)
    distortion = condition.log().square()
    stats, tails, cover = [], [], []
    for view in range(2):
        values = distortion[view][interior[view]]
        if values.numel() >= 20:
            stats.append(float(values.mean()))
            tails.append(float(values.topk(max(1, (values.numel()+9)//10)).values.mean()))
        # Source support proxy from occupied cells on a fixed 64 x 64 grid.
        cells = ((grid[view][valid[view]] + 1) * 32).long().clamp(0, 63)
        cover.append(float((cells[:, 1]*64 + cells[:, 0]).unique().numel() / 4096))
    return dict(source_coverage=float(np.mean(cover)),
                nonoverlap_shape=float(np.mean(stats)) if stats else None,
                nonoverlap_shape_tail10=float(np.mean(tails)) if tails else None,
                nonoverlap_fold_fraction=float((determinant[interior] <= 0).float().mean())
                if interior.any() else None)


def measure(result, source_size):
    images, support, grid, details = result
    if not all(torch.isfinite(tensor).all() for tensor in (images, support, grid)):
        raise InvalidPair("Non-finite renderer output")
    valid = support[:, 0] > 0.99
    overlap = valid.all(0)
    union = valid.any(0)
    if not union.any():
        raise InvalidPair("No valid output pixels")
    image = (images * valid[:, None]).permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
    mask = overlap.cpu().numpy()
    # Use the same 7x7 SSIM and support rule for both arms.
    interior = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    score = None
    if interior.any():
        _, score_map = structural_similarity(*image, channel_axis=-1, data_range=1.0,
                                             win_size=7, full=True)
        score = float(score_map[interior].mean())
    psnr = None
    if mask.any():
        mse = np.square(image[0][mask] - image[1][mask]).mean()
        psnr = float(-10*np.log10(max(float(mse), 1e-12)))
    fused = image.sum(0) / valid.sum(0).clamp_min(1).cpu().numpy()[..., None]
    metrics = dict(mpsnr_db=psnr, mssim=score, valid_canvas=float(union.float().mean()),
                   overlap_canvas=float(overlap.float().mean()),
                   overlap_pixels=int(overlap.sum()), canvas=list(overlap.shape), **details)
    metrics.update(geometry_metrics(grid, valid, source_size))
    return metrics, fused


def summarize(rows):
    reports = {}
    for label, group in (("all", rows), ("no_exact_train_image", [r for r in rows if r["train_overlap"] == 0])):
        metrics = {}
        for metric in METRICS:
            paired = [
                [r[arm].get(metric) for arm in ARMS] for r in group
                if all(r[arm].get(metric) is not None for arm in ARMS)
            ]
            if paired:
                a, b = np.asarray(paired).T
                delta = b - a
                indices = np.random.default_rng(2026).integers(0, len(delta), (2000, len(delta)))
                metrics[metric] = dict(n=len(delta), original=float(a.mean()), new=float(b.mean()),
                                       delta=float(delta.mean()), paired_bootstrap95=np.quantile(
                                           delta[indices].mean(1), [0.025, 0.975]).tolist())
        reports[label] = dict(pairs=len(group), metrics=metrics, arms={
            arm: dict(failed=sum("error" in r[arm] for r in group),
                      fallback=sum(bool(r[arm].get("fallback")) for r in group)) for arm in ARMS
        })
    return reports


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    state = read_checkpoint(args.checkpoint)
    if state["epoch"] != 150 or state["config"]["model"]["plane_mode"] != "fixed-half":
        raise ValueError("Expected the final 150-epoch fixed-half checkpoint")
    for key in MODEL_KEYS:
        setattr(args, key, state["config"]["model"][key])
    pairs = discover(args.data)
    if len(pairs) != 1105:
        raise ValueError(f"Expected 1105 matched test pairs, found {len(pairs)}")
    if args.limit:
        pairs = pairs[:args.limit]
    alignment = args.rop / "woCoefNet/model_homo/epoch100_model.pth"
    coefficient = args.rop / "wCoefNet/model_coef/epoch050_coefmodel.pth"
    original_code = args.rop / "wCoefNet/Codes"
    training = {digest for row in state["config"]["train"] for digest in row[1:]}
    manifest = [[name, file_hash(a), file_hash(b)] for name, a, b in pairs]
    plan = dict(model=state["config"]["model"], epoch=state["epoch"], data=str(args.data),
                pairs=manifest, max_original_search_iterations=5,
                checkpoints={str(p): file_hash(p) for p in (args.checkpoint, alignment, coefficient)},
                code={str(p): file_hash(p) for p in (Path(__file__), original_code / "test.py",
                                                      original_code / "network.py")},
                protocol="Same source pairs, common TPS evaluation renderer and canvas cap. Original "
                         "plane and TPS selected by its legacy search. Metrics use each output's "
                         "valid overlap. Source coverage is 64x64 occupied-bin fraction.")
    args.out.mkdir(parents=True)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2))
    for arm in ARMS:
        (args.out / arm).mkdir()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    model = FullModel(args, "checkpoint").cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    original = Original(original_code, alignment, coefficient)
    print(f"Ready: {len(pairs)} paired samples on {torch.cuda.get_device_name(0)}", flush=True)
    rows = []
    for i, (pair, hashes) in enumerate(zip(pairs, manifest), 1):
        name = pair[0]
        row = dict(id=name, train_overlap=sum(digest in training for digest in hashes[1:]))
        try:
            images = load_pair(pair)
            renderers = (lambda: original.render(images, model), lambda: refined(images, model))
            for arm, renderer in zip(ARMS, renderers):
                tick = time.perf_counter()
                try:
                    row[arm], fused = measure(renderer(), images.shape[-2:])
                    row[arm]["evaluation_seconds"] = time.perf_counter() - tick
                    image_file(args.out / arm / f"{name}.jpg", fused)
                except (InvalidPair, torch.linalg.LinAlgError, cv2.error) as exc:
                    row[arm] = dict(error=str(exc))
        except InvalidPair as exc:
            row.update({arm: dict(error=str(exc)) for arm in ARMS})
        rows.append(row)
        log_line(args.out / "pairs.jsonl", row)
        if i <= 2 or i % 25 == 0 or i == len(pairs):
            print(f"Evaluated {i}/{len(pairs)}: {name} "
                  f"mSSIM={[row[arm].get('mssim', row[arm].get('error')) for arm in ARMS]}", flush=True)
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (args.out / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "exact_train_images", *[f"{arm}_{m}" for arm in ARMS for m in METRICS]])
        for row in rows:
            writer.writerow([row["id"], row["train_overlap"],
                             *[row[arm].get(m) for arm in ARMS for m in METRICS]])
    print("COMPLETE", len(rows), flush=True)
    if not summary["all"]["metrics"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rop", "pixel", "data", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int)
    main(parser.parse_args())
