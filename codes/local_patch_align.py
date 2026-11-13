"""Bounded local alignment on the TPS + PixelStitch canvas before fusion."""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from compare_full_test import measure, refined
from full import MODEL_KEYS, read_checkpoint
from full_model import FullModel
from model import InvalidPair, grid_with_offset, pixel_grid, render, warp
from run import discover, file_hash, image_file, load_pair


PATCH_SIZE = 80
MAX_PATCHES = 8
MAX_SHIFT = 4.0
METRICS = ("mpsnr_db", "mssim")


def patch_windows(error, overlap):
    h, w = overlap.shape
    size = min(PATCH_SIZE, h, w)
    if size < 40:
        return
    # Give priority to neighborhoods with high local photometric error.
    scores = cv2.boxFilter(error * overlap, -1, (21, 21))
    scores[~overlap] = 0
    radius = size // 2
    for _ in range(MAX_PATCHES):
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score < 0.001:
            break
        top = min(max(0, y - radius), h - size)
        left = min(max(0, x - radius), w - size)
        scores[max(0, top - radius):top + size + radius,
               max(0, left - radius):left + size + radius] = 0
        yield slice(top, top + size), slice(left, left + size)


def patch_flow(dis, reference, moving, clearance, window):
    reference = np.ascontiguousarray(reference)
    moving = np.ascontiguousarray(moving)
    forward = dis.calc(reference, moving, None)
    backward = dis.calc(moving, reference, None)
    if not np.isfinite(forward).all() or not np.isfinite(backward).all():
        return None
    size = len(reference)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    backward_at_match = cv2.remap(backward, xx + forward[..., 0], yy + forward[..., 1],
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    consistency = np.linalg.norm(forward + backward_at_match, axis=-1)
    length = np.linalg.norm(forward, axis=-1)
    weight = np.clip((1.5 - consistency) / 1.0, 0, 1)
    weight *= np.clip((MAX_SHIFT + 2 - length) / 2, 0, 1)
    weight *= np.minimum(1, MAX_SHIFT / np.maximum(length, 1e-6))
    weight *= np.clip((clearance - MAX_SHIFT - 2) / 5, 0, 1)
    weight *= window
    return forward * weight[..., None]


def jacobian(grid, source_size):
    h, w = source_size
    dx = (grid[:-1, 1:] - grid[:-1, :-1]) * grid.new_tensor((w / 2, h / 2))
    dy = (grid[1:, :-1] - grid[:-1, :-1]) * grid.new_tensor((w / 2, h / 2))
    det = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    energy = dx.square().sum(-1) + dy.square().sum(-1)
    largest = (energy + (energy.square() - 4 * det.square()).clamp_min(0).sqrt()) / 2
    return det, (largest / det.abs().clamp_min(1e-8)).clamp_min(1).log()


def bounded_geometry(old, new, active, source_size):
    cells = active[:-1, :-1] | active[:-1, 1:] | active[1:, :-1]
    old_det, old_shape = (value[cells] for value in jacobian(old, source_size))
    new_det, new_shape = (value[cells] for value in jacobian(new, source_size))
    ratio = new_det / old_det.clamp_min(1e-8)
    min_ratio, max_ratio = ratio.amin(), ratio.amax()
    shape_increase = (new_shape - old_shape).amax()
    if bool((old_det > 0).all() & (min_ratio >= 0.65) & (max_ratio <= 1.5)
            & (shape_increase <= 0.32)):
        return float(min_ratio), float(max_ratio), float(shape_increase)
    return None


@torch.no_grad()
def repair(originals, result, dis):
    """Use the first view as anchor and locally resample the second view."""
    images, support, grid, details = result
    overlap = (support[:, 0] > 0.99).all(0).cpu().numpy().astype(np.uint8)
    color = images.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
    error = np.square(color[0] - color[1]).mean(-1)
    clearance = cv2.distanceTransform(np.pad(overlap, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    gray = [cv2.cvtColor(np.rint(im * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) for im in color]
    h, w = overlap.shape
    base = pixel_grid(h, w, grid)
    identity = grid_with_offset(base, torch.zeros_like(base))
    offset = grid.new_tensor((2 / w, 2 / h))
    repaired_images, repaired_grid = images.clone(), grid.clone()
    stats = dict(proposed_patches=0, accepted_patches=0, changed_pixels=0, max_shift_px=0.0,
                 min_area_ratio=1.0, max_area_ratio=1.0, max_log_condition_increase=0.0)
    size = min(PATCH_SIZE, h, w)
    # Two zero-weight border pixels keep the patch and its adjacent grid cells continuous.
    taper = np.pad(np.hanning(size - 2).astype(np.float32), 1) if size >= 40 else None
    window = np.outer(taper, taper) if taper is not None else None
    for y, x in patch_windows(error, clearance > MAX_SHIFT + 4):
        if (clearance[y, x] > MAX_SHIFT + 2).mean() < 0.25:
            continue
        stats["proposed_patches"] += 1
        flow = patch_flow(dis, gray[0][y, x], gray[1][y, x], clearance[y, x], window)
        if flow is None:
            continue
        displacement = torch.as_tensor(flow, device=grid.device)
        old_grid = grid[1, y, x]
        ref, moving = images[:, :, y, x].clamp(0, 1)
        for scale in (1.0, 0.5):
            shift = displacement * scale
            magnitude = shift.norm(dim=-1)
            active = magnitude >= 0.02
            if int(active.sum()) < 24:
                break
            query = identity[:, y, x] + shift * offset
            candidate = warp(grid[1:2].permute(0, 3, 1, 2), query,
                             padding="border")[0].permute(1, 2, 0)
            candidate = torch.where(active[..., None], candidate, old_grid)
            geometry = bounded_geometry(old_grid, candidate, active, originals.shape[-2:])
            if geometry is None:
                continue
            pixels, validity = render(originals[1:2], candidate[None])
            if not bool((validity[0, 0, active] > 0.99).all()):
                continue
            new = pixels[0, :, active].clamp(0, 1)
            previous_error = (ref[:, active] - moving[:, active]).square().mean()
            new_error = (ref[:, active] - new).square().mean()
            if new_error >= previous_error * 0.98:
                continue
            patch = repaired_images[1, :, y, x]
            patch[:, active] = pixels[0, :, active]
            repaired_grid[1, y, x] = candidate
            stats["accepted_patches"] += 1
            stats["changed_pixels"] += int(active.sum())
            stats["max_shift_px"] = max(stats["max_shift_px"], float(magnitude[active].max()))
            min_ratio, max_ratio, shape_increase = geometry
            stats["min_area_ratio"] = min(stats["min_area_ratio"], min_ratio)
            stats["max_area_ratio"] = max(stats["max_area_ratio"], max_ratio)
            stats["max_log_condition_increase"] = max(stats["max_log_condition_increase"],
                                                      shape_increase)
            break
    return repaired_images, support, repaired_grid, {**details, **stats}


def difficulty_map(pairs, plan, scores):
    if plan is None:
        return {}
    manifest = {name: (a, b) for name, a, b in json.loads(plan.read_text())["pairs"]}
    with scores.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(manifest) or {row["id"] for row in rows} != manifest.keys():
        raise ValueError("Original score IDs do not match the historical pair plan")
    for name, a, b in pairs:
        if manifest.get(name) != (file_hash(a), file_hash(b)):
            raise ValueError(f"Historical test pair does not match: {name}")
    tiers = {}
    for metric in METRICS:
        ranked = sorted(rows, key=lambda r: (-float(r[f"original_ropstitch_{metric}"]), r["id"]))
        easy, moderate = int(len(ranked) * 0.3), int(len(ranked) * 0.6)
        tiers[metric] = {row["id"]: level for level, group in
                         (("Easy", ranked[:easy]), ("Moderate", ranked[easy:moderate]),
                          ("Hard", ranked[moderate:])) for row in group}
    return tiers


def summarize(rows, tiers):
    summary = {}
    for metric in METRICS:
        groups = ("Easy", "Moderate", "Hard", "Average") if tiers else ("Average",)
        summary[metric] = {}
        for level in groups:
            paired = [(row[f"baseline_{metric}"], row[f"repair_{metric}"]) for row in rows
                      if row.get(f"baseline_{metric}") is not None
                      and row.get(f"repair_{metric}") is not None
                      and (level == "Average" or tiers[metric][row["id"]] == level)]
            entry = dict(pairs=len(paired))
            if paired:
                baseline, repair_scores = np.array(paired).T
                difference = repair_scores - baseline
                entry.update(baseline=float(baseline.mean()), repair=float(repair_scores.mean()),
                             delta=float(difference.mean()), improved=int((difference > 1e-6).sum()),
                             worsened=int((difference < -1e-6).sum()))
            summary[metric][level] = entry
    return summary


@torch.no_grad()
def compare(args):
    pairs = discover(args.data, args.list)
    pairs = pairs[:args.limit]
    tiers = difficulty_map(pairs, args.pair_plan, args.original_scores)
    if set(args.save_pair) - {pair[0] for pair in pairs}:
        raise ValueError("--save-pair must select an ID in the evaluation set")
    state = read_checkpoint(args.checkpoint)
    if state["config"]["model"]["plane_mode"] != "fixed-half":
        raise ValueError("Expected a regular fixed-half checkpoint")
    for key in MODEL_KEYS:
        setattr(args, key, state["config"]["model"][key])
    model = FullModel(args, "checkpoint").cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    dis.setFinestScale(0)
    args.out.mkdir(parents=True)
    if args.save_stitched:
        (args.out / "stitched").mkdir()
    diagnostics = ("proposed_patches", "accepted_patches", "changed_pixels", "max_shift_px",
                   "min_area_ratio", "max_area_ratio", "max_log_condition_increase")
    columns = ("id", "psnr_tier", "ssim_tier", *(
        f"{arm}_{metric}" for arm in ("baseline", "repair") for metric in
        (*METRICS, "overlap_pixels", "source_coverage", "nonoverlap_shape_tail10")),
        *diagnostics, "error")
    rows = []
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        for index, pair in enumerate(pairs, 1):
            name = pair[0]
            row = dict(id=name)
            if tiers:
                row.update(psnr_tier=tiers["mpsnr_db"][name], ssim_tier=tiers["mssim"][name])
            try:
                images = load_pair(pair)
                baseline = refined(images, model)
                repaired = repair(images, baseline, dis)
                for arm, result in (("baseline", baseline), ("repair", repaired)):
                    save_pair = name in args.save_pair
                    save_stitched = args.save_stitched and arm == "repair"
                    fusion = model.fuse if save_pair or save_stitched else None
                    scores, fused = measure(result, images.shape[-2:], fusion)
                    row.update({f"{arm}_{key}": value for key, value in scores.items()
                                if f"{arm}_{key}" in columns})
                    if save_pair:
                        image_file(args.out / f"{name}_{arm}.png", fused)
                    if save_stitched:
                        image_file(args.out / "stitched" / f"{name}.png", fused)
                row.update({key: repaired[3][key] for key in diagnostics})
            except (InvalidPair, torch.linalg.LinAlgError, cv2.error) as exc:
                row = dict(id=name, error=str(exc))
            rows.append(row)
            writer.writerow(row)
            if index == 1 or index % 25 == 0 or index == len(pairs):
                handle.flush()
                print(f"Evaluated {index}/{len(pairs)}", flush=True)
    summary = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint),
                   data=str(args.data), difficulty_reference=str(args.original_scores) if tiers else None,
                   epoch=state["epoch"], pairs=len(rows), failed=sum("error" in row for row in rows),
                   repaired_pairs=sum(row.get("accepted_patches", 0) > 0 for row in rows),
                   overlap_changed=sum(row.get("baseline_overlap_pixels") != row.get("repair_overlap_pixels")
                                       for row in rows if "error" not in row),
                   metrics=summarize(rows, tiers))
    report = json.dumps(summary, indent=2, allow_nan=False)
    (args.out / "summary.json").write_text(report + "\n")
    print(report, flush=True)
    if summary["failed"] == len(rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "checkpoint", "out", "rop", "pixel"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--list", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pair-plan", type=Path, help="Historical original RopStitch pair hashes")
    parser.add_argument("--original-scores", type=Path, help="CSV with original difficulty scores")
    parser.add_argument("--save-pair", action="append", default=[], help="Save fused A/B images for this ID")
    parser.add_argument("--save-stitched", action="store_true", help="Save the repaired stitched images")
    args = parser.parse_args()
    if args.out.exists() or (args.limit is not None and args.limit < 1) or args.gpu < 0:
        parser.error("Provide a new --out directory, a positive --limit and a nonnegative --gpu")
    if bool(args.pair_plan) != bool(args.original_scores):
        parser.error("Use --pair-plan and --original-scores together for difficulty tiers")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable on GPU {args.gpu}")
    compare(args)


if __name__ == "__main__":
    main()
