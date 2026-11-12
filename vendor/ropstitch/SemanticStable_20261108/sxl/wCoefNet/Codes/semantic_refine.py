"""Instance-aware refinement of frozen RopStitch meshes for one image pair."""

import math

import torch
import torch.nn.functional as F


def sample(image, grid, mode="bilinear"):
    # Coordinates represent the continuous image extent [0, 1], with pixel
    # centers at (index + 0.5) / size.
    return F.grid_sample(image, 2 * grid - 1, mode=mode, align_corners=False)


def plane_grid(bounds, size):
    low, high = bounds
    h, w = size
    y = (torch.arange(h, device=low.device, dtype=low.dtype) + 0.5) / h
    x = (torch.arange(w, device=low.device, dtype=low.dtype) + 0.5) / w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return (low + torch.stack((xx, yy), dim=-1) * (high - low))[None]


def tps_kernel(points, controls):
    squared = (points.square().sum(-1, keepdim=True)
               + controls.square().sum(-1).unsqueeze(-2)
               - 2 * (points @ controls.transpose(-1, -2))).clamp_min(0)
    return squared * squared.clamp_min(1e-10).log()


def affine_terms(points):
    return torch.cat((torch.ones_like(points[..., :1]), points), dim=-1)


def inverse_tps(mesh, source, grid, chunk=8192):
    """Fit destination-to-source TPS once and query the canvas in chunks."""
    shape = grid.shape
    mesh, source = mesh.flatten(1, 2), source.flatten(1, 2)
    n = mesh.shape[1]
    affine = affine_terms(mesh)
    kernel = tps_kernel(mesh, mesh) + 1e-6 * torch.eye(n, device=mesh.device, dtype=mesh.dtype)
    system = torch.cat((torch.cat((kernel, affine), -1),
                        torch.cat((affine.transpose(1, 2), mesh.new_zeros((1, 3, 3))), -1)), 1)
    params = torch.linalg.solve(system, F.pad(source, (0, 0, 0, 3)))
    results = []
    for points in grid.flatten(1, 2).split(chunk, dim=1):
        results.append(tps_kernel(points, mesh) @ params[:, :n]
                       + affine_terms(points) @ params[:, n:])
    return torch.cat(results, dim=1).reshape(shape)


def project(homography, grid):
    # ACA/DLT homographies have arbitrary homogeneous scale. Normalize the
    # denominator row before deciding whether a projected point is at infinity.
    scale = homography[:, 2:3].abs().amax(dim=-1, keepdim=True)
    homography = homography / scale.clamp_min(torch.finfo(homography.dtype).tiny)
    xy = 512 * grid.flatten(1, 2)
    points = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1) @ homography.transpose(1, 2)
    denom = points[..., 2:]
    valid = denom.abs() > 1e-8
    xy = points[..., :2] / torch.where(valid, denom, 1) / 512
    return torch.where(valid, xy, 2).reshape_as(grid)


def image_and_support(image, inverse_grid):
    support = sample(torch.ones_like(image[:, :1]), inverse_grid)
    warped = sample(image + 1, inverse_grid) / support.clamp_min(1e-5) - 1
    return warped, support


def instance_channels(ids, limit):
    labels, counts = torch.unique(ids[ids > 0], sorted=True, return_counts=True)
    order = torch.argsort(counts, descending=True, stable=True)[:limit]
    labels = labels[order]
    return (ids == labels[None, :, None, None]).float(), labels


def instance_matches(ids, images, homographies, grid, limit=24):
    """Reciprocal assignment in jointly visible space. IDs are view-local."""
    channels, labels = zip(*(instance_channels(ids_view, limit) for ids_view in ids))
    empty = (channels[0][:, :0], channels[1][:, :0], grid.new_empty(0), [])
    if not len(labels[0]) or not len(labels[1]):
        return empty
    maps = [project(h, grid) for h in homographies]
    colors, supports = zip(*(image_and_support(im, xy) for im, xy in zip(images, maps)))
    common = (supports[0] > 0.99) & (supports[1] > 0.99)
    masks = [sample(c, xy, "nearest") * common for c, xy in zip(channels, maps)]
    a, b = (m.flatten(2)[0] for m in masks)
    area_a, area_b = a.sum(1), b.sum(1)
    intersection = a @ b.T
    iou = intersection / (area_a[:, None] + area_b - intersection).clamp_min(1)
    mean_a = a @ colors[0].flatten(2)[0].T / area_a[:, None].clamp_min(1)
    mean_b = b @ colors[1].flatten(2)[0].T / area_b[:, None].clamp_min(1)
    appearance = torch.exp(-2 * (mean_a[:, None] - mean_b).abs().mean(-1))
    score = iou * (0.5 + 0.5 * appearance)
    rows = torch.arange(len(a), device=a.device)
    cols = score.argmax(dim=1)
    peak = score[rows, cols]
    second_a = score.topk(min(2, len(b)), dim=1).values[:, 1] if len(b) > 1 else peak * 0
    second_b = score.topk(min(2, len(a)), dim=0).values[1, cols] if len(a) > 1 else peak * 0
    keep = ((score.argmax(dim=0)[cols] == rows) & (peak >= 0.2)
            & (peak - torch.maximum(second_a, second_b) >= 0.05)
            & (intersection[rows, cols] >= 8))
    left, right, confidence = rows[keep], cols[keep], peak[keep]
    pairs = list(zip(labels[0][left].tolist(), labels[1][right].tolist(), confidence.tolist()))
    return channels[0][:, left], channels[1][:, right], confidence, pairs


def differences(value):
    return value[:, :-1, 1:] - value[:, :-1, :-1], value[:, 1:, :-1] - value[:, :-1, :-1]


def cross(x, y):
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def mesh_geometry(mesh, source, weight):
    dx, dy = differences(mesh)
    sx, sy = differences(source)
    dx = dx / sx[..., :1]
    dy = dy / sy[..., 1:]
    # A rotation with uniform scale has orthogonal axes of equal length.
    lx, ly = dx.norm(dim=-1).clamp_min(1e-5), dy.norm(dim=-1).clamp_min(1e-5)
    distortion = (lx.log() - ly.log()).abs() + ((dx * dy).sum(-1) / (lx * ly)).abs()
    dx_far = (mesh[:, 1:, 1:] - mesh[:, 1:, :-1]) / sx[..., :1]
    dy_far = (mesh[:, 1:, 1:] - mesh[:, :-1, 1:]) / sy[..., 1:]
    jacobian = torch.stack((cross(dx, dy), cross(dx_far, dy_far),
                            cross(dx, dy_far), cross(dx_far, dy)), -1)
    return (distortion * weight).sum() / weight.sum().clamp_min(1e-5), jacobian


def seam_blend(warped, supports, foreground, inverse, composition=None, weight=None):
    a, b = supports
    overlap = a * b
    if weight is not None:
        soft = weight
    elif composition is None:
        distances = [torch.minimum(xy, 1 - xy).amin(dim=-1)[:, None] for xy in inverse]
        soft = torch.sigmoid(32 * (distances[0] - distances[1]).clamp(-1, 1))
    else:
        soft = composition(*warped, *supports, *foreground)
    left = a * (1 - b) + overlap * soft
    right = b * (1 - a) + overlap * (1 - soft)
    stitched = (warped[0] + 1) * left + (warped[1] + 1) * right - 1
    return stitched, soft


def composition_weight_on_canvas(images, foreground, meshes, source, grid, composition, resolution):
    # Keep the coordinate frame and pixel scale seen by the seam network
    # during training, then transfer its blend weights onto the native canvas.
    low = grid.new_full((2,), -0.35)
    high = grid.new_full((2,), 1.35)
    model_grid = plane_grid((low, high), (resolution, resolution))
    inverse = [inverse_tps(m, source, model_grid) for m in meshes]
    warped, supports = zip(*(image_and_support(im, xy) for im, xy in zip(images, inverse)))
    masks = [sample(fg, xy) for fg, xy in zip(foreground, inverse)]
    weight = composition(*warped, *supports, *masks)
    locations = ((grid - low) / (high - low)).clamp(0.5 / resolution, 1 - 0.5 / resolution)
    return sample(weight, locations)


def assess(meshes, source, grid, images, foreground, matched, confidence, stiffness,
           composition=None, stable=False):
    inverse = [inverse_tps(mesh, source, grid) for mesh in meshes]
    warped, support = zip(*(image_and_support(im, xy) for im, xy in zip(images, inverse)))
    overlap = support[0] * support[1]
    union = support[0] + support[1] - overlap
    photo_error = (warped[0] - warped[1]).abs().mean(1, keepdim=True)
    alignment = (overlap * photo_error).sum() / overlap.sum().clamp_min(1)
    edge_alignment = alignment * 0
    if stable:
        for dimension in (-2, -1):
            gradient_error = (warped[0].diff(dim=dimension)
                              - warped[1].diff(dim=dimension)).abs().mean(1, keepdim=True)
            extent = overlap.size(dimension) - 1
            weights = torch.minimum(overlap.narrow(dimension, 0, extent),
                                    overlap.narrow(dimension, 1, extent))
            edge_alignment = edge_alignment + (gradient_error * weights).sum() / weights.sum().clamp_min(1)
    shape, jac = zip(*(mesh_geometry(m, source, w) for m, w in zip(meshes, stiffness)))
    warped_fg = [sample(m, xy) for m, xy in zip(foreground, inverse)]
    if stable and composition is not None:
        # Seam losses update the compositor. They cannot improve by moving
        # foreground or reducing the overlap through the geometry network.
        warped_for_seam, support_for_seam, fg_for_seam = (
            tuple(x.detach() for x in group) for group in (warped, support, warped_fg)
        )
        overlap_for_seam = overlap.detach()
        error_for_seam = photo_error.detach()
    else:
        warped_for_seam, support_for_seam, fg_for_seam = warped, support, warped_fg
        overlap_for_seam, error_for_seam = overlap, photo_error
    stitched, soft = seam_blend(warped_for_seam, support_for_seam, fg_for_seam, inverse, composition)
    anchor = alignment * 0
    if composition is not None:
        # Anchor the blend on either side of overlap so that a spatially
        # constant weight cannot minimize the seam loss on its own.
        def inside_boundary(mask):
            detached = mask.detach()
            dilated_outside = F.max_pool2d(1 - detached, 7, stride=1, padding=3)
            return detached * dilated_outside

        anchor_left = (support[0].detach() * inside_boundary(support[1])).clamp(0, 1)
        anchor_right = (support[1].detach() * inside_boundary(support[0])).clamp(0, 1)
        anchor = ((anchor_left * (1 - soft) + anchor_right * soft).sum()
                  / (anchor_left + anchor_right).sum().clamp_min(1))
    seam_detail = alignment * 0
    if composition is not None:
        change_x = (stitched[..., 1:] - stitched[..., :-1]).abs().mean(1, keepdim=True)
        change_y = (stitched[:, :, 1:] - stitched[:, :, :-1]).abs().mean(1, keepdim=True)
        seam_detail = ((change_x * (soft[..., 1:] - soft[..., :-1]).abs()).mean()
                       + (change_y * (soft[:, :, 1:] - soft[:, :, :-1]).abs()).mean())
    foreground_union = torch.maximum(*fg_for_seam)
    seam_x = (soft[..., 1:] - soft[..., :-1]).abs()
    seam_y = (soft[:, :, 1:] - soft[:, :, :-1]).abs()
    risk = foreground_union * overlap_for_seam
    seam = ((seam_x * torch.minimum(risk[..., 1:], risk[..., :-1])).mean()
            + (seam_y * torch.minimum(risk[:, :, 1:], risk[:, :, :-1])).mean())
    # Penalize blending inconsistent content even when a soft seam is diffuse.
    ghost_weight = overlap_for_seam * (0.25 + 0.75 * foreground_union) if stable else risk
    ghost = ((4 * soft * (1 - soft) * ghost_weight * error_for_seam).sum()
             / overlap_for_seam.sum().clamp_min(1))
    instance = alignment * 0
    if confidence.numel():
        left, right = [sample(m, xy) for m, xy in zip(matched, inverse)]
        intersection = (overlap * left * right).sum((0, 2, 3))
        mass = (overlap * (left + right)).sum((0, 2, 3))
        # A matched instance disappearing from overlap is penalized as well.
        instance = ((1 - 2 * intersection / mass.clamp_min(1e-6)) * confidence).sum() / confidence.sum()
    # Check the sampled inverse TPS too, because control cells alone do not
    # exclude folds between control points.
    dense_min = []
    for xy, mask in zip(inverse, support):
        dx, dy = differences(xy)
        area = cross(dx, dy) / cross(*differences(grid))
        interior = ((mask[:, 0, :-1, :-1] > 0.99) & (mask[:, 0, :-1, 1:] > 0.99)
                    & (mask[:, 0, 1:, :-1] > 0.99))
        dense_min.append(torch.where(interior, area, float("inf")).amin())
    return dict(alignment=alignment, instance=instance, shape=sum(shape) / 2, seam=seam,
                anchor=anchor, seam_detail=seam_detail, edge_alignment=edge_alignment,
                ghost=ghost, coverage=union.mean(), overlap=overlap.mean(),
                coverage_left=support[0].mean(), coverage_right=support[1].mean(),
                jacobian=torch.stack(jac).amin(), inverse_jacobian=torch.stack(dense_min).amin())


def refine(images, ids, foreground, homographies, meshes, source, *, mode, steps,
           max_shift_px, composition=None, resolution=128):
    """Return best feasible mesh pair on a fixed canvas, including the baseline."""
    meshes = tuple(m.detach() for m in meshes)
    if any(not torch.isfinite(m).all() for m in meshes):
        raise ValueError("RopStitch returned a non-finite mesh.")
    max_shift = max_shift_px / 512
    nodes = torch.cat([m.flatten(1, 2) for m in meshes], dim=1)
    bounds = (nodes.amin(dim=(0, 1)) - max_shift, nodes.amax(dim=(0, 1)) + max_shift)
    span = bounds[1] - bounds[0]
    if (span > 4).any() or (span <= 0).any():
        raise ValueError("RopStitch canvas is outside the supported range.")
    grid = plane_grid(bounds, (resolution, resolution))
    with torch.no_grad():
        left, right, confidence, pairs = instance_matches(ids, images, homographies, grid)
        matched = (left, right)
        rigid = tuple(1 + 2 * fg + 2 * (m * confidence[None, :, None, None]).sum(1, keepdim=True)
                      for fg, m in zip(foreground, matched))
        stiffness = tuple(sample(r, source, "nearest")[:, 0, :-1, :-1].clamp_min(1) for r in rigid)
        field_grid = plane_grid(bounds, (13, 13))
        plane_rigidity = [1 + sample(r - 1, project(h, field_grid))
                          for r, h in zip(rigid, homographies)]
        # Minimum of k1*||a*d||² + k2*||(1-a)*d||². Both sides share the
        # same displacement field in plane coordinates, with opposite signs.
        share = plane_rigidity[1] / (plane_rigidity[0] + plane_rigidity[1])
        if mode == "equal":
            share = torch.full_like(share, 0.5)

    def evaluate(candidate):
        return assess(candidate, source, grid, images, foreground, matched, confidence,
                      stiffness, composition)

    def objective(metrics):
        cost = metrics["alignment"] + 0.12 * metrics["instance"] + 0.06 * metrics["shape"]
        if mode == "full":
            cost = cost + 2 * metrics["seam"] + 0.15 * metrics["ghost"]
        return cost

    with torch.no_grad():
        baseline = evaluate(meshes)
    if (any(not torch.isfinite(v).item() for v in baseline.values())
            or min(baseline["jacobian"], baseline["inverse_jacobian"]) <= 0):
        raise ValueError("Baseline mesh fails finite-value or sampled fold checks.")
    if baseline["overlap"] < 0.01 and mode != "baseline":
        raise ValueError("Baseline overlap below 1%; semantic refinement is unsupported.")

    best, best_metrics, best_cost = meshes, baseline, float(objective(baseline))
    stats = dict(evaluated=0, feasible=0, selected_step=0, stop_reason="step_limit")
    if mode == "baseline":
        stats["stop_reason"] = "baseline"
        return best, bounds, share, pairs, baseline, best_metrics, stats

    field = torch.zeros_like(field_grid, requires_grad=True)
    optimizer = torch.optim.Adam([field], lr=0.12)
    locations = [(m - bounds[0]) / span for m in meshes]
    coverage_keys = ("coverage", "coverage_left", "coverage_right")
    for step in range(steps + 1):
        displacement = max_shift * field.tanh().permute(0, 3, 1, 2)
        offsets = [sample(d, xy).permute(0, 2, 3, 1)
                   for d, xy in zip((share * displacement, -(1 - share) * displacement), locations)]
        candidate = tuple(mesh + offset for mesh, offset in zip(meshes, offsets))
        try:
            metrics = evaluate(candidate)
        except torch.linalg.LinAlgError:
            stats["stop_reason"] = "singular_candidate"
            break
        stats["evaluated"] += int(step > 0)
        cost = objective(metrics)
        if not torch.isfinite(cost):
            stats["stop_reason"] = "nonfinite_objective"
            break
        with torch.no_grad():
            feasible = (all(torch.isfinite(v).item() for v in metrics.values())
                        and all(metrics[k] >= 0.99 * baseline[k] for k in coverage_keys)
                        and metrics["overlap"] >= 0.98 * baseline["overlap"]
                        and metrics["alignment"] <= baseline["alignment"] + 0.005
                        and min(metrics["jacobian"], metrics["inverse_jacobian"]) > 0.01)
            stats["feasible"] += int(step > 0 and feasible)
            if feasible and float(cost) < best_cost - 1e-6:
                stats["selected_step"] = step
                best, best_metrics, best_cost = (tuple(m.detach() for m in candidate),
                                                  {k: v.detach() for k, v in metrics.items()}, float(cost))
        if step == steps:
            break
        smooth_x, smooth_y = differences(displacement.permute(0, 2, 3, 1) / max_shift)
        smooth = smooth_x.square().mean() + smooth_y.square().mean()
        loss = (cost + 0.01 * smooth + 0.003 * (displacement / max_shift).square().mean()
                + sum(F.relu(0.99 * baseline[k] - metrics[k]) for k in coverage_keys)
                + F.relu(0.98 * baseline["overlap"] - metrics["overlap"])
                + F.relu(metrics["alignment"] - baseline["alignment"] - 0.005)
                + 0.1 * (F.relu(0.01 - metrics["jacobian"])
                         + F.relu(0.01 - metrics["inverse_jacobian"])))
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(field.grad).all():
            stats["stop_reason"] = "nonfinite_gradient"
            break
        optimizer.step()
    return best, bounds, share, pairs, baseline, best_metrics, stats


def output_size(bounds, image_size, max_pixels):
    span = (bounds[1] - bounds[0]).tolist()
    h = math.ceil(span[1] * image_size[0])
    w = math.ceil(span[0] * image_size[1])
    if h * w > max_pixels:
        raise ValueError(f"Canvas {w}×{h} exceeds --max-pixels={max_pixels}.")
    return h, w
