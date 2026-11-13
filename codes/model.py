"""RopStitch TPS followed by a trainable PixelStitch residual field."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F


class InvalidPair(ValueError):
    """A source pair for which the frozen geometric initializer is unusable."""


@dataclass
class Canvas:
    originals: torch.Tensor
    grid: torch.Tensor
    images: torch.Tensor
    support: torch.Tensor


def load_weights(module, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(state.get("model", state), strict=True)
    return module


def warp(image, grid, padding="zeros"):
    return F.grid_sample(image, grid, align_corners=False, padding_mode=padding)


def render(originals, grids):
    support = warp(torch.ones_like(originals[:, :1]), grids)
    # Normalize color at the partially valid boundary.
    return warp(originals, grids) / support.clamp_min(1e-5), support


def pixel_grid(h, w, like):
    y, x = torch.meshgrid(torch.arange(h, device=like.device, dtype=like.dtype),
                          torch.arange(w, device=like.device, dtype=like.dtype), indexing="ij")
    return torch.stack((x, y), dim=-1)[None]


def grid_with_offset(base, displacement):
    h, w = base.shape[1:3]
    return (base + displacement + 0.5) * base.new_tensor([2 / w, 2 / h]) - 1


def sample_displacement(flow, base, offset):
    return warp(flow.permute(0, 3, 1, 2), grid_with_offset(base, offset),
                padding="border").permute(0, 2, 3, 1)


class RopTPS:
    """Load the same frozen geometry and inverse TPS used by the existing baseline."""

    def __init__(self, root, alignment, coefficient, device, edge):
        path = Path(root) / "SemanticStable_20261108/sxl/wCoefNet/Codes"
        if not (path / "network.py").is_file():
            raise FileNotFoundError(f"Expected RopStitch semantic baseline at {path}")
        sys.path.insert(0, str(path))
        from network import Network, CoefNetwork, get_rigid_mesh
        from semantic_refine import inverse_tps, plane_grid

        self.net = load_weights(Network(pretrained=False), alignment).to(device).eval()
        self.coef = load_weights(CoefNetwork(), coefficient).to(device).eval()
        self.net.requires_grad_(False)
        self.coef.requires_grad_(False)
        self.rigid = get_rigid_mesh(1, 512, 512) / 512
        self.inverse_tps = inverse_tps
        self.plane_grid = plane_grid
        self.edge = edge

    @torch.no_grad()
    def __call__(self, images):
        # BGR, [0,1] for PixelStitch. RopStitch expects 512px and [-1,1].
        canonical = F.interpolate(images, (512, 512), mode="bilinear",
                                  align_corners=False, antialias=True)
        left, right = (canonical * 2 - 1).split(1)
        _, _, _, a, b = self.net(left, right, self.coef, alpha=0.5)
        return mesh_canvas(images, (a / 512, b / 512), self.rigid,
                           self.inverse_tps, self.plane_grid, self.edge)


def mesh_canvas(images, meshes, rigid, inverse_tps, plane_grid, edge):
    # Discrete canvas sizing is detached. The TPS and image sampler stay
    # differentiable with respect to the predicted control points.
    nodes = torch.cat([m.detach().flatten(1, 2) for m in meshes], dim=1)
    if not torch.isfinite(nodes).all():
        raise InvalidPair("Non-finite RopStitch mesh")
    low = nodes.amin((0, 1)) - 8 / 512
    high = nodes.amax((0, 1)) + 8 / 512
    span = high - low
    if (span <= 0).any() or (span > 4).any():
        raise InvalidPair("RopStitch canvas outside the supported range")
    native = span * span.new_tensor([images.shape[-1], images.shape[-2]])
    size = native * min(1.0, edge / float(native.max()))
    width, height = [max(32, int(v)) for v in size.ceil().tolist()]
    query = plane_grid((low, high), (height, width))
    try:
        grid = torch.cat([inverse_tps(mesh, rigid, query) for mesh in meshes])
    except torch.linalg.LinAlgError as exc:
        raise InvalidPair(f"Singular TPS: {exc}") from exc
    grid = 2 * grid - 1
    if not torch.isfinite(grid).all():
        raise InvalidPair("Non-finite inverse TPS")
    projected, support = render(images, grid)
    return Canvas(images, grid, projected, support)


def projection_weights(flow, base, support):
    """Inverse PN-Coef with centers measured on the TPS canvas.

    The denominator follows PixelStitch's inverse weighting. The equal split
    handles coincident view centers where the projection axis is undefined.
    """
    visible = support[:, 0]
    centers = (visible[..., None] * base).sum((1, 2)) / visible.sum((1, 2)).clamp_min(1)[:, None]
    direction = centers[0] - centers[1]
    length = direction.norm()
    unit = direction / length.clamp_min(1e-4)
    position = (base * unit).sum(-1, keepdim=True)
    low, high = position.amin(), position.amax()
    span = high - low
    component = (flow * unit).sum(-1, keepdim=True)
    floor = (0.1 * span).clamp_min(1e-4)
    left = (high - position) / (span + component[0:1]).clamp_min(floor)
    right = (position - low) / (span - component[1:2]).clamp_min(floor)
    weights = torch.cat((left, right)).clamp(0, 1)
    return torch.where(length > 1, weights, torch.full_like(weights, 0.5))


class PixelResidual(nn.Module):
    def __init__(self, pixel_root, checkpoint, max_shift, iterations):
        super().__init__()
        root = Path(pixel_root)
        if not (root / "modules/models/raft/raft.py").is_file():
            raise FileNotFoundError(f"Expected official PixelStitch source at {root}")
        sys.path.insert(0, str(root))
        from modules.models.raft.raft import RAFTStitch

        self.raft = RAFTStitch(argparse.Namespace(radius=6, ftdown=8))
        if checkpoint is not None:
            load_weights(self.raft, checkpoint)
        # PixelStitch's auxiliary matching head is unused here. The flow
        # network still receives gradients in eval mode with a two-view batch.
        self.raft.image_match.requires_grad_(False)
        self.raft.eval()
        self.max_shift = max_shift
        self.iterations = iterations

    def forward(self, canvas):
        self.raft.eval()
        h, w = canvas.images.shape[-2:]
        padded = F.pad(canvas.images * canvas.support, (0, (-w) % 8, 0, (-h) % 8))
        flows, _ = self.raft(padded, self.iterations)
        flow = flows[-1][..., :h, :w].permute(0, 2, 3, 1)
        raw = self.max_shift * torch.tanh(flow / self.max_shift)
        base = pixel_grid(h, w, raw)
        # Primary inter-view flow consistency estimates local reliability.
        cycle = raw + sample_displacement(raw.flip(0), base, raw)
        with torch.no_grad():
            confidence = torch.exp(-cycle.norm(dim=-1, keepdim=True) /
                                   (2 + 0.1 * raw.norm(dim=-1, keepdim=True)))
            overlap = canvas.support.amin(dim=0, keepdim=True)
            soft_support = F.avg_pool2d(overlap, 31, stride=1, padding=15,
                                        count_include_pad=False).permute(0, 2, 3, 1)
            gate = soft_support * confidence
        residual = raw * projection_weights(raw, base, canvas.support) * gate
        return residual, cycle, base


def residual_canvas(canvas, base, residual):
    # Compose canvas-to-TPS and TPS-to-source maps before sampling colors.
    query = grid_with_offset(base, residual)
    grid = warp(canvas.grid.permute(0, 3, 1, 2), query, padding="border").permute(0, 2, 3, 1)
    images, support = render(canvas.originals, grid)
    support = support * warp(torch.ones_like(canvas.support), query)
    return Canvas(canvas.originals, grid, images, support)


def refine(canvas, base, residual):
    result = residual_canvas(canvas, base, residual)
    return result.images, result.support


def fixed_overlap(canvas):
    mask = (canvas.support.amin(dim=0, keepdim=True) > 0.99).float()
    return (F.avg_pool2d(mask, 5, stride=1, padding=2) > 0.999).float()


def masked_mean(value, mask):
    expanded = mask.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1)


def photo(images, mask):
    difference = images[0:1] - images[1:2]
    return masked_mean((difference.square() + 1e-6).sqrt(), mask)


def field_terms(field):
    dx = field[:, :-1, 1:] - field[:, :-1, :-1]
    dy = field[:, 1:, :-1] - field[:, :-1, :-1]
    determinant = (1 + dx[..., 0]) * (1 + dy[..., 1]) - dx[..., 1] * dy[..., 0]
    smooth = (dx.square().mean() + dy.square().mean()) / 2
    return smooth, determinant


def objective(canvas, residual, cycle, base, weights):
    mask = fixed_overlap(canvas)
    result, support = refine(canvas, base, residual)
    smooth, det = field_terms(residual)
    nonoverlap = (canvas.support > 0.99).float() * (1 - mask)
    losses = dict(
        photo=photo(result, mask),
        cycle=masked_mean(cycle.permute(0, 3, 1, 2).abs(), mask),
        smooth=smooth,
        nonoverlap=masked_mean(residual.permute(0, 3, 1, 2).abs(), nonoverlap),
        coverage=masked_mean((1 - support.amin(dim=0, keepdim=True)).clamp_min(0), mask)
        + ((canvas.support - support).clamp_min(0) * (canvas.support > 0.99)).mean(),
        fold=F.relu(0.1 - det).square().mean(),
    )
    total = losses["photo"] + sum(weights[key] * losses[key] for key in weights)
    return total, losses


def grid_cells(valid):
    return valid[:, :-1, :-1] & valid[:, 1:, :-1] & valid[:, :-1, 1:] & valid[:, 1:, 1:]


def grid_determinants(grid):
    # Check both triangles of each source sampling cell.
    horizontal = grid[:, :, 1:] - grid[:, :, :-1]
    vertical = grid[:, 1:] - grid[:, :-1]
    top, bottom = horizontal[:, :-1], horizontal[:, 1:]
    left, right = vertical[:, :, :-1], vertical[:, :, 1:]
    first = top[..., 0] * left[..., 1] - top[..., 1] * left[..., 0]
    second = bottom[..., 0] * right[..., 1] - bottom[..., 1] * right[..., 0]
    return torch.minimum(first, second)


def residual_problems(residual, result, tps_det, tps_cells):
    _, det = field_terms(residual)
    current_det = grid_determinants(result.grid)
    visible_cells = grid_cells(result.support[:, 0] > 0.99)
    # Reject new flips and cells compressed below 5% of their TPS area.
    min_det = torch.where(tps_cells, tps_det.clamp_min(0) * 0.05, 0.0)
    new_or_positive = ~tps_cells | (tps_det > 0)
    composed = visible_cells & new_or_positive & (current_det <= min_det)
    return (det <= 0.05) | composed


def local_residual(residual, bad_cells, core, feather):
    # Expand each flagged cell to its four vertices and feather the suppression.
    vertices = F.max_pool2d(F.pad(bad_cells[:, None].float(), (1, 1, 1, 1)), 2, stride=1)
    radius = core + feather
    region = F.max_pool2d(vertices, (1, 2 * radius + 1), stride=1, padding=(0, radius))
    region = F.max_pool2d(region, (2 * radius + 1, 1), stride=1, padding=(radius, 0))
    # Replicate at the image boundary so a bad border vertex can still go to zero.
    blurred = F.avg_pool2d(F.pad(region, (feather, feather, 0, 0), mode="replicate"),
                           (1, 2 * feather + 1), stride=1)
    blurred = F.avg_pool2d(F.pad(blurred, (0, 0, feather, feather), mode="replicate"),
                           (2 * feather + 1, 1), stride=1)
    return residual * (1 - blurred).permute(0, 2, 3, 1)


@torch.no_grad()
def guarded_alignment(canvas, base, residual, min_retention, guard=True):
    """Return the actual rendered canvas, global scale and residual strategy."""
    mask = fixed_overlap(canvas)
    if int(mask.sum()) < 64 or not torch.isfinite(residual).all():
        return canvas, 0.0, "tps"
    full = residual_canvas(canvas, base, residual)
    if not guard:
        return full, 1.0, "unguarded"

    original_error = photo(canvas.images, mask)
    baseline = canvas.support > 0.99
    tps_det = grid_determinants(canvas.grid)
    tps_cells = grid_cells(baseline[:, 0])

    def preserves_alignment(candidate):
        visible = candidate.support > 0.99
        retained = (visible & baseline).sum((1, 2, 3)) / baseline.sum((1, 2, 3)).clamp_min(1)
        overlap_retained = masked_mean(visible.all(0, keepdim=True).float(), mask)
        return bool((retained.min() >= min_retention) & (overlap_retained >= min_retention)
                    & (photo(candidate.images, mask) <= original_error + 0.001))

    bad = residual_problems(residual, full, tps_det, tps_cells)
    if not bad.any() and preserves_alignment(full):
        return full, 1.0, "full"
    if bad.any():
        for core, feather in ((1, 5), (3, 9), (7, 15)):
            repaired = local_residual(residual, bad, core, feather)
            candidate = residual_canvas(canvas, base, repaired)
            remaining = residual_problems(repaired, candidate, tps_det, tps_cells)
            if not remaining.any() and preserves_alignment(candidate):
                return candidate, 1.0, "local"
            bad |= remaining
    for scale in (0.5, 0.25):
        scaled = residual * scale
        candidate = residual_canvas(canvas, base, scaled)
        if (not residual_problems(scaled, candidate, tps_det, tps_cells).any()
                and preserves_alignment(candidate)):
            return candidate, scale, "global"
    return canvas, 0.0, "tps"


def guarded_render(canvas, base, residual, min_retention, guard=True):
    result, scale, _ = guarded_alignment(canvas, base, residual, min_retention, guard)
    return result.images, result.support, scale
