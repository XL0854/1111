"""Differentiable RopStitch geometry and PixelStitch refinement."""

from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

from model import (InvalidPair, PixelResidual, fixed_overlap, load_weights, mesh_canvas,
                   objective, photo, pixel_grid, grid_with_offset, refine, render, warp)
from paper_modules import (DifferenceAwareSeamFusion, MultiScaleDualAttention,
                           attach_correlation_adapter, frequency_flow_objective,
                           frequency_photo, seam_objective)


PHASES = ("geometry", "plane", "residual", "fusion", "joint")
FLOW_WEIGHTS = dict(
    baseline=dict(cycle=0.01, smooth=0.015, nonoverlap=0.04, coverage=0.5, fold=0.15),
    freq_low=0.15,
    freq_high=0.05,
)


class Midpoint(nn.Module):
    def __init__(self, dlt, project):
        super().__init__()
        self.dlt, self.project = dlt, project

    def forward(self, left, right, motion):
        corners = motion.new_tensor([[[0, 0], [512, 0], [0, 512], [512, 512]]])
        target = corners + 0.5 * motion
        h = self.dlt(corners, corners + motion)
        ref = torch.linalg.solve(h, self.dlt(corners, target))
        return [self.project(ref, corners)], [target], [torch.full_like(motion[..., :1], 0.5)]


class FullModel(nn.Module):
    def __init__(self, args, init):
        super().__init__()
        base = args.rop / "SemanticStable_20261108/sxl/wCoefNet/Codes"
        if not (base / "network.py").is_file():
            raise FileNotFoundError(base / "network.py")
        sys.path.insert(0, str(base))
        from network import Network, CoefNetwork, H2Offset, H2Mesh, get_rigid_mesh
        from utils.torch_DLT import Tensor_ACA_rect
        from semantic_refine import inverse_tps, plane_grid

        self.geometry = Network(pretrained=False)
        self.alignment_adapter = (
            MultiScaleDualAttention() if args.alignment_adapter == "msda" else None
        )
        if self.alignment_adapter is not None:
            attach_correlation_adapter(self.geometry, self.alignment_adapter)
        self.fixed_plane = args.plane_mode == "fixed-half"
        self.coef = None if self.fixed_plane else CoefNetwork()
        self.flow = PixelResidual(args.pixel, args.pixel_weights if init == "pretrained" else None,
                                  args.max_shift, args.iters)
        self.fusion = (
            DifferenceAwareSeamFusion(args.fusion_width) if args.fusion_mode == "seam" else None
        )
        self.rigid_mesh = get_rigid_mesh
        self.dlt, self.h2mesh = Tensor_ACA_rect, H2Mesh
        self.inverse_tps, self.plane_grid = inverse_tps, plane_grid
        self.midpoint = Midpoint(Tensor_ACA_rect, H2Offset)
        self.edge, self.homo_bound, self.tps_bound = args.max_edge, args.homo_bound, args.tps_bound
        if init == "pretrained":
            load_weights(self.geometry, args.alignment)
            if self.coef is not None:
                load_weights(self.coef, args.coefficient)
        elif init == "scratch":
            heads = [self.geometry.regressHomo_part2[-1], self.geometry.regressTPS_part2[-1]]
            if self.coef is not None:
                heads.append(self.coef.regressCoef_part2[-1])
            for head in heads:
                nn.init.normal_(head.weight, std=1e-4)
                nn.init.zeros_(head.bias)

    def set_phase(self, phase):
        self.eval()
        self.geometry.requires_grad_(phase in ("geometry", "joint"))
        if self.alignment_adapter is not None:
            self.alignment_adapter.requires_grad_(phase in ("geometry", "joint"))
        if self.coef is not None:
            self.coef.requires_grad_(phase in ("plane", "joint"))
        self.flow.requires_grad_(phase in ("residual", "joint"))
        self.flow.raft.image_match.requires_grad_(False)
        if self.fusion is not None:
            self.fusion.requires_grad_(phase in ("fusion", "joint"))

    def fuse(self, images, support):
        if self.fusion is not None:
            return self.fusion(images, support)
        weights = support / support.sum(0, keepdim=True).clamp_min(1e-6)
        fused = (images * weights).sum(0, keepdim=True)
        return fused, None, weights

    def forward(self, images, phase, synthetic_target=None):
        # DDP requires training work to pass through Module.forward so its
        # reducer can synchronize gradients between pair-level workers.
        return training_objective(self, images, phase, synthetic_target)

    def build(self, images, phase):
        train_geometry = phase in ("geometry", "plane", "joint")
        with nullcontext() if train_geometry else torch.no_grad():
            canonical = F.interpolate(images, (512, 512), mode="bilinear",
                                      align_corners=False, antialias=True)
            left, right = (canonical * 2 - 1).split(1)
            captured = {}

            def capture(module, inputs):
                captured["motion"] = inputs[2]

            plane = self.midpoint if self.fixed_plane or phase == "geometry" else self.coef
            hook = plane.register_forward_pre_hook(capture)
            try:
                refs, tgts, _, a, b = self.geometry.forward_finetune(
                    left, right, plane, alpha=0.5, max_homo_px=self.homo_bound,
                    max_local_px=self.tps_bound,
                )
            finally:
                hook.remove()
            rigid = self.rigid_mesh(1, 512, 512)
            meshes = (a / 512, b / 512)
            canvas = mesh_canvas(images, meshes, rigid / 512,
                                 self.inverse_tps, self.plane_grid, self.edge)
            corners = canonical.new_tensor([[[0, 0], [512, 0], [0, 512], [512, 512]]])
            h = self.dlt(corners, corners + captured["motion"])
            plane_mesh = [self.h2mesh(self.dlt(corners, points[-1]), rigid) / 512
                          for points in (refs, tgts)]
        return canvas, meshes, plane_mesh, h, captured["motion"]


def local_mesh_penalty(meshes):
    view = []
    for mesh in meshes:
        gh, gw = mesh.shape[1] - 1, mesh.shape[2] - 1
        dx = gw * (mesh[:, :-1, 1:] - mesh[:, :-1, :-1])
        dy = gh * (mesh[:, 1:, :-1] - mesh[:, :-1, :-1])
        xnorm = dx.norm(dim=-1).clamp_min(1e-4)
        ynorm = dy.norm(dim=-1).clamp_min(1e-4)
        aspect = (xnorm.log() - ynorm.log()).square()
        angle = ((dx * dy).sum(-1) / (xnorm * ynorm)).square()
        det = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        values = (aspect + angle).flatten()
        tail = values.topk(max(1, values.numel() // 10)).values.mean()
        view.append(values.mean() + 0.25 * tail + 4 * F.relu(0.15 - det).square().mean())
    return torch.stack(view).amax()


def global_photo(images, matrix):
    # Match on the fixed first-view raster, so the loss constrains the
    # relative homography even when the common plane is also changing.
    small = F.interpolate(images, (192, 192), mode="bilinear", align_corners=False, antialias=True)
    xy = (pixel_grid(192, 192, small) + 0.5) * (512 / 192)
    points = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)
    projected = points @ matrix.transpose(1, 2)
    denominator = projected[..., 2:]
    signed_eps = torch.where(denominator < 0, -1e-6, 1e-6)
    denominator = torch.where(denominator.abs() < 1e-6, signed_eps, denominator)
    query = 2 * projected[..., :2] / denominator / 512 - 1
    target, support = render(small[1:], query)
    fixed = (support.detach() > 0.99).float()
    error = photo(torch.cat((small[:1], target)), fixed)
    return error + F.relu(0.25 - support.mean())


def synthetic_pair(images, seed):
    # The second image samples the first at x+dx, y+dy. Therefore the
    # desired first-to-second homography is the inverse translation.
    generator = torch.Generator().manual_seed(seed)
    dx, dy = (torch.rand(2, generator=generator) * 64 - 32).tolist()
    h, w = images.shape[-2:]
    base = pixel_grid(h, w, images)
    displacement = base.new_tensor([dx * w / 512, dy * h / 512])
    second = warp(images[:1], grid_with_offset(base, displacement), padding="border")
    return torch.cat((images[:1], second)), base.new_tensor([-dx, -dy]).view(1, 1, 2)


def training_objective(model, images, phase, synthetic_target):
    canvas, meshes, plane_mesh, h, motion = model.build(images, phase)
    fixed = fixed_overlap(canvas)
    if int(fixed.sum()) < 64:
        raise InvalidPair("Fewer than 64 fixed overlap pixels")
    if phase == "residual":
        return frequency_flow_objective(canvas, *model.flow(canvas), objective, FLOW_WEIGHTS)
    if phase == "fusion":
        if model.fusion is None:
            raise ValueError("The fusion phase requires --fusion-mode seam")
        with torch.no_grad():
            residual, _, base = model.flow(canvas)
            refined, support = refine(canvas, base, residual)
        fused, probability, _ = model.fuse(refined, support)
        return seam_objective(refined, support, fused, probability)

    terms = dict(tps_photo=photo(canvas.images, fixed),
                 tps_shape=local_mesh_penalty(meshes),
                 plane_shape=local_mesh_penalty(plane_mesh))
    total = terms["tps_photo"] + 0.045 * terms["tps_shape"] + 0.025 * terms["plane_shape"]
    terms["freq_low"], terms["freq_high"] = frequency_photo(canvas.images, fixed)
    total = total + FLOW_WEIGHTS["freq_low"] * terms["freq_low"] \
        + FLOW_WEIGHTS["freq_high"] * terms["freq_high"]
    if phase in ("geometry", "joint"):
        terms["global_photo"] = global_photo(images, h)
        total = total + 0.5 * terms["global_photo"]
    if synthetic_target is not None:
        terms["synthetic_h"] = F.smooth_l1_loss(motion / 32,
                                              synthetic_target.expand_as(motion) / 32)
        total = total + 0.5 * terms["synthetic_h"]
    if phase == "joint":
        residual, cycle, base = model.flow(canvas)
        flow_loss, flow_terms = frequency_flow_objective(
            canvas, residual, cycle, base, objective, FLOW_WEIGHTS
        )
        terms.update({f"flow_{k}": v for k, v in flow_terms.items()})
        total = total + flow_loss
        if model.fusion is not None:
            refined, support = refine(canvas, base, residual)
            fused, probability, _ = model.fuse(refined, support)
            seam_loss, seam_terms = seam_objective(refined, support, fused, probability)
            terms.update(seam_terms)
            total = total + seam_loss
    return total, terms
