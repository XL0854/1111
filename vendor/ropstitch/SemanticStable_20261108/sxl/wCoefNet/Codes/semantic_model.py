"""Trainable RopStitch, instance allocation and lkm composition."""

import torch
from torch import nn
import torch.nn.functional as F

from network import CoefNetwork, H2Offset, Network, get_rigid_mesh
from semantic_refine import (assess, differences, image_and_support, instance_matches,
                             mesh_geometry, plane_grid, project, sample)
from semantic_stitch import composition_model
from utils.torch_DLT import Tensor_ACA_rect


def corners_like(tensor):
    return tensor.new_tensor([[[0, 0], [512, 0], [0, 512], [512, 512]]])


def midplane(image1, image2, motion):
    corners = corners_like(motion)
    coefficients = torch.full_like(motion[..., :1], 0.5)
    right = corners + motion * coefficients
    relative = Tensor_ACA_rect(corners, corners + motion)
    h_right = Tensor_ACA_rect(corners, right)
    left = H2Offset(torch.linalg.solve(relative, h_right), corners)
    return [left], [right], [coefficients]


class SemanticStitchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.alignment = Network(pretrained=False)
        # Both branches must learn when neither starts with a pretrained prior.
        self.alignment.fronzen_resnet_backbone.requires_grad_(True)
        self.coef = CoefNetwork()
        self.composition = composition_model()
        # Legacy dilated blocks use padding=1 at all scales. Preserve feature
        # sizes in the trainable variant so a 64px raster survives four pools.
        for layer in self.composition.modules():
            if isinstance(layer, nn.Conv2d) and layer.dilation != (1, 1):
                layer.padding = layer.dilation
        self.allocation = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((13, 13)), nn.Conv2d(32, 3, 3, padding=1),
        )
        for layer in (self.alignment.regressHomo_part2[-1], self.alignment.regressTPS_part2[-1],
                      self.coef.regressCoef_part2[-1], self.allocation[-1], self.composition.out[0]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def configure_training(self, phase):
        for module, enabled in (
            (self.alignment, phase != "plane"), (self.coef, phase != "alignment"),
            (self.allocation, phase == "joint"), (self.composition, phase == "joint"),
        ):
            module.train(enabled)
            module.requires_grad_(enabled)

    def forward(self, images, ids, foreground, *, phase, alpha, resolution, shift=None, geometry=None,
                target_h=None, stable=False):
        source = get_rigid_mesh(1, 512, 512) / 512
        coef_model = midplane if phase == "alignment" else self.coef
        left, right, coefficients, mesh1, mesh2 = self.alignment.forward_finetune(
            *images, coef_model, alpha=alpha, max_homo_px=160, max_local_px=20,
        )
        homographies = tuple(Tensor_ACA_rect(corners_like(mesh1), c[-1]) for c in (left, right))
        meshes = (mesh1 / 512, mesh2 / 512)
        if geometry is not None:
            geometry.update(homographies=homographies, base_meshes=meshes)
        bounds = (mesh1.new_full((2,), -0.35), mesh1.new_full((2,), 1.35))
        grid = plane_grid(bounds, (resolution, resolution))
        inverse_global = [project(h, grid) for h in homographies]
        global_rgb, global_valid = zip(*(image_and_support(im, xy)
                                         for im, xy in zip(images, inverse_global)))
        global_overlap = global_valid[0] * global_valid[1]
        global_photo = ((global_rgb[0] - global_rgb[1]).abs().mean(1, keepdim=True)
                        * global_overlap).sum() / global_overlap.sum().clamp_min(1)
        with torch.no_grad():
            ids1, ids2, confidence, _ = instance_matches(ids, images, homographies, grid, limit=16)
            matched = (ids1, ids2)
            rigidity = tuple(1 + 2 * fg + 2 * (m * confidence[None, :, None, None]).sum(1, keepdim=True)
                             for fg, m in zip(foreground, matched))
            stiffness = tuple(sample(r, source, "nearest")[:, 0, :-1, :-1].clamp_min(1)
                              for r in rigidity)
        zero = global_photo * 0
        allocation_cost, smoothness = zero, zero
        if phase == "joint":
            plane_foreground = [sample(fg, xy) for fg, xy in zip(foreground, inverse_global)]
            plane_rigidity = [1 + sample(r - 1, xy) for r, xy in zip(rigidity, inverse_global)]
            features = torch.cat((*global_rgb, *global_valid, *plane_foreground, *plane_rigidity), dim=1)
            prediction = self.allocation(features)
            displacement = (12 / 512) * prediction[:, :2].tanh()
            k1, k2 = [F.adaptive_avg_pool2d(k, (13, 13)) for k in plane_rigidity]
            prior = k2 / (k1 + k2)
            share = (torch.logit(prior) + prediction[:, 2:].tanh()).sigmoid()
            allocation_cost = (share - prior).square().mean()
            dx, dy = differences(displacement.permute(0, 2, 3, 1) * 512 / 12)
            smoothness = dx.square().mean() + dy.square().mean()
            fields = (share * displacement, -(1 - share) * displacement)
            meshes = tuple(mesh + sample(field, (mesh - bounds[0]) / (bounds[1] - bounds[0]))
                           .permute(0, 2, 3, 1) for mesh, field in zip(meshes, fields))
        metrics = assess(meshes, source, grid, images, foreground, matched, confidence, stiffness,
                         self.composition if phase == "joint" else None, stable=stable)
        global_meshes = tuple(project(torch.linalg.inv(h), source) for h in homographies)
        plane_shape = sum(mesh_geometry(mesh, source, w)[0]
                          for mesh, w in zip(global_meshes, stiffness)) / 2
        coverage = (F.relu(0.12 - metrics["overlap"])
                    + F.relu(0.29 - metrics["coverage_left"])
                    + F.relu(0.29 - metrics["coverage_right"]))
        fold = F.relu(0.08 - metrics["jacobian"]) + F.relu(0.08 - metrics["inverse_jacobian"])
        synthetic = zero
        if shift is not None or target_h is not None:
            relative = homographies[1] @ torch.linalg.inv(homographies[0])
            target = project(target_h, source) if target_h is not None else source + shift
            synthetic = F.smooth_l1_loss(project(relative, source), target, beta=0.03)
        loss = (metrics["alignment"] + 0.5 * global_photo + 0.06 * metrics["shape"]
                + 0.1 * plane_shape + 0.5 * coverage + 0.4 * fold + 5 * synthetic)
        if phase == "joint":
            loss = (loss + 0.12 * metrics["instance"] + 0.02 * allocation_cost + 0.01 * smoothness
                    + 0.25 * metrics["anchor"] + 2 * metrics["seam"] + metrics["seam_detail"]
                    + 0.15 * metrics["ghost"])
        if stable:
            curvature = zero
            for mesh, base in zip(meshes, global_meshes):
                residual = (mesh - base) * (512 / 20)
                curvature = curvature + sum(residual.diff(n=2, dim=d).abs().mean() for d in (1, 2))
            loss = loss + 0.15 * metrics["edge_alignment"] + 0.025 * curvature
        metrics.update(loss=loss, global_photo=global_photo, plane_shape=plane_shape,
                       synthetic=synthetic, matched_instances=confidence.new_tensor(confidence.numel()))
        return meshes, source, coefficients[-1], metrics
