# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


def extract_mesh_spatial_state_maps(mesh, rigid_mesh, out_res=(32, 32)):
    """
    Extract a multi-channel spatial state map capturing local mesh distortion:
    Channels (5 per mesh):
      0: Folding risk (orientation-aware relative area reduction/inversion)
      1: Horizontal edge stretch deviation |len_x / rigid_len_x - 1|
      2: Vertical edge stretch deviation |len_y / rigid_len_y - 1|
      3: Discrete bending curvature (2nd-order Laplacian)
      4: Quad local scale variation |log(cell_area / rigid_area)|
    Returns:
      Tensor of shape [B, 5, out_res[0], out_res[1]]
    """
    B = mesh.size(0)
    eps = 1e-6

    # 1. Orientation-aware folding risk on triangular sub-cells
    v00 = mesh[:, :-1, :-1]
    v10 = mesh[:, 1:, :-1]
    v11 = mesh[:, 1:, 1:]
    v01 = mesh[:, :-1, 1:]

    r00 = rigid_mesh[:, :-1, :-1]
    r10 = rigid_mesh[:, 1:, :-1]
    r11 = rigid_mesh[:, 1:, 1:]
    r01 = rigid_mesh[:, :-1, 1:]

    # Tri 1: v00 -> v10 -> v11
    e1_t1 = v10 - v00
    e2_t1 = v11 - v00
    area_t1 = 0.5 * (e1_t1[..., 0] * e2_t1[..., 1] - e1_t1[..., 1] * e2_t1[..., 0])

    r_e1_t1 = r10 - r00
    r_e2_t1 = r11 - r00
    r_area_t1 = 0.5 * (r_e1_t1[..., 0] * r_e2_t1[..., 1] - r_e1_t1[..., 1] * r_e2_t1[..., 0])

    # Tri 2: v00 -> v11 -> v01
    e1_t2 = v11 - v00
    e2_t2 = v01 - v00
    area_t2 = 0.5 * (e1_t2[..., 0] * e2_t2[..., 1] - e1_t2[..., 1] * e2_t2[..., 0])

    r_e1_t2 = r11 - r00
    r_e2_t2 = r01 - r00
    r_area_t2 = 0.5 * (r_e1_t2[..., 0] * r_e2_t2[..., 1] - r_e1_t2[..., 1] * r_e2_t2[..., 0])

    # Reference triangle signs to correctly detect orientation reversal
    sign_r1 = torch.sign(r_area_t1)
    sign_r2 = torch.sign(r_area_t2)

    # Relative signed area ratio rho:
    #   rho = 1.0 for rigid/un-deformed mesh
    #   rho > 0.2 for normal/mild deformation
    #   0 <= rho <= 0.2 for severe compression / near-degenerate
    #   rho < 0 for flipped/inverted orientation
    rho_t1 = (area_t1 * sign_r1) / (torch.abs(r_area_t1) + eps)
    rho_t2 = (area_t2 * sign_r2) / (torch.abs(r_area_t2) + eps)
    min_rho = torch.minimum(rho_t1, rho_t2)

    # Folding risk threshold: safe margin tau = 0.2
    # Normal mesh has rho=1.0 -> fold_risk = 0.0
    fold_risk = F.relu(0.2 - min_rho)

    # 2. Local stretch on edges
    e_x = mesh[:, :, 1:, :] - mesh[:, :, :-1, :]
    r_x = rigid_mesh[:, :, 1:, :] - rigid_mesh[:, :, :-1, :]
    ratio_x = torch.norm(e_x, dim=-1) / (torch.norm(r_x, dim=-1) + eps)
    stretch_x_cells = 0.5 * (ratio_x[:, :-1, :] + ratio_x[:, 1:, :])  # cell-aligned [B, H, W]
    stretch_x_dev = torch.abs(stretch_x_cells - 1.0)

    e_y = mesh[:, 1:, :, :] - mesh[:, :-1, :, :]
    r_y = rigid_mesh[:, 1:, :, :] - rigid_mesh[:, :-1, :, :]
    ratio_y = torch.norm(e_y, dim=-1) / (torch.norm(r_y, dim=-1) + eps)
    stretch_y_cells = 0.5 * (ratio_y[:, :, :-1] + ratio_y[:, :, 1:])  # cell-aligned [B, H, W]
    stretch_y_dev = torch.abs(stretch_y_cells - 1.0)

    # 3. Scale variation
    cell_area = torch.abs(area_t1) + torch.abs(area_t2)
    cell_rigid_area = torch.abs(r_area_t1) + torch.abs(r_area_t2) + eps
    scale_var = torch.abs(torch.log(torch.clamp(cell_area / cell_rigid_area, min=1e-3, max=1e3)))

    # 4. Curvature (bending)
    d2_h = mesh[:, :, 2:, :] - 2 * mesh[:, :, 1:-1, :] + mesh[:, :, :-2, :]
    d2_v = mesh[:, 2:, :, :] - 2 * mesh[:, 1:-1, :, :] + mesh[:, :-2, :]
    bend_h = torch.norm(d2_h, dim=-1)
    bend_v = torch.norm(d2_v, dim=-1)
    # Pad to match (H, W) cells
    bend_map = F.pad(bend_h[:, 1:-1, :], (0, 1, 0, 1)) + F.pad(bend_v[:, :, 1:-1], (0, 1, 0, 1))
    step_ref = torch.mean(torch.norm(r_x, dim=-1)) + eps
    bend_normalized = bend_map / step_ref

    # Guard against NaN / Inf across all channels
    fold_risk = torch.nan_to_num(fold_risk, nan=0.0, posinf=1.0, neginf=0.0)
    stretch_x_dev = torch.nan_to_num(stretch_x_dev, nan=0.0, posinf=10.0, neginf=0.0)
    stretch_y_dev = torch.nan_to_num(stretch_y_dev, nan=0.0, posinf=10.0, neginf=0.0)
    bend_normalized = torch.nan_to_num(bend_normalized, nan=0.0, posinf=10.0, neginf=0.0)
    scale_var = torch.nan_to_num(scale_var, nan=0.0, posinf=10.0, neginf=0.0)

    # Stack 5 channels: [B, 5, grid_h, grid_w]
    state_grid = torch.stack([fold_risk, stretch_x_dev, stretch_y_dev, bend_normalized, scale_var], dim=1)

    # Interpolate to target resolution
    state_map = F.interpolate(state_grid, size=out_res, mode='bilinear', align_corners=False)
    return state_map


class TPSStateFeedbackCorrector(nn.Module):
    """
    Feedback Corrector Network predicting Delta z for virtual plane adjustment:
    Input:
      - Spatial state maps for both ref and tgt meshes (10 channels)
      - Initial coefficient C0 (4 values)
    Output:
      - Updated coefficient C1 = sigmoid(logit(C0) + Delta z)
      - Predicted delta_z (initialized to 0 at step 0)
    """
    def __init__(self, in_channels=10, out_channels=4):
        super(TPSStateFeedbackCorrector, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc_head = nn.Sequential(
            nn.Linear(128 + 4, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels, bias=True)
        )

        # Critical: Zero-initialize the final projection layer
        # This guarantees Delta z == 0 and C1 == C0 initially.
        final_linear = self.fc_head[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(self, mesh_ref, mesh_tgt, rigid_mesh, c0):
        """
        mesh_ref, mesh_tgt, rigid_mesh: [B, H+1, W+1, 2]
        c0: [B, 4] or [B, 4, 1]
        """
        if c0.dim() == 3:
            c0_flat = c0.squeeze(-1)
        else:
            c0_flat = c0

        # Extract spatial maps
        s_ref = extract_mesh_spatial_state_maps(mesh_ref, rigid_mesh)
        s_tgt = extract_mesh_spatial_state_maps(mesh_tgt, rigid_mesh)
        combined_state = torch.cat([s_ref, s_tgt], dim=1)  # [B, 10, 32, 32]

        feat = self.conv_block(combined_state)
        feat_flat = feat.view(feat.size(0), -1)  # [B, 128]

        fused = torch.cat([feat_flat, c0_flat], dim=1)  # [B, 132]
        delta_z = self.fc_head(fused)  # [B, 4]

        # Invert sigmoid safely via logit
        eps = 1e-6
        c0_clamped = torch.clamp(c0_flat, min=eps, max=1.0 - eps)
        logit_c0 = torch.log(c0_clamped / (1.0 - c0_clamped))

        # Updated coefficient
        c1 = torch.sigmoid(logit_c0 + delta_z)

        if c0.dim() == 3:
            c1 = c1.unsqueeze(-1)
            delta_z = delta_z.unsqueeze(-1)

        return c1, delta_z
