#!/usr/bin/env python3
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feedback_network import TPSStateFeedbackCorrector, extract_mesh_spatial_state_maps


def test_m2_identity_and_gradients():
    print("================== Testing M2 Feedback Corrector ==================")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    B, H_cells, W_cells = 2, 12, 12
    # 构造刚性网格与可导初始系数
    y = torch.linspace(0.0, 512.0, H_cells + 1, device=device)
    x = torch.linspace(0.0, 512.0, W_cells + 1, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    rigid_mesh = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    mesh_ref = rigid_mesh.clone()
    mesh_tgt = rigid_mesh.clone()

    c0 = torch.tensor([[0.3, 0.5, 0.7, 0.4], [0.6, 0.2, 0.8, 0.5]], device=device, requires_grad=True)

    corrector = TPSStateFeedbackCorrector().to(device)

    # 1. 验证零初始化恒等性
    c1, delta_z = corrector(mesh_ref, mesh_tgt, rigid_mesh, c0)

    print("delta_z initial values:", delta_z)
    max_dz = torch.max(torch.abs(delta_z)).item()
    max_diff_c = torch.max(torch.abs(c1 - c0)).item()

    assert max_dz == 0.0, f"Initial delta_z must be 0.0, got {max_dz}"
    assert max_diff_c < 1e-6, f"Initial C1 must be identical to C0, got diff {max_diff_c}"
    print(f"[PASS] 零初始化恒等性检验通过! max(|delta_z|) = {max_dz:.8f}, max(|C1 - C0|) = {max_diff_c:.8f}\n")

    # 2. 空间状态图响应性检验
    mesh_bent = rigid_mesh.clone()
    mesh_bent[..., 1] += 30.0 * torch.sin(rigid_mesh[..., 0] / 50.0)
    state_reg = extract_mesh_spatial_state_maps(rigid_mesh, rigid_mesh)
    state_bent = extract_mesh_spatial_state_maps(mesh_bent, rigid_mesh)

    bend_reg_mean = state_reg[:, 3].mean().item()
    bend_bent_mean = state_bent[:, 3].mean().item()
    print(f"Bending map mean (regular): {bend_reg_mean:.6f}")
    print(f"Bending map mean (bent):    {bend_bent_mean:.6f}")
    assert bend_bent_mean > bend_reg_mean + 0.05, "Bent mesh must trigger high response in bending channel"
    print("[PASS] 空间状态图对几何形变具有极高敏感度和可解释性!\n")

    # 3. 梯度传导检验
    target_c = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]], device=device)
    loss = torch.sum((c1 - target_c) ** 2)
    loss.backward()

    # 验证 corrector 参数可导且获得梯度
    fc_head_grad = corrector.fc_head[-1].weight.grad
    assert fc_head_grad is not None and torch.isfinite(fc_head_grad).all(), "Feedback head must have valid gradients"
    assert c0.grad is not None and torch.isfinite(c0.grad).all(), "C0 must have valid backward gradients"
    print("[PASS] 反向传播梯度无阻断完整流通!\n")

    print("================== ALL M2 MODULE TESTS PASSED ==================")


if __name__ == "__main__":
    test_m2_identity_and_gradients()
