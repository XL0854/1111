#!/usr/bin/env python3
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from network import Network, CoefNetwork, build_model
from loss import mesh_folding_loss, mesh_stretch_loss, mesh_bending_loss, tps_overlap_align_loss


def test_gradient_flow():
    print("================ Checking M1 Gradient Flow ================")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = Network().to(device)
    coef_net = CoefNetwork().to(device)

    # 1. 冻结原始配准网络 net 的所有参数
    for param in net.parameters():
        param.requires_grad = False

    # 2. 保持系数网络 coef_net 可导
    for param in coef_net.parameters():
        param.requires_grad = True

    # 构造虚拟输入
    B, C, H, W = 2, 3, 512, 512
    input1 = torch.randn(B, C, H, W, device=device)
    input2 = torch.randn(B, C, H, W, device=device)

    # 前向计算 (enable_tps_grad=True)
    out_dict = build_model(net, coef_net, input1, input2, is_training=True, enable_tps_grad=True)

    mesh_ref = out_dict['mesh_ref']
    mesh_tgt = out_dict['mesh_tgt']
    rigid_mesh = out_dict['rigid_mesh']
    output_tps_ref = out_dict['output_tps_ref']
    output_tps_tgt = out_dict['output_tps_tgt']

    # 计算各项网格与对齐损失
    loss_fold = mesh_folding_loss(mesh_ref, rigid_mesh) + mesh_folding_loss(mesh_tgt, rigid_mesh)
    loss_stretch = mesh_stretch_loss(mesh_ref, rigid_mesh) + mesh_stretch_loss(mesh_tgt, rigid_mesh)
    loss_bend = mesh_bending_loss(mesh_ref, rigid_mesh) + mesh_bending_loss(mesh_tgt, rigid_mesh)
    loss_align = tps_overlap_align_loss(output_tps_ref, output_tps_tgt)

    loss_total = loss_fold + loss_stretch + loss_bend + loss_align

    print(f"loss_fold:    {loss_fold.item():.6f}")
    print(f"loss_stretch: {loss_stretch.item():.6f}")
    print(f"loss_bend:    {loss_bend.item():.6f}")
    print(f"loss_align:   {loss_align.item():.6f}")
    print(f"loss_total:   {loss_total.item():.6f}")

    # 反向传播
    loss_total.backward()

    # 验证 net 参数没有梯度 (完全冻结)
    net_grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(net_grads) == 0, f"net parameters should have NO gradients, found {len(net_grads)}"
    print("[PASS] net 权重参数完全冻结，未产生任何梯度！")

    # 验证 coef_net 获得了有效非零梯度
    coef_grads_with_grad = []
    has_nan_or_inf = False
    for name, p in coef_net.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            coef_grads_with_grad.append((name, norm))
            if not torch.isfinite(p.grad).all():
                has_nan_or_inf = True

    assert len(coef_grads_with_grad) > 0, "coef_net must receive gradients from TPS mesh losses!"
    assert not has_nan_or_inf, "Gradients in coef_net contain NaN or Inf!"

    print(f"[PASS] coef_net 成功接收来自 TPS 几何与对齐损失的梯度！共有 {len(coef_grads_with_grad)} 组参数获得梯度更新！")
    print("样例梯度范数:", coef_grads_with_grad[-1])
    print("================ M1 GRADIENT FLOW VERIFICATION SUCCESSFUL ================")


if __name__ == "__main__":
    test_gradient_flow()
