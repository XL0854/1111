# coding: utf-8
import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from network import Network, CoefNetwork, build_model, get_rigid_mesh, get_norm_mesh, H2Mesh
from feedback_network import TPSStateFeedbackCorrector
from dataset import TrainDataset, TestDataset
from loss import (
    cal_ddmperception_loss,
    mesh_folding_loss,
    mesh_stretch_loss,
    mesh_bending_loss,
    tps_overlap_align_loss,
)
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_homo_ddm as torch_homo_ddm
import utils.torch_tps_transform as torch_tps_transform
import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model_m2')
SUMMARY_DIR = os.path.join(last_path, 'summary_m2')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

writer = SummaryWriter(log_dir=SUMMARY_DIR)


def forward_m2_full(net, coef_net, feedback_corrector, input1, input2, alpha=0.5):
    """
    Two-pass feedback loop:
    Pass 1: Predict C0 -> warp feature -> predict mesh0
    Feedback: Extract spatial state map from mesh0 -> predict Delta z -> C1 = sigmoid(logit(C0) + Delta z)
    Pass 2: Recompute homography with C1 -> predict refined mesh1
    """
    B, _, img_h, img_w = input1.size()

    # Pass 1
    dst_p_ref_list0, dst_p_tgt_list0, final_coef_list0, mesh_ref0, mesh_tgt0 = net(
        input1, input2, coef_net, alpha=alpha, enable_tps_grad=True
    )
    c0 = final_coef_list0[-1]
    rigid_mesh = get_rigid_mesh(B, img_h, img_w)

    # Feedback correction
    c1, delta_z = feedback_corrector(mesh_ref0, mesh_tgt0, rigid_mesh, c0)

    # Pass 2: Re-generate homography decomposition with refined C1
    with torch.no_grad():
        _, _, frozen_feat_2_64, _ = net.fronzen_resnet_backbone(input1, input2)
        active_feat_1_64, active_feat_1_32, active_feat_2_64, active_feat_2_32 = net.active_resnet_backbone(input1, input2)
        c_32 = (1 - alpha) * net.CCL(net.fronzen_resnet_backbone(input1, input2)[1], net.fronzen_resnet_backbone(input1, input2)[3]) \
               + alpha * net.CCL(active_feat_1_32, active_feat_2_32)
        t_1 = net.regressHomo_part1(c_32).view(B, -1)
        offset_1 = net.regressHomo_part2(t_1)

    H_motion = offset_1.reshape(-1, 4, 2)
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]], device=input1.device).unsqueeze(0).expand(B, -1, -1)
    dst_p = src_p + H_motion
    dst_p_tgt1 = src_p + H_motion * c1
    H = torch_DLT.Tensor_ACA_rect(src_p, dst_p)
    H_tgt1 = torch_DLT.Tensor_ACA_rect(src_p, dst_p_tgt1)
    H_ref1 = torch.matmul(torch.inverse(H), H_tgt1)

    ini_mesh_ref1 = H2Mesh(H_ref1, rigid_mesh)
    ini_mesh_tgt1 = H2Mesh(H_tgt1, rigid_mesh)

    # Warp features by updated homography
    H_ref_feat = torch_DLT.Tensor_ACA_rect(src_p/8, (src_p + H_motion * (1 - c1))/8)
    H_tgt_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_tgt1/8)

    M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                             [0., img_h/8 / 2.0, img_h/8 / 2.0],
                             [0., 0., 1.]], device=input1.device).unsqueeze(0).expand(B, -1, -1)
    M_inv = torch.inverse(M_tensor)

    H_mat_ref = torch.matmul(torch.matmul(M_inv, H_ref_feat), M_tensor)
    warp_f1_ref = torch_homo_transform.transformer(active_feat_1_64, H_mat_ref, (int(img_h/8), int(img_w/8)))
    H_mat_tgt = torch.matmul(torch.matmul(M_inv, H_tgt_feat), M_tensor)
    warp_f2_tgt = torch_homo_transform.transformer(active_feat_2_64, H_mat_tgt, (int(img_h/8), int(img_w/8)))

    # Refined TPS mesh
    tps1_ref = net.regressTPS_part1(torch.cat([warp_f1_ref, warp_f2_tgt], 1)).reshape(B, -1)
    motion_ref1 = net.regressTPS_part2(tps1_ref).reshape(-1, grid_h+1, grid_w+1, 2)

    tps1_tgt = net.regressTPS_part1(torch.cat([warp_f2_tgt, warp_f1_ref], 1)).reshape(B, -1)
    motion_tgt1 = net.regressTPS_part2(tps1_tgt).reshape(-1, grid_h+1, grid_w+1, 2)

    mesh_ref1 = ini_mesh_ref1 + motion_ref1
    mesh_tgt1 = ini_mesh_tgt1 + motion_tgt1

    # TPS warped image for alignment
    norm_rigid = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_ref1 = get_norm_mesh(mesh_ref1, img_h, img_w)
    norm_tgt1 = get_norm_mesh(mesh_tgt1, img_h, img_w)
    mask = torch.ones_like(input2)

    output_tps_ref1 = torch_tps_transform.transformer(torch.cat((input1+1, mask), 1), norm_ref1, norm_rigid, (img_h, img_w))
    output_tps_tgt1 = torch_tps_transform.transformer(torch.cat((input2+1, mask), 1), norm_tgt1, norm_rigid, (img_h, img_w))

    return {
        'c0': c0,
        'c1': c1,
        'delta_z': delta_z,
        'mesh_ref0': mesh_ref0,
        'mesh_tgt0': mesh_tgt0,
        'mesh_ref1': mesh_ref1,
        'mesh_tgt1': mesh_tgt1,
        'rigid_mesh': rigid_mesh,
        'output_tps_ref1': output_tps_ref1,
        'output_tps_tgt1': output_tps_tgt1,
    }


def train_m2(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=2, shuffle=True, drop_last=True)

    net = Network().cuda()
    coef_net = CoefNetwork().cuda()
    feedback_corrector = TPSStateFeedbackCorrector().cuda()

    # Freeze base registration backbone
    for p in net.parameters():
        p.requires_grad = False

    homo_checkpoint = torch.load(args.woCoefNet_path)
    net.load_state_dict(homo_checkpoint['model'])

    # Initialize coef_net from official baseline
    if args.resume_coef_path and os.path.exists(args.resume_coef_path):
        c_ckpt = torch.load(args.resume_coef_path)
        coef_net.load_state_dict(c_ckpt['model'])
        print(f"Loaded pretrained coef_net from {args.resume_coef_path}")

    # Optimize both coef_net and feedback_corrector
    optimizer = optim.Adam([
        {'params': coef_net.parameters(), 'lr': args.lr * 0.2},
        {'params': feedback_corrector.parameters(), 'lr': args.lr}
    ], betas=(0.9, 0.999), eps=1e-08)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    print("================== Start M2 State Feedback Training ==================")
    score_print_freq = 100
    glob_iter = 0

    for epoch in range(args.max_epoch):
        coef_net.train()
        feedback_corrector.train()
        net.eval()

        loss_sigma = 0.0
        fold_sigma = 0.0
        stretch_sigma = 0.0
        bend_sigma = 0.0
        align_sigma = 0.0
        reg_sigma = 0.0

        for i, batch_value in enumerate(train_loader):
            input1 = batch_value[0].float().cuda()
            input2 = batch_value[1].float().cuda()

            optimizer.zero_grad()
            res = forward_m2_full(net, coef_net, feedback_corrector, input1, input2)

            mesh_ref1 = res['mesh_ref1']
            mesh_tgt1 = res['mesh_tgt1']
            rigid_mesh = res['rigid_mesh']
            delta_z = res['delta_z']
            out_tps_ref1 = res['output_tps_ref1']
            out_tps_tgt1 = res['output_tps_tgt1']

            loss_fold = mesh_folding_loss(mesh_ref1, rigid_mesh) + mesh_folding_loss(mesh_tgt1, rigid_mesh)
            loss_stretch = mesh_stretch_loss(mesh_ref1, rigid_mesh) + mesh_stretch_loss(mesh_tgt1, rigid_mesh)
            loss_bend = mesh_bending_loss(mesh_ref1, rigid_mesh) + mesh_bending_loss(mesh_tgt1, rigid_mesh)
            loss_align = tps_overlap_align_loss(out_tps_ref1, out_tps_tgt1)

            # L2 regularization on delta_z to ensure conservative feedback
            loss_reg_dz = torch.mean(delta_z ** 2)

            total_loss = (
                args.lambda_align * loss_align
                + args.lambda_fold * loss_fold
                + args.lambda_stretch * loss_stretch
                + args.lambda_bend * loss_bend
                + 0.1 * loss_reg_dz
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(feedback_corrector.parameters(), max_norm=3.0)
            torch.nn.utils.clip_grad_norm_(coef_net.parameters(), max_norm=3.0)
            optimizer.step()

            loss_sigma += total_loss.item()
            fold_sigma += loss_fold.item()
            stretch_sigma += loss_stretch.item()
            bend_sigma += loss_bend.item()
            align_sigma += loss_align.item()
            reg_sigma += loss_reg_dz.item()

            if (i + 1) % score_print_freq == 0:
                print(f"M2 Epoch [{epoch+1:03d}/{args.max_epoch:03d}] Iter [{i+1:04d}/{len(train_loader):04d}] "
                      f"Total: {loss_sigma/score_print_freq:.4f} | Align: {align_sigma/score_print_freq:.4f} | "
                      f"Fold: {fold_sigma/score_print_freq:.4f} | Stretch: {stretch_sigma/score_print_freq:.4f} | "
                      f"Bend: {bend_sigma/score_print_freq:.4f} | |dz|: {reg_sigma/score_print_freq:.5f}")

                writer.add_scalar('M2/Total_Loss', loss_sigma / score_print_freq, glob_iter)
                writer.add_scalar('M2/Fold_Loss', fold_sigma / score_print_freq, glob_iter)
                writer.add_scalar('M2/Stretch_Loss', stretch_sigma / score_print_freq, glob_iter)
                writer.add_scalar('M2/Bend_Loss', bend_sigma / score_print_freq, glob_iter)

                loss_sigma, fold_sigma, stretch_sigma, bend_sigma, align_sigma, reg_sigma = 0, 0, 0, 0, 0, 0

            glob_iter += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.max_epoch:
            save_path = os.path.join(MODEL_DIR, f"epoch{epoch+1:03d}_m2_model.pth")
            torch.save({
                'coef_net': coef_net.state_dict(),
                'feedback_corrector': feedback_corrector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, save_path)
            print(f"Saved M2 checkpoint to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--woCoefNet_path', type=str, default='/workspace/python_pro/test1/RopStitch/woCoefNet/model_homo/epoch100_model.pth')
    parser.add_argument('--resume_coef_path', type=str, default='/workspace/python_pro/test1/RopStitch/wCoefNet/model_coef/epoch050_coefmodel.pth')
    parser.add_argument('--train_path', type=str, default='/workspace/python_pro/test1/RopStitch/data/UDIS-D/training/')
    parser.add_argument('--lambda_align', type=float, default=2.0)
    parser.add_argument('--lambda_fold', type=float, default=5.0)
    parser.add_argument('--lambda_stretch', type=float, default=10.0)
    parser.add_argument('--lambda_bend', type=float, default=10.0)

    args = parser.parse_args()
    train_m2(args)
