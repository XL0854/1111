import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network import Network, CoefNetwork, get_rigid_mesh, get_norm_mesh, H2Mesh
from tapr_refiner import TAPRRefiner
from loss_m3 import M3StitchingLoss
from loss import mesh_folding_loss, mesh_stretch_loss, mesh_bending_loss
from dataset import TrainDataset
import grid_res
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

MODEL_DIR = '/workspace/python_pro/test1/RopStitch/wCoefNet/model_m3'
SUMMARY_DIR = '/workspace/python_pro/test1/RopStitch/wCoefNet/summary_m3'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=SUMMARY_DIR)


def forward_m3_full(net, coef_net, tapr_refiner, input1, input2, alpha=0.5):
    """
    M3 Forward Pipeline:
    1. Base feature extraction & Homography regression (net backbone)
    2. CoefNet homography decomposition & TPS warping (Baseline B0)
    3. TAPR Residual Refinement: Predicts delta_ref and delta_tgt
    4. Refined TPS warping for final outputs
    """
    B, _, img_h, img_w = input1.size()

    # Pass 1: Base network inference
    dst_p_ref_list0, dst_p_tgt_list0, final_coef_list0, mesh_ref0, mesh_tgt0 = net(
        input1, input2, coef_net, alpha=alpha, enable_tps_grad=True
    )
    c0 = final_coef_list0[-1]
    rigid_mesh = get_rigid_mesh(B, img_h, img_w)

    # Feature representation for TAPR
    with torch.no_grad():
        _, _, frozen_feat_2_64, _ = net.fronzen_resnet_backbone(input1, input2)
        active_feat_1_64, active_feat_1_32, active_feat_2_64, active_feat_2_32 = net.active_resnet_backbone(input1, input2)
        c_32 = (1 - alpha) * net.CCL(net.fronzen_resnet_backbone(input1, input2)[1], net.fronzen_resnet_backbone(input1, input2)[3]) \
               + alpha * net.CCL(active_feat_1_32, active_feat_2_32)
        t_1 = net.regressHomo_part1(c_32).view(B, -1)
        offset_1 = net.regressHomo_part2(t_1)

    H_motion = offset_1.reshape(-1, 4, 2)
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]], device=input1.device).unsqueeze(0).expand(B, -1, -1)
    dst_p_tgt = src_p + H_motion * c0

    # Feature warping by base homography
    H_ref_feat = torch_DLT.Tensor_ACA_rect(src_p/8, (src_p + H_motion * (1 - c0))/8)
    H_tgt_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_tgt/8)

    M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                             [0., img_h/8 / 2.0, img_h/8 / 2.0],
                             [0., 0., 1.]], device=input1.device).unsqueeze(0).expand(B, -1, -1)
    M_inv = torch.inverse(M_tensor)

    H_mat_ref = torch.matmul(torch.matmul(M_inv, H_ref_feat), M_tensor)
    warp_f1_ref = torch_homo_transform.transformer(active_feat_1_64, H_mat_ref, (int(img_h/8), int(img_w/8)))
    H_mat_tgt = torch.matmul(torch.matmul(M_inv, H_tgt_feat), M_tensor)
    warp_f2_tgt = torch_homo_transform.transformer(active_feat_2_64, H_mat_tgt, (int(img_h/8), int(img_w/8)))

    # Base TPS warped images for computing photometric difference
    norm_rigid = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_ref0 = get_norm_mesh(mesh_ref0, img_h, img_w)
    norm_tgt0 = get_norm_mesh(mesh_tgt0, img_h, img_w)
    mask = torch.ones_like(input2)

    output_tps_ref0 = torch_tps_transform.transformer(torch.cat((input1+1, mask), 1), norm_ref0, norm_rigid, (img_h, img_w))
    output_tps_tgt0 = torch_tps_transform.transformer(torch.cat((input2+1, mask), 1), norm_tgt0, norm_rigid, (img_h, img_w))

    # Calculate photometric residual in overlapping region
    img_ref0 = output_tps_ref0[:, 0:3]
    img_tgt0 = output_tps_tgt0[:, 0:3]
    m_ref0 = output_tps_ref0[:, 3:6]
    m_tgt0 = output_tps_tgt0[:, 3:6]
    overlap_mask0 = m_ref0 * m_tgt0
    diff_img = torch.abs(img_ref0 - img_tgt0) * overlap_mask0

    # TAPR predicts residual displacement field (delta_ref, delta_tgt)
    delta_ref, delta_tgt = tapr_refiner(warp_f1_ref, warp_f2_tgt, diff_img)

    # Refined meshes (delta is zero-initialized -> equals base mesh at start)
    refined_mesh_ref = mesh_ref0 + delta_ref
    refined_mesh_tgt = mesh_tgt0 + delta_tgt

    # Final TPS warped output with refined meshes
    norm_ref_final = get_norm_mesh(refined_mesh_ref, img_h, img_w)
    norm_tgt_final = get_norm_mesh(refined_mesh_tgt, img_h, img_w)

    final_out_ref = torch_tps_transform.transformer(torch.cat((input1+1, mask), 1), norm_ref_final, norm_rigid, (img_h, img_w))
    final_out_tgt = torch_tps_transform.transformer(torch.cat((input2+1, mask), 1), norm_tgt_final, norm_rigid, (img_h, img_w))

    return {
        'rigid_mesh': rigid_mesh,
        'mesh_ref0': mesh_ref0,
        'mesh_tgt0': mesh_tgt0,
        'refined_mesh_ref': refined_mesh_ref,
        'refined_mesh_tgt': refined_mesh_tgt,
        'delta_ref': delta_ref,
        'delta_tgt': delta_tgt,
        'final_out_ref': final_out_ref,
        'final_out_tgt': final_out_tgt,
        'output_tps_ref0': output_tps_ref0,
        'output_tps_tgt0': output_tps_tgt0,
    }


def train_m3(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    print(f"Loading dataset from {args.train_path}...")
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    net = Network().cuda()
    coef_net = CoefNetwork().cuda()
    tapr_refiner = TAPRRefiner().cuda()

    # Freeze base feature extractor and initial TPS regressor (guarantee baseline stability)
    for p in net.parameters():
        p.requires_grad = False

    homo_checkpoint = torch.load(args.woCoefNet_path, map_location='cuda')
    net.load_state_dict(homo_checkpoint['model'])

    # Initialize coef_net from official baseline
    if args.resume_coef_path and os.path.exists(args.resume_coef_path):
        c_ckpt = torch.load(args.resume_coef_path, map_location='cuda')
        coef_net.load_state_dict(c_ckpt['model'])
        print(f"Loaded pretrained coef_net from {args.resume_coef_path}")

    # Loss function: L1 + SSIM + Smoothness
    criterion = M3StitchingLoss(w_l1=args.w_l1, w_ssim=args.w_ssim, w_smooth=args.w_smooth).cuda()

    # Optimizer: TAPR refiner has normal lr, coef_net gentle finetuning
    optimizer = optim.Adam([
        {'params': tapr_refiner.parameters(), 'lr': args.lr},
        {'params': coef_net.parameters(), 'lr': args.lr * 0.1}
    ], betas=(0.9, 0.999), eps=1e-08)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=args.lr * 0.05)

    print(f"================== Start M3 Training (TAPR + SSIM Loss) on GPU {args.gpu} ==================")
    score_print_freq = 50
    glob_iter = 0

    for epoch in range(args.max_epoch):
        tapr_refiner.train()
        coef_net.train()
        net.eval()

        loss_total_acc = 0.0
        loss_l1_acc = 0.0
        loss_ssim_acc = 0.0
        loss_smooth_acc = 0.0
        loss_geom_acc = 0.0

        for i, batch_value in enumerate(train_loader):
            input1 = batch_value[0].float().cuda()
            input2 = batch_value[1].float().cuda()

            optimizer.zero_grad()
            res = forward_m3_full(net, coef_net, tapr_refiner, input1, input2)

            final_out_ref = res['final_out_ref']
            final_out_tgt = res['final_out_tgt']
            delta_ref = res['delta_ref']
            delta_tgt = res['delta_tgt']
            refined_ref = res['refined_mesh_ref']
            refined_tgt = res['refined_mesh_tgt']
            rigid_mesh = res['rigid_mesh']

            # Primary M3 photometric & structural loss
            m3_loss, loss_details = criterion(final_out_ref, final_out_tgt, delta_ref, delta_tgt)

            # Mesh geometry topology constraint (anti-folding & anti-distortion)
            loss_fold = mesh_folding_loss(refined_ref, rigid_mesh) + mesh_folding_loss(refined_tgt, rigid_mesh)
            loss_bend = mesh_bending_loss(refined_ref, rigid_mesh) + mesh_bending_loss(refined_tgt, rigid_mesh)
            geom_loss = 2.0 * loss_fold + 2.0 * loss_bend

            total_loss = m3_loss + geom_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(tapr_refiner.parameters(), max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(coef_net.parameters(), max_norm=2.0)
            optimizer.step()

            loss_total_acc += total_loss.item()
            loss_l1_acc += loss_details['loss_l1']
            loss_ssim_acc += loss_details['loss_ssim']
            loss_smooth_acc += loss_details['loss_smooth']
            loss_geom_acc += geom_loss.item()

            if (i + 1) % score_print_freq == 0:
                avg_tot = loss_total_acc / score_print_freq
                avg_l1 = loss_l1_acc / score_print_freq
                avg_ssim = loss_ssim_acc / score_print_freq
                avg_sm = loss_smooth_acc / score_print_freq
                avg_geom = loss_geom_acc / score_print_freq

                print(f"M3 [Ep {epoch+1:02d}/{args.max_epoch:02d}][It {i+1:04d}/{len(train_loader):04d}] "
                      f"Tot: {avg_tot:.4f} | L1: {avg_l1:.4f} | SSIM_loss: {avg_ssim:.4f} | "
                      f"Smooth: {avg_sm:.4f} | Geom: {avg_geom:.4f}", flush=True)

                writer.add_scalar('M3/Total_Loss', avg_tot, glob_iter)
                writer.add_scalar('M3/L1_Loss', avg_l1, glob_iter)
                writer.add_scalar('M3/SSIM_Loss', avg_ssim, glob_iter)
                writer.add_scalar('M3/Smooth_Loss', avg_sm, glob_iter)
                writer.add_scalar('M3/Geom_Loss', avg_geom, glob_iter)

                loss_total_acc, loss_l1_acc, loss_ssim_acc, loss_smooth_acc, loss_geom_acc = 0, 0, 0, 0, 0

            glob_iter += 1

        scheduler.step()

        # Save checkpoint periodically
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.max_epoch:
            save_path = os.path.join(MODEL_DIR, f"epoch{epoch+1:03d}_m3_model.pth")
            torch.save({
                'tapr_refiner': tapr_refiner.state_dict(),
                'coef_net': coef_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, save_path)
            print(f"===> Saved M3 checkpoint: {save_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=0.8)
    parser.add_argument('--w_smooth', type=float, default=0.1)
    parser.add_argument('--woCoefNet_path', type=str, default='/workspace/python_pro/test1/RopStitch/woCoefNet/model_homo/epoch100_model.pth')
    parser.add_argument('--resume_coef_path', type=str, default='/workspace/python_pro/test1/RopStitch/wCoefNet/model_coef/epoch050_coefmodel.pth')
    parser.add_argument('--train_path', type=str, default='/workspace/python_pro/test1/RopStitch/data/UDIS-D/training/')

    args = parser.parse_args()
    train_m3(args)
