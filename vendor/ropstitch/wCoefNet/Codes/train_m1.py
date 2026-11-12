# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from network import build_model, Network, CoefNetwork
from dataset import TrainDataset, TestDataset
from loss import (
    cal_ddmperception_loss,
    mesh_folding_loss,
    mesh_stretch_loss,
    mesh_bending_loss,
    tps_overlap_align_loss,
)
import os
import glob
import random
import torch.backends.cudnn as cudnn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

setup_seed(114514)

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model_m1')
SUMMARY_DIR = os.path.join(last_path, 'summary_m1')

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

writer = SummaryWriter(log_dir=SUMMARY_DIR)


def train(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=2, shuffle=True, drop_last=True)

    coef_net = CoefNetwork()
    net = Network()
    vgg_model = models.vgg19(pretrained=True)

    if torch.cuda.is_available():
        net = net.cuda()
        coef_net = coef_net.cuda()
        vgg_model = vgg_model.cuda()

    # Freeze registration network and VGG
    for param in net.parameters():
        param.requires_grad = False
    for param in vgg_model.parameters():
        param.requires_grad = False

    # Load frozen Stage 1 registration weights
    homo_checkpoint = torch.load(args.woCoefNet_path)
    net.load_state_dict(homo_checkpoint['model'])
    print(f'Loaded frozen homo network from {args.woCoefNet_path}')

    # Optimizer only for coef_net
    optimizer = optim.Adam(coef_net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # Resume or warm-start from Stage 2 checkpoint
    start_epoch = 0
    glob_iter = 0

    if args.resume_coef_path and os.path.exists(args.resume_coef_path):
        checkpoint = torch.load(args.resume_coef_path)
        coef_net.load_state_dict(checkpoint['model'])
        if 'epoch' in checkpoint and args.resume_full_state:
            start_epoch = checkpoint['epoch']
            glob_iter = checkpoint.get('glob_iter', 0)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Initialized coef_net from {args.resume_coef_path}')
    else:
        existing_ckpts = sorted(glob.glob(os.path.join(MODEL_DIR, "*.pth")))
        if len(existing_ckpts) > 0:
            latest = existing_ckpts[-1]
            checkpoint = torch.load(latest)
            coef_net.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint.get('epoch', 0)
            glob_iter = checkpoint.get('glob_iter', 0)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'Resumed training from {latest} (epoch {start_epoch})')

    print("================== Start M1 Supervised Training ==================")
    print(f"Loss weights: lambda_align={args.lambda_align}, lambda_fold={args.lambda_fold}, "
          f"lambda_stretch={args.lambda_stretch}, lambda_bend={args.lambda_bend}")

    score_print_freq = 100

    for epoch in range(start_epoch, args.max_epoch):
        coef_net.train()
        net.eval()  # net evaluation mode, weights frozen

        loss_sigma = 0.0
        coef_loss_sigma = 0.0
        align_loss_sigma = 0.0
        fold_loss_sigma = 0.0
        stretch_loss_sigma = 0.0
        bend_loss_sigma = 0.0

        for i, batch_value in enumerate(train_loader):
            input1 = batch_value[0].float().cuda()
            input2 = batch_value[1].float().cuda()

            optimizer.zero_grad()

            # Forward with enable_tps_grad=True to maintain autograd back to coef_net
            batch_out = build_model(net, coef_net, input1, input2, is_training=True, enable_tps_grad=True)

            DDM_ref_list = batch_out['DDM_ref_list']
            DDM_tgt_list = batch_out['DDM_tgt_list']
            mesh_ref = batch_out['mesh_ref']
            mesh_tgt = batch_out['mesh_tgt']
            rigid_mesh = batch_out['rigid_mesh']
            output_tps_ref = batch_out['output_tps_ref']
            output_tps_tgt = batch_out['output_tps_tgt']

            # 1. Virtual plane distortion loss (DDM perception)
            loss_coef = cal_ddmperception_loss(vgg_model, input1, input2, DDM_ref_list, DDM_tgt_list) * 10.0

            # 2. Final TPS alignment loss
            loss_align = tps_overlap_align_loss(output_tps_ref, output_tps_tgt)

            # 3. Geometric regularity losses
            loss_fold = mesh_folding_loss(mesh_ref, rigid_mesh) + mesh_folding_loss(mesh_tgt, rigid_mesh)
            loss_stretch = mesh_stretch_loss(mesh_ref, rigid_mesh) + mesh_stretch_loss(mesh_tgt, rigid_mesh)
            loss_bend = mesh_bending_loss(mesh_ref, rigid_mesh) + mesh_bending_loss(mesh_tgt, rigid_mesh)

            total_loss = (
                loss_coef
                + args.lambda_align * loss_align
                + args.lambda_fold * loss_fold
                + args.lambda_stretch * loss_stretch
                + args.lambda_bend * loss_bend
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(coef_net.parameters(), max_norm=3.0, norm_type=2)
            optimizer.step()

            loss_sigma += total_loss.item()
            coef_loss_sigma += loss_coef.item()
            align_loss_sigma += loss_align.item()
            fold_loss_sigma += loss_fold.item()
            stretch_loss_sigma += loss_stretch.item()
            bend_loss_sigma += loss_bend.item()

            if (i + 1) % score_print_freq == 0:
                avg_total = loss_sigma / score_print_freq
                avg_coef = coef_loss_sigma / score_print_freq
                avg_align = align_loss_sigma / score_print_freq
                avg_fold = fold_loss_sigma / score_print_freq
                avg_stretch = stretch_loss_sigma / score_print_freq
                avg_bend = bend_loss_sigma / score_print_freq

                loss_sigma = 0.0
                coef_loss_sigma = 0.0
                align_loss_sigma = 0.0
                fold_loss_sigma = 0.0
                stretch_loss_sigma = 0.0
                bend_loss_sigma = 0.0

                print(f"Epoch [{epoch+1:03d}/{args.max_epoch:03d}] Iter [{i+1:04d}/{len(train_loader):04d}] "
                      f"Total: {avg_total:.4f} | Coef: {avg_coef:.4f} | Align: {avg_align:.4f} | "
                      f"Fold: {avg_fold:.4f} | Stretch: {avg_stretch:.4f} | Bend: {avg_bend:.4f}")

                writer.add_scalar('M1/Total_Loss', avg_total, glob_iter)
                writer.add_scalar('M1/Coef_Loss', avg_coef, glob_iter)
                writer.add_scalar('M1/Align_Loss', avg_align, glob_iter)
                writer.add_scalar('M1/Fold_Loss', avg_fold, glob_iter)
                writer.add_scalar('M1/Stretch_Loss', avg_stretch, glob_iter)
                writer.add_scalar('M1/Bend_Loss', avg_bend, glob_iter)

            glob_iter += 1

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.max_epoch:
            save_name = f"epoch{epoch+1:03d}_m1_model.pth"
            save_path = os.path.join(MODEL_DIR, save_name)
            torch.save({
                'model': coef_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'glob_iter': glob_iter,
            }, save_path)
            print(f"Saved M1 checkpoint: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--woCoefNet_path', type=str, default='/workspace/python_pro/test1/RopStitch/woCoefNet/model_homo/epoch100_model.pth')
    parser.add_argument('--resume_coef_path', type=str, default='/workspace/python_pro/test1/RopStitch/wCoefNet/model_coef/epoch050_coefmodel.pth')
    parser.add_argument('--resume_full_state', action='store_true', default=False)
    parser.add_argument('--train_path', type=str, default='/workspace/python_pro/test1/RopStitch/data/UDIS-D/training/')
    parser.add_argument('--test_path', type=str, default='/workspace/python_pro/test1/RopStitch/data/UDIS-D/testing/')
    parser.add_argument('--lambda_align', type=float, default=2.0)
    parser.add_argument('--lambda_fold', type=float, default=5.0)
    parser.add_argument('--lambda_stretch', type=float, default=10.0)
    parser.add_argument('--lambda_bend', type=float, default=10.0)

    args = parser.parse_args()
    train(args)
