import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(nn.Module):
    """
    Differentiable masked SSIM loss for image alignment.
    Directly maximizes perceptual structural similarity in overlapping zones.
    """
    def __init__(self, window_size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', create_window(window_size, channel))

    def forward(self, img1, img2, mask=None, eps=1e-6):
        # Expect img in range [-1, 1] or [0, 1]
        channel = img1.size(1)
        if channel == self.channel and self.window.data.type() == img1.data.type() and self.window.device == img1.device:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        padd = self.window_size // 2
        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)

        if mask is not None:
            # Downsample mask using window to match smoothed local region
            smooth_mask = F.conv2d(mask, window, padding=padd, groups=channel)
            ssim_val = torch.sum(ssim_map * smooth_mask) / (torch.sum(smooth_mask) + eps)
        else:
            ssim_val = torch.mean(ssim_map)

        return 1.0 - ssim_val


def mesh_smooth_loss(delta_mesh, eps=1e-6):
    """
    Penalizes abrupt displacement variations across neighboring vertices (Laplacian/Bending).
    delta_mesh: [B, H+1, W+1, 2]
    """
    diff_h = delta_mesh[:, :, 1:, :] - delta_mesh[:, :, :-1, :]
    diff_v = delta_mesh[:, 1:, :, :] - delta_mesh[:, :-1, :, :]
    
    loss_h = torch.mean(diff_h ** 2)
    loss_v = torch.mean(diff_v ** 2)
    
    # Second-order curvature smoothness
    if delta_mesh.size(2) > 2 and delta_mesh.size(1) > 2:
        d2_h = delta_mesh[:, :, 2:, :] - 2 * delta_mesh[:, :, 1:-1, :] + delta_mesh[:, :, :-2, :]
        d2_v = delta_mesh[:, 2:, :, :] - 2 * delta_mesh[:, 1:-1, :, :] + delta_mesh[:, :-2, :, :]
        loss_d2 = torch.mean(d2_h ** 2) + torch.mean(d2_v ** 2)
    else:
        loss_d2 = 0.0
        
    return loss_h + loss_v + 0.5 * loss_d2


class M3StitchingLoss(nn.Module):
    """
    M3 Composite Loss Module:
    Combines:
    1. Masked Overlap L1 Photometric Loss
    2. Differentiable Masked SSIM Loss (Direct target metric optimization)
    3. Mesh Smoothness Regularization on TAPR residual displacements
    """
    def __init__(self, w_l1=1.0, w_ssim=0.8, w_smooth=0.1):
        super(M3StitchingLoss, self).__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_smooth = w_smooth
        self.ssim_loss = SSIMLoss()

    def forward(self, out_ref, out_tgt, delta_ref=None, delta_tgt=None, eps=1e-6):
        """
        out_ref: [B, 6, H, W] (0:3 image, 3:6 mask)
        out_tgt: [B, 6, H, W]
        delta_ref: [B, 13, 13, 2] residual displacement
        delta_tgt: [B, 13, 13, 2] residual displacement
        """
        img_ref = out_ref[:, 0:3, :, :]
        img_tgt = out_tgt[:, 0:3, :, :]
        mask_ref = out_ref[:, 3:6, :, :]
        mask_tgt = out_tgt[:, 3:6, :, :]

        overlap_mask = mask_ref * mask_tgt

        # 1. Masked L1 loss
        diff = torch.abs(img_ref - img_tgt) * overlap_mask
        loss_l1 = torch.sum(diff) / (torch.sum(overlap_mask) + eps)

        # 2. Masked SSIM loss
        loss_ssim = self.ssim_loss(img_ref, img_tgt, overlap_mask)

        # 3. Mesh smoothness regularization
        loss_smooth = 0.0
        if delta_ref is not None:
            loss_smooth += mesh_smooth_loss(delta_ref)
        if delta_tgt is not None:
            loss_smooth += mesh_smooth_loss(delta_tgt)

        total_loss = self.w_l1 * loss_l1 + self.w_ssim * loss_ssim + self.w_smooth * loss_smooth

        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_l1': loss_l1.item(),
            'loss_ssim': loss_ssim.item(),
            'loss_smooth': loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else 0.0
        }
