import torch
import torch.nn as nn
import torch.nn.functional as F

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

min_w = (512 / grid_w) / 8
min_h = (512 / grid_h) / 8

# intra-grid constraint
def intra_grid_loss(pts):
    batch_size = pts.shape[0]

    delta_x = pts[:, :, 0:grid_w, 0] - pts[:, :, 1:grid_w + 1, 0]
    delta_y = pts[:, 0:grid_h, :, 1] - pts[:, 1:grid_h + 1, :, 1]

    loss_x = F.relu(delta_x + min_w)
    loss_y = F.relu(delta_y + min_h)

    loss = torch.mean(loss_x) + torch.mean(loss_y)
    return loss


# inter-grid constraint
def inter_grid_loss(train_mesh):
    w_edges = train_mesh[:, :, 0:grid_w, :] - train_mesh[:, :, 1:grid_w + 1, :]
    cos_w = torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 1:grid_w, :], 3) / \
            (torch.sqrt(torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 0:grid_w - 1, :], 3))
             * torch.sqrt(torch.sum(w_edges[:, :, 1:grid_w, :] * w_edges[:, :, 1:grid_w, :], 3)))
    delta_w_angle = 1 - cos_w

    h_edges = train_mesh[:, 0:grid_h, :, :] - train_mesh[:, 1:grid_h + 1, :, :]
    cos_h = torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 1:grid_h, :, :], 3) / \
            (torch.sqrt(torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 0:grid_h - 1, :, :], 3))
             * torch.sqrt(torch.sum(h_edges[:, 1:grid_h, :, :] * h_edges[:, 1:grid_h, :, :], 3)))
    delta_h_angle = 1 - cos_h

    loss = torch.mean(delta_w_angle) + torch.mean(delta_h_angle)
    return loss


def mesh_folding_loss(mesh, rigid_mesh, margin=0.05, eps=1e-6):
    """
    Differentiable triangle signed area loss to penalize mesh folding.
    mesh: [B, H+1, W+1, 2]
    rigid_mesh: [B, H+1, W+1, 2]
    """
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

    # Reference triangle orientation signs
    sign_r1 = torch.sign(r_area_t1)
    sign_r2 = torch.sign(r_area_t2)

    # Relative signed area ratios (1.0 for rigid mesh, < 0 for flipped/inverted orientation)
    rho_t1 = (area_t1 * sign_r1) / (torch.abs(r_area_t1) + eps)
    rho_t2 = (area_t2 * sign_r2) / (torch.abs(r_area_t2) + eps)

    # Penalize when relative area drops below margin (e.g. margin=0.05) or flips (< 0)
    loss_t1 = F.relu(margin - rho_t1)
    loss_t2 = F.relu(margin - rho_t2)

    return torch.mean(loss_t1) + torch.mean(loss_t2)


def mesh_stretch_loss(mesh, rigid_mesh, eps=1e-6):
    """
    Symmetric Dirichlet distortion loss on grid edge lengths.
    """
    e_x = mesh[:, :, 1:, :] - mesh[:, :, :-1, :]
    r_x = rigid_mesh[:, :, 1:, :] - rigid_mesh[:, :, :-1, :]
    len_e_x = torch.norm(e_x, dim=-1)
    len_r_x = torch.norm(r_x, dim=-1) + eps
    ratio_x = len_e_x / len_r_x

    e_y = mesh[:, 1:, :, :] - mesh[:, :-1, :, :]
    r_y = rigid_mesh[:, 1:, :, :] - rigid_mesh[:, :-1, :, :]
    len_e_y = torch.norm(e_y, dim=-1)
    len_r_y = torch.norm(r_y, dim=-1) + eps
    ratio_y = len_e_y / len_r_y

    pen_x = ratio_x + 1.0 / (ratio_x + eps) - 2.0
    pen_y = ratio_y + 1.0 / (ratio_y + eps) - 2.0

    return torch.mean(pen_x) + torch.mean(pen_y)


def mesh_bending_loss(mesh, rigid_mesh, eps=1e-6):
    """
    Discrete second-order curvature smoothness loss along grid lines.
    """
    d2_h = mesh[:, :, 2:, :] - 2 * mesh[:, :, 1:-1, :] + mesh[:, :, :-2, :]
    d2_v = mesh[:, 2:, :, :] - 2 * mesh[:, 1:-1, :, :] + mesh[:, :-2, :, :]

    step_x = torch.mean(torch.norm(rigid_mesh[:, :, 1:, :] - rigid_mesh[:, :, :-1, :], dim=-1)) + eps
    step_y = torch.mean(torch.norm(rigid_mesh[:, 1:, :, :] - rigid_mesh[:, :-1, :, :], dim=-1)) + eps

    bend_h = torch.sum(d2_h ** 2, dim=-1) / (step_x ** 2)
    bend_v = torch.sum(d2_v ** 2, dim=-1) / (step_y ** 2)

    return torch.mean(bend_h) + torch.mean(bend_v)


def tps_overlap_align_loss(output_tps_ref, output_tps_tgt, eps=1e-6):
    """
    L1 photometric alignment loss on the warped overlapping region.
    """
    img_ref = output_tps_ref[:, 0:3, :, :]
    img_tgt = output_tps_tgt[:, 0:3, :, :]
    mask_ref = output_tps_ref[:, 3:6, :, :]
    mask_tgt = output_tps_tgt[:, 3:6, :, :]

    overlap_mask = mask_ref * mask_tgt
    diff = torch.abs(img_ref - img_tgt) * overlap_mask
    align_loss = torch.sum(diff) / (torch.sum(overlap_mask) + eps)
    return align_loss

    

def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))


def cal_lp_loss(input1, output_H, output_H_ref, output_H_tgt, output_tps_ref, output_tps_tgt):
    batch_size, _, img_h, img_w = input1.size()

    # # part one:
    overlap = output_H[:,3:6,:,:]
    lp_loss_1_1 = l_num_loss(input1*overlap, output_H[:,0:3,:,:]*overlap, 1)

    overlap = output_H_ref[:,3:6,:,:] * output_H_tgt[:,3:6,:,:]
    lp_loss_1_2 = l_num_loss(output_H_ref[:,0:3,:,:]*overlap, output_H_tgt[:,0:3,:,:]*overlap, 1)

    lp_loss_1 = (lp_loss_1_1 + lp_loss_1_2) / 2.

    # # part two:
    overlap = output_tps_ref[:,3:6,:,:] * output_tps_tgt[:,3:6,:,:]
    lp_loss_2 = l_num_loss(output_tps_ref[:,0:3,:,:]*overlap, output_tps_tgt[:,0:3,:,:]*overlap, 1)


    lp_loss = 3. * lp_loss_1 + 1. * lp_loss_2
    return lp_loss



def cal_lp_loss2(output_tps_ref, output_tps_tgt):
    overlap = output_tps_ref[:,3:6,:,:] * output_tps_tgt[:,3:6,:,:]
    lp_loss_2 = l_num_loss(output_tps_ref[:,0:3,:,:]*overlap, output_tps_tgt[:,0:3,:,:]*overlap, 1)
    return lp_loss_2


def get_vgg19_FeatureMap(vgg_model, input_255, layer_index):
    
    vgg_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1))
    if torch.cuda.is_available():
        vgg_mean = vgg_mean.cuda()
    vgg_input = input_255-vgg_mean
    #x = vgg_model.features[0](vgg_input)
    #FeatureMap_list.append(x)

    x_list = []

    for i in range(0,layer_index+1):
        if i == 0:
            x = vgg_model.features[0](vgg_input)
        else:
            x = vgg_model.features[i](x)
            if i == 6 or i == 13 or i ==24:
                x_list.append(x)

    return x_list



def cal_ddmperception_loss(vgg_model, input1_tesnor, input2_tesnor, DDM_ref_list, DDM_tgt_list):
    bs = input1_tesnor.size(0)
    iter_num = len(DDM_ref_list)

    # resize DDM to the resolution of semantic features
    DDM_ref1 = F.interpolate(DDM_ref_list[0], (128, 128), mode='bicubic', align_corners=False)
    DDM_tgt1 = F.interpolate(DDM_tgt_list[0], (128, 128), mode='bicubic', align_corners=False)
    # DDM_ref2 = F.interpolate(DDM_ref_list[1], (128, 128), mode='bicubic', align_corners=False)
    # DDM_tgt2 = F.interpolate(DDM_tgt_list[1], (128, 128), mode='bicubic', align_corners=False)

    # get feature maps
    H_ref_feature_list = get_vgg19_FeatureMap(vgg_model, (input1_tesnor+1)*127.5, 13)
    H_tgt_feature_list = get_vgg19_FeatureMap(vgg_model, (input2_tesnor+1)*127.5, 13)
    feature_ref = torch.mean(H_ref_feature_list[-1], 1, keepdim=True)  # bs, 1, h, w
    feature_tgt = torch.mean(H_tgt_feature_list[-1], 1, keepdim=True)  # bs, 1, h, w

    # normalized ref features
    reshaped_ref = feature_ref.view(bs, -1)
    reshaped_ref_min = reshaped_ref.min(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_ref_max = reshaped_ref.max(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_tgt = feature_tgt.view(bs, -1)
    reshaped_tgt_min = reshaped_tgt.min(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_tgt_max = reshaped_tgt.max(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_min = torch.min(reshaped_ref_min, reshaped_tgt_min)
    reshaped_max = torch.max(reshaped_ref_max, reshaped_tgt_max)
    normalized_feature_ref = (feature_ref - reshaped_min) / (reshaped_max - reshaped_min + 1e-8)  
    normalized_feature_tgt = (feature_tgt - reshaped_min) / (reshaped_max - reshaped_min + 1e-8)  


    #loss_list = []
    total_loss = 0.
    for i in range(iter_num):
        DDM_ref1 = F.interpolate(DDM_ref_list[i], (128, 128), mode='bicubic', align_corners=False)
        DDM_tgt1 = F.interpolate(DDM_tgt_list[i], (128, 128), mode='bicubic', align_corners=False)
        loss_ref1 = torch.mean((normalized_feature_ref+0.1) * DDM_ref1)
        loss_tgt1 = torch.mean((normalized_feature_tgt+0.1) * DDM_tgt1)
        loss1 = torch.max(loss_ref1, loss_tgt1)
        #loss_list.append(loss1)
        total_loss = total_loss + loss1*(0.5**(iter_num-1-i))

    return total_loss

def cal_ddmperception_loss_test(vgg_model, input1_tesnor, input2_tesnor, DDM_ref, DDM_tgt):
    bs = input1_tesnor.size(0)

    # resize DDM to the resolution of semantic features
    DDM_ref1 = F.interpolate(DDM_ref, (128, 128), mode='bicubic', align_corners=False)
    DDM_tgt1 = F.interpolate(DDM_tgt, (128, 128), mode='bicubic', align_corners=False)

    # get feature maps
    H_ref_feature_list = get_vgg19_FeatureMap(vgg_model, (input1_tesnor+1)*127.5, 13)
    H_tgt_feature_list = get_vgg19_FeatureMap(vgg_model, (input2_tesnor+1)*127.5, 13)
    feature_ref = torch.mean(H_ref_feature_list[-1], 1, keepdim=True)  # bs, 1, h, w
    feature_tgt = torch.mean(H_tgt_feature_list[-1], 1, keepdim=True)  # bs, 1, h, w

    # normalized ref features
    reshaped_ref = feature_ref.view(bs, -1)
    reshaped_ref_min = reshaped_ref.min(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_ref_max = reshaped_ref.max(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_tgt = feature_tgt.view(bs, -1)
    reshaped_tgt_min = reshaped_tgt.min(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_tgt_max = reshaped_tgt.max(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_min = torch.min(reshaped_ref_min, reshaped_tgt_min)
    reshaped_max = torch.max(reshaped_ref_max, reshaped_tgt_max)
    normalized_feature_ref = (feature_ref - reshaped_min) / (reshaped_max - reshaped_min + 1e-8)  
    normalized_feature_tgt = (feature_tgt - reshaped_min) / (reshaped_max - reshaped_min + 1e-8)  

    loss_ref1 = torch.mean((normalized_feature_ref+0.1) * DDM_ref1)
    loss_tgt1 = torch.mean((normalized_feature_tgt+0.1) * DDM_tgt1)
    loss = torch.max(loss_ref1, loss_tgt1)

    return loss
