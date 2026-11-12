import argparse
import torch
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import time
import json
import csv
import torchvision.models as models
from skimage.metrics import structural_similarity as compare_ssim

from network import Network, CoefNetwork, H2Mesh, H2Offset, get_rigid_mesh, get_norm_mesh, resize_512
from feedback_network import TPSStateFeedbackCorrector
from dataset import TestDataset
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
from utils.grid_metrics import compute_grid_metrics
import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


def maskSSIM(image1, image2, mask):
    image1 = image1 * mask
    image2 = image2 * mask
    _, ssim = compare_ssim(image1, image2, data_range=255, channel_axis=2, full=True)
    ssim = np.sum(ssim * mask) / (np.sum(mask) + 1e-6)
    return ssim


def maskPSNR(image1, image2, mask):
    image1 = image1 * mask / 255.
    image2 = image2 * mask / 255.
    rmse = np.sqrt(np.sum((image1 - image2) ** 2) / (mask.sum() + 1e-6))
    if rmse == 0:
        return 100.0
    psnr = 20 * np.log10(1.0 / rmse)
    return psnr


def build_output_model_m2(net, coef_net, feedback_corrector, input1_tensor, input2_tensor, alpha=0.5, max_out_height=4000):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # Pass 1 on 512x512
    resized_input1 = resize_512(input1_tensor)
    resized_input2 = resize_512(input2_tensor)

    dst_p_ref_list0, dst_p_tgt_list0, final_coef_list0, mesh_ref0, mesh_tgt0 = net(
        resized_input1, resized_input2, coef_net, alpha=alpha, enable_tps_grad=False
    )
    c0 = final_coef_list0[-1]
    rigid_mesh_512 = get_rigid_mesh(batch_size, 512, 512)

    # Feedback correction
    c1, delta_z = feedback_corrector(mesh_ref0, mesh_tgt0, rigid_mesh_512, c0)

    # Pass 2: Re-generate homography with c1
    with torch.no_grad():
        _, _, frozen_feat_2_64, _ = net.fronzen_resnet_backbone(resized_input1, resized_input2)
        active_feat_1_64, active_feat_1_32, active_feat_2_64, active_feat_2_32 = net.active_resnet_backbone(resized_input1, resized_input2)
        c_32 = (1 - alpha) * net.CCL(net.fronzen_resnet_backbone(resized_input1, resized_input2)[1], net.fronzen_resnet_backbone(resized_input1, resized_input2)[3]) \
               + alpha * net.CCL(active_feat_1_32, active_feat_2_32)
        t_1 = net.regressHomo_part1(c_32).view(batch_size, -1)
        offset_1 = net.regressHomo_part2(t_1)

    H_motion = offset_1.reshape(-1, 4, 2)
    src_p = torch.tensor([[0., 0.], [512., 0.], [0., 512.], [512., 512.]], device=input1_tensor.device).unsqueeze(0).expand(batch_size, -1, -1)
    dst_p = src_p + H_motion
    dst_p_tgt1 = src_p + H_motion * c1
    H = torch_DLT.Tensor_ACA_rect(src_p, dst_p)
    H_tgt1 = torch_DLT.Tensor_ACA_rect(src_p, dst_p_tgt1)
    H_ref1 = torch.matmul(torch.inverse(H), H_tgt1)
    dst_p_ref1 = H2Offset(H_ref1, src_p)

    H_ref_dlt = torch_DLT.Tensor_ACA_rect(src_p, dst_p_ref1)
    ini_mesh_ref1 = H2Mesh(H_ref_dlt, rigid_mesh_512)
    ini_mesh_tgt1 = H2Mesh(H_tgt1, rigid_mesh_512)

    H_ref_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_ref1/8)
    H_tgt_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_tgt1/8)

    M_tensor = torch.tensor([[512/8 / 2.0, 0., 512/8 / 2.0],
                             [0., 512/8 / 2.0, 512/8 / 2.0],
                             [0., 0., 1.]], device=input1_tensor.device).unsqueeze(0).expand(batch_size, -1, -1)
    M_inv = torch.inverse(M_tensor)

    H_mat_ref = torch.matmul(torch.matmul(M_inv, H_ref_feat), M_tensor)
    warp_f1_ref = torch_homo_transform.transformer(active_feat_1_64, H_mat_ref, (int(512/8), int(512/8)))
    H_mat_tgt = torch.matmul(torch.matmul(M_inv, H_tgt_feat), M_tensor)
    warp_f2_tgt = torch_homo_transform.transformer(active_feat_2_64, H_mat_tgt, (int(512/8), int(512/8)))

    tps1_ref = net.regressTPS_part1(torch.cat([warp_f1_ref, warp_f2_tgt], 1)).reshape(batch_size, -1)
    motion_ref1 = net.regressTPS_part2(tps1_ref).reshape(-1, grid_h+1, grid_w+1, 2)

    tps1_tgt = net.regressTPS_part1(torch.cat([warp_f2_tgt, warp_f1_ref], 1)).reshape(batch_size, -1)
    motion_tgt1 = net.regressTPS_part2(tps1_tgt).reshape(-1, grid_h+1, grid_w+1, 2)

    mesh_ref1 = ini_mesh_ref1 + motion_ref1
    mesh_tgt1 = ini_mesh_tgt1 + motion_tgt1

    # Scale to full target resolution
    mesh_ref = torch.stack([mesh_ref1[..., 0] * img_w / 512.0, mesh_ref1[..., 1] * img_h / 512.0], 3)
    mesh_tgt = torch.stack([mesh_tgt1[..., 0] * img_w / 512.0, mesh_tgt1[..., 1] * img_h / 512.0], 3)
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)

    width_max = torch.maximum(torch.max(mesh_ref[..., 0]), torch.max(mesh_tgt[..., 0]))
    width_min = torch.minimum(torch.min(mesh_ref[..., 0]), torch.min(mesh_tgt[..., 0]))
    height_max = torch.maximum(torch.max(mesh_ref[..., 1]), torch.max(mesh_tgt[..., 1]))
    height_min = torch.minimum(torch.min(mesh_ref[..., 1]), torch.min(mesh_tgt[..., 1]))

    out_width = width_max - width_min
    out_height = height_max - height_min

    if max(out_height, out_width) >= max_out_height:
        return None, False

    mesh_trans_ref = torch.stack([mesh_ref[..., 0] - width_min, mesh_ref[..., 1] - height_min], 3)
    mesh_trans_tgt = torch.stack([mesh_tgt[..., 0] - width_min, mesh_tgt[..., 1] - height_min], 3)

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_trans_ref = get_norm_mesh(mesh_trans_ref, out_height, out_width)
    norm_mesh_trans_tgt = get_norm_mesh(mesh_trans_tgt, out_height, out_width)

    mask = torch.ones_like(input2_tensor).cuda()
    output_ref = torch_tps_transform.transformer(torch.cat((input1_tensor+1, mask), 1), norm_mesh_trans_ref, norm_rigid_mesh, (out_height.int(), out_width.int()))
    output_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor+1, mask), 1), norm_mesh_trans_tgt, norm_rigid_mesh, (out_height.int(), out_width.int()))

    out_dict = {
        'output_ref': output_ref,
        'output_tgt': output_tgt,
        'final_cdec': c1,
        'c0': c0,
        'delta_z': delta_z,
        'mesh_ref': mesh_ref,
        'mesh_tgt': mesh_tgt,
        'rigid_mesh': rigid_mesh,
        'out_height': int(out_height.item() if isinstance(out_height, torch.Tensor) else out_height),
        'out_width': int(out_width.item() if isinstance(out_width, torch.Tensor) else out_width),
    }
    return out_dict, True


def test_once_m2(net, coef_net, feedback_corrector, input1, input2, alpha=0.5, max_out_height=4000):
    with torch.no_grad():
        batch_out, flag_check = build_output_model_m2(net, coef_net, feedback_corrector, input1, input2, alpha=alpha, max_out_height=max_out_height)
    warp_fallback = not flag_check
    if not flag_check:
        batch_out = {}
        mask = torch.ones_like(input1).to(input1.device)
        batch_out['output_ref'] = torch.cat((input1+1, mask), 1)
        batch_out['output_tgt'] = torch.cat((input2+1, mask), 1)
        batch_out['final_cdec'] = None
        batch_out['c0'] = None
        batch_out['delta_z'] = None
        batch_out['mesh_ref'] = None
        batch_out['mesh_tgt'] = None
        batch_out['rigid_mesh'] = None
        batch_out['out_height'] = input1.shape[2]
        batch_out['out_width'] = input1.shape[3]

    output_tps_ref = batch_out['output_ref']
    output_tps_tgt = batch_out['output_tgt']

    output_ref = (output_tps_ref[0, 0:3, ...] * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
    output_tgt = (output_tps_tgt[0, 0:3, ...] * 127.5).cpu().detach().numpy().transpose(1, 2, 0)
    overlap_mask = output_tps_ref[0, 3:6, ...] * output_tps_tgt[0, 3:6, ...]
    overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1, 2, 0)

    ssim3 = maskSSIM(output_ref, output_tgt, overlap_mask)
    psnr3 = maskPSNR(output_ref, output_tgt, overlap_mask)

    fusion = output_ref * (output_ref / (output_ref + output_tgt + 1e-6)) + output_tgt * (output_tgt / (output_ref + output_tgt + 1e-6))

    details = {
        'warp_fallback': warp_fallback,
        'final_cdec': batch_out.get('final_cdec', None),
        'c0': batch_out.get('c0', None),
        'delta_z': batch_out.get('delta_z', None),
        'mesh_ref': batch_out.get('mesh_ref', None),
        'mesh_tgt': batch_out.get('mesh_tgt', None),
        'rigid_mesh': batch_out.get('rigid_mesh', None),
        'out_height': batch_out.get('out_height', input1.shape[2]),
        'out_width': batch_out.get('out_width', input1.shape[3]),
        'output_tps_ref': output_tps_ref,
        'output_tps_tgt': output_tps_tgt,
    }
    return ssim3, psnr3, fusion, details


def ternary_search_m2(net, coef_net, feedback_corrector, input1, input2, low=-1.0, high=2.0, max_iter=5, max_out_height=4000, tol=1e-6):
    best_alpha = None
    best_ssim = -float('inf')
    best_psnr = -float('inf')
    best_img = None
    best_details = None
    invalid_candidate_count = 0
    fallback_count = 0
    t_single_first = None

    t_search_start = time.perf_counter()

    for iter_idx in range(max_iter):
        if high - low < tol:
            break
        mid1 = low + (high - low) / 3.0
        mid2 = high - (high - low) / 3.0

        if t_single_first is None:
            t0 = time.perf_counter()
            ssim1, psnr1, img1, det1 = test_once_m2(net, coef_net, feedback_corrector, input1, input2, alpha=mid1, max_out_height=max_out_height)
            t_single_first = time.perf_counter() - t0
        else:
            ssim1, psnr1, img1, det1 = test_once_m2(net, coef_net, feedback_corrector, input1, input2, alpha=mid1, max_out_height=max_out_height)

        ssim2, psnr2, img2, det2 = test_once_m2(net, coef_net, feedback_corrector, input1, input2, alpha=mid2, max_out_height=max_out_height)

        if det1['warp_fallback']:
            invalid_candidate_count += 1
            fallback_count += 1
        if det2['warp_fallback']:
            invalid_candidate_count += 1
            fallback_count += 1

        if ssim1 > best_ssim:
            best_ssim = ssim1
            best_psnr = psnr1
            best_alpha = mid1
            best_img = img1
            best_details = det1
        if ssim2 > best_ssim:
            best_ssim = ssim2
            best_psnr = psnr2
            best_alpha = mid2
            best_img = img2
            best_details = det2

        if ssim1 < ssim2:
            low = mid1
        else:
            high = mid2

    t_search_total = time.perf_counter() - t_search_start

    search_meta = {
        't_single_first_s': t_single_first if t_single_first is not None else 0.0,
        't_search_total_s': t_search_total,
        'invalid_candidates': invalid_candidate_count,
        'fallback_count': fallback_count,
        'best_details': best_details,
    }
    return best_alpha, best_ssim, best_psnr, best_img, search_meta


def test_m2(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

    net = Network().cuda()
    coef_net = CoefNetwork().cuda()
    feedback_corrector = TPSStateFeedbackCorrector().cuda()

    homo_checkpoint = torch.load(args.woCoefNet_path)
    net.load_state_dict(homo_checkpoint['model'])
    print(f"Loaded homo net model from {args.woCoefNet_path}!")

    m2_checkpoint = torch.load(args.m2_model_path)
    coef_net.load_state_dict(m2_checkpoint['coef_net'])
    feedback_corrector.load_state_dict(m2_checkpoint['feedback_corrector'])
    print(f"Loaded M2 checkpoint from {args.m2_model_path}!")

    net.eval()
    coef_net.eval()
    feedback_corrector.eval()

    path_ave_fusion = '../ave_fusion_m2/'
    if not os.path.exists(path_ave_fusion):
        os.makedirs(path_ave_fusion)

    if args.save_meshes_dir and not os.path.exists(args.save_meshes_dir):
        os.makedirs(args.save_meshes_dir)

    ssim3_list = []
    psnr3_list = []
    per_sample_metrics = []
    per_sample_measurements = []

    print("################## Start M2 Testing #######################")

    for i, batch_value in enumerate(test_loader):
        input1_tensor = batch_value[0].float().cuda()
        input2_tensor = batch_value[1].float().cuda()

        best_alpha, best_ssim, best_psnr, best_img, s_meta = ternary_search_m2(
            net, coef_net, feedback_corrector, input1_tensor, input2_tensor,
            low=-1.0, high=2.0, max_iter=args.max_epoch, max_out_height=4000, tol=1e-6
        )
        t_fixed_single_s = s_meta['t_single_first_s']

        path = os.path.join(path_ave_fusion, f"{str(i+1).zfill(6)}.jpg")
        cv2.imwrite(path, best_img)
        ssim3_list.append(best_ssim)
        psnr3_list.append(best_psnr)

        best_det = s_meta['best_details']
        is_fallback = best_det['warp_fallback'] if best_det is not None else True
        is_success = not is_fallback

        f1_name = os.path.basename(test_data.datas['input1']['image'][i])
        f2_name = os.path.basename(test_data.datas['input2']['image'][i])
        per_sample_metrics.append({
            'sample_index': i + 1,
            'input1_filename': f1_name,
            'input2_filename': f2_name,
            'best_alpha': best_alpha,
            'psnr': best_psnr,
            'ssim': best_ssim,
            'valid': is_success,
            'failure_reason': '' if s_meta['fallback_count'] == 0 else f"warp_fallback:{s_meta['fallback_count']}",
        })

        cdec_vals = [0.0, 0.0, 0.0, 0.0]
        ref_metrics = {'folding_ratio': 0.0, 'degenerate_ratio': 0.0, 'avg_stretch': 0.0, 'max_stretch': 0.0, 'scale_variation': 0.0, 'bending_degree': 0.0}
        tgt_metrics = {'folding_ratio': 0.0, 'degenerate_ratio': 0.0, 'avg_stretch': 0.0, 'max_stretch': 0.0, 'scale_variation': 0.0, 'bending_degree': 0.0}
        effective_coverage = 1.0
        out_h = input1_tensor.shape[2]
        out_w = input1_tensor.shape[3]

        if best_det is not None and not is_fallback:
            out_h = int(best_det['out_height'])
            out_w = int(best_det['out_width'])
            cdec_t = best_det['final_cdec']
            if cdec_t is not None:
                cdec_np = cdec_t.detach().cpu().numpy().reshape(-1)
                cdec_vals = [float(v) for v in cdec_np[:4]]

            mesh_ref = best_det['mesh_ref']
            mesh_tgt = best_det['mesh_tgt']
            rigid_mesh = best_det['rigid_mesh']

            if mesh_ref is not None and rigid_mesh is not None:
                ref_metrics = compute_grid_metrics(mesh_ref, rigid_mesh)
            if mesh_tgt is not None and rigid_mesh is not None:
                tgt_metrics = compute_grid_metrics(mesh_tgt, rigid_mesh)

            ref_m = best_det['output_tps_ref'][0, 3, ...] > 0.5
            tgt_m = best_det['output_tps_tgt'][0, 3, ...] > 0.5
            union_m = ref_m | tgt_m
            effective_coverage = float(union_m.sum().item()) / float(max(1, union_m.numel()))

            if args.save_meshes_dir:
                mesh_path = os.path.join(args.save_meshes_dir, f"{str(i+1).zfill(6)}_mesh.npz")
                np.savez_compressed(
                    mesh_path,
                    mesh_ref=mesh_ref.detach().cpu().numpy(),
                    mesh_tgt=mesh_tgt.detach().cpu().numpy(),
                    rigid_mesh=rigid_mesh.detach().cpu().numpy(),
                    Cdec=np.array(cdec_vals),
                )

        per_sample_measurements.append({
            'sample_index': i + 1,
            'input1_filename': f1_name,
            'input2_filename': f2_name,
            'psnr': float(best_psnr),
            'ssim': float(best_ssim),
            'best_alpha': float(best_alpha) if best_alpha is not None else 0.0,
            'cdec_1': cdec_vals[0],
            'cdec_2': cdec_vals[1],
            'cdec_3': cdec_vals[2],
            'cdec_4': cdec_vals[3],
            'out_height': out_h,
            'out_width': out_w,
            't_fixed_single_s': t_fixed_single_s,
            't_search_total_s': s_meta['t_search_total_s'],
            'invalid_candidates': s_meta['invalid_candidates'],
            'fallback_count': s_meta['fallback_count'],
            'is_success': is_success,
            'ref_folding_ratio': ref_metrics['folding_ratio'],
            'tgt_folding_ratio': tgt_metrics['folding_ratio'],
            'ref_degenerate_ratio': ref_metrics['degenerate_ratio'],
            'tgt_degenerate_ratio': tgt_metrics['degenerate_ratio'],
            'ref_avg_stretch': ref_metrics['avg_stretch'],
            'tgt_avg_stretch': tgt_metrics['avg_stretch'],
            'ref_max_stretch': ref_metrics['max_stretch'],
            'tgt_max_stretch': tgt_metrics['max_stretch'],
            'ref_scale_variation': ref_metrics['scale_variation'],
            'tgt_scale_variation': tgt_metrics['scale_variation'],
            'ref_bending_degree': ref_metrics['bending_degree'],
            'tgt_bending_degree': tgt_metrics['bending_degree'],
            'effective_coverage': effective_coverage,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
            print(f"i = {i+1}, best alpha = {best_alpha:.4f}, best_ssim: {best_ssim:.4f}, best_psnr: {best_psnr:.2f}")

    # Sort metrics for top 30%, 30-60%, 60-100%
    psnr_arr = np.array(psnr3_list)
    ssim_arr = np.array(ssim3_list)
    sorted_psnr = np.sort(psnr_arr)[::-1]
    sorted_ssim = np.sort(ssim_arr)[::-1]
    n = len(psnr_arr)

    p_easy = float(np.mean(sorted_psnr[:int(n * 0.3)]))
    p_mod = float(np.mean(sorted_psnr[int(n * 0.3):int(n * 0.6)]))
    p_hard = float(np.mean(sorted_psnr[int(n * 0.6):]))
    p_avg = float(np.mean(psnr_arr))

    s_easy = float(np.mean(sorted_ssim[:int(n * 0.3)]))
    s_mod = float(np.mean(sorted_ssim[int(n * 0.3):int(n * 0.6)]))
    s_hard = float(np.mean(sorted_ssim[int(n * 0.6):]))
    s_avg = float(np.mean(ssim_arr))

    print(f"top 30% {p_easy:.5f}")
    print(f"top 30~60% {p_mod:.5f}")
    print(f"top 60~100% {p_hard:.5f}")
    print(f"average psnr: {p_avg:.6f}")
    print(f"top 30% {s_easy:.7f}")
    print(f"top 30~60% {s_mod:.7f}")
    print(f"top 60~100% {s_hard:.7f}")
    print(f"average ssim: {s_avg:.8f}")

    if args.metrics_csv:
        with open(args.metrics_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['sample_index', 'input1_filename', 'input2_filename', 'best_alpha', 'psnr', 'ssim', 'valid', 'failure_reason'])
            writer.writeheader()
            writer.writerows(per_sample_metrics)

    if args.measurement_csv:
        with open(args.measurement_csv, 'w', newline='') as f:
            fieldnames = list(per_sample_measurements[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sample_measurements)

    if args.measurement_summary_json:
        summary_data = {
            'protocol': 'measurement-m2',
            'sample_count': len(per_sample_measurements),
            'success_count': int(sum(1 for m in per_sample_measurements if m['is_success'])),
            'failure_count': int(sum(1 for m in per_sample_measurements if not m['is_success'])),
            'failure_rate': float(np.mean([0.0 if m['is_success'] else 1.0 for m in per_sample_measurements])),
            'total_invalid_candidates': int(sum(m['invalid_candidates'] for m in per_sample_measurements)),
            'total_fallbacks': int(sum(m['fallback_count'] for m in per_sample_measurements)),
            'alignment_metrics': {
                'psnr': {'easy': p_easy, 'moderate': p_mod, 'hard': p_hard, 'average': p_avg},
                'ssim': {'easy': s_easy, 'moderate': s_mod, 'hard': s_hard, 'average': s_avg},
            },
            'geometric_metrics_mean': {
                'ref_folding_ratio': float(np.mean([m['ref_folding_ratio'] for m in per_sample_measurements])),
                'tgt_folding_ratio': float(np.mean([m['tgt_folding_ratio'] for m in per_sample_measurements])),
                'mean_folding_ratio': float(np.mean([(m['ref_folding_ratio'] + m['tgt_folding_ratio']) / 2.0 for m in per_sample_measurements])),
                'ref_degenerate_ratio': float(np.mean([m['ref_degenerate_ratio'] for m in per_sample_measurements])),
                'tgt_degenerate_ratio': float(np.mean([m['tgt_degenerate_ratio'] for m in per_sample_measurements])),
                'ref_avg_stretch': float(np.mean([m['ref_avg_stretch'] for m in per_sample_measurements])),
                'tgt_avg_stretch': float(np.mean([m['tgt_avg_stretch'] for m in per_sample_measurements])),
                'ref_max_stretch': float(np.mean([m['ref_max_stretch'] for m in per_sample_measurements])),
                'tgt_max_stretch': float(np.mean([m['tgt_max_stretch'] for m in per_sample_measurements])),
                'ref_scale_variation': float(np.mean([m['ref_scale_variation'] for m in per_sample_measurements])),
                'tgt_scale_variation': float(np.mean([m['tgt_scale_variation'] for m in per_sample_measurements])),
                'ref_bending_degree': float(np.mean([m['ref_bending_degree'] for m in per_sample_measurements])),
                'tgt_bending_degree': float(np.mean([m['tgt_bending_degree'] for m in per_sample_measurements])),
                'effective_coverage': float(np.mean([m['effective_coverage'] for m in per_sample_measurements])),
            },
            'timing_mean': {
                't_fixed_single_s': float(np.mean([m['t_fixed_single_s'] for m in per_sample_measurements])),
                't_search_total_s': float(np.mean([m['t_search_total_s'] for m in per_sample_measurements])),
            }
        }
        with open(args.measurement_summary_json, 'w') as jf:
            json.dump(summary_data, jf, indent=2)

    print("################## End M2 Testing #######################")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_epoch', type=int, default=5)
    parser.add_argument('--woCoefNet_path', type=str, default='/workspace/python_pro/test1/RopStitch/woCoefNet/model_homo/epoch100_model.pth')
    parser.add_argument('--m2_model_path', type=str, default='/workspace/python_pro/test1/RopStitch/wCoefNet/model_m2/epoch020_m2_model.pth')
    parser.add_argument('--test_path', type=str, default='/workspace/python_pro/test1/RopStitch/data/UDIS-D/testing/')
    parser.add_argument('--metrics_csv', type=str, default='')
    parser.add_argument('--measurement_csv', type=str, default='')
    parser.add_argument('--save_meshes_dir', type=str, default='')
    parser.add_argument('--measurement_summary_json', type=str, default='')

    args = parser.parse_args()
    test_m2(args)
