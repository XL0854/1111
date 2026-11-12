# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_model, build_output_model, Network, CoefNetwork, H2Offset
from utils import torch_DLT
from dataset import *
import os
import cv2
import skimage
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import grid_res
import random
import torch.backends.cudnn as cudnn
import csv
import json
import time
from utils.grid_metrics import compute_grid_metrics

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

setup_seed(114514)

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

last_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_COEF_DIR = os.path.join(last_path, 'model_coef')


def draw_mesh_on_warp(warp, f_local):
    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp

def maskSSIM(image1, image2, mask):
    image1 = image1 * mask
    image2 = image2 * mask
    _, ssim = compare_ssim(image1, image2, data_range=255, channel_axis=2, full=True)
    ssim = np.sum(ssim * mask) / (np.sum(mask) + 1e-6)
    return ssim

def maskPSNR(image1, image2, mask):
    image1 = image1 * mask / 255.
    image2 = image2 * mask / 255.
    rmse = np.sqrt(np.sum((image1 - image2) ** 2) / mask.sum())
    psnr = 20 * np.log10(1 / rmse)
    return psnr

def test_once(net, coef_net, inpu1_tesnor, inpu2_tesnor, alpha=0.5, max_out_height=4000):
    with torch.no_grad():
        batch_out, flag_check = build_output_model(net, coef_net, inpu1_tesnor, inpu2_tesnor, alpha, max_out_height)
    warp_fallback = not flag_check
    if not flag_check:
        print("warp size is too huge {}! pass this image.".format(batch_out))
        batch_out = {}
        mask = torch.ones_like(inpu1_tesnor).to(inpu1_tesnor.device)
        batch_out['output_ref'] = torch.cat((inpu1_tesnor+1, mask), 1)
        batch_out['output_tgt'] = torch.cat((inpu2_tesnor+1, mask), 1)
        batch_out['final_cdec'] = None
        batch_out['mesh_ref'] = None
        batch_out['mesh_tgt'] = None
        batch_out['rigid_mesh'] = None
        batch_out['out_height'] = inpu1_tesnor.shape[2]
        batch_out['out_width'] = inpu1_tesnor.shape[3]

    output_tps_ref = batch_out['output_ref']
    output_tps_tgt = batch_out['output_tgt']

    # SSIM 3
    output_ref = (output_tps_ref[0,0:3,...]*127.5).cpu().detach().numpy().transpose(1,2,0)
    output_tgt = (output_tps_tgt[0,0:3,...]*127.5).cpu().detach().numpy().transpose(1,2,0)
    overlap_mask = output_tps_ref[0,3:6,...] * output_tps_tgt[0,3:6,...]
    overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
    ssim3 = maskSSIM(output_ref, output_tgt, overlap_mask)
    psnr3 = maskPSNR(output_ref, output_tgt, overlap_mask)

    fusion = output_ref * (output_ref/ (output_ref+output_tgt+1e-6)) + output_tgt * (output_tgt/ (output_ref+output_tgt+1e-6))

    details = {
        'warp_fallback': warp_fallback,
        'final_cdec': batch_out.get('final_cdec', None),
        'mesh_ref': batch_out.get('mesh_ref', None),
        'mesh_tgt': batch_out.get('mesh_tgt', None),
        'rigid_mesh': batch_out.get('rigid_mesh', None),
        'out_height': batch_out.get('out_height', inpu1_tesnor.shape[2]),
        'out_width': batch_out.get('out_width', inpu1_tesnor.shape[3]),
        'output_tps_ref': output_tps_ref,
        'output_tps_tgt': output_tps_tgt,
    }

    return ssim3, psnr3, fusion, details

def ternary_search(net, coef_net, input1, input2, low=0.0, high=1.0, max_iter=20, max_out_height=4000, tol=1e-6):
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
        
        # 计算两个三分点
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3
        
        # 计算 mid1 和 mid2 的 SSIM
        if t_single_first is None:
            t0 = time.perf_counter()
            ssim1, psnr1, img1, det1 = test_once(net, coef_net, input1, input2, alpha=mid1, max_out_height=max_out_height)
            t_single_first = time.perf_counter() - t0
        else:
            ssim1, psnr1, img1, det1 = test_once(net, coef_net, input1, input2, alpha=mid1, max_out_height=max_out_height)

        ssim2, psnr2, img2, det2 = test_once(net, coef_net, input1, input2, alpha=mid2, max_out_height=max_out_height)

        if det1['warp_fallback']:
            invalid_candidate_count += 1
            fallback_count += 1
        if det2['warp_fallback']:
            invalid_candidate_count += 1
            fallback_count += 1
        
        # 更新最优值
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
        
        # 调整搜索区间
        if ssim1 < ssim2:
            low = mid1  # 峰值在 [mid1, high]
        else:
            high = mid2  # 峰值在 [low, mid2]

    t_search_total = time.perf_counter() - t_search_start

    search_meta = {
        't_single_first_s': t_single_first if t_single_first is not None else 0.0,
        't_search_total_s': t_search_total,
        'invalid_candidates': invalid_candidate_count,
        'fallback_count': fallback_count,
        'best_details': best_details,
    }

    return best_alpha, best_ssim, best_psnr, best_img, search_meta


class FixedHalfPlane(nn.Module):
    def forward(self, input1, input2, motion):
        height, width = input1.shape[-2:]
        corners = motion.new_tensor([[0, 0], [width, 0], [0, height], [width, height]])
        corners = corners.unsqueeze(0).expand(motion.shape[0], -1, -1)
        coefficients = torch.full_like(motion[..., :1], 0.5)
        target = corners + motion * coefficients
        full_h = torch_DLT.Tensor_ACA_rect(corners, corners + motion)
        target_h = torch_DLT.Tensor_ACA_rect(corners, target)
        reference_h = torch.inverse(full_h) @ target_h
        return [H2Offset(reference_h, corners)], [target], [coefficients]


def infer_pair(args, net, plane, input1, input2):
    if args.legacy_search:
        return ternary_search(net, plane, input1, input2, low=-1.0, high=2.0,
                              max_iter=args.max_epoch, max_out_height=4000)
    alpha = 0.5
    start = time.perf_counter()
    ssim, psnr, image, details = test_once(net, plane, input1, input2, alpha=alpha)
    return alpha, ssim, psnr, image, {
        't_single_first_s': time.perf_counter() - start,
        't_search_total_s': 0.0,
        'invalid_candidates': int(details['warp_fallback']),
        'fallback_count': int(details['warp_fallback']),
        'best_details': details,
    }


def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    test_other_data = TestDataset(data_path=args.test_other_path)
    test_other_loader = DataLoader(dataset=test_other_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = Network()
    coef_net = CoefNetwork() if args.legacy_search else FixedHalfPlane()
    if torch.cuda.is_available():
        net = net.cuda()
        coef_net = coef_net.cuda()

    # load the existing models
    homo_checkpoint = torch.load(args.woCoefNet_path)
    net.load_state_dict(homo_checkpoint['model'])
    print('load homo net model from {}!'.format(args.woCoefNet_path))    

    if args.legacy_search:
        coef_ckpt_list = glob.glob(MODEL_COEF_DIR + "/*.pth")
        coef_ckpt_list.sort()

        if len(coef_ckpt_list) != 0:
            model_path = coef_ckpt_list[-1]
            checkpoint = torch.load(model_path)
            coef_net.load_state_dict(checkpoint['model'])
            print('load coef model from {}!'.format(model_path))
        else:
            print('No coef checkpoint found!')

    else:
        print("Fixed half-displacement plane, alpha=0.5, one pass per pair.")

    print("##################start testing#######################")

    path_ave_other_fusion = '../ave_other_fusion/'
    if not os.path.exists(path_ave_other_fusion):
        os.makedirs(path_ave_other_fusion)

    ssim_other_list = []
    psnr3_other_list = []
    net.eval()
    for i, batch_value in enumerate(test_other_loader):
        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        if inpu1_tesnor.shape != inpu2_tesnor.shape:
            print('input1 and input2 have different size!')
            continue

        best_alpha, best_ssim, best_psnr, best_img, _ = infer_pair(args, net, coef_net, inpu1_tesnor, inpu2_tesnor)

        path = path_ave_other_fusion + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, best_img)
        ssim_other_list.append(best_ssim)
        psnr3_other_list.append(best_psnr)

        print('i = {}, best alpha = {}, best_ssim:{}'.format(i+1, best_alpha, best_ssim))
        torch.cuda.empty_cache()

    psnr3_other_list.sort(reverse=True)
    list_len = len(psnr3_other_list)
    print("top 30%", np.mean(psnr3_other_list[:int(list_len*0.3)]))
    print("top 30~60%", np.mean(psnr3_other_list[int(list_len*0.3):int(list_len*0.6)]))
    print("top 60~100%", np.mean(psnr3_other_list[int(list_len*0.6):]))
    print('average psnr:', np.mean(psnr3_other_list))

    ssim_other_list.sort(reverse=True)
    print("top 30%", np.mean(ssim_other_list[:int(list_len*0.3)]))
    print("top 30~60%", np.mean(ssim_other_list[int(list_len*0.3):int(list_len*0.6)]))
    print("top 60~100%", np.mean(ssim_other_list[int(list_len*0.6):]))
    print('average ssim:', np.mean(ssim_other_list))

    #############################
    path_ave_fusion = '../ave_fusion/'
    if not os.path.exists(path_ave_fusion):
        os.makedirs(path_ave_fusion)

    if args.save_meshes_dir and not os.path.exists(args.save_meshes_dir):
        os.makedirs(args.save_meshes_dir)

    ssim3_list = []
    psnr3_list = []
    per_sample_metrics = []
    per_sample_measurements = []

    net.eval()
    for i, batch_value in enumerate(test_loader):
        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        best_alpha, best_ssim, best_psnr, best_img, s_meta = infer_pair(
            args, net, coef_net, inpu1_tesnor, inpu2_tesnor
        )
        t_fixed_single_s = s_meta['t_single_first_s']

        path = path_ave_fusion + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, best_img)
        ssim3_list.append(best_ssim)
        psnr3_list.append(best_psnr)

        best_det = s_meta['best_details']
        is_fallback = best_det['warp_fallback'] if best_det is not None else True
        is_success = not is_fallback

        # Basic record
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
            'failure_reason': '' if s_meta['fallback_count'] == 0 else f"official_warp_fallback:{s_meta['fallback_count']} candidates",
        })

        # Measurement record
        cdec_vals = [0.0, 0.0, 0.0, 0.0]
        ref_metrics = {'folding_ratio': 0.0, 'degenerate_ratio': 0.0, 'avg_stretch': 0.0, 'max_stretch': 0.0, 'scale_variation': 0.0, 'bending_degree': 0.0}
        tgt_metrics = {'folding_ratio': 0.0, 'degenerate_ratio': 0.0, 'avg_stretch': 0.0, 'max_stretch': 0.0, 'scale_variation': 0.0, 'bending_degree': 0.0}
        effective_coverage = 1.0
        out_h = inpu1_tesnor.shape[2]
        out_w = inpu1_tesnor.shape[3]

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
                    mesh_ref=mesh_ref.detach().cpu().numpy() if mesh_ref is not None else None,
                    mesh_tgt=mesh_tgt.detach().cpu().numpy() if mesh_tgt is not None else None,
                    rigid_mesh=rigid_mesh.detach().cpu().numpy() if rigid_mesh is not None else None,
                    cdec=np.array(cdec_vals, dtype=np.float32),
                )

        per_sample_measurements.append({
            'sample_index': i + 1,
            'input1_filename': f1_name,
            'input2_filename': f2_name,
            'psnr': best_psnr,
            'ssim': best_ssim,
            'best_alpha': best_alpha,
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

        print('i = {}, best alpha = {}, best_ssim:{}'.format(i+1, best_alpha, best_ssim))
        torch.cuda.empty_cache()

    # Sort copies for bucketed display (keeping original order intact)
    psnr_sorted = sorted(psnr3_list, reverse=True)
    ssim_sorted = sorted(ssim3_list, reverse=True)
    list_len = len(psnr3_list)
    idx30 = 331 if list_len == 1105 else int(list_len * 0.3)
    idx60 = 663 if list_len == 1105 else int(list_len * 0.6)

    print("top 30%", np.mean(psnr_sorted[:idx30]))
    print("top 30~60%", np.mean(psnr_sorted[idx30:idx60]))
    print("top 60~100%", np.mean(psnr_sorted[idx60:]))
    print('average psnr:', np.mean(psnr_sorted))

    print("top 30%", np.mean(ssim_sorted[:idx30]))
    print("top 30~60%", np.mean(ssim_sorted[idx30:idx60]))
    print("top 60~100%", np.mean(ssim_sorted[idx60:]))
    print('average ssim:', np.mean(ssim_sorted))

    # Write basic metrics csv
    if args.metrics_csv:
        metrics_dir = os.path.dirname(os.path.abspath(args.metrics_csv))
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        with open(args.metrics_csv, 'w', newline='') as metrics_file:
            fieldnames = ['sample_index', 'input1_filename', 'input2_filename', 'best_alpha', 'psnr', 'ssim', 'valid', 'failure_reason']
            writer_csv = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(per_sample_metrics)
        print('per-sample metrics:', os.path.abspath(args.metrics_csv))

    # Write measurement csv
    if args.measurement_csv:
        meas_dir = os.path.dirname(os.path.abspath(args.measurement_csv))
        if not os.path.exists(meas_dir):
            os.makedirs(meas_dir)
        with open(args.measurement_csv, 'w', newline='') as meas_file:
            m_fieldnames = [
                'sample_index', 'input1_filename', 'input2_filename', 'psnr', 'ssim', 'best_alpha',
                'cdec_1', 'cdec_2', 'cdec_3', 'cdec_4', 'out_height', 'out_width',
                't_fixed_single_s', 't_search_total_s', 'invalid_candidates', 'fallback_count', 'is_success',
                'ref_folding_ratio', 'tgt_folding_ratio', 'ref_degenerate_ratio', 'tgt_degenerate_ratio',
                'ref_avg_stretch', 'tgt_avg_stretch', 'ref_max_stretch', 'tgt_max_stretch',
                'ref_scale_variation', 'tgt_scale_variation', 'ref_bending_degree', 'tgt_bending_degree',
                'effective_coverage'
            ]
            m_writer = csv.DictWriter(meas_file, fieldnames=m_fieldnames)
            m_writer.writeheader()
            m_writer.writerows(per_sample_measurements)
        print('per-sample measurements:', os.path.abspath(args.measurement_csv))

    # Write measurement summary JSON
    if args.measurement_summary_json:
        sum_dir = os.path.dirname(os.path.abspath(args.measurement_summary_json))
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)

        total_samples = len(per_sample_measurements)
        success_samples = sum(1 for m in per_sample_measurements if m['is_success'])
        fail_samples = total_samples - success_samples

        summary_data = {
            'protocol': 'measurement-baseline',
            'sample_count': total_samples,
            'success_count': success_samples,
            'failure_count': fail_samples,
            'failure_rate': float(fail_samples / max(1, total_samples)),
            'total_invalid_candidates': sum(m['invalid_candidates'] for m in per_sample_measurements),
            'total_fallbacks': sum(m['fallback_count'] for m in per_sample_measurements),
            'alignment_metrics': {
                'psnr': {
                    'easy': float(np.mean(psnr_sorted[:idx30])),
                    'moderate': float(np.mean(psnr_sorted[idx30:idx60])),
                    'hard': float(np.mean(psnr_sorted[idx60:])),
                    'average': float(np.mean(psnr_sorted)),
                },
                'ssim': {
                    'easy': float(np.mean(ssim_sorted[:idx30])),
                    'moderate': float(np.mean(ssim_sorted[idx30:idx60])),
                    'hard': float(np.mean(ssim_sorted[idx60:])),
                    'average': float(np.mean(ssim_sorted)),
                }
            },
            'geometric_metrics_mean': {
                'ref_folding_ratio': float(np.mean([m['ref_folding_ratio'] for m in per_sample_measurements])),
                'tgt_folding_ratio': float(np.mean([m['tgt_folding_ratio'] for m in per_sample_measurements])),
                'mean_folding_ratio': float(np.mean([0.5*(m['ref_folding_ratio']+m['tgt_folding_ratio']) for m in per_sample_measurements])),
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
        print('measurement summary:', os.path.abspath(args.measurement_summary_json))

    print("##################end testing#######################")


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--legacy_search', action='store_true', help='Run the original iterative search for comparisons')
    parser.add_argument('--max_epoch', type=int, default=5, help='Outer search rounds with --legacy_search')
    parser.add_argument('--woCoefNet_path', type=str, default='/workspace/python_pro/test1/RopStitch/woCoefNet/model_homo/epoch100_model.pth')
    parser.add_argument('--test_path', type=str, default='/workspace/python_pro/test1/RopStitch/data/UDIS-D/testing/')
    parser.add_argument('--test_other_path', type=str, default='/workspace/python_pro/test1/RopStitch/data/stitch_real/')
    parser.add_argument('--metrics_csv', type=str, default='')
    parser.add_argument('--measurement_csv', type=str, default='')
    parser.add_argument('--save_meshes_dir', type=str, default='')
    parser.add_argument('--measurement_summary_json', type=str, default='')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
