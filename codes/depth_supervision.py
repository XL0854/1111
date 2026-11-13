"""Relative depth supervision on the actual TPS and composed sampling grids.

Prepare floating point depth with the CLI, then pass --depth-data to train.
Depth is used during training only. Constant/invalid depth has zero weight.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def depth_paths(root, pair):
    return [root / side / (pair[0] + '.npy') for side in ('input1', 'input2')]


def fingerprint(root, pairs):
    from run import file_hash
    digest = hashlib.sha256()
    for pair in pairs:
        for source, path in zip(pair[1:], depth_paths(root, pair)):
            record = path.with_suffix('.sha256')
            if not record.exists() or record.read_text() != file_hash(source):
                raise ValueError(f'Depth/source mismatch; prepare depth again: {path}')
            array = np.load(path, allow_pickle=False)
            if array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError(f'Invalid floating point depth map: {path}')
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_depth(root, pair, images):
    result = []
    for path in depth_paths(root, pair):
        value = np.load(path, allow_pickle=False).astype(np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f'Invalid depth map: {path}')
        tensor = torch.as_tensor(value, device=images.device)[None, None]
        result.append(F.interpolate(tensor, images.shape[-2:], mode='bilinear', align_corners=False))
    return torch.cat(result)


def depth_loss(depths, grid, support):
    from model import warp
    sampled = warp(depths, grid)
    valid = (support > 0.99).all(0, keepdim=True)
    valid = F.avg_pool2d(valid.float(), 5, stride=1, padding=2) > 0.999
    count = valid.sum()
    if int(count) < 64:
        return sampled.sum() * 0
    # Detached statistics stop the model changing the normalization to lower loss.
    with torch.no_grad():
        mean = (sampled * valid).sum((2, 3), keepdim=True) / count
        std = (((sampled - mean).square() * valid).sum((2, 3), keepdim=True) / count).sqrt()
    if bool((std < 1e-4).any()):
        return sampled.sum() * 0
    normalized = (sampled - mean) / std.clamp_min(1e-4)
    # Robust penalty limits the influence of occlusions and depth-estimation errors.
    difference = normalized[0:1] - normalized[1:2]
    penalty = torch.sqrt(1 + difference.square()) - 1
    return (penalty * valid).sum() / count


def prepare(args):
    os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[2] / 'depth_cache'))
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from run import discover, file_hash
    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, revision=args.revision).to(device).eval()
    pairs = discover(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    config = dict(model=args.model, revision=getattr(model.config, '_commit_hash', None),
                  data=str(args.data.resolve()))
    plan = args.out / 'depth_config.json'
    if plan.exists() and json.loads(plan.read_text()) != config:
        raise ValueError('Output directory was prepared with another model or dataset')
    plan.write_text(json.dumps(config, indent=2))
    for index, pair in enumerate(pairs, 1):
        for source, target in zip(pair[1:], depth_paths(args.out, pair)):
            target.parent.mkdir(parents=True, exist_ok=True)
            record = target.with_suffix('.sha256')
            digest = file_hash(source)
            if target.exists() and record.exists() and record.read_text() == digest:
                continue
            with Image.open(source) as image:
                image = image.convert('RGB')
                inputs = processor(images=image, return_tensors='pt').to(device)
                with torch.inference_mode():
                    predicted = model(**inputs).predicted_depth[:, None]
                    predicted = F.interpolate(predicted, (image.height, image.width),
                                              mode='bilinear', align_corners=False)[0, 0]
                value = predicted.float().cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite depth for {source}')
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, value)
            temporary.replace(target)
            record.write_text(digest)
        if index == 1 or index % 100 == 0 or index == len(pairs):
            print(f'Depth prepared {index}/{len(pairs)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='depth-anything/Depth-Anything-V2-Small-hf')
    parser.add_argument('--revision', default='5426e4f0f36572d16453bbda7a8389317b1bef99')
    parser.add_argument('--device', default='cuda:0')
    prepare(parser.parse_args())
