"""Offline, single-pair entry point for semantic RopStitch."""

import argparse
import json
import os
from pathlib import Path
import time


def parse_args():
    sxl = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image1", type=Path, required=True)
    parser.add_argument("--image2", type=Path, required=True)
    parser.add_argument("--sam1", type=Path, required=True, help="View-local SAM .npz")
    parser.add_argument("--sam2", type=Path, required=True, help="View-local SAM .npz")
    parser.add_argument("--output", type=Path, required=True, help="A new directory")
    parser.add_argument("--alignment", type=Path, default=sxl / "woCoefNet/model_homo/epoch100_model.pth")
    parser.add_argument("--coef", type=Path, default=sxl / "wCoefNet/model_coef/epoch050_coefmodel.pth")
    parser.add_argument("--composition", type=Path, help="Optional lkm Composition checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1"), default="cpu")
    parser.add_argument("--mode", choices=("baseline", "equal", "semantic", "full", "compare"), default="full")
    parser.add_argument("--alpha", type=float, default=0.5, help="RopStitch correlation fusion weight")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--max-shift", type=float, default=12, help="Per-axis bound in 512-scale pixels")
    parser.add_argument("--max-pixels", type=int, default=2_000_000)
    parser.add_argument("--resolution", type=int, default=128, help="Square refinement raster")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not (0 <= args.alpha <= 1 and 0 <= args.steps <= 250
            and 0 < args.max_shift <= 32 and 64 <= args.resolution <= 256
            and args.max_pixels > 0 and args.cpu_threads > 0):
        parser.error("Check alpha [0,1], steps [0,250], max-shift (0,32], resolution [64,256] and positive limits.")
    if args.output.exists():
        parser.error("--output must not already exist")
    for path in (args.image1, args.image2, args.sam1, args.sam2, args.alignment, args.coef, args.composition):
        if path is not None and not path.is_file():
            parser.error(f"File not found: {path}")
    return args


def read_image(path, device):
    import cv2
    import numpy as np
    import torch

    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    # Original models were trained on OpenCV BGR in [-1, 1].
    tensor = torch.from_numpy(image).to(device, torch.float32)
    return tensor.permute(2, 0, 1)[None] / 127.5 - 1


def read_sam(path, device):
    import numpy as np
    import torch

    with np.load(path, allow_pickle=False) as data:
        ids = data["id_map"]
        foreground = data["binary_mask"]
    if (ids.ndim != 2 or ids.size == 0 or not np.issubdtype(ids.dtype, np.integer)
            or ids.min() < 0 or ids.max() >= 2**24 or foreground.shape != ids.shape
            or not np.isin(foreground, (0, 1, 255)).all()
            or np.any((ids > 0) & (foreground == 0))):
        raise ValueError(f"Invalid SAM id_map or binary_mask: {path}")
    return (torch.from_numpy(ids.astype(np.int64)).to(device)[None, None],
            torch.from_numpy((foreground != 0).astype(np.float32)).to(device)[None, None])


def load_checkpoint(model, path, device, label):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state.get("model", state), strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Incompatible {label} checkpoint: {path}") from exc
    return model.to(device).requires_grad_(False).eval()


def composition_model():
    from importlib.util import module_from_spec, spec_from_file_location

    source = Path(__file__).resolve().parents[3] / "Composition/Codes/network.py"
    spec = spec_from_file_location("lkm_composition_network", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Network()


def load_composition(path, device):
    return load_checkpoint(composition_model(), path, device, "Composition")


def write_png(path, values, *, rgb_values):
    import cv2
    import torch

    scaled = (values + 1) * 127.5 if rgb_values else values * 255
    array = scaled[0].clamp(0, 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    if array.shape[2] == 1:
        array = array[:, :, 0]
    ok, encoded = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError(f"Could not encode: {path}")
    encoded.tofile(path)


def main():
    args = parse_args()
    # Legacy RopStitch helpers allocate on the process's default CUDA device.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else args.device[-1]
    import torch
    import torch.nn.functional as F
    from network import CoefNetwork, Network, get_rigid_mesh
    from semantic_refine import image_and_support, inverse_tps, output_size, plane_grid, refine, sample, seam_blend
    from utils.torch_DLT import Tensor_ACA_rect

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu" if args.device == "cpu" else "cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {args.device} is not available.")
    images = tuple(read_image(path, device) for path in (args.image1, args.image2))
    if images[0].shape != images[1].shape:
        raise ValueError("This entry point requires equal image dimensions for both views.")
    ids, foreground = zip(*(read_sam(path, device) for path in (args.sam1, args.sam2)))
    for id_map in ids:
        if id_map.shape[-2:] not in (images[0].shape[-2:], (512, 512)):
            raise ValueError("SAM masks must use the original image dimensions or the project's 512×512 preprocessing.")
    alignment = load_checkpoint(Network(pretrained=False), args.alignment, device, "alignment")
    coef = load_checkpoint(CoefNetwork(), args.coef, device, "CoefNet")
    composition = load_composition(args.composition, device) if args.composition else None
    start = time.perf_counter()
    with torch.no_grad():
        scaled = tuple(F.interpolate(im, (512, 512), mode="bilinear", align_corners=False, antialias=True)
                       for im in images)
        scaled_ids = tuple(F.interpolate(m.float(), (512, 512), mode="nearest-exact").long() for m in ids)
        scaled_foreground = tuple(F.interpolate(m, (512, 512), mode="nearest-exact") for m in foreground)
        left, right, coefficients, mesh1, mesh2 = alignment(*scaled, coef, alpha=args.alpha)
        corners = mesh1.new_tensor([[[0, 0], [512, 0], [0, 512], [512, 512]]])
        homographies = tuple(Tensor_ACA_rect(corners, c) for c in (left[-1], right[-1]))
        source = get_rigid_mesh(1, 512, 512) / 512
        initial = (mesh1 / 512, mesh2 / 512)
    if device.type == "cuda":
        torch.cuda.synchronize()
    backbone_seconds = time.perf_counter() - start
    modes = ("baseline", "equal", "semantic", "full") if args.mode == "compare" else (args.mode,)
    report = dict(
        alignment_checkpoint=str(args.alignment.resolve()),
        coef_checkpoint=str(args.coef.resolve()),
        composition_checkpoint=str(args.composition.resolve()) if args.composition else None,
        inputs=[str(path.resolve()) for path in (args.image1, args.image2, args.sam1, args.sam2)],
        alpha=args.alpha, plane_coefficients=coefficients[-1].flatten().tolist(),
        device=args.device, steps=args.steps, max_shift_px=args.max_shift,
        evaluation_resolution=args.resolution, backbone_seconds=backbone_seconds, runs={},
    )
    for mode in modes:
        start = time.perf_counter()
        meshes, bounds, share, pairs, before, after, search = refine(
            scaled, scaled_ids, scaled_foreground, homographies, initial, source, mode=mode, steps=args.steps,
            max_shift_px=args.max_shift, composition=composition, resolution=args.resolution,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        refinement_seconds = time.perf_counter() - start
        with torch.no_grad():
            h, w = output_size(bounds, images[0].shape[-2:], args.max_pixels)
            grid = plane_grid(bounds, (h, w))
            inverse = [inverse_tps(mesh, source, grid) for mesh in meshes]
            warped, supports = zip(*(image_and_support(im, xy) for im, xy in zip(images, inverse)))
            warped_fg = [sample(fg, xy, "nearest") for fg, xy in zip(foreground, inverse)]
            stitched, soft = seam_blend(warped, supports, warped_fg, inverse, composition)
        if not torch.isfinite(stitched).all():
            raise ValueError(f"Non-finite native-resolution render for mode {mode}.")
        if mode == modes[0]:
            args.output.mkdir(parents=True, exist_ok=False)
        output = args.output / mode
        output.mkdir()
        write_png(output / "stitched.png", stitched, rgb_values=True)
        write_png(output / "blend_weight.png", soft, rgb_values=False)
        for i, (warp, valid, fg) in enumerate(zip(warped, supports, warped_fg), start=1):
            write_png(output / f"warp{i}.png", warp, rgb_values=True)
            write_png(output / f"mask{i}.png", valid, rgb_values=False)
            write_png(output / f"inst{i}.png", fg, rgb_values=False)
        report["runs"][mode] = dict(
            before={k: float(v) for k, v in before.items()},
            after={k: float(v) for k, v in after.items()},
            matches=pairs, share_range=[float(share.min()), float(share.max())],
            search=search,
            updated=any(not torch.equal(m, b) for m, b in zip(meshes, initial)),
            max_control_shift_px=max(float((m - b).abs().max()) for m, b in zip(meshes, initial)) * 512,
            refinement_seconds=refinement_seconds,
            canvas_size=[h, w],
        )
    with (args.output / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(args.output / "metrics.json")


if __name__ == "__main__":
    main()
