import argparse
import json
import os
from typing import List

import cv2
import numpy as np
import torch

from .dataset import compute_crop_box, transform_points_from_crop, CropConfig
from .heatmap import decode_heatmaps_argmax
from .model import BBox8PoseNet
from .utils import draw_corners, ensure_dir, save_json, solve_pnp_from_bbox8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer 8 bbox corners from RGB images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image path or a directory of images")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bbox_json", default=None, help="Optional JSON mapping image stems/paths to [x, y, w, h] boxes for crop-based inference.")
    parser.add_argument("--crop_mode", choices=["full", "bbox"], default="full")
    parser.add_argument("--crop_padding", type=float, default=0.25)
    parser.add_argument("--crop_square", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera_json", default=None, help="Optional json with key cam_K or a flat 3x3 list")
    parser.add_argument("--bbox3d_json", default=None, help="Optional object_bbox_3d.json for solvePnP")
    return parser.parse_args()


def collect_images(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full) and os.path.splitext(name.lower())[1] in {".png", ".jpg", ".jpeg", ".bmp"}:
                files.append(full)
        return files
    return [path]


def load_camera_matrix(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    cam_k = obj.get("cam_K", obj)
    return np.asarray(cam_k, dtype=np.float32).reshape(3, 3)


def load_bbox_mapping(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def lookup_bbox(bbox_map, image_path: str):
    if not bbox_map:
        return None
    stem = os.path.splitext(os.path.basename(image_path))[0]
    abs_path = os.path.abspath(image_path)
    return bbox_map.get(abs_path) or bbox_map.get(image_path) or bbox_map.get(stem)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    args.image_width = ckpt_args.get("image_width", args.image_width)
    args.image_height = ckpt_args.get("image_height", args.image_height)
    if args.crop_mode == "full" and ckpt_args.get("crop_mode") == "bbox":
        args.crop_mode = "bbox"
    if abs(args.crop_padding - 0.25) < 1e-9:
        args.crop_padding = ckpt_args.get("crop_padding", args.crop_padding)
    if args.crop_square is True:
        args.crop_square = ckpt_args.get("crop_square", args.crop_square)
    model = BBox8PoseNet(
        backbone=ckpt_args.get("backbone", "resnet18"),
        pretrained_backbone=False,
        base_channels=ckpt_args.get("base_channels", 32),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    image_paths = collect_images(args.input)
    camera_K = load_camera_matrix(args.camera_json) if args.camera_json else None
    corners_3d = None
    if args.bbox3d_json:
        with open(args.bbox3d_json, "r", encoding="utf-8") as f:
            corners_3d = np.asarray(json.load(f)["corners_3d_object"], dtype=np.float32)
    bbox_map = load_bbox_mapping(args.bbox_json)
    crop_cfg = CropConfig(mode=args.crop_mode, padding=args.crop_padding, square=args.crop_square)

    for image_path in image_paths:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]
        bbox_xywh = lookup_bbox(bbox_map, image_path)
        crop_box = compute_crop_box((orig_w, orig_h), bbox_xywh, crop_cfg)
        x1, y1, x2, y2 = crop_box
        image_crop = image_rgb[y1:y2, x1:x2]
        if image_crop.size == 0:
            image_crop = image_rgb
            crop_box = (0, 0, orig_w, orig_h)
        resized = cv2.resize(image_crop, (args.image_width, args.image_height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        with torch.no_grad():
            pred_heatmaps = model(tensor)
            pred_xy = decode_heatmaps_argmax(pred_heatmaps, image_size=(args.image_width, args.image_height))[0].cpu().numpy()

        pred_xy_orig = transform_points_from_crop(
            torch.from_numpy(pred_xy).float(),
            torch.tensor(crop_box, dtype=torch.float32),
            image_size=(args.image_width, args.image_height),
        ).cpu().numpy()
        valid_mask = np.ones(8, dtype=np.float32)

        result = {
            "image_path": os.path.abspath(image_path),
            "corners_2d": pred_xy_orig.tolist(),
            "crop_box_xyxy": list(crop_box),
        }
        if camera_K is not None and corners_3d is not None:
            pnp = solve_pnp_from_bbox8(corners_3d, pred_xy_orig, camera_K, valid_mask)
            if pnp is not None:
                result.update(pnp)

        stem = os.path.splitext(os.path.basename(image_path))[0]
        save_json(os.path.join(args.output_dir, f"{stem}.json"), result)

        vis = draw_corners(image_rgb, pred_xy_orig, valid_mask)
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_vis.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
