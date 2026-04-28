import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

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
    parser.add_argument("--camera_json", default=None, help="Optional json with key cam_K or a flat 3x3 list")
    parser.add_argument("--bbox3d_json", default=None, help="Optional object_bbox_3d.json for solvePnP")
    parser.add_argument(
        "--bboxes",
        default=None,
        help="Optional semicolon-separated boxes for a single image: x1,y1,x2,y2;x1,y1,x2,y2",
    )
    parser.add_argument(
        "--bboxes_json",
        default=None,
        help=(
            "Optional JSON mapping image basename/stem/path to boxes. "
            "Each box can be [x1,y1,x2,y2] or {'bbox':[x1,y1,x2,y2]}."
        ),
    )
    parser.add_argument("--crop_margin", type=float, default=0.15, help="Relative padding around each input bbox.")
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
    if isinstance(obj, dict):
        if "cam_K" in obj:
            cam_k = obj["cam_K"]
            return np.asarray(cam_k, dtype=np.float32).reshape(3, 3)
        if all(k in obj for k in ("fx", "fy", "cx", "cy")):
            return np.asarray(
                [
                    [obj["fx"], 0.0, obj["cx"]],
                    [0.0, obj["fy"], obj["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        # Support nested BOP-like single-entry dicts if ever passed in.
        if len(obj) == 1:
            only_value = next(iter(obj.values()))
            if isinstance(only_value, dict):
                if "cam_K" in only_value:
                    return np.asarray(only_value["cam_K"], dtype=np.float32).reshape(3, 3)
                if all(k in only_value for k in ("fx", "fy", "cx", "cy")):
                    return np.asarray(
                        [
                            [only_value["fx"], 0.0, only_value["cx"]],
                            [0.0, only_value["fy"], only_value["cy"]],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    )
    return np.asarray(obj, dtype=np.float32).reshape(3, 3)


def parse_inline_bboxes(spec: Optional[str]) -> List[List[float]]:
    if not spec:
        return []
    boxes = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(v.strip()) for v in chunk.split(",")]
        if len(vals) != 4:
            raise ValueError(f"Expected 4 comma-separated values per bbox, got: {chunk}")
        boxes.append(vals)
    return boxes


def load_bboxes_json(path: Optional[str]) -> Dict[str, List[List[float]]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {"*": normalize_boxes(raw)}
    if not isinstance(raw, dict):
        raise ValueError("--bboxes_json must be a dict or list")
    return {str(k): normalize_boxes(v) for k, v in raw.items()}


def normalize_boxes(obj) -> List[List[float]]:
    if isinstance(obj, dict):
        if "boxes" in obj:
            obj = obj["boxes"]
        elif "instances" in obj:
            obj = obj["instances"]
        elif "bbox" in obj:
            obj = [obj]
    boxes = []
    for item in obj:
        box = item.get("bbox", item) if isinstance(item, dict) else item
        if len(box) < 4:
            raise ValueError(f"Invalid bbox entry: {item}")
        boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
    return boxes


def boxes_for_image(image_path: str, bbox_map: Dict[str, List[List[float]]], inline_boxes: List[List[float]]) -> List[List[float]]:
    if inline_boxes:
        return inline_boxes
    if not bbox_map:
        return []
    abs_path = os.path.abspath(image_path)
    base = os.path.basename(image_path)
    stem = os.path.splitext(base)[0]
    for key in (abs_path, image_path, base, stem, "*"):
        if key in bbox_map:
            return bbox_map[key]
    return []


def expand_box(box: Sequence[float], width: int, height: int, margin: float) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = float(margin) * max(bw, bh)
    x1 = max(0, int(np.floor(x1 - pad)))
    y1 = max(0, int(np.floor(y1 - pad)))
    x2 = min(width, int(np.ceil(x2 + pad)))
    y2 = min(height, int(np.ceil(y2 + pad)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid expanded bbox: {box}")
    return [x1, y1, x2, y2]


def infer_one_image(model, image_rgb: np.ndarray, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    resized = cv2.resize(image_rgb, (args.image_width, args.image_height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        pred_heatmaps = model(tensor)
        pred_xy = decode_heatmaps_argmax(pred_heatmaps, image_size=(args.image_width, args.image_height))[0].cpu().numpy()
    return pred_xy


def draw_instances(image_rgb: np.ndarray, instances: List[Dict]) -> np.ndarray:
    canvas = image_rgb.copy()
    for inst_idx, inst in enumerate(instances):
        box = inst.get("bbox")
        if box is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 220, 80), 2)
            cv2.putText(canvas, f"inst {inst_idx}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
        canvas = draw_corners(canvas, np.asarray(inst["corners_2d"], dtype=np.float32), np.ones(8, dtype=np.float32))
    return canvas


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    args.image_width = ckpt_args.get("image_width", args.image_width)
    args.image_height = ckpt_args.get("image_height", args.image_height)
    model = BBox8PoseNet(
        backbone=ckpt_args.get("backbone", "resnet18"),
        pretrained_backbone=False,
        base_channels=ckpt_args.get("base_channels", 32),
        decoder=ckpt_args.get("decoder", "boxdreamer_lite"),
        decoder_dim=ckpt_args.get("decoder_dim", 192),
        decoder_depth=ckpt_args.get("decoder_depth", 3),
        decoder_heads=ckpt_args.get("decoder_heads", 8),
        decoder_patch_size=ckpt_args.get("decoder_patch_size", 4),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    image_paths = collect_images(args.input)
    camera_K = load_camera_matrix(args.camera_json) if args.camera_json else None
    inline_boxes = parse_inline_bboxes(args.bboxes)
    bbox_map = load_bboxes_json(args.bboxes_json)
    corners_3d = None
    if args.bbox3d_json:
        with open(args.bbox3d_json, "r", encoding="utf-8") as f:
            corners_3d = np.asarray(json.load(f)["corners_3d_object"], dtype=np.float32)

    for image_path in image_paths:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]
        valid_mask = np.ones(8, dtype=np.float32)

        boxes = boxes_for_image(image_path, bbox_map, inline_boxes)
        instances = []
        if boxes:
            for box in boxes:
                crop_box = expand_box(box, orig_w, orig_h, args.crop_margin)
                x1, y1, x2, y2 = crop_box
                crop_rgb = image_rgb[y1:y2, x1:x2]
                pred_xy = infer_one_image(model, crop_rgb, args, device)
                pred_xy_orig = pred_xy.copy()
                pred_xy_orig[:, 0] = x1 + pred_xy_orig[:, 0] * ((x2 - x1) / float(args.image_width))
                pred_xy_orig[:, 1] = y1 + pred_xy_orig[:, 1] * ((y2 - y1) / float(args.image_height))
                inst = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "input_bbox": [float(v) for v in box[:4]],
                    "corners_2d": pred_xy_orig.tolist(),
                }
                if camera_K is not None and corners_3d is not None:
                    pnp = solve_pnp_from_bbox8(corners_3d, pred_xy_orig, camera_K, valid_mask)
                    if pnp is not None:
                        inst.update(pnp)
                instances.append(inst)
        else:
            pred_xy = infer_one_image(model, image_rgb, args, device)
            scale_x = orig_w / float(args.image_width)
            scale_y = orig_h / float(args.image_height)
            pred_xy_orig = pred_xy.copy()
            pred_xy_orig[:, 0] *= scale_x
            pred_xy_orig[:, 1] *= scale_y
            inst = {
                "bbox": None,
                "corners_2d": pred_xy_orig.tolist(),
            }
            if camera_K is not None and corners_3d is not None:
                pnp = solve_pnp_from_bbox8(corners_3d, pred_xy_orig, camera_K, valid_mask)
                if pnp is not None:
                    inst.update(pnp)
            instances.append(inst)

        result = {
            "image_path": os.path.abspath(image_path),
            "instances": instances,
        }
        if len(instances) == 1:
            result["corners_2d"] = instances[0]["corners_2d"]
        if camera_K is not None and corners_3d is not None and len(instances) == 1:
            pred_xy_orig = np.asarray(instances[0]["corners_2d"], dtype=np.float32)
            pnp = solve_pnp_from_bbox8(corners_3d, pred_xy_orig, camera_K, valid_mask)
            if pnp is not None:
                result.update(pnp)

        stem = os.path.splitext(os.path.basename(image_path))[0]
        save_json(os.path.join(args.output_dir, f"{stem}.json"), result)

        vis = draw_instances(image_rgb, instances)
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_vis.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
