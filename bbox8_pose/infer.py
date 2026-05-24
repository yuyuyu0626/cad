import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

from .heatmap import decode_heatmaps_argmax
from .dataset import transform_corners_from_crop
from .model import BBox8PoseNet
from .utils import draw_corners, ensure_dir, project_bbox8_from_pose, save_json, solve_pnp_from_bbox8


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
    parser.add_argument(
        "--reference_corners",
        default=None,
        help=(
            "Optional prior corner predictions/labels used only to derive crop boxes. "
            "Accepts an infer output directory, one JSON file, or annotations-style JSONL."
        ),
    )
    parser.add_argument("--crop_margin", type=float, default=None, help="Relative padding around each input bbox. Defaults to checkpoint crop_margin when available.")
    parser.add_argument("--yolo_model", default=None, help="Optional Ultralytics YOLO model for automatic object bboxes.")
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_iou", type=float, default=0.7)
    parser.add_argument("--yolo_imgsz", type=int, default=960)
    parser.add_argument("--yolo_max_det", type=int, default=20)
    parser.add_argument("--yolo_classes", default=None, help="Optional comma-separated YOLO class ids to keep.")
    parser.add_argument(
        "--no_full_image_fallback",
        action="store_true",
        help="If no manual/YOLO boxes are found, skip the image instead of predicting one full-image instance.",
    )
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


def corners_to_box(corners, valid_mask=None) -> List[float]:
    pts = np.asarray(corners, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"Invalid corners_2d shape: {pts.shape}")
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=np.float32) > 0
        if valid.shape[0] == pts.shape[0] and int(valid.sum()) >= 2:
            pts = pts[valid]
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return [x1, y1, x2, y2]


def add_reference_boxes(box_map: Dict[str, List[List[float]]], keys: List[str], boxes: List[List[float]]) -> None:
    if not boxes:
        return
    for key in keys:
        if key:
            box_map.setdefault(str(key), []).extend(boxes)


def reference_keys_from_path(path: Optional[str]) -> List[str]:
    if not path:
        return []
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return [os.path.abspath(path), path, base, stem]


def boxes_from_reference_record(record: Dict) -> List[List[float]]:
    boxes = []
    instances = record.get("instances")
    if isinstance(instances, list):
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            if "corners_2d" in inst:
                boxes.append(corners_to_box(inst["corners_2d"], inst.get("corner_valid_mask")))
            elif "bbox" in inst and inst["bbox"] is not None:
                boxes.append(box_xyxy(inst["bbox"]))
            elif "input_bbox" in inst and inst["input_bbox"] is not None:
                boxes.append(box_xyxy(inst["input_bbox"]))
    elif "corners_2d" in record:
        boxes.append(corners_to_box(record["corners_2d"], record.get("corner_valid_mask")))
    elif "bbox" in record and record["bbox"] is not None:
        boxes.append(box_xyxy(record["bbox"]))
    elif "input_bbox" in record and record["input_bbox"] is not None:
        boxes.append(box_xyxy(record["input_bbox"]))
    return boxes


def load_reference_corners(path: Optional[str]) -> Dict[str, List[List[float]]]:
    if not path:
        return {}
    if os.path.isdir(path):
        box_map: Dict[str, List[List[float]]] = {}
        for name in sorted(os.listdir(path)):
            if not name.lower().endswith(".json"):
                continue
            if name.endswith("_vis.json"):
                continue
            json_path = os.path.join(path, name)
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            boxes = boxes_from_reference_record(record)
            stem = os.path.splitext(name)[0]
            keys = [name, stem]
            keys.extend(reference_keys_from_path(record.get("image_path")))
            add_reference_boxes(box_map, keys, boxes)
        return box_map

    box_map = {}
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                boxes = boxes_from_reference_record(record)
                keys = []
                for path_key in ("image_path", "rgb_path", "path", "file_name"):
                    keys.extend(reference_keys_from_path(record.get(path_key)))
                sample_id = record.get("sample_id")
                if sample_id is not None:
                    keys.append(str(sample_id))
                add_reference_boxes(box_map, keys, boxes)
        return box_map

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        if "instances" in raw or "corners_2d" in raw:
            boxes = boxes_from_reference_record(raw)
            keys = reference_keys_from_path(raw.get("image_path"))
            if not keys:
                keys = ["*"]
            add_reference_boxes(box_map, keys, boxes)
            return box_map
        for key, value in raw.items():
            if isinstance(value, dict):
                boxes = boxes_from_reference_record(value)
            else:
                boxes = normalize_boxes(value)
            add_reference_boxes(box_map, [str(key)], boxes)
        return box_map
    if isinstance(raw, list):
        has_records = any(isinstance(item, dict) and ("instances" in item or "corners_2d" in item) for item in raw)
        if has_records:
            for record in raw:
                if not isinstance(record, dict):
                    continue
                boxes = boxes_from_reference_record(record)
                keys = []
                for path_key in ("image_path", "rgb_path", "path", "file_name"):
                    keys.extend(reference_keys_from_path(record.get(path_key)))
                sample_id = record.get("sample_id")
                if sample_id is not None:
                    keys.append(str(sample_id))
                add_reference_boxes(box_map, keys, boxes)
        else:
            add_reference_boxes(box_map, ["*"], normalize_boxes(raw))
        return box_map
    raise ValueError("--reference_corners must be a directory, JSON file, or JSONL file")


def expand_box(box: Sequence[float], width: int, height: int, margin: float) -> List[int]:
    x1, y1, x2, y2 = box_xyxy(box)
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


def box_xyxy(box) -> List[float]:
    if isinstance(box, dict):
        box = box.get("bbox", box.get("xyxy"))
    if box is None or len(box) < 4:
        raise ValueError(f"Invalid bbox: {box}")
    return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]


def parse_yolo_classes(spec: Optional[str]) -> Optional[List[int]]:
    if spec is None or spec.strip() == "":
        return None
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def load_yolo_model(path: Optional[str]):
    if not path:
        return None
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Using --yolo_model requires the ultralytics package in this environment.") from exc
    return YOLO(path)


def detect_yolo_boxes(yolo_model, image_rgb: np.ndarray, args: argparse.Namespace) -> List[Dict]:
    if yolo_model is None:
        return []
    classes = parse_yolo_classes(args.yolo_classes)
    device_arg = 0 if str(args.device).startswith("cuda") else "cpu"
    results = yolo_model.predict(
        image_rgb,
        conf=args.yolo_conf,
        iou=args.yolo_iou,
        imgsz=args.yolo_imgsz,
        max_det=args.yolo_max_det,
        classes=classes,
        device=device_arg,
        verbose=False,
    )
    if not results:
        return []
    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return []
    xyxy = boxes_obj.xyxy.detach().cpu().numpy()
    conf = boxes_obj.conf.detach().cpu().numpy() if boxes_obj.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
    cls = boxes_obj.cls.detach().cpu().numpy() if boxes_obj.cls is not None else np.full((xyxy.shape[0],), -1, dtype=np.float32)
    detections = []
    for box, score, cls_id in zip(xyxy, conf, cls):
        detections.append(
            {
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "score": float(score),
                "class_id": int(cls_id),
            }
        )
    detections.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return detections


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
    for inst in instances:
        raw_corners = np.asarray(inst["corners_2d"], dtype=np.float32)
        canvas = draw_corners(canvas, raw_corners, np.ones(8, dtype=np.float32), draw_edges=True)
    return canvas


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    args.image_width = ckpt_args.get("image_width", args.image_width)
    args.image_height = ckpt_args.get("image_height", args.image_height)
    if args.crop_margin is None:
        args.crop_margin = float(ckpt_args.get("crop_margin", 0.15))
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
    yolo_model = load_yolo_model(args.yolo_model)

    image_paths = collect_images(args.input)
    camera_K = load_camera_matrix(args.camera_json) if args.camera_json else None
    inline_boxes = parse_inline_bboxes(args.bboxes)
    bbox_map = load_bboxes_json(args.bboxes_json)
    reference_bbox_map = load_reference_corners(args.reference_corners)
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
        if not boxes:
            boxes = boxes_for_image(image_path, reference_bbox_map, [])
            if boxes:
                print(f"[INFO] {os.path.basename(image_path)}: reference corners provided {len(boxes)} boxes")
        if not boxes and yolo_model is not None:
            boxes = detect_yolo_boxes(yolo_model, image_rgb, args)
            print(f"[INFO] {os.path.basename(image_path)}: YOLO detected {len(boxes)} boxes")
        instances = []
        if boxes:
            for box in boxes:
                crop_box = expand_box(box, orig_w, orig_h, args.crop_margin)
                x1, y1, x2, y2 = crop_box
                crop_rgb = image_rgb[y1:y2, x1:x2]
                pred_xy = infer_one_image(model, crop_rgb, args, device)
                pred_xy_orig = transform_corners_from_crop(pred_xy, crop_box, (args.image_width, args.image_height))
                inst = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "input_bbox": box_xyxy(box),
                    "corners_2d": pred_xy_orig.tolist(),
                }
                if isinstance(box, dict):
                    if "score" in box:
                        inst["det_score"] = float(box["score"])
                    if "class_id" in box:
                        inst["det_class_id"] = int(box["class_id"])
                if camera_K is not None and corners_3d is not None:
                    pnp = solve_pnp_from_bbox8(corners_3d, pred_xy_orig, camera_K, valid_mask)
                    if pnp is not None:
                        inst.update(pnp)
                        projected = project_bbox8_from_pose(
                            corners_3d,
                            np.asarray(pnp["cam_R_m2c"], dtype=np.float32).reshape(3, 3),
                            np.asarray(pnp["cam_t_m2c"], dtype=np.float32),
                            camera_K,
                        )
                        inst["projected_corners_2d"] = projected.tolist()
                instances.append(inst)
        elif args.no_full_image_fallback:
            print(f"[WARN] {os.path.basename(image_path)}: no boxes found, skipped full-image fallback")
        else:
            if yolo_model is not None:
                print(f"[WARN] {os.path.basename(image_path)}: no YOLO boxes found, using full-image fallback")
            if ckpt_args.get("crop_to_bbox", False):
                print(f"[WARN] {os.path.basename(image_path)}: checkpoint was crop-trained; full-image fallback is distribution-mismatched")
            pred_xy = infer_one_image(model, image_rgb, args, device)
            pred_xy_orig = transform_corners_from_crop(pred_xy, (0, 0, orig_w, orig_h), (args.image_width, args.image_height))
            inst = {
                "bbox": None,
                "corners_2d": pred_xy_orig.tolist(),
            }
            if camera_K is not None and corners_3d is not None:
                pnp = solve_pnp_from_bbox8(corners_3d, pred_xy_orig, camera_K, valid_mask)
                if pnp is not None:
                    inst.update(pnp)
                    projected = project_bbox8_from_pose(
                        corners_3d,
                        np.asarray(pnp["cam_R_m2c"], dtype=np.float32).reshape(3, 3),
                        np.asarray(pnp["cam_t_m2c"], dtype=np.float32),
                        camera_K,
                    )
                    inst["projected_corners_2d"] = projected.tolist()
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
