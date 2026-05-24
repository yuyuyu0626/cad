import argparse
import json
import os
import random
import re
import shutil
from typing import Dict, List, Optional, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Labelme 8-corner point labels to bbox8_pose training labels.")
    parser.add_argument("--labelme_dir", required=True, help="Directory containing Labelme JSON files.")
    parser.add_argument("--output_dir", required=True, help="Output labels_root directory for bbox8_pose.train.")
    parser.add_argument("--camera_json", required=True, help="JSON containing cam_K or fx/fy/cx/cy.")
    parser.add_argument("--bbox3d_json", required=True, help="object_bbox_3d.json containing corners_3d_object.")
    parser.add_argument("--image_root", default=None, help="Optional root used to resolve relative imagePath values.")
    parser.add_argument("--obj_id", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_valid_corners", type=int, default=4, help="Minimum valid corners required to keep an instance.")
    parser.add_argument("--strict", action="store_true", help="Raise an error instead of skipping instances with too few corners.")
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_camera_matrix(path: str) -> List[float]:
    obj = load_json(path)
    if isinstance(obj, dict):
        if "cam_K" in obj:
            return [float(v) for v in obj["cam_K"]]
        if all(k in obj for k in ("fx", "fy", "cx", "cy")):
            return [
                float(obj["fx"]),
                0.0,
                float(obj["cx"]),
                0.0,
                float(obj["fy"]),
                float(obj["cy"]),
                0.0,
                0.0,
                1.0,
            ]
        if len(obj) == 1:
            only_value = next(iter(obj.values()))
            if isinstance(only_value, dict):
                if "cam_K" in only_value:
                    return [float(v) for v in only_value["cam_K"]]
                if all(k in only_value for k in ("fx", "fy", "cx", "cy")):
                    return [
                        float(only_value["fx"]),
                        0.0,
                        float(only_value["cx"]),
                        0.0,
                        float(only_value["fy"]),
                        float(only_value["cy"]),
                        0.0,
                        0.0,
                        1.0,
                    ]
    values = [float(v) for v in obj]
    if len(values) != 9:
        raise ValueError(f"Expected a 3x3 camera matrix in {path}")
    return values


def corner_index(label: str) -> Optional[int]:
    text = str(label).strip().lower()
    if text.isdigit():
        idx = int(text)
        return idx if 0 <= idx < 8 else None
    match = re.search(r"(?:corner|kp|point|pt|p|c)[_\-\s]*(\d+)$", text)
    if match:
        idx = int(match.group(1))
        return idx if 0 <= idx < 8 else None
    return None


def instance_key(shape: Dict, fallback: str) -> str:
    group_id = shape.get("group_id")
    if group_id is not None:
        return str(group_id)
    label = str(shape.get("label", ""))
    match = re.search(r"(?:inst|obj|id)[_\-\s]*(\d+)", label.lower())
    if match:
        return match.group(1)
    return fallback


def resolve_image_path(label_path: str, image_path: str, image_root: Optional[str]) -> str:
    image_path = image_path.replace("\\", os.sep)
    if os.path.isabs(image_path):
        return os.path.abspath(image_path)
    roots = []
    if image_root:
        roots.append(image_root)
    roots.append(os.path.dirname(label_path))
    for root in roots:
        candidate = os.path.abspath(os.path.join(root, image_path))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(os.path.join(roots[0], image_path))


def collect_instances(label_path: str, label_obj: Dict) -> Dict[str, Dict[int, Tuple[float, float]]]:
    instances: Dict[str, Dict[int, Tuple[float, float]]] = {}
    fallback_key = "0"
    for shape in label_obj.get("shapes", []):
        idx = corner_index(shape.get("label", ""))
        if idx is None:
            continue
        points = shape.get("points", [])
        if not points:
            continue
        key = instance_key(shape, fallback_key)
        x, y = points[0]
        instances.setdefault(key, {})[idx] = (float(x), float(y))
    if not instances:
        raise ValueError(f"No corner point labels found in {label_path}")
    return instances


def load_image_size(path: str) -> Optional[Tuple[int, int]]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def scale_instances(
    instances: Dict[str, Dict[int, Tuple[float, float]]],
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0 or (src_w, src_h) == (dst_w, dst_h):
        return instances
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    scaled: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for inst_key, points in instances.items():
        scaled[inst_key] = {idx: (xy[0] * sx, xy[1] * sy) for idx, xy in points.items()}
    return scaled


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cam_k = load_camera_matrix(args.camera_json)
    bbox3d = load_json(args.bbox3d_json)
    corners_3d = bbox3d.get("corners_3d_object")
    if corners_3d is None:
        raise ValueError(f"{args.bbox3d_json} must contain corners_3d_object")

    records = []
    for name in sorted(os.listdir(args.labelme_dir)):
        if not name.lower().endswith(".json"):
            continue
        label_path = os.path.join(args.labelme_dir, name)
        label_obj = load_json(label_path)
        image_path = resolve_image_path(label_path, label_obj["imagePath"], args.image_root)
        label_size = (int(label_obj.get("imageWidth", 0)), int(label_obj.get("imageHeight", 0)))
        actual_size = load_image_size(image_path)
        image_size_tuple = actual_size or label_size
        image_size = [int(image_size_tuple[0]), int(image_size_tuple[1])]
        stem = os.path.splitext(name)[0]
        instances = collect_instances(label_path, label_obj)
        if actual_size is not None and label_size != actual_size:
            print(f"[INFO] scaling {name}: label_size={label_size} image_size={actual_size}")
            instances = scale_instances(instances, label_size, actual_size)
        for inst_key in sorted(instances.keys(), key=lambda x: (len(x), x)):
            points_by_idx = instances[inst_key]
            corners = [[0.0, 0.0] for _ in range(8)]
            valid = [0 for _ in range(8)]
            for idx, xy in points_by_idx.items():
                corners[idx] = [xy[0], xy[1]]
                valid[idx] = 1
            if sum(valid) < args.min_valid_corners:
                message = (
                    f"{label_path} instance {inst_key} has {sum(valid)} valid corners, "
                    f"fewer than {args.min_valid_corners}"
                )
                if args.strict:
                    raise ValueError(message)
                print(f"[WARN] skipped: {message}")
                continue
            records.append(
                {
                    "sample_id": f"{stem}/{inst_key}",
                    "image_id": stem,
                    "ann_idx": int(inst_key) if str(inst_key).isdigit() else inst_key,
                    "obj_id": args.obj_id,
                    "rgb_path": image_path,
                    "image_size": image_size,
                    "cam_K": cam_k,
                    "corners_3d_object": corners_3d,
                    "corners_2d": corners,
                    "corner_valid_mask": valid,
                    "visib_fract": 1.0,
                    "source": "labelme_real",
                }
            )

    if not records:
        raise ValueError(f"No records converted from {args.labelme_dir}")

    ann_path = os.path.join(args.output_dir, "annotations.jsonl")
    with open(ann_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ids = [rec["sample_id"] for rec in records]
    rnd = random.Random(args.seed)
    rnd.shuffle(ids)
    n_val = max(1, int(round(len(ids) * args.val_ratio))) if len(ids) > 1 else 0
    val_ids = set(ids[:n_val])
    train_ids = [sample_id for sample_id in ids if sample_id not in val_ids]
    val_ids_sorted = [sample_id for sample_id in ids if sample_id in val_ids]

    with open(os.path.join(args.output_dir, "train.txt"), "w", encoding="utf-8") as f:
        for sample_id in train_ids:
            f.write(sample_id + "\n")
    with open(os.path.join(args.output_dir, "val.txt"), "w", encoding="utf-8") as f:
        for sample_id in val_ids_sorted:
            f.write(sample_id + "\n")

    shutil.copyfile(args.bbox3d_json, os.path.join(args.output_dir, "object_bbox_3d.json"))
    print(f"[OK] wrote {len(records)} records to {ann_path}")
    print(f"[OK] train={len(train_ids)} val={len(val_ids_sorted)}")


if __name__ == "__main__":
    main()
