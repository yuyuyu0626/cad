import argparse
import json
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract crop boxes from YOLO/corner visualization frames.")
    parser.add_argument("--video", default=None, help="Reference visualization mp4 with drawn boxes.")
    parser.add_argument("--overlay", default=None, help="Reference visualization image or directory with drawn boxes.")
    parser.add_argument("--input", required=True, help="Image path or directory whose frame names should receive boxes.")
    parser.add_argument("--output_json", required=True, help="Output JSON mapping image basename/stem to [x1,y1,x2,y2] boxes.")
    parser.add_argument("--max_boxes", type=int, default=2, help="Maximum boxes to keep per frame.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Read every Nth video frame.")
    parser.add_argument("--frame_offset", type=int, default=0, help="Skip this many video frames before matching image 0.")
    parser.add_argument("--min_area", type=float, default=400.0, help="Minimum connected component box area.")
    parser.add_argument("--min_side", type=int, default=16, help="Minimum component width and height.")
    parser.add_argument("--dilate", type=int, default=5, help="Morphological dilation kernel size for joining wireframe strokes.")
    parser.add_argument("--color", choices=["blue", "white"], default="blue", help="Overlay color to extract.")
    parser.add_argument("--value_thresh", type=int, default=120, help="HSV value threshold for overlay pixels.")
    parser.add_argument("--sat_thresh", type=int, default=80, help="HSV saturation threshold for white overlay pixels.")
    parser.add_argument("--blue_hue_min", type=int, default=95, help="Minimum HSV hue for blue overlay pixels.")
    parser.add_argument("--blue_hue_max", type=int, default=135, help="Maximum HSV hue for blue overlay pixels.")
    parser.add_argument("--blue_sat_min", type=int, default=80, help="Minimum HSV saturation for blue overlay pixels.")
    parser.add_argument("--margin", type=float, default=0.0, help="Optional extra relative padding written into the boxes.")
    parser.add_argument("--debug_dir", default=None, help="Optional directory to write debug masks/box overlays.")
    return parser.parse_args()


def collect_images(path: str) -> List[str]:
    if os.path.isdir(path):
        paths = []
        for name in sorted(os.listdir(path)):
            ext = os.path.splitext(name.lower())[1]
            full = os.path.join(path, name)
            if ext in IMAGE_EXTS and os.path.isfile(full):
                paths.append(full)
        return paths
    return [path]


def collect_overlays(path: Optional[str]) -> List[str]:
    if not path:
        return []
    return collect_images(path)


def expand_box(box: Sequence[float], width: int, height: int, margin: float) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = float(margin) * max(bw, bh)
    return [
        max(0.0, x1 - pad),
        max(0.0, y1 - pad),
        min(float(width), x2 + pad),
        min(float(height), y2 + pad),
    ]


def component_boxes(mask: np.ndarray, min_area: float, min_side: int) -> List[List[float]]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        box_area = float(w * h)
        if w < min_side or h < min_side:
            continue
        if box_area < min_area or area < min_area * 0.03:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
    return boxes


def merge_overlapping_boxes(boxes: List[List[float]], iou_thresh: float = 0.15) -> List[List[float]]:
    merged: List[List[float]] = []
    for box in sorted(boxes, key=lambda b: (b[0], b[1])):
        bx1, by1, bx2, by2 = box
        consumed = False
        for existing in merged:
            ex1, ey1, ex2, ey2 = existing
            ix1 = max(bx1, ex1)
            iy1 = max(by1, ey1)
            ix2 = min(bx2, ex2)
            iy2 = min(by2, ey2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
            area_e = max(1.0, (ex2 - ex1) * (ey2 - ey1))
            iou = inter / max(1.0, area_b + area_e - inter)
            if iou >= iou_thresh or inter / min(area_b, area_e) > 0.5:
                existing[0] = min(ex1, bx1)
                existing[1] = min(ey1, by1)
                existing[2] = max(ex2, bx2)
                existing[3] = max(ey2, by2)
                consumed = True
                break
        if not consumed:
            merged.append(box.copy())
    return merged


def extract_boxes(frame_bgr: np.ndarray, args: argparse.Namespace) -> Tuple[List[List[float]], np.ndarray]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    if args.color == "blue":
        mask_bool = (
            (hue >= args.blue_hue_min)
            & (hue <= args.blue_hue_max)
            & (sat >= args.blue_sat_min)
            & (val >= args.value_thresh)
        )
    else:
        mask_bool = (val >= args.value_thresh) & (sat <= args.sat_thresh)
    mask = mask_bool.astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    if args.dilate > 0:
        k = max(1, int(args.dilate))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    height, width = frame_bgr.shape[:2]
    boxes = component_boxes(mask, min_area=args.min_area, min_side=args.min_side)
    boxes = merge_overlapping_boxes(boxes)
    boxes = [expand_box(box, width, height, args.margin) for box in boxes]
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    boxes = boxes[: args.max_boxes]
    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes, mask


def read_matched_frame(cap: cv2.VideoCapture, target_index: int, stride: int) -> Optional[np.ndarray]:
    frame = None
    for _ in range(max(1, stride)):
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


def debug_write(debug_dir: str, image_path: str, frame_bgr: np.ndarray, mask: np.ndarray, boxes: List[List[float]]) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    overlay = frame_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_mask.png"), mask)
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_boxes.jpg"), overlay)


def main() -> None:
    args = parse_args()
    if bool(args.video) == bool(args.overlay):
        raise ValueError("Pass exactly one of --video or --overlay")
    image_paths = collect_images(args.input)
    if not image_paths:
        raise ValueError(f"No images found: {args.input}")

    cap = None
    overlay_paths = collect_overlays(args.overlay)
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {args.video}")
        for _ in range(max(0, args.frame_offset)):
            ok, _ = cap.read()
            if not ok:
                raise ValueError(f"Video ended while applying frame_offset={args.frame_offset}")
    elif len(overlay_paths) != len(image_paths):
        raise ValueError(f"Overlay count ({len(overlay_paths)}) does not match input image count ({len(image_paths)})")

    box_map = {}
    matched = 0
    missing = 0
    for idx, image_path in enumerate(image_paths):
        if cap is not None:
            frame = read_matched_frame(cap, idx, args.frame_stride)
            if frame is None:
                missing += 1
                break
        else:
            frame = cv2.imread(overlay_paths[idx], cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to read overlay image: {overlay_paths[idx]}")
        boxes, mask = extract_boxes(frame, args)
        base = os.path.basename(image_path)
        stem = os.path.splitext(base)[0]
        box_map[base] = boxes
        box_map[stem] = boxes
        if args.debug_dir:
            debug_write(args.debug_dir, image_path, frame, mask, boxes)
        matched += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(box_map, f, indent=2)

    print(f"[INFO] matched_images={matched} missing_video_frames={missing}")
    print(f"[INFO] wrote: {args.output_json}")


if __name__ == "__main__":
    main()
