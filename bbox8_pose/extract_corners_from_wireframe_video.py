import argparse
import json
import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover approximate bbox8 corner candidates from a wireframe visualization video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--debug_dir", default=None)
    parser.add_argument("--max_instances", type=int, default=4)
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--min_box_area", type=float, default=500.0)
    parser.add_argument("--min_side", type=int, default=16)
    parser.add_argument("--bright_thresh", type=int, default=190)
    parser.add_argument("--channel_diff", type=int, default=40)
    parser.add_argument("--dilate", type=int, default=3)
    parser.add_argument("--group_dilate", type=int, default=21)
    parser.add_argument("--cluster_radius", type=float, default=7.0)
    parser.add_argument("--min_corner_count", type=int, default=5)
    return parser.parse_args()


def make_wire_mask(frame_bgr: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    b, g, r = cv2.split(frame_bgr)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    neutral_bright = (
        (r >= args.bright_thresh)
        & (g >= args.bright_thresh)
        & (b >= args.bright_thresh)
        & ((maxc - minc) <= args.channel_diff)
    )

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    pale_green = (hue >= 35) & (hue <= 95) & (sat >= 25) & (val >= args.bright_thresh)

    mask = (neutral_bright | pale_green).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    if args.dilate > 0:
        k = max(1, int(args.dilate))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def component_candidates(mask: np.ndarray, args: argparse.Namespace) -> List[Tuple[int, int, int, int, int]]:
    group_mask = mask
    if args.group_dilate > 0:
        k = max(1, int(args.group_dilate))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        group_mask = cv2.dilate(mask, kernel, iterations=1)
        group_mask = cv2.morphologyEx(group_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    num, _, stats, _ = cv2.connectedComponentsWithStats(group_mask, connectivity=8)
    comps = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if w < args.min_side or h < args.min_side:
            continue
        if float(w * h) < args.min_box_area:
            continue
        # The wireframe is sparse; dense white blobs are often non-overlay highlights.
        fill = float(area) / float(max(1, w * h))
        if fill > 0.75:
            continue
        comps.append((int(x), int(y), int(w), int(h), int(area)))
    comps.sort(key=lambda item: item[2] * item[3], reverse=True)
    return comps[: args.max_instances]


def cluster_points(points: np.ndarray, radius: float) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2)
    remaining = points.astype(np.float32).tolist()
    centers = []
    while remaining:
        seed = np.asarray(remaining.pop(0), dtype=np.float32)
        changed = True
        cluster = [seed]
        while changed:
            changed = False
            keep = []
            center = np.mean(cluster, axis=0)
            for pt in remaining:
                arr = np.asarray(pt, dtype=np.float32)
                if float(np.linalg.norm(arr - center)) <= radius:
                    cluster.append(arr)
                    changed = True
                else:
                    keep.append(pt)
            remaining = keep
        centers.append(np.mean(cluster, axis=0))
    return np.asarray(centers, dtype=np.float32)


def farthest_subset(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points
    center = points.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    selected = [first]
    while len(selected) < count:
        selected_pts = points[selected]
        d = np.linalg.norm(points[:, None, :] - selected_pts[None, :, :], axis=2).min(axis=1)
        d[selected] = -1
        selected.append(int(np.argmax(d)))
    return points[selected]


def order_points_canonical(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.reshape(0, 2)
    pts = points.astype(np.float32)
    if len(pts) < 8:
        pad = np.repeat(pts[-1:], 8 - len(pts), axis=0)
        pts = np.concatenate([pts, pad], axis=0)
    elif len(pts) > 8:
        pts = farthest_subset(pts, 8)

    # Canonical image-space order: upper four clockwise, then lower four clockwise.
    # This is deterministic for training/debugging, but it is not guaranteed to match
    # the original 3D z-face ordering when only a rendered wireframe is available.
    idx_y = np.argsort(pts[:, 1])
    upper = pts[idx_y[:4]]
    lower = pts[idx_y[4:]]

    def clockwise(face: np.ndarray) -> np.ndarray:
        c = face.mean(axis=0)
        ang = np.arctan2(face[:, 1] - c[1], face[:, 0] - c[0])
        ordered = face[np.argsort(ang)]
        start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        return np.roll(ordered, -start, axis=0)

    return np.concatenate([clockwise(upper), clockwise(lower)], axis=0)


def corners_from_component(mask: np.ndarray, box: Sequence[int], args: argparse.Namespace) -> np.ndarray:
    x, y, w, h, _ = box
    pad = 6
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
    crop = mask[y0:y1, x0:x1]

    pts = cv2.goodFeaturesToTrack(crop, maxCorners=40, qualityLevel=0.01, minDistance=3, blockSize=3)
    if pts is None:
        return np.zeros((0, 2), dtype=np.float32)
    pts = pts.reshape(-1, 2)
    pts[:, 0] += float(x0)
    pts[:, 1] += float(y0)
    centers = cluster_points(pts, args.cluster_radius)
    return order_points_canonical(centers)


def draw_debug(frame_bgr: np.ndarray, instances: List[dict]) -> np.ndarray:
    out = frame_bgr.copy()
    colors = [(0, 0, 255), (0, 180, 255), (0, 255, 0), (255, 0, 0)]
    for inst_idx, inst in enumerate(instances):
        color = colors[inst_idx % len(colors)]
        x1, y1, x2, y2 = [int(round(v)) for v in inst["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        for pt_idx, pt in enumerate(inst["corners_2d"]):
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(out, (x, y), 3, color, -1)
            cv2.putText(out, str(pt_idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {args.video}")

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)

    results = {}
    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, args.frame_stride) != 0:
            frame_idx += 1
            continue
        if args.frame_limit is not None and saved_idx >= args.frame_limit:
            break

        mask = make_wire_mask(frame, args)
        comps = component_candidates(mask, args)
        instances = []
        for comp in comps:
            corners = corners_from_component(mask, comp, args)
            if len(corners) < args.min_corner_count:
                continue
            x, y, w, h, area = comp
            instances.append(
                {
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "corners_2d": corners.tolist(),
                    "corner_valid_mask": [1] * 8,
                    "source": "wireframe_video",
                    "component_area": int(area),
                }
            )
        instances.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))

        key = f"{saved_idx + 1:06d}"
        results[key] = {
            "video_path": os.path.abspath(args.video),
            "frame_index": int(frame_idx),
            "instances": instances,
        }

        if args.debug_dir and (saved_idx < 50 or saved_idx % 100 == 0):
            cv2.imwrite(os.path.join(args.debug_dir, f"{key}_mask.png"), mask)
            cv2.imwrite(os.path.join(args.debug_dir, f"{key}_debug.jpg"), draw_debug(frame, instances))

        frame_idx += 1
        saved_idx += 1

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": os.path.abspath(args.video),
                "num_frames": saved_idx,
                "note": "Corners are recovered from rendered wireframe pixels. Point order is deterministic image-space canonical order, not guaranteed original bbox8 channel order.",
                "frames": results,
            },
            f,
            indent=2,
        )
    print(f"[INFO] wrote {args.output_json}")
    print(f"[INFO] frames={saved_idx}")


if __name__ == "__main__":
    main()
