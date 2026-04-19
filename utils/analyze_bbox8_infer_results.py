import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_camera_matrix(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        if "cam_K" in obj:
            return np.asarray(obj["cam_K"], dtype=np.float32).reshape(3, 3)
        if all(k in obj for k in ("fx", "fy", "cx", "cy")):
            return np.asarray(
                [
                    [obj["fx"], 0.0, obj["cx"]],
                    [0.0, obj["fy"], obj["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
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


def load_corners_3d(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return np.asarray(obj["corners_3d_object"], dtype=np.float32)


def rotation_delta_deg(R_prev: np.ndarray, R_curr: np.ndarray) -> float:
    rel = R_prev.T @ R_curr
    trace = float(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(trace))


def project_points(corners_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    cam = (R @ corners_3d.T).T + t.reshape(1, 3)
    proj = (K @ cam.T).T
    proj_xy = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)
    return proj_xy


def summarize_numeric(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def frame_sort_key(path: str) -> Tuple[int, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (int(stem), stem)
    except ValueError:
        digits = "".join(ch for ch in stem if ch.isdigit())
        return (int(digits) if digits else 0, stem)


def analyze_frame(
    item: Dict,
    camera_K: Optional[np.ndarray],
    corners_3d: Optional[np.ndarray],
) -> Dict:
    image_path = item["image_path"]
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = image_bgr.shape[:2]

    corners = np.asarray(item["corners_2d"], dtype=np.float32)
    bbox_min = corners.min(axis=0)
    bbox_max = corners.max(axis=0)
    bbox_w = float(bbox_max[0] - bbox_min[0])
    bbox_h = float(bbox_max[1] - bbox_min[1])
    bbox_area = bbox_w * bbox_h
    center = corners.mean(axis=0)
    in_frame = (
        (corners[:, 0] >= 0.0)
        & (corners[:, 0] < float(w))
        & (corners[:, 1] >= 0.0)
        & (corners[:, 1] < float(h))
    )
    in_frame_ratio = float(in_frame.mean())

    edge_lengths = [float(np.linalg.norm(corners[i] - corners[j])) for i, j in CUBOID_EDGES]
    result = {
        "frame_stem": os.path.splitext(os.path.basename(image_path))[0],
        "image_path": image_path,
        "image_width": int(w),
        "image_height": int(h),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "bbox_x": float(bbox_min[0]),
        "bbox_y": float(bbox_min[1]),
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_area": bbox_area,
        "bbox_area_ratio": float(bbox_area / max(1.0, float(w * h))),
        "in_frame_ratio": in_frame_ratio,
        "mean_edge_len_2d": float(np.mean(edge_lengths)),
        "std_edge_len_2d": float(np.std(edge_lengths)),
    }

    if "cam_t_m2c" in item:
        t = np.asarray(item["cam_t_m2c"], dtype=np.float32).reshape(3)
        result["tx"] = float(t[0])
        result["ty"] = float(t[1])
        result["tz"] = float(t[2])
        result["t_norm"] = float(np.linalg.norm(t))
    if "cam_R_m2c" in item:
        R = np.asarray(item["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
        result["rot_det"] = float(np.linalg.det(R))
        if camera_K is not None and corners_3d is not None and "cam_t_m2c" in item:
            t = np.asarray(item["cam_t_m2c"], dtype=np.float32).reshape(3)
            reproj = project_points(corners_3d, R, t, camera_K)
            reproj_err = np.linalg.norm(reproj - corners, axis=1)
            result["pose_reproj_mean"] = float(reproj_err.mean())
            result["pose_reproj_max"] = float(reproj_err.max())
    return result


def compute_temporal_metrics(rows: List[Dict]) -> List[Dict]:
    temporal = []
    for prev, curr in zip(rows[:-1], rows[1:]):
        entry = {
            "frame_stem": curr["frame_stem"],
            "prev_frame_stem": prev["frame_stem"],
            "center_disp": float(
                math.hypot(curr["center_x"] - prev["center_x"], curr["center_y"] - prev["center_y"])
            ),
            "bbox_area_ratio_change": float(
                abs(curr["bbox_area"] - prev["bbox_area"]) / max(1.0, prev["bbox_area"])
            ),
            "mean_edge_len_change": float(abs(curr["mean_edge_len_2d"] - prev["mean_edge_len_2d"])),
        }
        if all(k in curr for k in ("tx", "ty", "tz")) and all(k in prev for k in ("tx", "ty", "tz")):
            t_prev = np.array([prev["tx"], prev["ty"], prev["tz"]], dtype=np.float32)
            t_curr = np.array([curr["tx"], curr["ty"], curr["tz"]], dtype=np.float32)
            entry["trans_delta"] = float(np.linalg.norm(t_curr - t_prev))
            entry["tz_delta"] = float(abs(curr["tz"] - prev["tz"]))
        temporal.append(entry)
    return temporal


def attach_rotation_deltas(rows: List[Dict], json_items: List[Dict], temporal: List[Dict]) -> None:
    rot_map = {}
    for item in json_items:
        if "cam_R_m2c" in item:
            rot_map[os.path.splitext(os.path.basename(item["image_path"]))[0]] = np.asarray(item["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    for entry in temporal:
        prev_key = entry["prev_frame_stem"]
        curr_key = entry["frame_stem"]
        if prev_key in rot_map and curr_key in rot_map:
            entry["rot_delta_deg"] = float(rotation_delta_deg(rot_map[prev_key], rot_map[curr_key]))


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: Dict) -> str:
    lines = []
    lines.append("BBox8 Inference Analysis Report")
    lines.append("")
    lines.append(f"Frames analyzed: {summary['num_frames']}")
    lines.append(f"In-frame corner ratio mean: {summary['frame_metrics']['in_frame_ratio']['mean']:.4f}")
    lines.append(f"BBox area ratio mean: {summary['frame_metrics']['bbox_area_ratio']['mean']:.6f}")
    lines.append(f"2D edge length mean: {summary['frame_metrics']['mean_edge_len_2d']['mean']:.3f}")
    if summary["frame_metrics"].get("tz", {}).get("mean") is not None:
        lines.append(f"Pose tz mean: {summary['frame_metrics']['tz']['mean']:.3f}")
    if summary["frame_metrics"].get("pose_reproj_mean", {}).get("mean") is not None:
        lines.append(f"Pose reprojection error mean: {summary['frame_metrics']['pose_reproj_mean']['mean']:.3f}")
    lines.append("")
    lines.append("Temporal stability")
    lines.append(f"Center displacement mean: {summary['temporal_metrics']['center_disp']['mean']:.3f}")
    lines.append(f"BBox area ratio change mean: {summary['temporal_metrics']['bbox_area_ratio_change']['mean']:.6f}")
    if summary["temporal_metrics"].get("trans_delta", {}).get("mean") is not None:
        lines.append(f"Translation delta mean: {summary['temporal_metrics']['trans_delta']['mean']:.3f}")
    if summary["temporal_metrics"].get("rot_delta_deg", {}).get("mean") is not None:
        lines.append(f"Rotation delta mean (deg): {summary['temporal_metrics']['rot_delta_deg']['mean']:.3f}")

    suspicious = summary.get("suspicious_frames", [])
    lines.append("")
    lines.append(f"Suspicious frames flagged: {len(suspicious)}")
    for item in suspicious[:20]:
        lines.append(
            f"- {item['frame_stem']}: in_frame_ratio={item['in_frame_ratio']:.3f}, "
            f"bbox_area_ratio={item['bbox_area_ratio']:.6f}, mean_edge_len_2d={item['mean_edge_len_2d']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bbox8 inference outputs without GT metrics.")
    parser.add_argument("--infer_dir", required=True, help="Directory containing infer JSONs and *_vis.jpg files.")
    parser.add_argument("--output_dir", default=None, help="Directory to write analysis outputs. Defaults to infer_dir/analysis")
    parser.add_argument("--camera_json", default=None, help="Optional camera.json for pose reprojection checks.")
    parser.add_argument("--bbox3d_json", default=None, help="Optional object_bbox_3d.json for pose reprojection checks.")
    args = parser.parse_args()

    infer_dir = os.path.abspath(args.infer_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(infer_dir, "analysis"))
    ensure_dir(output_dir)

    json_paths = []
    for name in os.listdir(infer_dir):
        if name.lower().endswith(".json"):
            json_paths.append(os.path.join(infer_dir, name))
    json_paths.sort(key=frame_sort_key)
    if not json_paths:
        raise ValueError(f"No JSON files found in {infer_dir}")

    camera_K = load_camera_matrix(args.camera_json) if args.camera_json else None
    corners_3d = load_corners_3d(args.bbox3d_json) if args.bbox3d_json else None

    json_items = []
    frame_rows = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        json_items.append(item)
        frame_rows.append(analyze_frame(item, camera_K, corners_3d))

    temporal_rows = compute_temporal_metrics(frame_rows)
    attach_rotation_deltas(frame_rows, json_items, temporal_rows)

    frame_summary_keys = [
        "in_frame_ratio",
        "bbox_area_ratio",
        "mean_edge_len_2d",
        "std_edge_len_2d",
        "tx",
        "ty",
        "tz",
        "t_norm",
        "pose_reproj_mean",
        "pose_reproj_max",
    ]
    temporal_summary_keys = [
        "center_disp",
        "bbox_area_ratio_change",
        "mean_edge_len_change",
        "trans_delta",
        "tz_delta",
        "rot_delta_deg",
    ]

    frame_metrics = {
        key: summarize_numeric([row[key] for row in frame_rows if key in row])
        for key in frame_summary_keys
    }
    temporal_metrics = {
        key: summarize_numeric([row[key] for row in temporal_rows if key in row])
        for key in temporal_summary_keys
    }

    suspicious_frames = []
    for row in frame_rows:
        bad = (
            row["in_frame_ratio"] < 0.75
            or row["bbox_area_ratio"] < 0.002
            or row["bbox_area_ratio"] > 0.25
        )
        if bad:
            suspicious_frames.append(row)

    summary = {
        "infer_dir": infer_dir,
        "num_frames": len(frame_rows),
        "frame_metrics": frame_metrics,
        "temporal_metrics": temporal_metrics,
        "suspicious_frames": suspicious_frames,
    }

    write_csv(os.path.join(output_dir, "frame_metrics.csv"), frame_rows)
    write_csv(os.path.join(output_dir, "temporal_metrics.csv"), temporal_rows)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(build_report(summary))

    print(f"[OK] analyzed {len(frame_rows)} frames")
    print(f"[OK] frame metrics -> {os.path.join(output_dir, 'frame_metrics.csv')}")
    print(f"[OK] temporal metrics -> {os.path.join(output_dir, 'temporal_metrics.csv')}")
    print(f"[OK] summary -> {os.path.join(output_dir, 'summary.json')}")
    print(f"[OK] report -> {os.path.join(output_dir, 'report.txt')}")


if __name__ == "__main__":
    main()
