import argparse
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter bbox8 infer frames by simple geometry and temporal consistency.")
    parser.add_argument("--infer_dir", required=True, help="Directory containing infer *.json and *_vis.jpg files.")
    parser.add_argument("--output_dir", required=True, help="Directory to write filtered *_vis.jpg frames.")
    parser.add_argument("--expected_instances", type=int, default=2, help="Required number of predicted instances.")
    parser.add_argument("--max_center_jump", type=float, default=220.0, help="Maximum per-instance center jump from previous kept frame.")
    parser.add_argument("--min_area", type=float, default=200.0, help="Minimum corner bbox area in pixels.")
    parser.add_argument("--max_area", type=float, default=250000.0, help="Maximum corner bbox area in pixels.")
    parser.add_argument(
        "--mode",
        choices=["compact", "hold_previous"],
        default="compact",
        help="compact drops bad frames; hold_previous keeps length by repeating the previous good visualization.",
    )
    parser.add_argument("--copy_json", action="store_true", help="Also copy kept JSON files.")
    return parser.parse_args()


def load_instances(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    instances = obj.get("instances")
    if isinstance(instances, list):
        return [inst for inst in instances if isinstance(inst, dict) and "corners_2d" in inst]
    if "corners_2d" in obj:
        return [{"corners_2d": obj["corners_2d"]}]
    return []


def vis_path_for_json(infer_dir: str, stem: str) -> Optional[str]:
    for ext in IMAGE_EXTS:
        path = os.path.join(infer_dir, f"{stem}_vis{ext}")
        if os.path.exists(path):
            return path
    return None


def corner_box(inst: Dict) -> Optional[Tuple[np.ndarray, float]]:
    corners = np.asarray(inst.get("corners_2d"), dtype=np.float32)
    if corners.ndim != 2 or corners.shape[0] < 4 or corners.shape[1] < 2:
        return None
    finite = np.isfinite(corners[:, :2]).all(axis=1)
    if int(finite.sum()) < 4:
        return None
    pts = corners[finite, :2]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    area = float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
    return center, area


def frame_geometry(instances: List[Dict]) -> Optional[List[Tuple[np.ndarray, float]]]:
    geoms = []
    for inst in instances:
        geom = corner_box(inst)
        if geom is None:
            return None
        geoms.append(geom)
    geoms.sort(key=lambda item: float(item[0][0]))
    return geoms


def valid_frame(
    geoms: List[Tuple[np.ndarray, float]],
    prev_geoms: Optional[List[Tuple[np.ndarray, float]]],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    if len(geoms) != args.expected_instances:
        return False, f"instance_count={len(geoms)}"
    for _, area in geoms:
        if area < args.min_area:
            return False, f"area_too_small={area:.1f}"
        if area > args.max_area:
            return False, f"area_too_large={area:.1f}"
    if prev_geoms is not None and len(prev_geoms) == len(geoms):
        for idx, ((center, _), (prev_center, _)) in enumerate(zip(geoms, prev_geoms)):
            jump = float(np.linalg.norm(center - prev_center))
            if jump > args.max_center_jump:
                return False, f"center_jump_inst{idx}={jump:.1f}"
    return True, "ok"


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_names = sorted(name for name in os.listdir(args.infer_dir) if name.endswith(".json"))
    prev_good_geoms = None
    prev_good_vis = None
    kept = []
    rejected = []
    out_idx = 0

    for name in json_names:
        stem = os.path.splitext(name)[0]
        json_path = os.path.join(args.infer_dir, name)
        vis_path = vis_path_for_json(args.infer_dir, stem)
        instances = load_instances(json_path)
        geoms = frame_geometry(instances)
        if vis_path is None:
            ok, reason = False, "missing_vis"
        elif geoms is None:
            ok, reason = False, "bad_corners"
        else:
            ok, reason = valid_frame(geoms, prev_good_geoms, args)

        if ok:
            dst_vis = os.path.join(args.output_dir, f"{out_idx:06d}_vis.jpg")
            shutil.copy2(vis_path, dst_vis)
            if args.copy_json:
                shutil.copy2(json_path, os.path.join(args.output_dir, f"{out_idx:06d}.json"))
            kept.append({"stem": stem, "output_index": out_idx, "reason": reason})
            prev_good_geoms = geoms
            prev_good_vis = vis_path
            out_idx += 1
        else:
            rejected.append({"stem": stem, "reason": reason})
            if args.mode == "hold_previous" and prev_good_vis is not None:
                dst_vis = os.path.join(args.output_dir, f"{out_idx:06d}_vis.jpg")
                shutil.copy2(prev_good_vis, dst_vis)
                kept.append({"stem": stem, "output_index": out_idx, "reason": f"held_previous:{reason}"})
                out_idx += 1

    manifest = {
        "infer_dir": os.path.abspath(args.infer_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "mode": args.mode,
        "num_input": len(json_names),
        "num_output": out_idx,
        "num_rejected": len(rejected),
        "kept": kept,
        "rejected": rejected,
    }
    manifest_path = os.path.join(args.output_dir, "filter_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] input={len(json_names)} output={out_idx} rejected={len(rejected)}")
    print(f"[INFO] manifest: {manifest_path}")
    if rejected:
        print("[INFO] first rejected:")
        for item in rejected[:20]:
            print(f"  {item['stem']}: {item['reason']}")


if __name__ == "__main__":
    main()
