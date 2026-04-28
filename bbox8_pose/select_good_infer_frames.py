import argparse
import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select good bbox8 infer visualization frames by GT corner error.")
    parser.add_argument("--infer_dir", required=True, help="Directory containing infer *.json and *_vis.jpg files.")
    parser.add_argument("--labels_root", required=True, help="Directory containing annotations.jsonl.")
    parser.add_argument("--output_dir", required=True, help="Directory to write selected sequential *_vis.jpg frames.")
    parser.add_argument("--top_k", type=int, default=120, help="Number of best frames to select.")
    parser.add_argument("--max_mean_l2", type=float, default=None, help="Optional maximum mean corner L2 threshold in pixels.")
    parser.add_argument("--min_valid_corners", type=int, default=8, help="Minimum valid GT corners required for scoring.")
    parser.add_argument("--video_path", default=None, help="Optional output mp4 path. If set, ffmpeg is invoked.")
    parser.add_argument("--fps", type=int, default=6, help="FPS for optional video generation.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable.")
    return parser.parse_args()


def norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def path_suffix_key(path: str, parts: int = 4) -> str:
    chunks = os.path.normpath(path).split(os.sep)
    return os.path.join(*chunks[-parts:]) if len(chunks) >= parts else os.path.normpath(path)


def load_annotations(labels_root: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    ann_path = os.path.join(labels_root, "annotations.jsonl")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(ann_path)

    by_abs: Dict[str, Dict] = {}
    by_suffix: Dict[str, Dict] = {}
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rgb_path = rec.get("rgb_path")
            if not rgb_path:
                continue
            by_abs[norm_path(rgb_path)] = rec
            by_suffix[path_suffix_key(rgb_path)] = rec
    return by_abs, by_suffix


def find_annotation(image_path: str, by_abs: Dict[str, Dict], by_suffix: Dict[str, Dict]) -> Optional[Dict]:
    key = norm_path(image_path)
    if key in by_abs:
        return by_abs[key]
    suffix = path_suffix_key(image_path)
    return by_suffix.get(suffix)


def score_prediction(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, min_valid: int) -> Optional[float]:
    valid = valid.astype(bool)
    if int(valid.sum()) < min_valid:
        return None
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    return float(np.linalg.norm(pred_valid - gt_valid, axis=1).mean())


def load_pred_instances(result: Dict) -> List[Dict]:
    if "instances" in result and isinstance(result["instances"], list):
        return result["instances"]
    if "corners_2d" in result:
        return [{"corners_2d": result["corners_2d"]}]
    return []


def vis_path_for_json(infer_dir: str, stem: str) -> Optional[str]:
    for ext in IMAGE_EXTS:
        path = os.path.join(infer_dir, f"{stem}_vis{ext}")
        if os.path.exists(path):
            return path
    return None


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    by_abs, by_suffix = load_annotations(args.labels_root)
    scored = []
    skipped = 0

    for name in sorted(os.listdir(args.infer_dir)):
        if not name.endswith(".json"):
            continue
        json_path = os.path.join(args.infer_dir, name)
        stem = os.path.splitext(name)[0]
        with open(json_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        image_path = result.get("image_path")
        if not image_path:
            skipped += 1
            continue
        ann = find_annotation(image_path, by_abs, by_suffix)
        if ann is None:
            skipped += 1
            continue

        gt = np.asarray(ann["corners_2d"], dtype=np.float32)
        valid = np.asarray(ann.get("corner_valid_mask", [1] * len(gt)), dtype=np.float32)
        best_score = None
        best_inst = -1
        for inst_idx, inst in enumerate(load_pred_instances(result)):
            if "corners_2d" not in inst:
                continue
            pred = np.asarray(inst["corners_2d"], dtype=np.float32)
            if pred.shape != gt.shape:
                continue
            score = score_prediction(pred, gt, valid, args.min_valid_corners)
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_inst = inst_idx

        if best_score is None:
            skipped += 1
            continue
        if args.max_mean_l2 is not None and best_score > args.max_mean_l2:
            continue

        vis_path = vis_path_for_json(args.infer_dir, stem)
        if vis_path is None:
            skipped += 1
            continue
        scored.append(
            {
                "stem": stem,
                "score_mean_l2": best_score,
                "instance_index": best_inst,
                "image_path": image_path,
                "json_path": json_path,
                "vis_path": vis_path,
            }
        )

    scored.sort(key=lambda item: item["score_mean_l2"])
    selected = scored[: args.top_k]

    manifest_path = os.path.join(args.output_dir, "selected_manifest.json")
    for out_idx, item in enumerate(selected):
        dst = os.path.join(args.output_dir, f"{out_idx:06d}_vis.jpg")
        shutil.copy2(item["vis_path"], dst)
        item["selected_frame"] = dst

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "infer_dir": os.path.abspath(args.infer_dir),
                "labels_root": os.path.abspath(args.labels_root),
                "num_candidates": len(scored),
                "num_selected": len(selected),
                "num_skipped": skipped,
                "mean_l2_min": selected[0]["score_mean_l2"] if selected else None,
                "mean_l2_max": selected[-1]["score_mean_l2"] if selected else None,
                "frames": selected,
            },
            f,
            indent=2,
        )

    print(f"[INFO] candidates={len(scored)} selected={len(selected)} skipped={skipped}")
    if selected:
        print(f"[INFO] selected mean_l2 range: {selected[0]['score_mean_l2']:.3f} - {selected[-1]['score_mean_l2']:.3f}")
    print(f"[INFO] manifest: {manifest_path}")

    if args.video_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
        cmd = [
            args.ffmpeg,
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            os.path.join(args.output_dir, "%06d_vis.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            args.video_path,
        ]
        print("[INFO] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
