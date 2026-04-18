import json
import os
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def draw_corners(image_rgb: np.ndarray, corners_xy: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    canvas = image_rgb.copy()
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]
    for idx, pt in enumerate(corners_xy):
        if valid_mask is not None and valid_mask[idx] <= 0:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(canvas, (x, y), 4, colors[idx], -1)
        cv2.putText(canvas, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[idx], 1, cv2.LINE_AA)
    return canvas


def solve_pnp_from_bbox8(
    corners_3d: np.ndarray,
    corners_2d: np.ndarray,
    camera_K: np.ndarray,
    valid_mask: np.ndarray,
) -> Optional[Dict[str, List[float]]]:
    valid_idx = valid_mask > 0
    if valid_idx.sum() < 4:
        return None

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=corners_3d[valid_idx].astype(np.float32),
        imagePoints=corners_2d[valid_idx].astype(np.float32),
        cameraMatrix=camera_K.astype(np.float32),
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    return {
        "cam_R_m2c": R.reshape(-1).tolist(),
        "cam_t_m2c": tvec.reshape(-1).tolist(),
    }
