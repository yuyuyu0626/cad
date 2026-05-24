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


BBOX8_EDGES = [
    (2, 3),
    (3, 0),
    (0, 1),
    (1, 2),
    (6, 7),
    (7, 4),
    (4, 5),
    (5, 6),
    (2, 6),
    (3, 7),
    (0, 4),
    (1, 5),
]


def draw_corners(
    image_rgb: np.ndarray,
    corners_xy: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    draw_edges: bool = True,
) -> np.ndarray:
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
    if draw_edges and len(corners_xy) >= 8:
        for a, b in BBOX8_EDGES:
            if valid_mask is not None and (valid_mask[a] <= 0 or valid_mask[b] <= 0):
                continue
            pa = corners_xy[a]
            pb = corners_xy[b]
            p1 = (int(round(pa[0])), int(round(pa[1])))
            p2 = (int(round(pb[0])), int(round(pb[1])))
            cv2.line(canvas, p1, p2, (255, 80, 20), 4, cv2.LINE_AA)
    for idx, pt in enumerate(corners_xy):
        if valid_mask is not None and valid_mask[idx] <= 0:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(canvas, (x, y), 4, colors[idx], -1)
        cv2.putText(canvas, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[idx], 1, cv2.LINE_AA)
    return canvas


def draw_bbox8_edges(
    image_rgb: np.ndarray,
    corners_xy: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    color: tuple = (255, 80, 20),
    thickness: int = 4,
) -> np.ndarray:
    canvas = image_rgb.copy()
    if len(corners_xy) < 8:
        return canvas
    for a, b in BBOX8_EDGES:
        if valid_mask is not None and (valid_mask[a] <= 0 or valid_mask[b] <= 0):
            continue
        pa = corners_xy[a]
        pb = corners_xy[b]
        p1 = (int(round(pa[0])), int(round(pa[1])))
        p2 = (int(round(pb[0])), int(round(pb[1])))
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)
    return canvas


def project_bbox8_from_pose(
    corners_3d: np.ndarray,
    cam_R_m2c: np.ndarray,
    cam_t_m2c: np.ndarray,
    camera_K: np.ndarray,
) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(cam_R_m2c, dtype=np.float32).reshape(3, 3))
    tvec = np.asarray(cam_t_m2c, dtype=np.float32).reshape(3, 1)
    projected, _ = cv2.projectPoints(
        objectPoints=np.asarray(corners_3d, dtype=np.float32),
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=np.asarray(camera_K, dtype=np.float32).reshape(3, 3),
        distCoeffs=None,
    )
    return projected.reshape(-1, 2)


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
