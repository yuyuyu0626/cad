import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap import generate_corner_heatmaps


@dataclass
class AugmentConfig:
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    blur_prob: float = 0.1


def _load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_split_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _clip_box_xyxy(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x2))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y2))))
    return x1, y1, x2, y2


def corners_to_crop_box(
    corners: np.ndarray,
    valid_mask: np.ndarray,
    width: int,
    height: int,
    margin: float,
    jitter: float = 0.0,
    rng: Optional[random.Random] = None,
) -> Tuple[int, int, int, int]:
    valid = valid_mask > 0
    if int(valid.sum()) < 2:
        valid = np.ones((corners.shape[0],), dtype=bool)

    pts = corners[valid]
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = float(margin) * max(bw, bh)

    if jitter > 0 and rng is not None:
        low = max(0.0, 1.0 - float(jitter))
        factors = [rng.uniform(low, 1.0 + float(jitter)) for _ in range(4)]
    else:
        factors = [1.0, 1.0, 1.0, 1.0]

    return _clip_box_xyxy(
        (
            x1 - pad * factors[0],
            y1 - pad * factors[1],
            x2 + pad * factors[2],
            y2 + pad * factors[3],
        ),
        width,
        height,
    )


def transform_corners_to_crop(
    corners: torch.Tensor,
    crop_box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> torch.Tensor:
    x1, y1, x2, y2 = crop_box
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    out = corners.clone()
    out[:, 0] = (out[:, 0] - float(x1)) * (float(image_size[0]) / float(crop_w))
    out[:, 1] = (out[:, 1] - float(y1)) * (float(image_size[1]) / float(crop_h))
    return out


def transform_corners_from_crop(
    corners: np.ndarray,
    crop_box: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in crop_box]
    crop_w = max(1.0, x2 - x1)
    crop_h = max(1.0, y2 - y1)
    out = corners.copy()
    out[:, 0] = x1 + out[:, 0] * (crop_w / float(image_size[0]))
    out[:, 1] = y1 + out[:, 1] * (crop_h / float(image_size[1]))
    return out


class BBox8PoseDataset(Dataset):
    def __init__(
        self,
        labels_root: str,
        split: str,
        image_size: Tuple[int, int] = (256, 256),
        heatmap_size: Tuple[int, int] = (64, 64),
        sigma: float = 2.5,
        use_soft_mask_filter: bool = True,
        augment: bool = False,
        seed: int = 42,
        crop_to_bbox: bool = False,
        crop_margin: float = 0.15,
        crop_jitter: float = 0.0,
    ) -> None:
        self.labels_root = os.path.abspath(labels_root)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.use_soft_mask_filter = use_soft_mask_filter
        self.augment = augment
        self.rng = random.Random(seed)
        self.augment_cfg = AugmentConfig()
        self.crop_to_bbox = bool(crop_to_bbox)
        self.crop_margin = float(crop_margin)
        self.crop_jitter = float(crop_jitter)

        ann_path = os.path.join(self.labels_root, "annotations.jsonl")
        split_path = os.path.join(self.labels_root, f"{split}.txt")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(ann_path)
        if not os.path.exists(split_path):
            raise FileNotFoundError(split_path)

        all_records = _load_jsonl(ann_path)
        split_ids = set(_load_split_ids(split_path))
        self.records = [rec for rec in all_records if rec["sample_id"] in split_ids]
        self.records.sort(key=lambda x: x["sample_id"])
        if not self.records:
            raise ValueError(f"No samples found for split={split} in {self.labels_root}")

        bbox_meta_path = os.path.join(self.labels_root, "object_bbox_3d.json")
        self.object_bbox_3d = None
        if os.path.exists(bbox_meta_path):
            with open(bbox_meta_path, "r", encoding="utf-8") as f:
                self.object_bbox_3d = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if self.rng.random() < 0.9:
            alpha = 1.0 + self.rng.uniform(-self.augment_cfg.contrast, self.augment_cfg.contrast)
            beta = 255.0 * self.rng.uniform(-self.augment_cfg.brightness, self.augment_cfg.brightness)
            image = image * alpha + beta
        if self.rng.random() < self.augment_cfg.blur_prob:
            image = cv2.GaussianBlur(image, (5, 5), 0)
        return np.clip(image, 0, 255).astype(np.uint8)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec = self.records[index]
        image = cv2.imread(rec["rgb_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {rec['rgb_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        corners = torch.tensor(rec["corners_2d"], dtype=torch.float32)
        valid_mask = torch.tensor(rec["corner_valid_mask"], dtype=torch.float32)
        if self.use_soft_mask_filter and rec.get("visib_fract", 1.0) <= 0:
            valid_mask.zero_()

        crop_box = (0, 0, orig_w, orig_h)
        if self.crop_to_bbox:
            crop_box = corners_to_crop_box(
                corners.numpy(),
                valid_mask.numpy(),
                width=orig_w,
                height=orig_h,
                margin=self.crop_margin,
                jitter=self.crop_jitter if self.augment else 0.0,
                rng=self.rng,
            )
        x1, y1, x2, y2 = crop_box
        crop = image[y1:y2, x1:x2]

        if self.augment:
            crop = self._apply_augmentation(crop)
        resized = cv2.resize(crop, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        corners_resized = transform_corners_to_crop(corners, crop_box, self.image_size)
        heatmaps = generate_corner_heatmaps(
            corners_xy=corners_resized,
            valid_mask=valid_mask,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
        )

        return {
            "image": image_tensor,
            "heatmaps": heatmaps,
            "corners_xy": corners_resized,
            "corners_xy_orig": corners,
            "valid_mask": valid_mask,
            "camera_K": torch.tensor(rec["cam_K"], dtype=torch.float32).view(3, 3),
            "corners_3d": torch.tensor(rec["corners_3d_object"], dtype=torch.float32),
            "sample_id": rec["sample_id"],
            "rgb_path": rec["rgb_path"],
            "orig_size": torch.tensor([orig_w, orig_h], dtype=torch.float32),
            "crop_box": torch.tensor(crop_box, dtype=torch.float32),
        }


def collate_bbox8(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    collated: Dict[str, torch.Tensor] = {}
    tensor_keys = [
        "image",
        "heatmaps",
        "corners_xy",
        "corners_xy_orig",
        "valid_mask",
        "camera_K",
        "corners_3d",
        "orig_size",
        "crop_box",
    ]
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)
    collated["sample_id"] = [item["sample_id"] for item in batch]
    collated["rgb_path"] = [item["rgb_path"] for item in batch]
    return collated
