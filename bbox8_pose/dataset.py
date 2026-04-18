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


@dataclass
class CropConfig:
    mode: str = "full"
    padding: float = 0.25
    square: bool = True
    min_size: int = 32


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


def compute_crop_box(
    orig_size: Tuple[int, int],
    bbox_xywh: Optional[List[float]],
    crop_cfg: CropConfig,
) -> Tuple[int, int, int, int]:
    orig_w, orig_h = orig_size
    if crop_cfg.mode == "full" or bbox_xywh is None:
        return (0, 0, orig_w, orig_h)

    x, y, w, h = [float(v) for v in bbox_xywh]
    if w <= 1 or h <= 1 or x < 0 or y < 0:
        return (0, 0, orig_w, orig_h)

    cx = x + 0.5 * w
    cy = y + 0.5 * h
    crop_w = w * (1.0 + crop_cfg.padding * 2.0)
    crop_h = h * (1.0 + crop_cfg.padding * 2.0)
    if crop_cfg.square:
        side = max(crop_w, crop_h, float(crop_cfg.min_size))
        crop_w = side
        crop_h = side
    else:
        crop_w = max(crop_w, float(crop_cfg.min_size))
        crop_h = max(crop_h, float(crop_cfg.min_size))

    x1 = int(round(cx - 0.5 * crop_w))
    y1 = int(round(cy - 0.5 * crop_h))
    x2 = int(round(cx + 0.5 * crop_w))
    y2 = int(round(cy + 0.5 * crop_h))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(orig_w, x2)
    y2 = min(orig_h, y2)

    if x2 <= x1:
        x2 = min(orig_w, x1 + crop_cfg.min_size)
        x1 = max(0, x2 - crop_cfg.min_size)
    if y2 <= y1:
        y2 = min(orig_h, y1 + crop_cfg.min_size)
        y1 = max(0, y2 - crop_cfg.min_size)
    return (x1, y1, x2, y2)


def transform_points_to_crop(
    corners_xy: torch.Tensor,
    crop_box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> torch.Tensor:
    x1, y1, x2, y2 = crop_box
    crop_w = max(1.0, float(x2 - x1))
    crop_h = max(1.0, float(y2 - y1))
    corners_crop = corners_xy.clone()
    corners_crop[:, 0] = (corners_crop[:, 0] - float(x1)) * (image_size[0] / crop_w)
    corners_crop[:, 1] = (corners_crop[:, 1] - float(y1)) * (image_size[1] / crop_h)
    return corners_crop


def transform_points_from_crop(
    corners_xy: torch.Tensor,
    crop_box: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    if crop_box.ndim == 1:
        crop_box = crop_box.unsqueeze(0)
        corners_xy = corners_xy.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    x1 = crop_box[:, 0].unsqueeze(-1)
    y1 = crop_box[:, 1].unsqueeze(-1)
    crop_w = (crop_box[:, 2] - crop_box[:, 0]).clamp(min=1.0).unsqueeze(-1)
    crop_h = (crop_box[:, 3] - crop_box[:, 1]).clamp(min=1.0).unsqueeze(-1)

    coords = corners_xy.clone()
    coords[..., 0] = coords[..., 0] * (crop_w / float(image_size[0])) + x1
    coords[..., 1] = coords[..., 1] * (crop_h / float(image_size[1])) + y1
    return coords[0] if squeeze else coords


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
        crop_mode: str = "full",
        crop_padding: float = 0.25,
        crop_square: bool = True,
        seed: int = 42,
    ) -> None:
        self.labels_root = os.path.abspath(labels_root)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.use_soft_mask_filter = use_soft_mask_filter
        self.augment = augment
        self.rng = random.Random(seed)
        self.augment_cfg = AugmentConfig()
        self.crop_cfg = CropConfig(mode=crop_mode, padding=crop_padding, square=crop_square)

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
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec = self.records[index]
        image = cv2.imread(rec["rgb_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {rec['rgb_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        if self.augment:
            image = self._apply_augmentation(image)

        bbox_xywh = rec.get("bbox_visib") or rec.get("bbox_obj")
        crop_box = compute_crop_box((orig_w, orig_h), bbox_xywh, self.crop_cfg)
        x1, y1, x2, y2 = crop_box
        image_crop = image[y1:y2, x1:x2]
        if image_crop.size == 0:
            image_crop = image
            crop_box = (0, 0, orig_w, orig_h)

        resized = cv2.resize(image_crop, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        corners = torch.tensor(rec["corners_2d"], dtype=torch.float32)
        valid_mask = torch.tensor(rec["corner_valid_mask"], dtype=torch.float32)
        corners_resized = transform_points_to_crop(corners, crop_box, self.image_size)

        if self.use_soft_mask_filter and rec.get("visib_fract", 1.0) <= 0:
            valid_mask.zero_()

        heatmaps = generate_corner_heatmaps(
            corners_xy=corners_resized,
            valid_mask=valid_mask,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
        )

        sample = {
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
        return sample


def collate_bbox8(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    collated: Dict[str, torch.Tensor] = {}
    tensor_keys = ["image", "heatmaps", "corners_xy", "corners_xy_orig", "valid_mask", "camera_K", "corners_3d", "orig_size", "crop_box"]
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)
    collated["sample_id"] = [item["sample_id"] for item in batch]
    collated["rgb_path"] = [item["rgb_path"] for item in batch]
    return collated
