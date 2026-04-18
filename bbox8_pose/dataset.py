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
    ) -> None:
        self.labels_root = os.path.abspath(labels_root)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.use_soft_mask_filter = use_soft_mask_filter
        self.augment = augment
        self.rng = random.Random(seed)
        self.augment_cfg = AugmentConfig()

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

        resized = cv2.resize(image, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        corners = torch.tensor(rec["corners_2d"], dtype=torch.float32)
        valid_mask = torch.tensor(rec["corner_valid_mask"], dtype=torch.float32)
        scale_x = self.image_size[0] / float(orig_w)
        scale_y = self.image_size[1] / float(orig_h)
        corners_resized = corners.clone()
        corners_resized[:, 0] *= scale_x
        corners_resized[:, 1] *= scale_y

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
        }
        return sample


def collate_bbox8(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    collated: Dict[str, torch.Tensor] = {}
    tensor_keys = ["image", "heatmaps", "corners_xy", "corners_xy_orig", "valid_mask", "camera_K", "corners_3d", "orig_size"]
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)
    collated["sample_id"] = [item["sample_id"] for item in batch]
    collated["rgb_path"] = [item["rgb_path"] for item in batch]
    return collated
