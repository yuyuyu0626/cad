import math
from typing import Tuple

import torch


def draw_gaussian(
    heatmap: torch.Tensor,
    center_xy: Tuple[float, float],
    sigma: float,
) -> None:
    radius = max(1, int(math.ceil(3 * sigma)))
    x0, y0 = center_xy
    height, width = heatmap.shape[-2:]

    x_min = max(0, int(math.floor(x0 - radius)))
    x_max = min(width - 1, int(math.ceil(x0 + radius)))
    y_min = max(0, int(math.floor(y0 - radius)))
    y_max = min(height - 1, int(math.ceil(y0 + radius)))

    if x_min > x_max or y_min > y_max:
        return

    ys = torch.arange(y_min, y_max + 1, dtype=heatmap.dtype, device=heatmap.device)
    xs = torch.arange(x_min, x_max + 1, dtype=heatmap.dtype, device=heatmap.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gaussian = torch.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma * sigma))

    patch = heatmap[y_min : y_max + 1, x_min : x_max + 1]
    heatmap[y_min : y_max + 1, x_min : x_max + 1] = torch.maximum(patch, gaussian)


def generate_corner_heatmaps(
    # 根据（缩放后的）2d角点坐标生成热图
    corners_xy: torch.Tensor,
    valid_mask: torch.Tensor,
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    num_keypoints = corners_xy.shape[0]
    heatmaps = torch.zeros((num_keypoints, heatmap_size[0], heatmap_size[1]), dtype=torch.float32)

    scale_x = heatmap_size[1] / float(image_size[0])
    scale_y = heatmap_size[0] / float(image_size[1])

    for idx in range(num_keypoints):
        if valid_mask[idx] <= 0:
            continue
        x = corners_xy[idx, 0].item() * scale_x
        y = corners_xy[idx, 1].item() * scale_y
        draw_gaussian(heatmaps[idx], (x, y), sigma)

    return heatmaps


def decode_heatmaps_argmax(heatmaps: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    # 根据热图解码出2d角点坐标
    batch, channels, hm_h, hm_w = heatmaps.shape
    flat_idx = heatmaps.view(batch, channels, -1).argmax(dim=-1)
    ys = (flat_idx // hm_w).float()
    xs = (flat_idx % hm_w).float()

    scale_x = image_size[0] / float(hm_w)
    scale_y = image_size[1] / float(hm_h)
    coords = torch.stack((xs * scale_x, ys * scale_y), dim=-1)
    return coords


def decode_heatmaps_softargmax(
    heatmaps: torch.Tensor,
    image_size: Tuple[int, int],
    temperature: float = 1.0,
) -> torch.Tensor:
    batch, channels, hm_h, hm_w = heatmaps.shape
    probs = torch.softmax(heatmaps.view(batch, channels, -1) / temperature, dim=-1)

    xs = torch.linspace(0, image_size[0], hm_w, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0, image_size[1], hm_h, device=heatmaps.device, dtype=heatmaps.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    pred_x = (probs * xx).sum(dim=-1)
    pred_y = (probs * yy).sum(dim=-1)
    return torch.stack((pred_x, pred_y), dim=-1)
