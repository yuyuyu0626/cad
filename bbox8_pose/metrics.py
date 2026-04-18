from typing import Dict

import torch


def corner_l2_error(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    dist = torch.norm(pred_xy - gt_xy, dim=-1)
    valid = valid_mask > 0
    if valid.any():
        mean_l2 = dist[valid].mean().item()
    else:
        mean_l2 = 0.0
    return {"mean_corner_l2": mean_l2}
