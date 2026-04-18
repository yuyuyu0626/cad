import torch
import torch.nn as nn


class WeightedHeatmapMSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_channel = ((pred_heatmaps - target_heatmaps) ** 2).mean(dim=(-2, -1))
        weights = valid_mask.float()
        denom = weights.sum().clamp_min(1.0)
        return (per_channel * weights).sum() / denom
