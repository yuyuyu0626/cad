import torch
import torch.nn as nn


class WeightedHeatmapMSELoss(nn.Module):
    # 这里指的是对每个通道的MSE进行加权平均，权重由valid_mask提供，表示哪些通道是有效的（即对应的角点是可见的）。最终输出是一个标量，表示加权后的平均MSE损失。
    # MSE指的是均方误差（Mean Squared Error），是回归问题中常用的损失函数，计算预测值与真实值之间的平均平方差。
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
