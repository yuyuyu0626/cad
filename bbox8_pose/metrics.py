from typing import Dict

import torch


def corner_l2_error(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    # 这里计算的是预测角点坐标与真实角点坐标之间的平均 L2 距离，valid_mask 用于过滤掉不可见或无效的角点。
    # L2距离计算方法是: 对每个角点，计算预测坐标与真实坐标之间的欧氏距离（即 sqrt((x_pred - x_gt)^2 + (y_pred - y_gt)^2)），然后对所有有效角点的距离取平均。
    # norm函数计算的是欧氏距离，dim=-1 表示在最后一个维度上计算（即对每个角点的 x 和 y 坐标进行计算）。valid_mask > 0 用于筛选出有效的角点，最后返回一个字典，包含平均 L2 距离的值。
    # 这里的计算结果是不是跟MSE损失一样的？不完全一样，MSE是平均平方误差，而这里是平均L2距离（即平均欧氏距离）。L2距离是MSE的平方根，因此它们在数值上会有区别，尤其是在误差较大的情况下。
    dist = torch.norm(pred_xy - gt_xy, dim=-1)
    valid = valid_mask > 0
    if valid.any():
        mean_l2 = dist[valid].mean().item()
    else:
        mean_l2 = 0.0
    return {"mean_corner_l2": mean_l2}
