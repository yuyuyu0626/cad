import argparse
import os
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import BBox8PoseDataset, collate_bbox8
from .heatmap import decode_heatmaps_argmax, decode_heatmaps_softargmax
from .losses import WeightedHeatmapMSELoss
from .metrics import corner_l2_error
from .model import BBox8PoseNet
from .utils import draw_corners, ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    """
    summary
        解析训练脚本所需的命令行参数，并返回统一的配置对象。

    参数分析 (Args)
        无显式函数参数；参数均来自命令行输入。
        --labels_root: str，数据标注根目录，必填，应指向 bbox8_labels_obj_xxxxxx。
        --output_dir: str，输出目录，必填，用于保存 checkpoint、日志与可视化结果。
        --epochs: int，总训练轮数，默认值 30，应为正整数。
        --batch_size: int，批大小，默认值 16，应为正整数。
        --num_workers: int，DataLoader 的并行加载进程数，默认值 4。
        --lr: float，优化器学习率，默认值 1e-3，通常应为正数。
        --image_width: int，输入图像宽度，默认值 256。
        --image_height: int，输入图像高度，默认值 256。
        --heatmap_width: int，热图宽度，默认值 64。
        --heatmap_height: int，热图高度，默认值 64。
        --sigma: float，生成高斯热图时的标准差，默认值 2.5。
        --device: str，运行设备，默认优先使用 cuda，否则 cpu。
        --save_every: int，每隔多少个 epoch 额外保存一次周期性 checkpoint，默认值 5。
        --backbone: str，骨干网络类型，可选 simple / resnet18 / resnet34。
        --pretrained_backbone: bool，是否使用 torchvision 预训练权重；仅当外部实现支持时生效。
        --base_channels: int，仅在 backbone=simple 时使用的基础通道数，默认值 32。
        --decoder: str，解码器结构类型，可选 fpn / boxdreamer_lite。
        --decoder_dim: int，解码器特征维度，默认值 192。
        --decoder_depth: int，解码器层数，默认值 3。
        --decoder_heads: int，解码器头数，默认值 8。
        --vis_every: int，每隔多少个 epoch 导出一次验证集可视化，默认值 1。
        --vis_num_samples: int，每次导出的验证样本数量，默认值 4。

    返回值 (Returns)
        argparse.Namespace:
            解析后的命令行参数对象，可通过 args.xxx 访问各字段。

    变量维度分析
        该函数本身不直接处理张量。
        image_width / image_height / heatmap_width / heatmap_height 为标量超参数，
        后续通常分别参与构造图像尺寸 (W, H) 与热图尺寸 (Hh, Wh)。

    举例
        命令行示例:
            python train.py --labels_root ./data/bbox8_labels_obj_000001 --output_dir ./runs/exp1
        返回后可通过:
            args.batch_size
            args.backbone
            args.decoder
        等字段控制训练流程。
    """
    parser = argparse.ArgumentParser(description="Train a single-image bbox8 heatmap network.")
    parser.add_argument("--labels_root", required=True, help="Path to bbox8_labels_obj_xxxxxx")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_width", type=int, default=256)
    parser.add_argument("--image_height", type=int, default=256)
    parser.add_argument("--heatmap_width", type=int, default=64)
    parser.add_argument("--heatmap_height", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--backbone", choices=["simple", "resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--pretrained_backbone", action="store_true", help="Use torchvision pretrained weights if available.")
    parser.add_argument("--base_channels", type=int, default=32, help="Only used when backbone=simple.")
    parser.add_argument("--decoder", choices=["fpn", "boxdreamer_lite", "boxdreamer"], default="boxdreamer_lite")
    parser.add_argument("--decoder_dim", type=int, default=192)
    parser.add_argument("--decoder_depth", type=int, default=3)
    parser.add_argument("--decoder_heads", type=int, default=8)
    parser.add_argument("--decoder_patch_size", type=int, default=4)
    parser.add_argument("--coord_loss_weight", type=float, default=0.0)
    parser.add_argument("--coord_softargmax_temp", type=float, default=0.05)
    parser.add_argument("--vis_every", type=int, default=1, help="Export validation visualizations every N epochs.")
    parser.add_argument("--vis_num_samples", type=int, default=4, help="Number of validation samples to visualize.")
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    """
    summary
        根据配置构建训练集与验证集对应的 DataLoader。

    参数分析 (Args)
        args: argparse.Namespace，训练配置对象；至少应包含 labels_root、batch_size、num_workers、
            image_width、image_height、heatmap_width、heatmap_height、sigma 等字段。

    返回值 (Returns)
        Dict[str, DataLoader]:
            包含两个键:
            - "train": 训练集 DataLoader
            - "val": 验证集 DataLoader

    变量维度分析
        image_size:
            tuple[int, int]，图像尺寸，当前代码约定为 (W, H)。
        heatmap_size:
            tuple[int, int]，热图尺寸，当前代码约定为 (Hh, Wh)。
        DataLoader 输出的 batch:
            依赖 BBox8PoseDataset 与 collate_bbox8 的实现。
            通常包含:
            - image: (B, 3, H, W)
            - heatmaps: (B, K, Hh, Wh)，其中 K 通常为 8 个角点热图
            - valid_mask: 维度由数据集实现决定，通常与角点数 K 对齐
            - corners_xy: (B, K, 2)

    举例
        若 args.image_width=256, args.image_height=256, args.heatmap_height=64, args.heatmap_width=64，
        则 image_size=(256, 256)，heatmap_size=(64, 64)。
        返回的 loaders["train"] 可直接用于:
            for batch in loaders["train"]:
                ...
    """
    image_size = (args.image_width, args.image_height)  # image_size表示: 输入网络的图像尺寸配置；维度为: 2元组 (W, H)；分别表示宽与高
    heatmap_size = (args.heatmap_height, args.heatmap_width)  # heatmap_size表示: 监督热图尺寸配置；维度为: 2元组 (Hh, Wh)；与图像尺寸分开设置以降低预测分辨率

    #! BBox8PoseDataset: 外部数据集类；负责读取图像、角点标注并生成监督热图/有效性掩码等训练样本
    train_set = BBox8PoseDataset(
        labels_root=args.labels_root,
        split="train",
        image_size=image_size,
        heatmap_size=heatmap_size,
        sigma=args.sigma,
        augment=True,
    )
    val_set = BBox8PoseDataset(
        labels_root=args.labels_root,
        split="val",
        image_size=image_size,
        heatmap_size=heatmap_size,
        sigma=args.sigma,
        augment=False,
    )
    return {
        "train": DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_bbox8,#函数接口：用于拼接batch维度
            pin_memory=True,
        ),
        "val": DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_bbox8,
            pin_memory=True,
        ),
    }


def masked_normalized_l1_coord_loss(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    valid_mask: torch.Tensor,
    image_size,
) -> torch.Tensor:
    scale = pred_xy.new_tensor([float(image_size[0]), float(image_size[1])]).view(1, 1, 2)
    diff = (pred_xy - gt_xy).abs() / scale
    weights = valid_mask.float().unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0) * 2.0
    return (diff * weights).sum() / denom


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    image_size,
    train: bool,
    coord_loss_weight: float = 0.0,
    coord_softargmax_temp: float = 0.05,
):
    """
    summary
        执行一个完整的训练或验证轮次，并返回平均损失与平均角点 L2 误差。

    参数分析 (Args)
        model:
            PyTorch 模型实例；输入图像张量，输出角点热图预测。
        loader:
            DataLoader，按批提供训练或验证样本。
        criterion:
            损失函数对象；此处用于比较预测热图与目标热图，并结合有效掩码计算损失。
        optimizer:
            优化器；仅在 train=True 时参与参数更新。
        device:
            torch.device 或兼容设备描述；控制张量与模型所在设备。
        image_size:
            tuple[int, int]，当前代码约定为 (W, H)，供热图解码到图像坐标时使用。
        train: bool
            True 表示训练模式，会启用梯度并执行反向传播与优化；
            False 表示验证模式，不更新参数。

    返回值 (Returns)
        dict:
            - "loss": float，当前 epoch 所有 batch 的平均热图损失
            - "mean_corner_l2": float，当前 epoch 所有 batch 的平均角点欧氏距离误差

    变量维度分析
        images:
            通常为 (B, 3, H, W)，B 为批大小，3 为 RGB 通道数。
        target_heatmaps:
            通常为 (B, K, Hh, Wh)，K 为角点数，Hh/Wh 为热图高宽。
        valid_mask:
            维度依赖数据集与损失实现，通常与角点维 K 对齐，例如 (B, K)。
        gt_xy:
            通常为 (B, K, 2)，最后一维表示 (x, y)。
        pred_heatmaps:
            形状应与 target_heatmaps 对齐。
        pred_xy:
            解码后的角点坐标，通常为 (B, K, 2)。

    举例
        当 B=16、K=8、输入图像尺寸为 256x256、热图尺寸为 64x64 时:
            images.shape = (16, 3, 256, 256)
            pred_heatmaps.shape ≈ (16, 8, 64, 64)
            pred_xy.shape ≈ (16, 8, 2)
        若 train=False，则本函数只前向推理并统计指标，不会更新模型参数。
    """
    model.train(train)  #* 根据 train 标志切换模型模式；这会影响 BatchNorm / Dropout 等存在 train/eval 差异的模块行为
    total_loss = 0.0  # total_loss表示: 当前 epoch 累积的 batch 损失和；标量；后续会除以 batch 数得到平均损失
    total_metric = 0.0  # total_metric表示: 当前 epoch 累积的 mean_corner_l2 指标和；标量；后续会除以 batch 数得到平均指标
    num_batches = 0  # num_batches表示: 当前 epoch 已处理的 batch 数量；标量整数；用于做均值归一化

    # * 接下来这段代码按 batch 遍历一个完整数据集切分，并在训练/验证模式下分别执行前向、反向与指标统计
    for batch in loader:
        images = batch["image"].to(device)  # images表示: 输入图像批；维度通常为 (B, 3, H, W)；images[b] 表示第 b 个样本的归一化图像张量
        target_heatmaps = batch["heatmaps"].to(device)  # target_heatmaps表示: 目标角点热图批；维度通常为 (B, K, Hh, Wh)；与模型输出逐元素对齐
        valid_mask = batch["valid_mask"].to(device)  # valid_mask表示: 角点有效性掩码；维度依赖实现，通常与 K 对齐；用于屏蔽无效角点的损失与评估
        gt_xy = batch["corners_xy"].to(device)  # gt_xy表示: 角点真值坐标；维度通常为 (B, K, 2)；最后一维为 (x, y)

        # * 这里利用 torch.set_grad_enabled 在同一套代码中统一处理训练与验证两种路径
        with torch.set_grad_enabled(train):
            pred_heatmaps = model(images)  # pred_heatmaps表示: 模型预测热图；输入通常为 (B, 3, H, W)，输出通常为 (B, K, Hh, Wh)
            # 再次啰嗦一嘴
            # critieon传入的是MSE均方误差
            # 也就是计算预测热图和真是热图之间的平均平方差
            heatmap_loss = criterion(pred_heatmaps, target_heatmaps, valid_mask)
            loss = heatmap_loss
            if coord_loss_weight > 0:
                pred_xy_soft = decode_heatmaps_softargmax(
                    pred_heatmaps,
                    image_size=image_size,
                    temperature=coord_softargmax_temp,
                )
                coord_loss = masked_normalized_l1_coord_loss(pred_xy_soft, gt_xy, valid_mask, image_size)
                loss = loss + float(coord_loss_weight) * coord_loss
            if train:
                optimizer.zero_grad(set_to_none=True)  #* 清空历史梯度；set_to_none=True 可减少不必要的内存写入，通常更高效
                loss.backward()  #* 反向传播当前 batch 的损失，计算模型参数梯度
                optimizer.step()  #* 根据当前梯度更新模型参数
        # * 条件分支结束后，pred_heatmaps 表示当前 batch 的热图预测；维度通常为 (B, K, Hh, Wh)；loss 表示当前 batch 的标量损失

        #! decode_heatmaps_argmax: 外部热图解码函数；通常在每个热图上取最大响应位置，并映射回图像坐标系
        pred_xy = decode_heatmaps_argmax(pred_heatmaps.detach(), image_size=image_size)  # pred_xy表示: 从预测热图中解码出的角点坐标；通常为 (B, K, 2)；detach() 用于阻断梯度，避免评估路径保留计算图
        #! corner_l2_error: 外部评估函数；输入预测/真值角点与有效掩码，输出角点欧氏距离误差统计字典
        metrics = corner_l2_error(pred_xy.cpu(), gt_xy.cpu(), valid_mask.cpu())  # metrics表示: 当前 batch 的误差统计结果；字典；至少包含 "mean_corner_l2" 键
        total_loss += loss.item()  #* 将当前 batch 的标量损失累加到 epoch 级统计量
        total_metric += metrics["mean_corner_l2"]  #* 将当前 batch 的平均角点 L2 误差累加到 epoch 级统计量
        num_batches += 1  #* 已处理 batch 数量加一

    # * 循环结束后，total_loss / total_metric 分别表示整个 epoch 内的累计损失和累计指标；num_batches 表示总 batch 数
    denom = max(1, num_batches)  # denom表示: 平均化分母；标量整数；使用 max(1, ...) 防止极端情况下空 loader 导致除零
    return {
        "loss": total_loss / denom,
        "mean_corner_l2": total_metric / denom,
    }


def export_val_visualizations(model, dataset, device, image_size, output_dir, epoch: int, num_samples: int) -> None:
    """
    summary
        从验证集中抽取若干样本，导出真值与预测角点叠加可视化结果。

    参数分析 (Args)
        model:
            已训练或当前 epoch 的模型，用于生成热图预测。
        dataset:
            验证数据集对象；支持按索引读取单个样本。
        device:
            torch.device 或兼容设备描述；用于模型推理时放置输入张量。
        image_size:
            tuple[int, int]，当前代码约定为 (W, H)；用于将热图解码结果映射到网络输入图像坐标系。
        output_dir:
            str，实验输出目录；内部会在其下创建 val_vis/epoch_xxx 子目录。
        epoch: int
            当前 epoch 序号，用于命名可视化输出目录。
        num_samples: int
            希望导出的样本数量；函数内部会裁剪到 [1, len(dataset)] 范围内。

    返回值 (Returns)
        None:
            无显式返回值；副作用是在磁盘上写入可视化图片。

    变量维度分析
        sample["image"]:
            通常为 (3, H, W)，unsqueeze(0) 后变为 (1, 3, H, W)。
        pred_heatmaps:
            通常为 (1, K, Hh, Wh)。
        pred_xy:
            解码后通常为 (1, K, 2)，取第 0 个样本后为 (K, 2)。
        gt_xy / gt_xy_orig:
            通常为 (K, 2)。
        valid_mask:
            通常为 (K,) 或与角点 K 对齐的掩码结构。
        merged:
            通过宽度维拼接真值图与预测图后，形状通常为 (H0, 2*W0, 3)。

    举例
        当 num_samples=4、数据集长度为 100 时，会从索引区间 [0, 99] 线性抽取 4 个样本。
        每张导出图通常由左侧真值角点可视化与右侧预测角点可视化拼接而成。
    """
    vis_dir = os.path.join(output_dir, "val_vis", f"epoch_{epoch:03d}")  # vis_dir表示: 当前 epoch 的可视化输出目录；字符串路径；形如 output_dir/val_vis/epoch_001
    ensure_dir(vis_dir)  #! ensure_dir: 外部工具函数；确保目录存在，不存在则创建

    model.eval()  #* 显式切换到评估模式，保证可视化导出时使用稳定的推理行为
    # led(dataset)
    # 会调用dataset类，也就是BBox8PoseDataset类的__len_方法
    # 返回self.records的长度，也就是当前mode的样本数目
    max_count = min(len(dataset), max(1, int(num_samples)))  # max_count表示: 实际导出样本数；标量整数；至少为 1，至多为数据集长度
    indices = np.linspace(0, len(dataset) - 1, num=max_count, dtype=int)  # indices表示: 均匀采样得到的样本索引数组；维度为 (max_count,)；用于覆盖验证集不同位置的样本

    # * 接下来这段代码逐个样本执行前向推理、坐标反缩放、可视化绘制与磁盘保存
    for vis_idx, data_idx in enumerate(indices.tolist()):
        sample = dataset[data_idx]  # sample表示: 单个验证样本；字典；具体键由 BBox8PoseDataset 决定
        # dataset[idx]，会调用dataset类，也就是BBox8PoseDataset类的__getitem_方法
        # 具体返回的内容定义，在BBox8PoseDataset类中定义
        # 这里简单说一下：
        # 主要包括根据idx从records中读取的对应样本的信息；
        # 返回一个经过输入的image_size缩放后的image以及相应缩放后的2d角点坐标，以及根据这个生成的缩放后的热图监督信息
        # 原始的2d角点坐标
        # 同时会返回3d空间角点坐标，相应的相机参数等等。

        # 顺便梳理一下整个这个逻辑：我的目标是拿到真实的原始坐标系中的2d角点坐标+预测的原始坐标系的2d角点坐标，把这两个放到原图上进行一个可视化对比
        # sample中已经有了原始坐标系的2d角点坐标，以及原图的路径，所以我目前只需要拿到预测的原始坐标系的2d角点坐标；
        # 用model，输入缩放后的image，得到预测的热图，解码得到预测的2d角点坐标，但是是缩放过的；
        # 所以要找到一个扩大比例，就可以直接用原图的尺寸/image_size即可得到扩大比例，也就拿到了原始坐标系的2d角点坐标。
        image = sample["image"].unsqueeze(0).to(device)  # image表示: 单样本输入图像批；(3, H, W) -> (1, 3, H, W)；通过增加 batch 维适配模型前向
        with torch.no_grad():
            pred_heatmaps = model(image)  # pred_heatmaps表示: 单样本预测热图；通常为 (1, K, Hh, Wh)
            pred_xy = decode_heatmaps_argmax(pred_heatmaps, image_size=image_size)[0].cpu().numpy()  # pred_xy表示: 单样本预测角点坐标；通常为 (K, 2)；[0] 取出批中的唯一一个样本

        gt_xy = sample["corners_xy"].cpu().numpy()  # gt_xy表示: 当前样本在网络输入坐标系中的真值角点；通常为 (K, 2)
        valid_mask = sample["valid_mask"].cpu().numpy()  # valid_mask表示: 当前样本角点有效性掩码；维度通常与 K 对齐
        rgb_path = sample["rgb_path"]  # rgb_path表示: 当前样本原始 RGB 图像路径；字符串；用于读取未缩放的原图做可视化
        image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # image_bgr表示: OpenCV 读取的 BGR 原图；维度为 (H0, W0, 3)；uint8
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # image_rgb表示: 转为 RGB 顺序后的原图；(H0, W0, 3)；便于后续可视化函数使用
        orig_h, orig_w = image_rgb.shape[:2]  # orig_h / orig_w表示: 原图高宽；标量整数；由磁盘读取的真实图像大小决定
        pred_xy_orig = pred_xy.copy()  # pred_xy_orig表示: 将要映射回原图坐标系的预测角点副本；维度为 (K, 2)
        pred_xy_orig[:, 0] *= orig_w / float(image_size[0])  #* 将 x 坐标从网络输入宽度尺度映射到原图宽度尺度；列 0 仍表示所有角点的 x 坐标
        pred_xy_orig[:, 1] *= orig_h / float(image_size[1])  #* 将 y 坐标从网络输入高度尺度映射到原图高度尺度；列 1 仍表示所有角点的 y 坐标
        gt_xy_orig = sample["corners_xy_orig"].cpu().numpy()  # gt_xy_orig表示: 原图坐标系下的真值角点；通常为 (K, 2)；无需再次缩放

        #! draw_corners: 外部可视化函数；将角点和有效掩码叠加到 RGB 图像上，返回绘制后的图像
        pred_vis = draw_corners(image_rgb, pred_xy_orig, valid_mask)  # pred_vis表示: 绘制预测角点后的 RGB 图像；维度通常为 (H0, W0, 3)
        gt_vis = draw_corners(image_rgb, gt_xy_orig, valid_mask)  # gt_vis表示: 绘制真值角点后的 RGB 图像；维度通常为 (H0, W0, 3)
        merged = np.concatenate([gt_vis, pred_vis], axis=1)  # 宽度维拼接两张可视化图；(H0, W0, 3) + (H0, W0, 3) -> (H0, 2*W0, 3)；左真值右预测便于直观对比
        stem = os.path.splitext(os.path.basename(rgb_path))[0]  # stem表示: 原图文件名去扩展名后的主干字符串；用于构造更可读的输出文件名
        out_path = os.path.join(vis_dir, f"{vis_idx:02d}_{stem}_gt_pred.jpg")  # out_path表示: 当前可视化图的输出路径；字符串
        cv2.imwrite(out_path, cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))  #* OpenCV 写盘默认按 BGR 理解，因此这里先把 RGB 结果转回 BGR 再保存
    # * 循环结束后，vis_dir 下会包含若干张 gt/pred 对比图；每张图都对应该 epoch 的一个验证样本可视化结果


def main() -> None:
    """
    summary
        组织完整训练流程，包括参数解析、数据构建、模型初始化、逐 epoch 训练验证、checkpoint 保存与可视化导出。

    参数分析 (Args)
        无显式函数参数；所有配置通过 parse_args 从命令行解析得到。

    返回值 (Returns)
        None:
            无显式返回值；主要副作用包括模型训练、写入 checkpoint、写入历史日志与保存验证可视化结果。

    变量维度分析
        loaders["train"] / loaders["val"]:
            分别输出训练与验证 batch。
        model:
            输入通常为 (B, 3, H, W)，输出通常为 (B, K, Hh, Wh)。
        history:
            list[dict]，每个元素保存一个 epoch 的训练/验证统计结果。
        epoch_stats:
            dict，包含 epoch、train_loss、train_mean_corner_l2、val_loss、val_mean_corner_l2、lr 等字段。

    举例
        运行命令:
            python train.py --labels_root ./data/xxx --output_dir ./runs/exp1 --epochs 50
        执行后会在 output_dir 中得到:
            - last.pt
            - best.pt
            - best_metric.pt
            - train_history.json
            - val_vis/epoch_xxx/*.jpg
    """
    args = parse_args()  # args表示: 命令行解析结果；Namespace；统一承载训练配置
    ensure_dir(args.output_dir)  #* 确保实验输出目录存在，避免后续保存 checkpoint / json / 可视化时路径不存在
    loaders = build_dataloaders(args)  # loaders表示: 训练与验证 DataLoader 字典；键为 "train" / "val"
    image_size = (args.image_width, args.image_height)  # image_size表示: 解码热图所需的输入图像尺寸；2元组 (W, H)

    device = torch.device(args.device)  # device表示: 当前运行设备对象；如 cuda / cpu；后续模型与输入张量都会移动到该设备
    #! BBox8PoseNet: 外部网络定义；根据指定 backbone 与 decoder 预测 8 个角点热图
    model = BBox8PoseNet(
        backbone=args.backbone,
        pretrained_backbone=args.pretrained_backbone,
        base_channels=args.base_channels,
        decoder=args.decoder,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        decoder_patch_size=args.decoder_patch_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # optimizer表示: Adam 优化器；负责根据梯度更新模型参数
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # scheduler表示: 余弦退火学习率调度器；在 epochs 范围内平滑衰减学习率
    #! WeightedHeatmapMSELoss: 外部损失函数；通常对预测热图和目标热图做加权 MSE，并结合有效掩码忽略无效位置
    # MSE:表示计算真是热图和预测热图之间的平均平方差。
    criterion = WeightedHeatmapMSELoss()

    best_val = float("inf")  # best_val表示: 目前观察到的最优验证损失；标量；初始化为正无穷便于首次比较
    best_metric = float("inf")  # best_metric表示: 目前观察到的最优验证角点 L2 指标；标量；越小越好
    history = []  # history表示: 训练全过程的历史统计列表；每个元素是一个 epoch 的字典记录

    # * 接下来这段代码执行标准训练主循环：先训练、再验证、再更新调度器，随后保存日志、checkpoint 与可视化结果
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            image_size,
            train=True,
            coord_loss_weight=args.coord_loss_weight,
            coord_softargmax_temp=args.coord_softargmax_temp,
        )  # train_stats表示: 当前 epoch 的训练统计；字典；至少含 loss 与 mean_corner_l2
        val_stats = run_epoch(
            model,
            loaders["val"],
            criterion,
            optimizer,
            device,
            image_size,
            train=False,
            coord_loss_weight=0.0,
            coord_softargmax_temp=args.coord_softargmax_temp,
        )  # val_stats表示: 当前 epoch 的验证统计；字典；至少含 loss 与 mean_corner_l2
        scheduler.step()  #* 在每个 epoch 末尾推进一次学习率调度器

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_mean_corner_l2": train_stats["mean_corner_l2"],
            "val_loss": val_stats["loss"],
            "val_mean_corner_l2": val_stats["mean_corner_l2"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_stats)  #* 将当前 epoch 的训练/验证结果与学习率状态追加到完整历史中
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_stats['loss']:.6f} "
            f"val_loss={val_stats['loss']:.6f} "
            f"val_corner_l2={val_stats['mean_corner_l2']:.3f}"
        )

        last_ckpt = os.path.join(args.output_dir, "last.pt")  # last_ckpt表示: 始终覆盖的“最后一次训练状态”保存路径；字符串
        torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, last_ckpt)  #* 无条件保存最近一次模型参数、配置与历史

        # * 当验证损失刷新最优时，额外保存一个以 loss 为准的最佳模型
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]  #* 更新当前最优验证损失
            best_ckpt = os.path.join(args.output_dir, "best.pt")  # best_ckpt表示: 按验证损失选择的最佳模型保存路径；字符串
            torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, best_ckpt)

        # * 当验证角点 L2 误差刷新最优时，额外保存一个以 metric 为准的最佳模型
        if val_stats["mean_corner_l2"] < best_metric:
            best_metric = val_stats["mean_corner_l2"]  #* 更新当前最优验证指标
            best_metric_ckpt = os.path.join(args.output_dir, "best_metric.pt")  # best_metric_ckpt表示: 按角点 L2 指标选择的最佳模型保存路径；字符串
            torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, best_metric_ckpt)

        # * 根据 save_every 周期性额外保存带 epoch 编号的 checkpoint，便于回溯不同训练阶段的模型状态
        if epoch % args.save_every == 0:
            epoch_ckpt = os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt")  # epoch_ckpt表示: 周期性 checkpoint 路径；字符串
            torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, epoch_ckpt)

        # * 根据 vis_every 周期导出验证集可视化，便于人工观察角点预测质量与错误模式
        if epoch % args.vis_every == 0:
            export_val_visualizations(
                model=model,
                dataset=loaders["val"].dataset,
                # loaders["val"] 是 DataLoader，而 .dataset 是把里面包着的原始 BBox8PoseDataset 取出来。
                device=device,
                image_size=image_size,
                output_dir=args.output_dir,
                epoch=epoch,
                num_samples=args.vis_num_samples,
            )
    # * 循环结束后，history 表示完整训练过程的逐 epoch 统计记录；长度为 args.epochs；每个元素均为一个字典

    save_json(os.path.join(args.output_dir, "train_history.json"), {"history": history})  #! save_json: 外部工具函数；将历史统计持久化为 JSON，便于后续分析与绘图


if __name__ == "__main__":
    main()
