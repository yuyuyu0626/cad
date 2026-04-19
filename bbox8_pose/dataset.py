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
    """
    summary
        定义数据增强阶段使用的若干图像扰动超参数配置。

    参数分析 (Args)
        brightness: float，亮度扰动幅度，默认值 0.2；实际用于生成加性偏移 beta 的范围。
        contrast: float，对比度扰动幅度，默认值 0.2；实际用于生成乘性系数 alpha 的范围。
        saturation: float，饱和度扰动幅度，默认值 0.2；当前文件中尚未实际使用，仅作为保留配置。
        blur_prob: float，应用高斯模糊的概率，默认值 0.1；取值通常在 [0, 1]。

    返回值 (Returns)
        AugmentConfig:
            一个 dataclass 配置对象，供数据集内部读取增强超参数。

    变量维度分析
        本类不直接处理张量。
        各字段均为标量超参数，用于控制图像增强的强弱与触发概率。

    举例
        若 brightness=0.2、contrast=0.2，则图像可能执行:
            image' = image * alpha + beta
        其中 alpha 约在 [0.8, 1.2]，beta 约在 [-51, 51] 范围内变化。
    """
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    blur_prob: float = 0.1


def _load_jsonl(path: str) -> List[Dict]:
    """
    summary
        从 JSONL 文件中逐行读取记录，并解析为字典列表。

    参数分析 (Args)
        path: str，JSONL 文件路径；文件中每一行应为一个合法的 JSON 对象字符串。

    返回值 (Returns)
        List[Dict]:
            解析后的样本记录列表；每个元素对应 JSONL 中的一行字典记录。

    变量维度分析
        records:
            Python 列表，长度为 N，N 为 JSONL 中的有效非空行数。
        每个 records[i]:
            字典，键值结构依赖数据文件定义；通常包含 sample_id、rgb_path、corners_2d 等字段。

    举例
        若文件包含两行:
            {"sample_id": "0001", "rgb_path": "..."}
            {"sample_id": "0002", "rgb_path": "..."}
        则返回:
            [dict(...), dict(...)]
    """
    records: List[Dict] = []  # records表示: 解析后的样本记录列表；长度为 N；records[i] 表示第 i 条样本元数据字典
    with open(path, "r", encoding="utf-8") as f:
        # * 接下来这段代码逐行读取 JSONL 文件，并跳过空行
        for line in f:
            line = line.strip()  # line表示: 当前行去除首尾空白后的字符串；空串表示该行无有效内容
            if line:
                records.append(json.loads(line))  #* 将当前非空 JSON 字符串反序列化为字典并追加到记录列表
        # * 循环结束后，records 表示文件中全部有效样本记录；长度等于非空 JSON 行数
    return records


def _load_split_ids(path: str) -> List[str]:
    """
    summary
        从划分文件中读取样本 ID 列表。

    参数分析 (Args)
        path: str，划分文件路径；文件中通常每行一个 sample_id。

    返回值 (Returns)
        List[str]:
            样本 ID 字符串列表；顺序与文件中的有效非空行顺序一致。

    变量维度分析
        返回列表长度为 M，M 为文件中的有效非空行数。
        每个元素为一个样本 ID 字符串，不涉及张量维度。

    举例
        若文件内容为:
            000001
            000005
            000010
        则返回:
            ["000001", "000005", "000010"]
    """
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


#! Dataset: PyTorch 的数据集抽象基类；需要实现 __len__ 和 __getitem__，供 DataLoader 按索引取样本
class BBox8PoseDataset(Dataset):
    """
    summary
        读取单张 RGB 图像与对应 8 个 2D/3D 角点标注，并生成热图监督信号的目标检测/位姿估计数据集。

    参数分析 (Args)
        labels_root: str，标注根目录；内部应包含 annotations.jsonl、train.txt / val.txt 等文件。
        split: str，数据划分名称，例如 "train" 或 "val"；用于选择对应样本子集。
        image_size: Tuple[int, int]，输入图像缩放尺寸，默认值 (256, 256)；当前代码约定为 (W, H)。
        heatmap_size: Tuple[int, int]，监督热图尺寸，默认值 (64, 64)；当前代码沿用 generate_corner_heatmaps 的输入约定。
        sigma: float，高斯热图的标准差，默认值 2.5；控制角点响应峰值的扩散范围。
        use_soft_mask_filter: bool，是否根据 visib_fract 进一步过滤无效样本角点，默认值 True。
        augment: bool，是否在取样时执行图像增强，默认值 False；通常训练集为 True，验证集为 False。
        seed: int，随机增强所用伪随机种子，默认值 42。

    返回值 (Returns)
        BBox8PoseDataset:
            一个可被 DataLoader 包装的数据集对象；按索引返回单样本字典。

    变量维度分析
        self.records:
            长度为 N 的样本记录列表；N 为当前 split 的样本数。
        单个样本通常包含:
            - image: (3, H, W)
            - heatmaps: (K, Hh, Wh)，K 通常为 8 个 bbox 角点
            - corners_xy: (K, 2)
            - corners_xy_orig: (K, 2)
            - valid_mask: (K,)
            - camera_K: (3, 3)
            - corners_3d: (K, 3)，通常 K=8
            - orig_size: (2,)，内容为 [orig_w, orig_h]

    举例
        若某样本原图大小为 640x480，image_size=(256, 256)，则:
            image.shape = (3, 256, 256)
            corners_xy 为按比例缩放后的 2D 角点
            corners_xy_orig 保留原图坐标系下的角点位置
    """

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
        """
        summary
            初始化数据集，加载标注文件、划分信息，并准备增强与可选 3D bbox 元数据。

        参数分析 (Args)
            labels_root: str，标注根目录，会被转为绝对路径。
            split: str，数据划分名；需存在同名的 split 文件，例如 train.txt / val.txt。
            image_size: Tuple[int, int]，网络输入图像尺寸，默认值 (256, 256)，约定为 (W, H)。
            heatmap_size: Tuple[int, int]，角点热图尺寸，默认值 (64, 64)。
            sigma: float，角点高斯热图半径相关标准差，默认值 2.5。
            use_soft_mask_filter: bool，是否根据 visib_fract<=0 时将 valid_mask 置零。
            augment: bool，是否开启增强。
            seed: int，随机数种子；用于内部 random.Random，保证增强可复现。

        返回值 (Returns)
            None:
                无显式返回值；主要副作用是完成路径检查、记录筛选与成员变量初始化。

        变量维度分析
            self.image_size:
                2 元组 (W, H)。
            self.heatmap_s  ize:
                2 元组，具体顺序依赖 generate_corner_heatmaps 的接口约定；当前代码直接透传。
            self.records:
                长度为 N 的样本字典列表。
            self.object_bbox_3d:
                Optional[dict]，若存在 object_bbox_3d.json 则加载其内容，否则为 None。

        举例
            当 split="train" 时，本函数会:
                1. 读取 annotations.jsonl
                2. 读取 train.txt 中的 sample_id
                3. 按 sample_id 过滤样本
                4. 对结果按 sample_id 排序
        """
        self.labels_root = os.path.abspath(labels_root)  # self.labels_root表示: 标注根目录的绝对路径；字符串；后续所有元数据路径都基于该目录构造
        self.image_size = image_size  # self.image_size表示: 网络输入图像尺寸；2元组 (W, H)
        self.heatmap_size = heatmap_size  # self.heatmap_size表示: 监督热图尺寸；2元组；具体高宽语义依赖热图生成函数约定
        self.sigma = sigma  # self.sigma表示: 高斯热图扩散尺度；标量；越大则角点响应区域越宽
        self.use_soft_mask_filter = use_soft_mask_filter  # self.use_soft_mask_filter表示: 是否根据样本可见性进一步过滤有效角点；布尔值
        self.augment = augment  # self.augment表示: 是否在取样时执行随机图像增强；布尔值
        self.rng = random.Random(seed)  # self.rng表示: 数据增强专用伪随机数生成器；避免直接依赖全局 random 状态
        self.augment_cfg = AugmentConfig()  # self.augment_cfg表示: 增强超参数配置对象；用于控制亮度/对比度/模糊等扰动

        ann_path = os.path.join(self.labels_root, "annotations.jsonl")  # ann_path表示: 主标注文件路径；字符串
        # annotations.jsonl文件中的主要内容：
        # 一共N行，表示N个训练图片的信息；每一行信息主要有这些：
        # 样本的身份信息——包括来自哪个scene的第几个image,以及是这个image中的第几个实例、对应的物体id是多少，记忆相应的rgb和mask的路径
        # image的信息：包括尺寸，相机内参、物体的位姿信息（6D pose标注，由R和T组成）；
        # 物体的2D/3D角点信息：包括bbox的大小，3d角点坐标（世界坐标系和相机坐标系的），映射到2d图像后的2d角点坐标，以及角点是否有效。
        split_path = os.path.join(self.labels_root, f"{split}.txt")  # split_path表示: 当前划分对应的样本 ID 文件路径；字符串
        # train/val.txt 文件中的主要内容：
        # 每行一个样本，表示train/val中每个样本的身份信息，也就是来自于哪个scene的第一个image的第几个实例这种；
        if not os.path.exists(ann_path):
            raise FileNotFoundError(ann_path)
        if not os.path.exists(split_path):
            raise FileNotFoundError(split_path)

        all_records = _load_jsonl(ann_path)  # all_records表示: 标注文件中的全部样本记录；长度为 N_all；每个元素为字典
        split_ids = set(_load_split_ids(split_path))  # split_ids表示: 当前划分包含的 sample_id 集合；长度为 N_split；集合加速后续成员测试
        self.records = [rec for rec in all_records if rec["sample_id"] in split_ids]  # self.records表示: 当前 split 实际使用的样本记录列表；长度为 N；N<=N_all
        self.records.sort(key=lambda x: x["sample_id"])  #* 按 sample_id 稳定排序，保证数据访问顺序可重复

        if not self.records:
            raise ValueError(f"No samples found for split={split} in {self.labels_root}")

        bbox_meta_path = os.path.join(self.labels_root, "object_bbox_3d.json")  # bbox_meta_path表示: 可选 3D bbox 元数据文件路径；字符串
        # 这里表示的是原始的.ply文件中，物体在自身坐标系（通常是原点选取使得整体满足中心对称）下的角点坐标等等。
        self.object_bbox_3d = None  # self.object_bbox_3d表示: 可选的对象 3D bbox 元数据；字典或 None；当前文件中仅加载未直接使用
        if os.path.exists(bbox_meta_path):
            with open(bbox_meta_path, "r", encoding="utf-8") as f:
                self.object_bbox_3d = json.load(f)

    def __len__(self) -> int:
        """
        summary
            返回当前数据集的样本数量。

        参数分析 (Args)
            无。

        返回值 (Returns)
            int:
                当前 split 下的样本总数。

        变量维度分析
            返回值为标量整数，不涉及张量维度。
            其数值等于 self.records 的长度。

        举例
            若 train split 过滤后共有 1000 条样本，则 __len__() 返回 1000。
        """
        return len(self.records)

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        summary
            对输入 RGB 图像执行轻量级随机增强，包括亮度/对比度扰动和可选高斯模糊。

        参数分析 (Args)
            image: np.ndarray，输入 RGB 图像；通常维度为 (H, W, 3)，元素值范围通常在 [0, 255]。

        返回值 (Returns)
            np.ndarray:
                增强后的 RGB 图像；维度仍为 (H, W, 3)，数据类型为 uint8。

        变量维度分析
            image:
                输入时通常为 (H, W, 3)；增强过程中会先转为 float32，再裁剪并恢复为 uint8。
            alpha:
                标量，对比度缩放系数。
            beta:
                标量，亮度加性偏移，单位与像素值一致。

        举例
            若原图某像素值为 100，且采样到 alpha=1.1, beta=10，则增强后该像素近似为:
                100 * 1.1 + 10 = 120
            最终会经过 clip 限制在 [0, 255]。
        """
        image = image.astype(np.float32)  # image表示: 转为浮点后的图像；维度为 (H, W, 3)；便于执行线性亮度/对比度变换
        if self.rng.random() < 0.9:
            alpha = 1.0 + self.rng.uniform(-self.augment_cfg.contrast, self.augment_cfg.contrast)  # alpha表示: 对比度缩放系数；标量；中心为 1.0
            beta = 255.0 * self.rng.uniform(-self.augment_cfg.brightness, self.augment_cfg.brightness)  # beta表示: 亮度偏移量；标量；按 0~255 像素尺度进行采样
            image = image * alpha + beta  # 逐像素线性变换；输入/输出均为 (H, W, 3)；语义上同时调整亮度与对比度
        if self.rng.random() < self.augment_cfg.blur_prob:
            image = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊；(H, W, 3) -> (H, W, 3)；用于模拟轻微失焦或运动模糊
        image = np.clip(image, 0, 255).astype(np.uint8)  #* 将增强结果裁剪到合法像素范围并转回 uint8，便于后续 OpenCV / PyTorch 处理
        return image

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        summary
            按索引读取单个样本，完成图像读取、缩放、角点坐标同步变换、热图生成与字段组装。

        参数分析 (Args)
            index: int，样本索引，应满足 0 <= index < len(self.records)。

        返回值 (Returns)
            Dict[str, torch.Tensor]:
                返回一个样本字典，典型字段包括:
                - "image": torch.Tensor，缩放后的 RGB 图像，形状通常为 (3, H, W)
                - "heatmaps": torch.Tensor，角点监督热图，形状通常为 (K, Hh, Wh)
                - "corners_xy": torch.Tensor，缩放后 2D 角点，形状通常为 (K, 2)
                - "corners_xy_orig": torch.Tensor，原图坐标系 2D 角点，形状通常为 (K, 2)
                - "valid_mask": torch.Tensor，角点有效掩码，形状通常为 (K,)
                - "camera_K": torch.Tensor，相机内参矩阵，形状为 (3, 3)
                - "corners_3d": torch.Tensor，物体坐标系 3D 角点，形状通常为 (K, 3)
                - "sample_id": str，样本 ID
                - "rgb_path": str，原始 RGB 图像路径
                - "orig_size": torch.Tensor，原图宽高，形状为 (2,)

        变量维度分析
            resized:
                (H, W, 3)，其中 H/W 由 self.image_size 指定。
            image_tensor:
                (3, H, W)，由 numpy 图像转为 CHW 格式并归一化到 [0, 1]。
            corners:
                通常为 (K, 2)，K 常为 8。
            valid_mask:
                通常为 (K,)。
            heatmaps:
                通常为 (K, Hh, Wh)。
            camera_K:
                (3, 3)。
            orig_size:
                (2,)，内容顺序为 [orig_w, orig_h]。

        举例
            若原图大小为 640x480、K=8、目标输入大小为 256x256，则:
                image_tensor.shape = (3, 256, 256)
                corners_xy 为按宽高比例缩放后的 8 个角点
                heatmaps.shape 通常为 (8, 64, 64)
        """
        rec = self.records[index]  # rec表示: 当前索引对应的样本元数据字典；键值结构来自 annotations.jsonl
        image = cv2.imread(rec["rgb_path"], cv2.IMREAD_COLOR)  # image表示: 从磁盘读取的 BGR 图像；维度通常为 (H0, W0, 3)；dtype=uint8
        if image is None:
            raise ValueError(f"Failed to read image: {rec['rgb_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB；(H0, W0, 3) 保持不变；语义上转为更常见的深度学习图像通道顺序
        orig_h, orig_w = image.shape[:2]  # orig_h / orig_w表示: 原图高宽；标量整数；由实际读取图像决定

        if self.augment:
            image = self._apply_augmentation(image)  #* 仅在训练增强开启时对 RGB 图像做随机扰动；不改变空间尺寸

        resized = cv2.resize(image, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_LINEAR)  # 空间缩放；(H0, W0, 3) -> (H, W, 3)；目标尺寸由 image_size=(W, H) 指定
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0  # HWC -> CHW；(H, W, 3) -> (3, H, W)；并归一化到 [0, 1] 浮点范围

        corners = torch.tensor(rec["corners_2d"], dtype=torch.float32)  # corners表示: 原图坐标系下的 2D 角点；维度通常为 (K, 2)；最后一维为 (x, y)
        valid_mask = torch.tensor(rec["corner_valid_mask"], dtype=torch.float32)  # valid_mask表示: 角点有效掩码；维度通常为 (K,)；1 表示有效，0 表示无效
        scale_x = self.image_size[0] / float(orig_w)  # scale_x表示: x 方向缩放比例；标量；将原图宽坐标映射到输入宽度
        scale_y = self.image_size[1] / float(orig_h)  # scale_y表示: y 方向缩放比例；标量；将原图高坐标映射到输入高度
        corners_resized = corners.clone()  # corners_resized表示: 输入图像坐标系下的 2D 角点副本；维度为 (K, 2)
        corners_resized[:, 0] *= scale_x  #* 仅缩放 x 坐标列；(K, 2) 形状不变；语义上把原图 x 坐标映射到网络输入坐标系
        corners_resized[:, 1] *= scale_y  #* 仅缩放 y 坐标列；(K, 2) 形状不变；语义上把原图 y 坐标映射到网络输入坐标系

        if self.use_soft_mask_filter and rec.get("visib_fract", 1.0) <= 0:
            valid_mask.zero_()  #* 当样本整体不可见时，将所有角点标记为无效；原地修改 valid_mask，维度保持 (K,)

        #! generate_corner_heatmaps: 外部热图生成函数；根据 2D 角点、有效掩码和尺寸配置，生成每个角点的高斯监督热图
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
        # * 返回字典中同时保留网络输入坐标系和原图坐标系下的角点，便于训练监督与可视化/误差分析分别使用
        return sample


def collate_bbox8(batch: List[Dict]) -> Dict[str, torch.Tensor]:#
    """
    summary
        将单样本字典列表拼接为一个批次字典，供 DataLoader 输出。

    参数分析 (Args)
        batch: List[Dict]，长度为 B 的样本列表；每个元素通常来自 BBox8PoseDataset.__getitem__。

    返回值 (Returns)
        Dict[str, torch.Tensor]:
            批次字典，其中张量字段会在第 0 维堆叠形成 batch 维，字符串字段保留为列表。
            典型字段包括:
            - image: (B, 3, H, W)
            - heatmaps: (B, K, Hh, Wh)
            - corners_xy: (B, K, 2)
            - corners_xy_orig: (B, K, 2)
            - valid_mask: (B, K)
            - camera_K: (B, 3, 3)
            - corners_3d: (B, K, 3)
            - orig_size: (B, 2)
            - sample_id: 长度为 B 的字符串列表
            - rgb_path: 长度为 B 的字符串列表

    变量维度分析
        batch:
            长度为 B 的列表。
        collated[key]:
            对张量字段使用 torch.stack(..., dim=0) 后形成批次维。
            例如 image: (3, H, W) -> (B, 3, H, W)。

    举例
        若 batch_size=4，则:
            collated["image"].shape 通常为 (4, 3, H, W)
            collated["sample_id"] 为长度 4 的字符串列表
    """
    collated: Dict[str, torch.Tensor] = {}  # collated表示: 批次级输出字典；键与单样本字段基本一致，但张量字段会增加 batch 维
    tensor_keys = ["image", "heatmaps", "corners_xy", "corners_xy_orig", "valid_mask", "camera_K", "corners_3d", "orig_size"]  # tensor_keys表示: 需要执行张量堆叠的字段名列表；长度为 8
    # * 接下来这段代码对所有张量字段逐个做 batch 维堆叠
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch], dim=0)  # 在第 0 维堆叠；若单样本为 (...)，则批次后为 (B, ...)
    # * 循环结束后，collated 中的张量字段均已增加 batch 维；维度通常从 (...) 变为 (B, ...)

    collated["sample_id"] = [item["sample_id"] for item in batch]  # collated["sample_id"]表示: 批次内样本 ID 列表；长度为 B；保留字符串不做张量化
    collated["rgb_path"] = [item["rgb_path"] for item in batch]  # collated["rgb_path"]表示: 批次内原图路径列表；长度为 B；便于后续可视化或错误追踪
    return collated