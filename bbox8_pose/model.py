import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models as tv_models
except Exception:  # pragma: no cover - optional dependency at import time
    tv_models = None


class ConvBnRelu(nn.Module):
    """
    summary
        以 Conv2d + BatchNorm2d + ReLU 组成的基础卷积块，用于局部特征提取与通道变换。

    参数分析 (Args)
        in_channels: int，输入通道数，应为正整数。
        out_channels: int，输出通道数，应为正整数。

    返回值 (Returns)
        ConvBnRelu:
            一个可调用模块；输入特征图后输出相同空间分辨率、不同通道数的特征图。

    变量维度分析
        输入 x:
            形状为 (B, C_in, H, W)，其中:
            B: batch size
            C_in: 输入通道数
            H/W: 特征图高宽
        输出:
            形状为 (B, C_out, H, W)，空间尺寸保持不变。

    举例
        当 in_channels=64, out_channels=128 时:
            输入 x.shape = (8, 64, 32, 32)
            输出 y.shape = (8, 128, 32, 32)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        summary
            初始化基础卷积块，内部按顺序构建卷积、归一化与激活层。

        参数分析 (Args)
            in_channels: int，输入通道数。
            out_channels: int，输出通道数。

        返回值 (Returns)
            None:
                无显式返回值；主要副作用是创建 self.block 子模块。

        变量维度分析
            self.block:
                nn.Sequential 容器。
                Conv2d: (B, in_channels, H, W) -> (B, out_channels, H, W)
                BatchNorm2d: 保持形状不变
                ReLU: 保持形状不变

        举例
            若输入为 (B, 3, 256, 256)，且 out_channels=32，则卷积后输出为 (B, 32, 256, 256)。
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        summary
            对输入特征图执行卷积、批归一化和 ReLU 激活。

        参数分析 (Args)
            x: torch.Tensor，输入特征图，形状为 (B, C_in, H, W)。

        返回值 (Returns)
            torch.Tensor:
                输出特征图，形状为 (B, C_out, H, W)。

        变量维度分析
            x:
                (B, C_in, H, W)
            self.block(x):
                (B, C_out, H, W)

        举例
            输入 x.shape=(4, 64, 32, 32) 时，若本模块 out_channels=128，则输出为 (4, 128, 32, 32)。
        """
        return self.block(x)


class SimpleBackbone(nn.Module):
    """
    summary
        一个轻量级多阶段卷积骨干网络，逐步下采样并输出 3 个层级的特征图。

    参数分析 (Args)
        base_channels: int，基础通道数，默认值 32；后续各 stage 通道数按 2 倍增长。

    返回值 (Returns)
        SimpleBackbone:
            一个骨干网络模块；前向时返回三个尺度的特征图 (x2, x3, x4)。

    变量维度分析
        输入 x:
            (B, 3, H, W)
        输出 x2:
            (B, 2*base_channels, H/2, W/2)
        输出 x3:
            (B, 4*base_channels, H/4, W/4)
        输出 x4:
            (B, 8*base_channels, H/8, W/8)
        上述 H/W 是否能整除由输入尺寸决定；实际结果遵循 MaxPool2d 的下采样规则。

    举例
        当 base_channels=32，输入为 (2, 3, 256, 256) 时，输出通常为:
            x2: (2, 64, 128, 128)
            x3: (2, 128, 64, 64)
            x4: (2, 256, 32, 32)
    """

    def __init__(self, base_channels: int = 32) -> None:
        """
        summary
            构建轻量级卷积骨干网络，包括 stem 与 3 个逐级下采样阶段。

        参数分析 (Args)
            base_channels: int，初始卷积通道数，默认值 32。
                - stem 输出通道为 base_channels
                - stage1 输出通道为 2*base_channels
                - stage2 输出通道为 4*base_channels
                - stage3 输出通道为 8*base_channels

        返回值 (Returns)
            None:
                无显式返回值；主要副作用是创建 stem、stage1、stage2、stage3 四组子模块。

        变量维度分析
            stem:
                (B, 3, H, W) -> (B, base_channels, H, W)
            stage1:
                经 MaxPool2d(2) 下采样后输出 (B, 2*base_channels, H/2, W/2)
            stage2:
                输出 (B, 4*base_channels, H/4, W/4)
            stage3:
                输出 (B, 8*base_channels, H/8, W/8)

        举例
            若 base_channels=16，则各层输出通道依次为:
                stem: 16
                stage1: 32
                stage2: 64
                stage3: 128
        """
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnRelu(3, base_channels),
            ConvBnRelu(base_channels, base_channels),
        )
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBnRelu(base_channels, base_channels * 2),
            ConvBnRelu(base_channels * 2, base_channels * 2),
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBnRelu(base_channels * 2, base_channels * 4),
            ConvBnRelu(base_channels * 4, base_channels * 4),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBnRelu(base_channels * 4, base_channels * 8),
            ConvBnRelu(base_channels * 8, base_channels * 8),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        summary
            对输入图像执行逐级卷积与下采样，返回三个尺度的中高层特征图。

        参数分析 (Args)
            x: torch.Tensor，输入图像张量，形状通常为 (B, 3, H, W)。

        返回值 (Returns)
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x2: 第 1 个下采样阶段输出，形状通常为 (B, 2C, H/2, W/2)
                - x3: 第 2 个下采样阶段输出，形状通常为 (B, 4C, H/4, W/4)
                - x4: 第 3 个下采样阶段输出，形状通常为 (B, 8C, H/8, W/8)
                其中 C=base_channels。

        变量维度分析
            x1:
                stem 输出，(B, C, H, W)
            x2:
                stage1 输出，(B, 2C, H/2, W/2)
            x3:
                stage2 输出，(B, 4C, H/4, W/4)
            x4:
                stage3 输出，(B, 8C, H/8, W/8)

        举例
            输入 x.shape=(1, 3, 256, 256)，base_channels=32 时:
                x2.shape=(1, 64, 128, 128)
                x3.shape=(1, 128, 64, 64)
                x4.shape=(1, 256, 32, 32)
        """
        x1 = self.stem(x)  # x1表示: stem 提取后的浅层特征；维度为 (B, C, H, W)
        x2 = self.stage1(x1)  # x2表示: 第 1 个下采样阶段特征；(B, C, H, W) -> (B, 2C, H/2, W/2)
        x3 = self.stage2(x2)  # x3表示: 第 2 个下采样阶段特征；(B, 2C, H/2, W/2) -> (B, 4C, H/4, W/4)
        x4 = self.stage3(x3)  # x4表示: 第 3 个下采样阶段特征；(B, 4C, H/4, W/4) -> (B, 8C, H/8, W/8)
        return x2, x3, x4


class ResNetBackbone(nn.Module):
    """
    summary
        基于 torchvision ResNet 的骨干网络包装器，输出三个中高层特征图供后续解码器使用。

    参数分析 (Args)
        name: str，ResNet 结构名称，仅支持 "resnet18" 或 "resnet34"。
        pretrained: bool，是否加载 torchvision 预训练权重；默认值 False。

    返回值 (Returns)
        ResNetBackbone:
            一个骨干网络模块；前向时返回 (c2, c3, c4) 三个层级特征图。

    变量维度分析
        输入 x:
            (B, 3, H, W)
        输出 c2:
            通常为 (B, 64, H/4, W/4)
        输出 c3:
            通常为 (B, 128, H/8, W/8)
        输出 c4:
            通常为 (B, 256, H/16, W/16)
        具体空间分辨率受 conv1 + maxpool + layer2/layer3 的步长影响。

    举例
        当输入为 (2, 3, 256, 256) 时，resnet18 / resnet34 一般输出:
            c2: (2, 64, 64, 64)
            c3: (2, 128, 32, 32)
            c4: (2, 256, 16, 16)
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = False) -> None:
        """
        summary
            初始化指定的 ResNet 骨干网络，并抽取前几层作为多尺度特征提取器。

        参数分析 (Args)
            name: str，支持 "resnet18" 或 "resnet34"。
            pretrained: bool，是否启用 torchvision 默认预训练权重。
                当 torchvision 版本较新时尝试使用 weights 参数；
                否则退回旧版 pretrained 参数接口。

        返回值 (Returns)
            None:
                无显式返回值；主要副作用是构建 conv1、bn1、relu、maxpool、layer1、layer2、layer3 等成员。

        变量维度分析
            self.out_channels:
                一个 3 元组，表示 c2/c3/c4 的通道数；当前 resnet18 与 resnet34 都设为 (64, 128, 256)。

        举例
            当 name="resnet18" 且 pretrained=True 时，会尽可能加载 torchvision 的默认 ImageNet 预训练权重。
        """
        super().__init__()
        if tv_models is None:
            raise ImportError("torchvision is required for ResNetBackbone")
        if name not in {"resnet18", "resnet34"}:
            raise ValueError(f"Unsupported resnet backbone: {name}")

        builder = getattr(tv_models, name)
        # getattr指的是: Python 内置函数；用于动态获取模块或对象的属性；这里用于从 torchvision.models 模块中获取指定名称的 ResNet 构建函数。
        # tv_models 是 torchvision.models 模块；getattr(tv_models, name) 等价于 tv_models.resnet18 或 tv_models.resnet34；返回一个函数接口，用于构建指定 ResNet 模型实例
        # builder指的是: torchvision.models.resnet18 或 torchvision.models.resnet34；一个函数接口，用于构建指定 ResNet 模型实例
        try:
            weights = "DEFAULT" if pretrained else None  # weights表示: torchvision 新接口的权重配置；预训练时为默认权重，否则为 None
            net = builder(weights=weights)
        except Exception:
            # Fallback for older torchvision APIs
            net = builder(pretrained=pretrained)

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        if name == "resnet18":
            self.out_channels = (64, 128, 256)  # self.out_channels表示: c2/c3/c4 的通道数配置；3 元组
        else:
            self.out_channels = (64, 128, 256)  # self.out_channels表示: c2/c3/c4 的通道数配置；resnet34 在此实现下与 resnet18 相同

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        summary
            对输入图像执行 ResNet 前几层前向传播，并返回 3 个尺度的特征图。

        参数分析 (Args)
            x: torch.Tensor，输入图像张量，形状通常为 (B, 3, H, W)。

        返回值 (Returns)
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - c2: layer1 输出
                - c3: layer2 输出
                - c4: layer3 输出
                三者分别用于后续 FPN 或 Transformer 风格解码器。

        变量维度分析
            x 经 conv1+bn1+relu+maxpool 后:
                通常从 (B, 3, H, W) 变为 (B, 64, H/4, W/4)
            c2:
                (B, 64, H/4, W/4)
            c3:
                (B, 128, H/8, W/8)
            c4:
                (B, 256, H/16, W/16)

        举例
            输入 x.shape=(4, 3, 256, 256) 时，典型输出为:
                c2.shape=(4, 64, 64, 64)
                c3.shape=(4, 128, 32, 32)
                c4.shape=(4, 256, 16, 16)
        """
        x = self.conv1(x)  # 卷积下采样；(B, 3, H, W) -> (B, 64, H/2, W/2)
        x = self.bn1(x)  # 批归一化；形状不变；train/eval 模式下统计量来源不同
        x = self.relu(x)  # 激活；形状不变
        x = self.maxpool(x)  # 最大池化；(B, 64, H/2, W/2) -> (B, 64, H/4, W/4)
        c2 = self.layer1(x)  # c2表示: 第 1 组残差层输出；维度通常为 (B, 64, H/4, W/4)
        c3 = self.layer2(c2)  # c3表示: 第 2 组残差层输出；(B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        c4 = self.layer3(c3)  # c4表示: 第 3 组残差层输出；(B, 128, H/8, W/8) -> (B, 256, H/16, W/16)
        return c2, c3, c4


class HeatmapHead(nn.Module):
    """
    summary
        将解码后的高层特征映射为关键点热图输出的预测头。

    参数分析 (Args)
        in_channels: int，输入特征通道数。
        num_keypoints: int，关键点数量，也是输出热图通道数。

    返回值 (Returns)
        HeatmapHead:
            一个热图预测头模块；前向时输出每个关键点对应的一张热图。

    变量维度分析
        输入 x:
            (B, C_in, H, W)
        输出:
            (B, K, H, W)，其中 K=num_keypoints

    举例
        若输入特征为 (8, 192, 64, 64)，num_keypoints=8，则输出为 (8, 8, 64, 64)。
    """

    def __init__(self, in_channels: int, num_keypoints: int) -> None:
        """
        summary
            初始化热图预测头，先通过两层卷积块细化特征，再用 1x1 卷积映射到关键点通道。

        参数分析 (Args)
            in_channels: int，输入通道数。
            num_keypoints: int，关键点数量，对应输出通道数。

        返回值 (Returns)
            None:
                无显式返回值；主要副作用是创建 self.block。

        变量维度分析
            self.block:
                ConvBnRelu(in_channels, 128)
                ConvBnRelu(128, 64)
                Conv2d(64, num_keypoints, kernel_size=1)
            最终输出形状为 (B, num_keypoints, H, W)。

        举例
            当 in_channels=128, num_keypoints=8 时，输出热图通道数为 8。
        """
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_channels, 128),
            ConvBnRelu(128, 64),
            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        summary
            将输入特征图变换为关键点热图。

        参数分析 (Args)
            x: torch.Tensor，输入特征图，形状为 (B, C_in, H, W)。

        返回值 (Returns)
            torch.Tensor:
                热图张量，形状为 (B, K, H, W)。

        变量维度分析
            x:
                (B, C_in, H, W)
            self.block(x):
                (B, K, H, W)

        举例
            输入 x.shape=(2, 128, 64, 64)，K=8 时，输出为 (2, 8, 64, 64)。
        """
        return self.block(x)


def build_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    summary
        构建二维正弦余弦位置编码，并以 token 序列形式返回。

    参数分析 (Args)
        height: int，特征图高度 H，应为正整数。
        width: int，特征图宽度 W，应为正整数。
        dim: int，位置编码维度 D，必须能被 4 整除。
        device: torch.device，输出张量所在设备。
        dtype: torch.dtype，输出张量数据类型。

    返回值 (Returns)
        torch.Tensor:
            位置编码张量，形状为 (1, H*W, D)。
            第 1 维是 batch 维占位，便于与 token 序列按广播方式相加。

    变量维度分析
        y, x:
            由 meshgrid 生成的二维网格坐标，形状均为 (H, W)。
        omega:
            频率向量，形状为 (D/4,)。
        y / x 在 reshape 后:
            (H*W, 1) 与 (1, D/4) 广播相乘，得到 (H*W, D/4)。
        pos:
            由 sin/cos 拼接后得到 (H*W, D)。
        pos.unsqueeze(0):
            (1, H*W, D)。

    举例
        当 height=8, width=8, dim=192 时:
            返回位置编码形状为 (1, 64, 192)。
    """
    if dim % 4 != 0:
        raise ValueError(f"Positional embedding dim must be divisible by 4, got {dim}")

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=dtype)  # omega表示: 位置编码频率基；维度为 (D/4,)
    omega = 1.0 / (10000 ** (omega / max(1, dim // 4 - 1)))  #* 将频率索引映射为不同尺度的周期，避免全部通道编码频率相同

    y = y.reshape(-1, 1) * omega.reshape(1, -1)  # 展平并广播相乘；(H, W) -> (H*W, 1) 与 (1, D/4) 得到 (H*W, D/4)；编码纵向位置信息
    x = x.reshape(-1, 1) * omega.reshape(1, -1)  # 展平并广播相乘；(H, W) -> (H*W, 1) 与 (1, D/4) 得到 (H*W, D/4)；编码横向位置信息
    pos = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)  # 特征维拼接；4 个 (H*W, D/4) -> (H*W, D)；融合 x/y 的正余弦位置编码
    return pos.unsqueeze(0)  # 增加 batch 维；(H*W, D) -> (1, H*W, D)


class FeedForward(nn.Module):
    """
    summary
        Transformer 风格的前馈网络，对每个 token 独立执行两层 MLP 变换。

    参数分析 (Args)
        dim: int，输入与输出 token 维度 D。
        mlp_ratio: float，隐藏层维度扩张倍率，默认值 4.0。
        dropout: float，Dropout 概率，默认值 0.0；仅在训练模式下生效。

    返回值 (Returns)
        FeedForward:
            一个前馈网络模块；保持 token 数不变，仅在特征维上变换。

    变量维度分析
        输入 x:
            (B, N, D)
        hidden_dim:
            int(dim * mlp_ratio)
        输出:
            (B, N, D)

    举例
        若 dim=192, mlp_ratio=4，则 hidden_dim=768。
        输入 x.shape=(2, 1024, 192) 时，输出仍为 (2, 1024, 192)。
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """
        summary
            初始化前馈网络，包括线性扩维、GELU 激活、Dropout 和线性回投影。

        参数分析 (Args)
            dim: int，输入/输出维度。
            mlp_ratio: float，隐藏层扩张倍率。
            dropout: float，Dropout 比例。

        返回值 (Returns)
            None:
                无显式返回值；主要副作用是构建 self.net。

        变量维度分析
            self.net:
                Linear(D, D_hidden) -> GELU -> Dropout -> Linear(D_hidden, D) -> Dropout
            对输入 (B, N, D) 保持 batch 和 token 维不变。

        举例
            当 dim=128, mlp_ratio=2 时，隐藏层维度为 256。
        """
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)  # hidden_dim表示: 前馈网络隐藏层维度；标量整数；控制 MLP 容量
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        summary
            对输入 token 序列执行逐 token 的前馈变换。

        参数分析 (Args)
            x: torch.Tensor，token 序列，形状为 (B, N, D)。

        返回值 (Returns)
            torch.Tensor:
                变换后的 token 序列，形状仍为 (B, N, D)。

        变量维度分析
            x:
                (B, N, D)
            self.net(x):
                (B, N, D)

        举例
            输入 x.shape=(1, 64, 192) 时，输出仍为 (1, 64, 192)。
        """
        return self.net(x)


class TransformerDecoderBlock(nn.Module):
    """
    summary
        一个简化的 Transformer 解码块，依次执行自注意力、交叉注意力和前馈网络，并采用残差连接。

    参数分析 (Args)
        dim: int，token 特征维度 D。
        num_heads: int，多头注意力头数，默认值 8。
        mlp_ratio: float，前馈网络隐藏层扩张倍率，默认值 4.0。
        dropout: float，注意力与 MLP 中的 Dropout 概率，默认值 0.0。

    返回值 (Returns)
        TransformerDecoderBlock:
            一个可复用的解码块模块；输入查询序列 q 和 memory，输出更新后的 q。

    变量维度分析
        q:
            (B, Nq, D)，查询 token 序列。
        memory:
            (B, Nm, D)，记忆 token 序列。
        输出:
            (B, Nq, D)，token 数与维度与输入 q 保持一致。

    举例
        当 q.shape=(2, 1024, 192)，memory.shape=(2, 80, 192) 时，
        输出形状仍为 (2, 1024, 192)。
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        """
        summary
            初始化解码块中的层归一化、自注意力、交叉注意力和前馈网络模块。

        参数分析 (Args)
            dim: int，特征维度 D。
            num_heads: int，多头注意力头数。
            mlp_ratio: float，前馈网络扩张倍率。
            dropout: float，Dropout 比例。

        返回值 (Returns)
            None:
                无显式返回值；主要副作用是创建 norm、attention 与 ffn 模块。

        变量维度分析
            self.self_attn:
                MultiheadAttention，输入/输出均为 (B, Nq, D)。
            self.cross_attn:
                MultiheadAttention，query 为 (B, Nq, D)，key/value 为 (B, Nm, D)，输出 (B, Nq, D)。
            self.ffn:
                输入/输出均为 (B, Nq, D)。

        举例
            当 dim=192, num_heads=8 时，每个头的理论特征维约为 24。
        """
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        #! MultiheadAttention: PyTorch 多头注意力模块；输入 query/key/value 序列并输出注意力聚合后的序列表示
        self.self_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_q3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        summary
            对查询序列先做自注意力，再与 memory 做交叉注意力，最后经前馈网络细化。

        参数分析 (Args)
            q: torch.Tensor，查询 token 序列，形状为 (B, Nq, D)。
            memory: torch.Tensor，记忆 token 序列，形状为 (B, Nm, D)。

        返回值 (Returns)
            torch.Tensor:
                更新后的查询 token 序列，形状为 (B, Nq, D)。

        变量维度分析
            q_norm:
                与 q 同形状，为 LayerNorm 后的查询。
            self_attn 输出:
                (B, Nq, D)
            cross_attn 输出:
                (B, Nq, D)
            ffn 输出:
                (B, Nq, D)

        举例
            输入 q.shape=(4, 4096, 192)，memory.shape=(4, 80, 192) 时，
            输出仍为 (4, 4096, 192)。
        """
        q_norm = self.norm_q1(q)  # q_norm表示: 自注意力前归一化后的查询序列；维度为 (B, Nq, D)
        q = q + self.self_attn(q_norm, q_norm, q_norm, need_weights=False)[0]  # 残差自注意力；输入/输出均为 (B, Nq, D)；只取注意力输出，不保留权重
        q_norm = self.norm_q2(q)  # q_norm表示: 交叉注意力前归一化后的查询序列；维度为 (B, Nq, D)
        q = q + self.cross_attn(q_norm, memory, memory, need_weights=False)[0]  # 残差交叉注意力；query=(B, Nq, D), key/value=(B, Nm, D) -> 输出 (B, Nq, D)
        q = q + self.ffn(self.norm_q3(q))  # 残差前馈网络；(B, Nq, D) -> (B, Nq, D)；进一步做逐 token 非线性映射
        return q


class BoxDreamerLiteDecoder(nn.Module):
    """
    summary
        结合 FPN 式多尺度融合与轻量 Transformer 解码的特征解码器，用于输出更强的高分辨率关键点表征。

    参数分析 (Args)
        c2_channels: int，来自骨干网络 c2 特征图的通道数。
        c3_channels: int，来自骨干网络 c3 特征图的通道数。
        c4_channels: int，来自骨干网络 c4 特征图的通道数。
        dim: int，统一投影后的特征维度，默认值 192。
        depth: int，Transformer 解码块层数，默认值 3。
        num_heads: int，多头注意力头数，默认值 8。
        dropout: float，Transformer 中 Dropout 概率，默认值 0.0。

    返回值 (Returns)
        BoxDreamerLiteDecoder:
            一个解码器模块；输入三个尺度特征图，输出融合后的高分辨率特征图。

    变量维度分析
        输入 c2:
            (B, C2, H2, W2)
        输入 c3:
            (B, C3, H3, W3)
        输入 c4:
            (B, C4, H4, W4)
        p2 / p3 / p4:
            统一到通道 dim 后的多尺度特征图。
        q:
            来自 p2 的查询 token，形状为 (B, H2*W2, dim)
        memory:
            来自池化后的 p3/p4 token 拼接，形状为 (B, Nmem, dim)
        输出:
            (B, dim, H2, W2)

    举例
        若 c2.shape=(2, 64, 64, 64), c3.shape=(2, 128, 32, 32), c4.shape=(2, 256, 16, 16)，
        且 dim=192，则输出特征通常为 (2, 192, 64, 64)。
    """

    def __init__(
        self,
        c2_channels: int,
        c3_channels: int,
        c4_channels: int,
        dim: int = 192,
        depth: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """
        summary
            初始化多尺度投影层、平滑卷积层、记忆池化层、Transformer 块和输出细化层。

        参数分析 (Args)
            c2_channels: int，c2 输入通道数。
            c3_channels: int，c3 输入通道数。
            c4_channels: int，c4 输入通道数。
            dim: int，统一特征维度。
            depth: int，Transformer 解码层数。
            num_heads: int，多头注意力头数。
            dropout: float，Dropout 概率。

        返回值 (Returns)
            None:
                无显式返回值；主要副作用是构造整个解码器的所有子模块。

        变量维度分析
            proj2/proj3/proj4:
                1x1 卷积，将输入通道统一映射到 dim。
            memory_pool3:
                将 p3 池化到 (8, 8)。
            memory_pool4:
                将 p4 池化到 (4, 4)。
            blocks:
                长度为 depth 的 TransformerDecoderBlock 列表。
            out:
                输出细化卷积块，保持 (B, dim, H2, W2)。

        举例
            当 depth=3 时，q 会依次经过 3 个解码块进行更新。
        """
        super().__init__()
        self.dim = dim  # self.dim表示: 解码器统一特征维度；标量整数；后续 token 与输出通道都以此为准
        self.proj4 = nn.Conv2d(c4_channels, dim, kernel_size=1)
        self.proj3 = nn.Conv2d(c3_channels, dim, kernel_size=1)
        self.proj2 = nn.Conv2d(c2_channels, dim, kernel_size=1)
        self.smooth3 = ConvBnRelu(dim, dim)
        self.smooth2 = ConvBnRelu(dim, dim)
        self.memory_pool3 = nn.AdaptiveAvgPool2d((8, 8))
        self.memory_pool4 = nn.AdaptiveAvgPool2d((4, 4))
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout) for _ in range(depth)]
        )
        self.out = nn.Sequential(
            ConvBnRelu(dim, dim),
            ConvBnRelu(dim, dim),
        )

    def _flatten_with_pos(self, feat: torch.Tensor) -> torch.Tensor:
        """
        summary
            将二维特征图展平为 token 序列，并加上二维正弦余弦位置编码。

        参数分析 (Args)
            feat: torch.Tensor，输入特征图，形状为 (B, C, H, W)。

        返回值 (Returns)
            torch.Tensor:
                加入位置编码后的 token 序列，形状为 (B, H*W, C)。

        变量维度分析
            feat:
                (B, C, H, W)
            tokens:
                flatten + transpose 后为 (B, H*W, C)
            pos:
                (1, H*W, C)
            tokens + pos:
                广播相加后为 (B, H*W, C)

        举例
            输入 feat.shape=(2, 192, 8, 8) 时，输出为 (2, 64, 192)。
        """
        b, c, h, w = feat.shape  # b/c/h/w表示: 输入特征图的 batch、大通道、高、宽；均由运行时决定
        tokens = feat.flatten(2).transpose(1, 2)  # 展平空间维并交换维度顺序；(B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)；每个空间位置变为一个 token
        pos = build_2d_sincos_pos_embed(h, w, c, feat.device, feat.dtype)  # pos表示: 当前位置的二维正余弦位置编码；维度为 (1, H*W, C)
        return tokens + pos  # 广播相加；(B, H*W, C) + (1, H*W, C) -> (B, H*W, C)

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        """
        summary
            先做 FPN 风格自顶向下融合，再把高分辨率特征作为查询、低分辨率特征作为记忆送入 Transformer 细化。

        参数分析 (Args)
            c2: torch.Tensor，骨干网络较高分辨率特征，形状为 (B, C2, H2, W2)。
            c3: torch.Tensor，中等分辨率特征，形状为 (B, C3, H3, W3)。
            c4: torch.Tensor，较低分辨率高语义特征，形状为 (B, C4, H4, W4)。

        返回值 (Returns)
            torch.Tensor:
                解码后的高分辨率特征图，形状为 (B, dim, H2, W2)。

        变量维度分析
            p4:
                (B, dim, H4, W4)
            p3:
                (B, dim, H3, W3)
            p2:
                (B, dim, H2, W2)
            q:
                (B, H2*W2, dim)
            mem3:
                (B, 64, dim)，因为 8*8=64
            mem4:
                (B, 16, dim)，因为 4*4=16
            memory:
                (B, 80, dim)
            q_map:
                (B, dim, H2, W2)

        举例
            若 p2 的空间大小为 64x64，则 q 的 token 数为 4096；
            memory 由 64+16=80 个 token 组成。
        """
        p4 = self.proj4(c4)  # p4表示: 将 c4 通道投影到 dim 后的低分辨率语义特征；(B, C4, H4, W4) -> (B, dim, H4, W4)
        p3 = self.proj3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)  # 顶层特征上采样并与 c3 融合；(B, dim, H4, W4) -> (B, dim, H3, W3)，再与 proj3(c3) 相加得到 (B, dim, H3, W3)
        p3 = self.smooth3(p3)  # p3表示: 平滑后的中尺度融合特征；维度保持 (B, dim, H3, W3)
        p2 = self.proj2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)  # 将 p3 上采样到 c2 分辨率并相加；得到高分辨率融合特征 (B, dim, H2, W2)
        p2 = self.smooth2(p2)  # p2表示: 平滑后的高分辨率融合特征；维度保持 (B, dim, H2, W2)

        q = self._flatten_with_pos(p2)  # q表示: 由高分辨率特征展开得到的查询 token；(B, dim, H2, W2) -> (B, H2*W2, dim)
        mem3 = self._flatten_with_pos(self.memory_pool3(p3))  # 对 p3 自适应池化到 8x8 再展开；(B, dim, H3, W3) -> (B, dim, 8, 8) -> (B, 64, dim)
        mem4 = self._flatten_with_pos(self.memory_pool4(p4))  # 对 p4 自适应池化到 4x4 再展开；(B, dim, H4, W4) -> (B, dim, 4, 4) -> (B, 16, dim)
        memory = torch.cat([mem3, mem4], dim=1)  # token 维拼接记忆序列；(B, 64, dim) + (B, 16, dim) -> (B, 80, dim)；融合中低分辨率上下文

        # * 接下来这段代码使用多个 Transformer 解码块迭代更新查询序列 q
        for block in self.blocks:
            q = block(q, memory)  # q表示: 经当前解码块更新后的查询 token；维度始终为 (B, H2*W2, dim)
        # * 循环结束后，q 表示融合了 memory 全局上下文后的高分辨率 token 序列；维度为 (B, H2*W2, dim)

        b, _, h, w = p2.shape  # b/h/w表示: 目标重排回特征图时所需的 batch 与空间尺寸；均来自高分辨率特征 p2
        q_map = q.transpose(1, 2).reshape(b, self.dim, h, w)  # 先交换 token/通道维，再恢复为空间特征图；(B, H2*W2, dim) -> (B, dim, H2*W2) -> (B, dim, H2, W2)
        return self.out(q_map)  # 输出细化；(B, dim, H2, W2) -> (B, dim, H2, W2)


class BBox8PoseNet(nn.Module):
    """
    summary
        面向 8 个关键点热图预测的主网络，支持 simple/ResNet 骨干以及 FPN/BoxDreamerLite 两类解码器。

    参数分析 (Args)
        num_keypoints: int，关键点数量，默认值 8。
        base_channels: int，仅在 simple backbone 下使用的基础通道数，默认值 32。
        backbone: str，骨干网络类型，支持 "simple"、"resnet18"、"resnet34"。
        pretrained_backbone: bool，是否为 ResNet 骨干加载预训练权重。
        decoder: str，解码器类型，支持 "fpn"、"boxdreamer_lite"。
        decoder_dim: int，当 decoder="boxdreamer_lite" 时的统一特征维度，默认值 192。
        decoder_depth: int，当 decoder="boxdreamer_lite" 时 Transformer 解码层数，默认值 3。
        decoder_heads: int，当 decoder="boxdreamer_lite" 时多头注意力头数，默认值 8。

    返回值 (Returns)
        BBox8PoseNet:
            一个完整关键点热图预测网络；前向时输入图像，输出关键点热图。

    变量维度分析
        输入 x:
            (B, 3, H, W)
        backbone 输出:
            c2, c3, c4 三个尺度特征图
        neck 输出 feat:
            若 decoder="fpn"，通常为 (B, 128, H2, W2)
            若 decoder="boxdreamer_lite"，通常为 (B, decoder_dim, H2, W2)
        head 输出:
            (B, num_keypoints, H2, W2)

    举例
        当输入为 (4, 3, 256, 256)、backbone="resnet18"、decoder="boxdreamer_lite"、num_keypoints=8 时，
        输出热图通常为 (4, 8, 64, 64)。
    """

    def __init__(
        self,
        num_keypoints: int = 8,
        base_channels: int = 32,
        backbone: str = "resnet18",
        pretrained_backbone: bool = False,
        decoder: str = "boxdreamer_lite",
        decoder_dim: int = 192,
        decoder_depth: int = 3,
        decoder_heads: int = 8,
    ) -> None:
        """
        summary
            根据配置构建骨干网络、解码器 neck 和热图预测头，形成完整模型。

        参数分析 (Args)
            num_keypoints: int，输出热图通道数。
            base_channels: int，simple backbone 的基础通道数。
            backbone: str，骨干网络类型。
            pretrained_backbone: bool，ResNet 是否使用预训练权重。
            decoder: str，解码器类型。
            decoder_dim: int，BoxDreamerLite 的统一特征维度。
            decoder_depth: int，BoxDreamerLite 的 Transformer 层数。
            decoder_heads: int，BoxDreamerLite 的注意力头数。

        返回值 (Returns)
            None:
                无显式返回值；主要副作用是按配置创建 self.backbone、self.neck 和 self.head。

        变量维度分析
            c2/c3/c4:
                记录骨干输出通道数的中间标量配置。
            head_in_channels:
                热图头输入通道数；FPN 情况下为 128，BoxDreamerLite 情况下为 decoder_dim。
            self.backbone_name / self.decoder_name:
                字符串配置缓存，用于 forward 阶段分支选择。

        举例
            若 backbone="simple", base_channels=32, decoder="fpn"，则:
                c2=64, c3=128, c4=256, head_in_channels=128。
        """
        super().__init__()
        self.backbone_name = backbone
        self.decoder_name = decoder

        if backbone == "simple":
            self.backbone = SimpleBackbone(base_channels=base_channels)
            c2 = base_channels * 2
            c3 = base_channels * 4
            c4 = base_channels * 8
        elif backbone in {"resnet18", "resnet34"}:
            self.backbone = ResNetBackbone(name=backbone, pretrained=pretrained_backbone)
            c2, c3, c4 = self.backbone.out_channels
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if decoder == "fpn":
            self.neck = nn.ModuleDict(
                {
                    "lateral4": nn.Conv2d(c4, 128, kernel_size=1),
                    "lateral3": nn.Conv2d(c3, 128, kernel_size=1),
                    "lateral2": nn.Conv2d(c2, 128, kernel_size=1),
                    "smooth3": ConvBnRelu(128, 128),
                    "smooth2": ConvBnRelu(128, 128),
                }
            )
            head_in_channels = 128
        elif decoder == "boxdreamer_lite":
            self.neck = BoxDreamerLiteDecoder(
                c2_channels=c2,
                c3_channels=c3,
                c4_channels=c4,
                dim=decoder_dim,
                depth=decoder_depth,
                num_heads=decoder_heads,
            )
            head_in_channels = decoder_dim
        else:
            raise ValueError(f"Unsupported decoder: {decoder}")
        self.head = HeatmapHead(head_in_channels, num_keypoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        summary
            执行完整前向传播：骨干提取多尺度特征，neck 融合解码，head 输出关键点热图。

        参数分析 (Args)
            x: torch.Tensor，输入图像张量，形状通常为 (B, 3, H, W)。

        返回值 (Returns)
            torch.Tensor:
                关键点热图，形状通常为 (B, K, H_out, W_out)，其中 K=num_keypoints。

        变量维度分析
            c2/c3/c4:
                backbone 输出的多尺度特征图。
            若 decoder_name == "fpn":
                p4:
                    (B, 128, H4, W4)
                p3:
                    (B, 128, H3, W3)
                p2/feat:
                    (B, 128, H2, W2)
            若 decoder_name != "fpn":
                feat:
                    通常为 (B, decoder_dim, H2, W2)
            self.head(feat):
                (B, K, H2, W2)

        举例
            输入 x.shape=(2, 3, 256, 256)，backbone=resnet18, decoder=fpn 时，
            输出热图一般为 (2, 8, 64, 64)。
        """
        c2, c3, c4 = self.backbone(x)  # c2/c3/c4表示: 骨干网络提取的三层多尺度特征图；空间分辨率逐级降低、语义逐级增强
        if self.decoder_name == "fpn":
            p4 = self.neck["lateral4"](c4)  # p4表示: c4 经过 1x1 侧向投影后的特征；(B, C4, H4, W4) -> (B, 128, H4, W4)
            p3 = self.neck["lateral3"](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)  # FPN 融合；上采样 p4 到 c3 分辨率并相加，得到 (B, 128, H3, W3)
            p3 = self.neck["smooth3"](p3)  # p3表示: 平滑后的中尺度融合特征；维度保持 (B, 128, H3, W3)
            p2 = self.neck["lateral2"](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)  # 再次上采样并与 c2 融合；得到 (B, 128, H2, W2)
            feat = self.neck["smooth2"](p2)  # feat表示: FPN 最终高分辨率特征；维度为 (B, 128, H2, W2)
        else:
            feat = self.neck(c2, c3, c4)  # feat表示: BoxDreamerLite 解码器输出的融合特征；维度通常为 (B, decoder_dim, H2, W2)
        return self.head(feat)  # 关键点热图预测；(B, C_feat, H2, W2) -> (B, K, H2, W2)