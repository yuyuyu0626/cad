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
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleBackbone(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
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
        x1 = self.stem(x)
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        return x2, x3, x4


class ResNetBackbone(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = False) -> None:
        super().__init__()
        if tv_models is None:
            raise ImportError("torchvision is required for ResNetBackbone")
        if name not in {"resnet18", "resnet34"}:
            raise ValueError(f"Unsupported resnet backbone: {name}")

        builder = getattr(tv_models, name)
        try:
            weights = "DEFAULT" if pretrained else None
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
            self.out_channels = (64, 128, 256)
        else:
            self.out_channels = (64, 128, 256)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        return c2, c3, c4


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int, num_keypoints: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_channels, 128),
            ConvBnRelu(128, 64),
            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def build_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"Positional embedding dim must be divisible by 4, got {dim}")

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(1, dim // 4 - 1)))

    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    return pos.unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_q3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q1(q)
        q = q + self.self_attn(q_norm, q_norm, q_norm, need_weights=False)[0]
        q_norm = self.norm_q2(q)
        q = q + self.cross_attn(q_norm, memory, memory, need_weights=False)[0]
        q = q + self.ffn(self.norm_q3(q))
        return q


class BoxDreamerLiteDecoder(nn.Module):
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
        super().__init__()
        self.dim = dim
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
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        pos = build_2d_sincos_pos_embed(h, w, c, feat.device, feat.dtype)
        return tokens + pos

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        p4 = self.proj4(c4)
        p3 = self.proj3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)
        p2 = self.proj2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)

        q = self._flatten_with_pos(p2)
        mem3 = self._flatten_with_pos(self.memory_pool3(p3))
        mem4 = self._flatten_with_pos(self.memory_pool4(p4))
        memory = torch.cat([mem3, mem4], dim=1)

        for block in self.blocks:
            q = block(q, memory)

        b, _, h, w = p2.shape
        q_map = q.transpose(1, 2).reshape(b, self.dim, h, w)
        return self.out(q_map)


class BBox8PoseNet(nn.Module):
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
        c2, c3, c4 = self.backbone(x)
        if self.decoder_name == "fpn":
            p4 = self.neck["lateral4"](c4)
            p3 = self.neck["lateral3"](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
            p3 = self.neck["smooth3"](p3)
            p2 = self.neck["lateral2"](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
            feat = self.neck["smooth2"](p2)
        else:
            feat = self.neck(c2, c3, c4)
        return self.head(feat)
