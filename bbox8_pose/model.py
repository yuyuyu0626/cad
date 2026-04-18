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


class BBox8PoseNet(nn.Module):
    def __init__(
        self,
        num_keypoints: int = 8,
        base_channels: int = 32,
        backbone: str = "resnet18",
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone

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

        self.lateral4 = nn.Conv2d(c4, 128, kernel_size=1)
        self.lateral3 = nn.Conv2d(c3, 128, kernel_size=1)
        self.lateral2 = nn.Conv2d(c2, 128, kernel_size=1)
        self.smooth3 = ConvBnRelu(128, 128)
        self.smooth2 = ConvBnRelu(128, 128)
        self.head = HeatmapHead(128, num_keypoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c2, c3, c4 = self.backbone(x)
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.smooth3(p3)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.smooth2(p2)
        return self.head(p2)
