"""
vision/backbone_resnet.py — ResNet 骨干网络支持
借鉴论文《大型船舶钢板面高反光工况2D视觉检测技术调研报告》

论文要点：
  "深度学习算法：利用 ResNet-18 和 Faster R-CNN 等模型增强边缘检测能力"

本模块为 FLARE 网络提供可选的 ResNet-18/34/50 编码器骨干，
替代原有的 U-Net 编码器，便于与论文中的方法进行对比实验。

用法::

    from vision.backbone_resnet import ResNetBackbone
    from vision.feature_extraction import FLARE

    # 使用 ResNet-18 骨干构建 FLARE
    backbone = ResNetBackbone(variant="resnet18", pretrained=True)
    model = FLARE(backbone=backbone, in_channels=3, base_ch=64)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models

__all__ = [
    "ResNetBackbone",
    "ResNetEncoder",
    "build_resnet_backbone",
]


# ============================================================
# 1. ResNet 编码器（输出多尺度特征，适配 U-Net 解码器）
# ============================================================

class ResNetEncoder(nn.Module):
    """
    ResNet 编码器，输出 4 级特征图（1/4, 1/8, 1/16, 1/32），
    与 U-Net 解码器的跳跃连接完全兼容。
    """

    def __init__(self, variant: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.variant = variant

        if variant == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            self.channels = [64, 64, 128, 256, 512]
        elif variant == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.channels = [64, 64, 128, 256, 512]
        elif variant == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        # 拆分 ResNet 阶段
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            features: [stem_out, layer1_out, layer2_out, layer3_out, layer4_out]
                      对应尺度 [1/2, 1/4, 1/8, 1/16, 1/32]
        """
        x0 = self.stem(x)      # 1/2
        x1 = self.layer1(x0)   # 1/4
        x2 = self.layer2(x1)   # 1/8
        x3 = self.layer3(x2)   # 1/16
        x4 = self.layer4(x3)   # 1/32
        return [x0, x1, x2, x3, x4]

    def get_channels(self) -> List[int]:
        """返回每级特征图的通道数"""
        return self.channels


# ============================================================
# 2. ResNetBackbone — 包装类（与 FLARE 编码器接口对齐）
# ============================================================

class ResNetBackbone(nn.Module):
    """
    ResNet 骨干包装器，提供与 FLARE 原有编码器一致的接口。

    特点：
      - 支持 ResNet-18/34/50
      - ImageNet 预训练权重加速收敛
      - 冻结早期层（layer1 之前）可做微调策略
    """

    def __init__(
        self,
        variant: str = "resnet18",
        pretrained: bool = True,
        freeze_stem: bool = False,
        freeze_layer1: bool = False,
    ):
        super().__init__()
        self.encoder = ResNetEncoder(variant=variant, pretrained=pretrained)
        self._channels = self.encoder.get_channels()

        # 冻结策略（用于微调）
        if freeze_stem:
            for p in self.encoder.stem.parameters():
                p.requires_grad = False
        if freeze_layer1:
            for p in self.encoder.layer1.parameters():
                p.requires_grad = False

    @property
    def base_ch(self) -> int:
        """返回 stem 输出通道数（兼容 FLARE 配置）"""
        return self._channels[0]

    @property
    def channel_stages(self) -> List[int]:
        """返回各阶段通道数 [stem, layer1, layer2, layer3, layer4]"""
        return self._channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)

    def get_skip_channels(self) -> List[int]:
        """
        返回用于 U-Net 跳跃连接的通道数。
        解码器阶段 0-3 分别对应 encoder 的 layer1-layer4。
        """
        return self._channels[1:5]


# ============================================================
# 3. 构建函数
# ============================================================

def build_resnet_backbone(
    variant: str = "resnet18",
    pretrained: bool = True,
    **kwargs,
) -> ResNetBackbone:
    """工厂函数：构建 ResNet 骨干"""
    return ResNetBackbone(variant=variant, pretrained=pretrained, **kwargs)


# ============================================================
# 4. 与 FLARE 集成的适配层（可选）
# ============================================================

class ResNetFLAREAdapter(nn.Module):
    """
    将 ResNet 编码器与 FLARE 解码器桥接的适配层。
    处理通道数不匹配问题（ResNet-50 的 layer4 输出 2048 通道，而 FLARE 可能只需要 512）。
    """

    def __init__(self, in_channels: List[int], out_channels: List[int]):
        super().__init__()
        assert len(in_channels) == len(out_channels), "通道列表长度必须一致"
        self.projects = nn.ModuleList([
            nn.Conv2d(ic, oc, 1, bias=False) if ic != oc else nn.Identity()
            for ic, oc in zip(in_channels, out_channels)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [proj(f) for proj, f in zip(self.projects, features)]


# ============================================================
# 5. 简单的对比实验工具
# ============================================================

if __name__ == "__main__":
    # 快速验证
    for variant in ["resnet18", "resnet34", "resnet50"]:
        backbone = ResNetBackbone(variant=variant, pretrained=False)
        x = torch.randn(1, 3, 512, 512)
        feats = backbone(x)
        print(f"{variant}: channels={backbone.channel_stages}")
        for i, f in enumerate(feats):
            print(f"  stage {i}: {f.shape}")
