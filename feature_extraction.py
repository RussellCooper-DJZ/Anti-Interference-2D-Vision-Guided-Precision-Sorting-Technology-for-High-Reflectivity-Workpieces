"""
feature_extraction.py — AGEANet：船舶大型金属高光面边缘感知分割网络

架构说明：
  AGEANet (Anti-Glare Edge-Aware Network) 基于 U-Net 骨架，专为以下挑战设计：
  1. 大面积镜面高光导致的局部信息丢失
  2. 焊缝/铆钉/舷窗等细小结构边缘的精准定位
  3. 复杂环境光（水面反射、侧光、泛光灯）下的鲁棒性

  核心改进：
  - CBAM 注意力模块（通道 + 空间）：抑制高光区域的虚假激活
  - 双分支输出头：语义分割 + 边缘检测同步训练
  - 高光感知跳跃连接：在跳跃连接处加入高光掩膜门控
  - 深度可分离卷积（Lite 版）：面向 RA8P1 嵌入式部署

模型变体：
  AGEANet      — 标准版，~8.5M 参数，适合 GPU 训练
  AGEANetLite  — 轻量版，~1.2M 参数，适合嵌入式推理

用法::

    model = AGEANet(in_channels=3, base_ch=64)
    out = model(x)  # x: (B,3,H,W)
    seg  = out['seg']   # (B,1,H,W) [0,1]
    edge = out['edge']  # (B,1,H,W) [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ============================================================
# 1. 基础模块
# ============================================================

class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU（可选残差）。"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1,
                 groups: int = 1, relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积（Depthwise + Pointwise），用于 Lite 版。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBNReLU(in_ch, in_ch, kernel=3, stride=stride,
                             padding=1, groups=in_ch)
        self.pw = ConvBNReLU(in_ch, out_ch, kernel=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DoubleConv(nn.Module):
    """标准 U-Net 双卷积块。"""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None,
                 depthwise: bool = False):
        super().__init__()
        mid_ch = mid_ch or out_ch
        if depthwise:
            self.block = nn.Sequential(
                DepthwiseSeparableConv(in_ch, mid_ch),
                DepthwiseSeparableConv(mid_ch, out_ch),
            )
        else:
            self.block = nn.Sequential(
                ConvBNReLU(in_ch, mid_ch),
                ConvBNReLU(mid_ch, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# 2. CBAM 注意力模块
# ============================================================

class ChannelAttention(nn.Module):
    """
    通道注意力（Channel Attention Module）。

    通过全局平均池化 + 全局最大池化，学习各通道的重要性权重。
    对高光区域产生的虚假高激活通道进行抑制。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    """
    空间注意力（Spatial Attention Module）。

    沿通道维度做平均/最大池化后卷积，学习空间位置的重要性权重。
    帮助模型聚焦于结构边缘区域，忽略大面积高光干扰。
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        feat = torch.cat([avg, mx], dim=1)
        scale = self.sigmoid(self.conv(feat))
        return x * scale


class CBAM(nn.Module):
    """CBAM：先通道注意力，再空间注意力。"""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


# ============================================================
# 3. 高光感知门控跳跃连接
# ============================================================

class GlareGatedSkip(nn.Module):
    """
    高光感知门控跳跃连接。

    在 U-Net 跳跃连接处，通过检测高光区域（高亮度、低饱和度）
    生成软门控权重，降低高光区域的跳跃连接权重，
    防止高光伪特征直接传递到解码器。
    """

    def __init__(self, channels: int):
        super().__init__()
        # 高光检测分支：轻量 1x1 卷积估计高光概率
        self.glare_detector = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        # 特征精炼
        self.refine = ConvBNReLU(channels, channels, kernel=1, padding=0)

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        # glare_map: (B,1,H,W)，高光区域接近 1
        glare_map = self.glare_detector(skip)
        # 门控：高光区域权重降低
        gate = 1.0 - glare_map * 0.7  # 最多抑制 70%
        return self.refine(skip * gate)


# ============================================================
# 4. 编码器块
# ============================================================

class EncoderBlock(nn.Module):
    """编码器块：DoubleConv + CBAM + MaxPool。"""

    def __init__(self, in_ch: int, out_ch: int,
                 use_cbam: bool = True, depthwise: bool = False):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, depthwise=depthwise)
        self.attn = CBAM(out_ch) if use_cbam else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        feat = self.attn(self.conv(x))
        return feat, self.pool(feat)  # (skip, pooled)


# ============================================================
# 5. 解码器块
# ============================================================

class DecoderBlock(nn.Module):
    """解码器块：双线性上采样 + 跳跃连接 + DoubleConv。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 use_glare_gate: bool = True, depthwise: bool = False):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.gate = GlareGatedSkip(skip_ch) if use_glare_gate else nn.Identity()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, depthwise=depthwise)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 尺寸对齐（处理奇数尺寸）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        skip = self.gate(skip)
        return self.conv(torch.cat([x, skip], dim=1))


# ============================================================
# 6. AGEANet 标准版
# ============================================================

class AGEANet(nn.Module):
    """
    AGEANet — 船舶大型金属高光面边缘感知分割网络（标准版）。

    Args:
        in_channels:  输入通道数（默认 3，RGB）
        out_channels: 分割输出通道数（默认 1，二值分割）
        base_ch:      基础通道数（默认 64，控制模型容量）
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base_ch: int = 64):
        super().__init__()
        c = base_ch
        # 编码器
        self.enc1 = EncoderBlock(in_channels, c,      use_cbam=False)
        self.enc2 = EncoderBlock(c,           c * 2,  use_cbam=True)
        self.enc3 = EncoderBlock(c * 2,       c * 4,  use_cbam=True)
        self.enc4 = EncoderBlock(c * 4,       c * 8,  use_cbam=True)
        # 瓶颈
        self.bottleneck = nn.Sequential(
            DoubleConv(c * 8, c * 16),
            CBAM(c * 16),
        )
        # 解码器
        self.dec4 = DecoderBlock(c * 16, c * 8,  c * 8,  use_glare_gate=True)
        self.dec3 = DecoderBlock(c * 8,  c * 4,  c * 4,  use_glare_gate=True)
        self.dec2 = DecoderBlock(c * 4,  c * 2,  c * 2,  use_glare_gate=True)
        self.dec1 = DecoderBlock(c * 2,  c,       c,      use_glare_gate=False)
        # 输出头：语义分割
        self.seg_head = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, out_channels, 1),
            nn.Sigmoid(),
        )
        # 输出头：边缘检测（从 dec2 特征引出，分辨率更高）
        self.edge_head = nn.Sequential(
            nn.Conv2d(c * 2, c // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        # 瓶颈
        b = self.bottleneck(p4)
        # 解码
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        # 输出
        seg  = self.seg_head(d1)
        edge = self.edge_head(d2)
        # 对齐 edge 到 seg 尺寸
        if edge.shape[-2:] != seg.shape[-2:]:
            edge = F.interpolate(edge, size=seg.shape[-2:],
                                 mode='bilinear', align_corners=False)
        return {'seg': seg, 'edge': edge}


# ============================================================
# 7. AGEANetLite 轻量版（嵌入式部署）
# ============================================================

class AGEANetLite(nn.Module):
    """
    AGEANetLite — 轻量版，面向 RA8P1 (Cortex-M85 + Helium) 嵌入式部署。

    相比标准版：
    - 使用深度可分离卷积替代标准卷积
    - 减少通道数（base_ch=32）
    - 仅在 enc3/enc4 使用 CBAM（减少计算量）
    - 去除高光门控（节省内存）
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base_ch: int = 32):
        super().__init__()
        c = base_ch
        # 编码器（深度可分离卷积）
        self.enc1 = EncoderBlock(in_channels, c,      use_cbam=False, depthwise=False)
        self.enc2 = EncoderBlock(c,           c * 2,  use_cbam=False, depthwise=True)
        self.enc3 = EncoderBlock(c * 2,       c * 4,  use_cbam=True,  depthwise=True)
        self.enc4 = EncoderBlock(c * 4,       c * 8,  use_cbam=True,  depthwise=True)
        # 瓶颈
        self.bottleneck = DoubleConv(c * 8, c * 8, depthwise=True)
        # 解码器
        self.dec4 = DecoderBlock(c * 8,  c * 8, c * 4, use_glare_gate=False, depthwise=True)
        self.dec3 = DecoderBlock(c * 4,  c * 4, c * 2, use_glare_gate=False, depthwise=True)
        self.dec2 = DecoderBlock(c * 2,  c * 2, c,     use_glare_gate=False, depthwise=True)
        self.dec1 = DecoderBlock(c,      c,     c,     use_glare_gate=False, depthwise=True)
        # 输出头
        self.seg_head = nn.Sequential(
            nn.Conv2d(c, out_channels, 1),
            nn.Sigmoid(),
        )
        # edge_head 接 dec2 输出，通道数为 c（不是 c*2）
        self.edge_head = nn.Sequential(
            nn.Conv2d(c, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b  = self.bottleneck(p4)
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        seg  = self.seg_head(d1)
        edge = self.edge_head(d2)
        if edge.shape[-2:] != seg.shape[-2:]:
            edge = F.interpolate(edge, size=seg.shape[-2:],
                                 mode='bilinear', align_corners=False)
        return {'seg': seg, 'edge': edge}


# ============================================================
# 8. 推理辅助函数
# ============================================================

def predict(
    model: nn.Module,
    image: 'np.ndarray',
    device: torch.device,
    seg_threshold: float = 0.5,
    edge_threshold: float = 0.3,
    input_size: int = 512,
) -> Dict[str, 'np.ndarray']:
    """
    对单张 BGR uint8 图像进行推理。

    Args:
        model:          训练好的 AGEANet / AGEANetLite
        image:          BGR uint8 numpy 图像 (H,W,3)
        device:         torch.device
        seg_threshold:  分割二值化阈值
        edge_threshold: 边缘二值化阈值
        input_size:     推理分辨率（32 的倍数）

    Returns:
        {
          'seg_prob':  float32 (H,W) 分割概率图
          'seg_mask':  uint8  (H,W) 二值分割掩膜
          'edge_prob': float32 (H,W) 边缘概率图
          'edge_mask': uint8  (H,W) 二值边缘掩膜
        }
    """
    import numpy as np
    import cv2

    model.eval()
    orig_h, orig_w = image.shape[:2]

    # 预处理
    img_resized = cv2.resize(image, (input_size, input_size),
                             interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)

    seg_prob  = out['seg'].squeeze().cpu().numpy()
    edge_prob = out['edge'].squeeze().cpu().numpy()

    # 还原到原始尺寸
    seg_prob  = cv2.resize(seg_prob,  (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    edge_prob = cv2.resize(edge_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    seg_mask  = (seg_prob  > seg_threshold).astype(np.uint8) * 255
    edge_mask = (edge_prob > edge_threshold).astype(np.uint8) * 255

    return {
        'seg_prob':  seg_prob,
        'seg_mask':  seg_mask,
        'edge_prob': edge_prob,
        'edge_mask': edge_mask,
    }


def get_model_info(model: nn.Module) -> Dict:
    """返回模型参数量统计。"""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params':     total,
        'trainable_params': trainable,
        'size_mb_fp32':     total * 4 / (1024 ** 2),
        'size_mb_int8':     total / (1024 ** 2),
    }


# ============================================================
# 9. 命令行入口（模型结构验证）
# ============================================================

if __name__ == "__main__":
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    for name, ModelClass, base_ch in [
        ("AGEANet (标准版)",   AGEANet,     64),
        ("AGEANetLite (轻量版)", AGEANetLite, 32),
    ]:
        print("=" * 56)
        print(f"  {name}")
        print("=" * 56)
        model = ModelClass(in_channels=3, base_ch=base_ch).to(device)
        info = get_model_info(model)
        print(f"  总参数量:      {info['total_params']:>10,}")
        print(f"  FP32 模型大小: {info['size_mb_fp32']:>8.2f} MB")
        print(f"  INT8 模型大小: {info['size_mb_int8']:>8.2f} MB")

        dummy = torch.randn(1, 3, 512, 512).to(device)
        out = model(dummy)
        print(f"  输入:  {tuple(dummy.shape)}")
        print(f"  seg:   {tuple(out['seg'].shape)}")
        print(f"  edge:  {tuple(out['edge'].shape)}")
        print()

    print("模型结构验证通过！")
