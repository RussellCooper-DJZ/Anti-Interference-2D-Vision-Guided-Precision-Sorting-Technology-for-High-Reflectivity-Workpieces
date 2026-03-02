"""
feature_extraction.py — 高反光工件边缘分割模型

核心架构：Anti-Glare Edge-Aware U-Net (AGEANet)
  - 4 层编码器 + 4 层解码器，通道数 [64, 128, 256, 512]
  - CBAM (Convolutional Block Attention Module) 注意力机制
  - 独立的边缘感知分支 (Edge-Aware Branch)
  - 高光抑制前端 (Specular Suppression Frontend)
  - 支持 TFLite 导出的轻量化变体 (AGEANet-Lite)

Anti-Glare Edge-Aware U-Net (AGEANet) for high-reflectivity workpiece edge segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. 注意力模块 — CBAM (Channel + Spatial Attention)
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """通道注意力：学习哪些通道对边缘特征更重要，抑制反光通道响应。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = x.mean(dim=[2, 3])                       # (B, C)
        max_pool = x.amax(dim=[2, 3])                       # (B, C)
        attn = torch.sigmoid(self.fc(avg_pool) + self.fc(max_pool))  # (B, C)
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """空间注意力：学习哪些空间位置是真实边缘，抑制高光伪影区域。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """CBAM: 先通道注意力再空间注意力，联合抑制反光伪影。"""

    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


# ---------------------------------------------------------------------------
# 2. 高光抑制前端 (Specular Suppression Frontend)
# ---------------------------------------------------------------------------

class SpecularSuppression(nn.Module):
    """
    可学习的高光抑制前端：
    - 检测过曝区域 (亮度 > 阈值)
    - 通过可学习卷积生成抑制掩膜
    - 将原始图像与抑制后的图像融合
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.detect = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        glare_mask = self.detect(x)          # (B, 1, H, W) 高光概率图
        refined = self.refine(x)             # (B, 3, H, W) 修复后的图像
        out = x * (1 - glare_mask) + refined * glare_mask
        return out, glare_mask


# ---------------------------------------------------------------------------
# 3. 基础构建块
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """双卷积块 + 可选 CBAM 注意力。"""

    def __init__(self, in_c, out_c, use_cbam=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_c) if use_cbam else nn.Identity()

    def forward(self, x):
        return self.cbam(self.conv(x))


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块 — 用于轻量化变体。"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class LiteConvBlock(nn.Module):
    """轻量化双卷积块 — 用于嵌入式部署。"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_c, out_c),
            DepthwiseSeparableConv(out_c, out_c),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# 4. 边缘感知分支 (Edge-Aware Branch)
# ---------------------------------------------------------------------------

class EdgeBranch(nn.Module):
    """
    独立的边缘检测分支：
    - 从多尺度特征中提取边缘信息
    - 使用 Sobel 先验引导学习
    - 输出边缘概率图，用于辅助主分割分支
    """

    def __init__(self, encoder_channels):
        super().__init__()
        # 每个编码器层级的边缘提取头
        self.edge_heads = nn.ModuleList()
        for ch in encoder_channels:
            self.edge_heads.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 2, 1, 1),
            ))
        # 融合多尺度边缘图
        self.fuse = nn.Sequential(
            nn.Conv2d(len(encoder_channels), 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, encoder_features, target_size):
        edge_maps = []
        for feat, head in zip(encoder_features, self.edge_heads):
            e = head(feat)
            e = F.interpolate(e, size=target_size, mode='bilinear', align_corners=False)
            edge_maps.append(e)
        fused = self.fuse(torch.cat(edge_maps, dim=1))
        return fused


# ---------------------------------------------------------------------------
# 5. 主模型：AGEANet (Anti-Glare Edge-Aware Network)
# ---------------------------------------------------------------------------

class AGEANet(nn.Module):
    """
    Anti-Glare Edge-Aware Network — 高反光工件精准边缘分割网络。

    架构特点：
    1. 高光抑制前端 — 可学习地检测并修复高光区域
    2. 4 层 U-Net 编码器-解码器 + CBAM 注意力
    3. 边缘感知分支 — 多尺度边缘检测，提供边缘先验
    4. 双输出头 — 分割 mask + 边缘 map

    输入: (B, 3, H, W) — 经 HDR 融合后的 RGB 图像
    输出: dict {
        'seg':   (B, 1, H, W) — 工件分割概率图
        'edge':  (B, 1, H, W) — 边缘概率图
        'glare': (B, 1, H, W) — 高光检测图 (训练时用于辅助监督)
    }
    """

    def __init__(self, in_channels=3, out_channels=1, base_ch=64):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]  # [64, 128, 256, 512]

        # --- 高光抑制前端 ---
        self.specular_suppress = SpecularSuppression(in_channels)

        # --- 编码器 ---
        self.enc1 = ConvBlock(in_channels, chs[0])
        self.enc2 = ConvBlock(chs[0], chs[1])
        self.enc3 = ConvBlock(chs[1], chs[2])
        self.enc4 = ConvBlock(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2, 2)

        # --- 瓶颈层 ---
        self.bottleneck = ConvBlock(chs[3], chs[3] * 2)  # 1024

        # --- 解码器 ---
        self.up4 = nn.ConvTranspose2d(chs[3] * 2, chs[3], 2, stride=2)
        self.dec4 = ConvBlock(chs[3] * 2, chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = ConvBlock(chs[2] * 2, chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = ConvBlock(chs[1] * 2, chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = ConvBlock(chs[0] * 2, chs[0])

        # --- 分割输出头 ---
        self.seg_head = nn.Conv2d(chs[0], out_channels, 1)

        # --- 边缘感知分支 ---
        self.edge_branch = EdgeBranch(chs)

    @staticmethod
    def _pad_to_match(x, target):
        """确保上采样后的特征图尺寸与跳跃连接匹配。"""
        dh = target.size(2) - x.size(2)
        dw = target.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [0, dw, 0, dh])
        return x

    def forward(self, x):
        # 高光抑制
        x_suppressed, glare_map = self.specular_suppress(x)

        # 编码器
        e1 = self.enc1(x_suppressed)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 瓶颈
        b = self.bottleneck(self.pool(e4))

        # 解码器 + 跳跃连接
        d4 = self.up4(b)
        d4 = self._pad_to_match(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self._pad_to_match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # 分割输出
        seg = torch.sigmoid(self.seg_head(d1))

        # 边缘分支
        target_size = (x.size(2), x.size(3))
        edge = self.edge_branch([e1, e2, e3, e4], target_size)

        return {'seg': seg, 'edge': edge, 'glare': glare_map}


# ---------------------------------------------------------------------------
# 6. 轻量化变体：AGEANet-Lite (用于嵌入式 RA8P1 部署)
# ---------------------------------------------------------------------------

class AGEANetLite(nn.Module):
    """
    AGEANet 的轻量化版本，适合嵌入式部署 (RA8P1 + Helium)。
    - 使用深度可分离卷积替代标准卷积
    - 通道数减半 [32, 64, 128, 256]
    - 去掉 CBAM 注意力 (减少计算量)
    - 保留边缘分支 (轻量化版)
    """

    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # 编码器
        self.enc1 = LiteConvBlock(in_channels, chs[0])
        self.enc2 = LiteConvBlock(chs[0], chs[1])
        self.enc3 = LiteConvBlock(chs[1], chs[2])
        self.enc4 = LiteConvBlock(chs[2], chs[3])
        self.pool = nn.MaxPool2d(2, 2)

        # 瓶颈
        self.bottleneck = LiteConvBlock(chs[3], chs[3])

        # 解码器
        self.up4 = nn.ConvTranspose2d(chs[3], chs[3], 2, stride=2)
        self.dec4 = LiteConvBlock(chs[3] * 2, chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = LiteConvBlock(chs[2] * 2, chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = LiteConvBlock(chs[1] * 2, chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = LiteConvBlock(chs[0] * 2, chs[0])

        # 输出头
        self.seg_head = nn.Conv2d(chs[0], out_channels, 1)

        # 轻量化边缘分支
        self.edge_head = nn.Sequential(
            nn.Conv2d(chs[0], chs[0] // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(chs[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(chs[0] // 2, 1, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _pad_to_match(x, target):
        dh = target.size(2) - x.size(2)
        dw = target.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [0, dw, 0, dh])
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self._pad_to_match(self.up4(b), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self._pad_to_match(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._pad_to_match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._pad_to_match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        seg = torch.sigmoid(self.seg_head(d1))
        edge = self.edge_head(d1)

        return {'seg': seg, 'edge': edge}


# ---------------------------------------------------------------------------
# 7. 向后兼容：保留 SimpleUNet 接口
# ---------------------------------------------------------------------------

class SimpleUNet(AGEANet):
    """向后兼容旧接口。内部使用 AGEANet 架构。"""
    pass


# ---------------------------------------------------------------------------
# 8. 推理辅助函数
# ---------------------------------------------------------------------------

def predict_contour(model, image, device, threshold=0.5):
    """
    使用训练好的模型预测工件轮廓。

    参数:
        model:     训练好的 AGEANet 或 AGEANetLite 模型
        image:     BGR 格式的 numpy 图像 (H, W, 3), uint8
        device:    torch.device
        threshold: 二值化阈值

    返回:
        contour_mask: 二值化轮廓掩膜 (H, W), uint8, 值为 0 或 255
    """
    model.eval()
    h, w = image.shape[:2]

    # 预处理：归一化 + 调整尺寸到 32 的倍数
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    with torch.no_grad():
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
        img_tensor = img_tensor.to(device)

        output = model(img_tensor)
        seg_map = output['seg'].squeeze().cpu().numpy()

    # 裁剪回原始尺寸
    seg_map = seg_map[:h, :w]
    contour_mask = (seg_map > threshold).astype(np.uint8) * 255
    return contour_mask


def predict_edge(model, image, device, threshold=0.3):
    """
    使用训练好的模型预测工件边缘。

    返回:
        edge_mask: 边缘概率图 (H, W), float32, 值 [0, 1]
    """
    model.eval()
    h, w = image.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    with torch.no_grad():
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, [0, pad_w, 0, pad_h], mode='reflect')
        img_tensor = img_tensor.to(device)

        output = model(img_tensor)
        edge_map = output['edge'].squeeze().cpu().numpy()

    edge_map = edge_map[:h, :w]
    return edge_map


def get_model_info(model):
    """打印模型参数量和计算量估计。"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_mb': total_params * 4 / (1024 ** 2),  # float32
    }


# ---------------------------------------------------------------------------
# 9. 入口点
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试标准模型
    print("=" * 60)
    print("AGEANet (Standard) — 高反光工件边缘分割网络")
    print("=" * 60)
    model = AGEANet(in_channels=3, out_channels=1).to(device)
    info = get_model_info(model)
    print(f"  总参数量: {info['total_params']:,}")
    print(f"  可训练参数: {info['trainable_params']:,}")
    print(f"  模型大小 (float32): {info['total_params_mb']:.1f} MB")

    # 测试前向传播
    dummy = torch.randn(1, 3, 256, 256).to(device)
    out = model(dummy)
    print(f"  输入尺寸: {dummy.shape}")
    print(f"  分割输出: {out['seg'].shape}")
    print(f"  边缘输出: {out['edge'].shape}")
    print(f"  高光检测: {out['glare'].shape}")

    # 测试轻量化模型
    print("\n" + "=" * 60)
    print("AGEANet-Lite — 嵌入式轻量化版本")
    print("=" * 60)
    model_lite = AGEANetLite(in_channels=3, out_channels=1).to(device)
    info_lite = get_model_info(model_lite)
    print(f"  总参数量: {info_lite['total_params']:,}")
    print(f"  可训练参数: {info_lite['trainable_params']:,}")
    print(f"  模型大小 (float32): {info_lite['total_params_mb']:.1f} MB")

    out_lite = model_lite(dummy)
    print(f"  输入尺寸: {dummy.shape}")
    print(f"  分割输出: {out_lite['seg'].shape}")
    print(f"  边缘输出: {out_lite['edge'].shape}")
    print(f"\n模型架构验证通过！")
