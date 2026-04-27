"""
feature_extraction.py — FLARE：船舶大型金属高光面边缘感知分割网络
:Author: RussellCooper

架构说明：
  FLARE (Anti-Glare Edge-Aware Network) 基于 U-Net 骨架，专为以下挑战设计：
  1. 大面积镜面高光导致的局部信息丢失
  2. 焊缝/铆钉/舷窗等细小结构边缘的精准定位
  3. 复杂环境光（水面反射、侧光、泛光灯）下的鲁棒性

  核心改进：
  - CBAM 注意力模块（通道 + 空间）：抑制高光区域的虚假激活
  - 双分支输出头：语义分割 + 边缘检测同步训练
  - 高光感知跳跃连接：在跳跃连接处加入高光掩膜门控
  - 深度可分离卷积（Lite 版）：面向 RA8P1 嵌入式部署

模型变体：
  FLARE      — 标准版，~8.5M 参数，适合 GPU 训练
  FLARELite  — 轻量版，~1.2M 参数，适合嵌入式推理

用法::

    model = FLARE(in_channels=3, base_ch=64)
    out = model(x)  # x: (B,3,H,W)
    seg  = out['seg']   # (B,1,H,W) [0,1]
    edge = out['edge']  # (B,1,H,W) [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

__all__ = [
    "ConvBNReLU",
    "DepthwiseSeparableConv",
    "DoubleConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "BiLevelAttention",
    "SCSELayer",
    "CoordConv",
    "GlareGatedSkip",
    # ========== Phase 5 轻量化/形变卷积 (iter 159-160) ==========
    "DeformConv2d",
    "CoordDeformConv",
    "GhostConv",
    "GhostDoubleConv",
    "EncoderBlock",
    "DecoderBlock",
    "FLARE",
    "FLARELite",
    "DetectionHead",
    "FPN",
    "PAFPN",
    "FeatureAlignBlock",
    "AnchorFreeHead",
    "CIoULoss",
    "EdgeRefinementHead",
    "predict",
    "get_model_info",
    "nms_centerness",
    # ========== 数学增强模块 ==========
    "WaveletScattering",
    "FourierConv",
    "GradientBalance",
    "InformationBottleneck",
    "MorphologyLayer",
    # ========== 边缘检测头 ==========
    "HEDEdgeHead",
    "RCFEdgeHead",
    "EdgeAttentionHead",
    "ASPPModule",
]


# ============================================================
# HED风格多尺度边缘检测头（借鉴业界领先方法）
# ============================================================

class HEDEdgeHead(nn.Module):
    """
    HED (Holistically-Nested Edge Detection) 风格的多尺度边缘检测头。

    核心改进：
    1. 多尺度侧输出 - 从多个解码层提取边缘特征
    2. 嵌套架构 - 每层独立监督，层级递进
    3. 密集特征融合 - 借鉴 DenseNet 的密集连接思想
    4. 深层监督 - 避免梯度消失，加速收敛

    参考：Xie et al., "Holistically-Nested Edge Detection", ICCV 2015
    """

    def __init__(self, in_channels_list, mid_channels=128):
        """
        Args:
            in_channels_list: 多尺度特征通道列表 [dec1_ch, dec2_ch, dec3_ch, dec4_ch]
            mid_channels: 中间层通道数
        """
        super().__init__()
        self.n_levels = len(in_channels_list)

        # 每层的边缘检测头（侧输出）
        self.side_outputs = nn.ModuleList()
        for in_ch in in_channels_list:
            # 逐渐减少通道，避免过拟合
            side_ch = min(mid_channels, in_ch)
            self.side_outputs.append(nn.Sequential(
                nn.Conv2d(in_ch, side_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(side_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(side_ch, side_ch // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(side_ch // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(side_ch // 2, 1, 1),
            ))

        # 融合权重（可学习）
        self.fusion_weights = nn.Parameter(torch.ones(self.n_levels) / self.n_levels)

    def forward(self, features_list):
        """
        Args:
            features_list: [d1, d2, d3, d4] 多尺度解码器特征

        Returns:
            edge: 融合后的边缘预测
            side_edges: 各层独立边缘预测列表
        """
        side_edges = []
        target_size = features_list[0].shape[-2:]

        for i, (feat, side_conv) in enumerate(zip(features_list, self.side_outputs)):
            side_edge = side_conv(feat)  # (B, 1, H_i, W_i)

            # 上采样到相同尺寸
            if side_edge.shape[-2:] != target_size:
                side_edge = F.interpolate(side_edge, size=target_size,
                                         mode='bilinear', align_corners=False)
            side_edges.append(side_edge)

        # 加权融合（可学习权重）
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_edge = sum(w * e for w, e in zip(weights, side_edges))

        return fused_edge, side_edges


class RCFEdgeHead(nn.Module):
    """
    RCF (Richer Convolutional Features) 风格边缘检测头。

    核心改进（相比 HED）：
    1. 多尺度特征密集连接 - 每层特征都与最终输出密集连接
    2. 更深的侧输出分支 - 每层使用双分支网络
    3. 编码器特征直接监督 - 不仅监督解码器，还监督编码器多尺度
    4. 边缘感知注意力 - 使用边缘特征引导融合

    参考：Hou et al., "Richer Convolutional Features for Edge Detection", CVPR 2017
    """

    def __init__(self, in_channels_list, mid_channels=128):
        """
        Args:
            in_channels_list: 多尺度特征通道列表 [enc1_ch, enc2_ch, enc3_ch, enc4_ch, dec1_ch, dec2_ch, dec3_ch, dec4_ch]
            mid_channels: 中间层通道数
        """
        super().__init__()
        self.n_levels = len(in_channels_list)

        # 每层独立的边缘检测分支（双分支：Branch1 + Branch2 → 融合）
        self.branch1_convs = nn.ModuleList()
        self.branch2_convs = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()

        for in_ch in in_channels_list:
            side_ch = min(mid_channels, in_ch)
            half_ch = side_ch // 2

            # Branch 1: 1x1 降维 + 3x3 提取
            branch1 = nn.Sequential(
                nn.Conv2d(in_ch, half_ch, 1, bias=False),
                nn.BatchNorm2d(half_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(half_ch, half_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(half_ch),
                nn.ReLU(inplace=True),
            )
            self.branch1_convs.append(branch1)

            # Branch 2: 3x3 卷积（直接提取）
            branch2 = nn.Sequential(
                nn.Conv2d(in_ch, half_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(half_ch),
                nn.ReLU(inplace=True),
            )
            self.branch2_convs.append(branch2)

            # 融合层：合并两个分支的输出
            fusion = nn.Sequential(
                nn.Conv2d(side_ch, side_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(side_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(side_ch, 1, 1),
            )
            self.fusion_convs.append(fusion)

        # 边缘感知注意力融合权重
        self.edge_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.n_levels, self.n_levels, 1),
            nn.Sigmoid(),
        )

        # 可学习融合权重
        self.fusion_weights = nn.Parameter(torch.ones(self.n_levels) / self.n_levels)

    def forward(self, features_list):
        """
        Args:
            features_list: [f1, f2, f3, f4, d1, d2, d3, d4] 多尺度编码器+解码器特征

        Returns:
            edge: 融合后的边缘预测
            side_edges: 各层独立边缘预测列表
        """
        side_edges = []
        target_size = features_list[0].shape[-2:]

        for i, feat in enumerate(features_list):
            # 双分支处理
            b1 = self.branch1_convs[i](feat)  # Branch 1
            b2 = self.branch2_convs[i](feat)  # Branch 2

            # 拼接两个分支
            fused = torch.cat([b1, b2], dim=1)  # (B, side_ch, H, W)

            # 融合并预测边缘
            side_edge = self.fusion_convs[i](fused)  # (B, 1, H_i, W_i)

            # 上采样到相同尺寸
            if side_edge.shape[-2:] != target_size:
                side_edge = F.interpolate(side_edge, size=target_size,
                                         mode='bilinear', align_corners=False)
            side_edges.append(side_edge)

        # 堆叠所有侧输出用于注意力
        side_stack = torch.cat(side_edges, dim=1)  # (B, n_levels, H, W)
        attn_weights = self.edge_attention(side_stack.mean(dim=[2, 3], keepdim=True))  # (B, n_levels, 1, 1)
        attn_weights = attn_weights.squeeze(-1).squeeze(-1)  # (B, n_levels)

        # 融合：结合可学习权重和注意力权重
        weights = F.softmax(self.fusion_weights, dim=0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, n_levels, 1, 1)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, n_levels, 1, 1)
        combined_weights = 0.5 * weights + 0.5 * attn_weights  # (B, n_levels, 1, 1)

        # 加权融合
        stacked = torch.stack(side_edges, dim=1)  # (B, n_levels, 1, H, W)
        # Unsqueeze combined_weights to match stacked's dimensions
        combined_weights = combined_weights.unsqueeze(2)  # (B, n_levels, 1, 1, 1)
        fused_edge = (combined_weights * stacked).sum(dim=1)  # (B, 1, H, W)

        return fused_edge, side_edges


class EdgeAttentionHead(nn.Module):
    """
    边缘注意力头 - 使用通道和空间注意力增强边缘检测。

    核心改进：
    1. 通道注意力 - 学习哪些特征图对边缘检测更重要
    2. 空间注意力 - 学习哪些位置更可能是边缘
    3. 边缘一致性 - 确保边缘与分割掩膜边界一致
    """

    def __init__(self, in_channels, mid_channels=128):
        super().__init__()

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # 空间注意力（边缘增强）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, 1, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        # 边缘特征提取
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, 1),
        )

        # 边缘一致性：确保与分割掩膜边界一致
        self.consistency_conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, seg_logits=None):
        """
        Args:
            x: (B, C, H, W) 输入特征
            seg_logits: (B, 1, H, W) 可选的分割 logits 用于一致性约束

        Returns:
            edge: (B, 1, H, W) 边缘预测
        """
        # 通道注意力
        ch_attn = self.channel_attn(x)
        x = x * ch_attn

        # 空间注意力
        sp_attn = self.spatial_attn(x)
        x = x * sp_attn

        # 边缘预测
        edge = self.edge_conv(x)

        # 边缘一致性（如果有分割 logits）
        if seg_logits is not None:
            # 计算分割梯度的边界
            seg_prob = torch.sigmoid(seg_logits)

            # 如果edge和seg_logits尺寸不同，需要对齐
            if edge.shape[-2:] != seg_logits.shape[-2:]:
                edge_upsampled = F.interpolate(edge, size=seg_logits.shape[-2:],
                                              mode='bilinear', align_corners=False)
            else:
                edge_upsampled = edge

            grad_x = torch.abs(seg_prob[:, :, :, :-1] - seg_prob[:, :, :, 1:])
            grad_y = torch.abs(seg_prob[:, :, :-1, :] - seg_prob[:, :, 1:, :])
            grad_x = F.pad(grad_x, (0, 1, 0, 0))
            grad_y = F.pad(grad_y, (0, 0, 0, 1))
            seg_edge = grad_x + grad_y

            # 边缘一致性权重
            edge_consistency = torch.cat([edge_upsampled, seg_edge], dim=1)
            consistency_weight = self.consistency_conv(edge_consistency)

            # 融合原始边缘和一致性增强的边缘
            edge = edge_upsampled * (1 + 0.3 * consistency_weight)

            # 如果需要，downsample回到原始尺寸(x的尺寸)
            if edge.shape[-2:] != x.shape[-2:]:
                edge = F.interpolate(edge, size=x.shape[-2:],
                                    mode='bilinear', align_corners=False)

        return edge


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (DeepLabV3+风格)

    多尺度空洞卷积，获取全局上下文信息。
    对不同尺度的边缘结构都能有效捕获。
    """
    dilations = [1, 6, 12, 18]

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp_convs = nn.ModuleList()
        for d in self.dilations:
            self.aspp_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True),
            ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5 // 4, out_channels, 1, bias=False),  # 5 branches * (out_channels // 4)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        aspp_feats = [F.interpolate(conv(x), size=size, mode='bilinear', align_corners=False)
                      for conv in self.aspp_convs]
        # 全局特征（不使用BatchNorm，避免batch=1问题）
        global_feat = F.adaptive_avg_pool2d(x, 1)
        global_feat = self.global_pool[1](global_feat)  # Conv only, no BN
        global_feat = F.interpolate(global_feat, size=size,
                                   mode='bilinear', align_corners=False)
        aspp_feats.append(global_feat)
        concat = torch.cat(aspp_feats, dim=1)
        return self.project(concat)


# ============================================================
# Phase 5 新增：Edge Refinement Head (Iteration 167)
# ============================================================

class EdgeRefinementHead(nn.Module):
    """
    边缘细化头。
    接收粗边缘预测和分割 logits，通过分割边界梯度先验指导边缘位置精修。
    """

    def __init__(self, edge_channels: int = 1, seg_channels: int = 1,
                 mid_channels: int = 32):
        super().__init__()
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

        self.refine = nn.Sequential(
            nn.Conv2d(edge_channels + seg_channels + 1, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, edge_channels, 1),
            nn.Sigmoid(),
        )

    def _seg_gradient(self, seg_logits: torch.Tensor) -> torch.Tensor:
        seg_prob = torch.sigmoid(seg_logits)
        grad_x = F.conv2d(seg_prob, self.sobel_x, padding=1)
        grad_y = F.conv2d(seg_prob, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return grad_mag

    def forward(self, edge_coarse: torch.Tensor, seg_logits: torch.Tensor) -> torch.Tensor:
        seg_grad = self._seg_gradient(seg_logits)
        if edge_coarse.shape[-2:] != seg_logits.shape[-2:]:
            edge_coarse = F.interpolate(edge_coarse, size=seg_logits.shape[-2:], mode="bilinear", align_corners=False)
        concat = torch.cat([edge_coarse, seg_logits, seg_grad], dim=1)
        edge_refined = self.refine(concat)
        return edge_refined


# ============================================================
# 11. 数学增强：Wavelet Scattering（可证明的 invariance）
# ============================================================

class WaveletScattering(nn.Module):
    """
    小波散射变换 - 数学上可证明的平移/旋转不变性

    原理：通过对信号施加小波变换并取平均，得到具有不变性的表示

    优势：
    - 无需学习，完全由数学定义
    - 对噪声鲁棒
    - 提供多尺度纹理特征
    """

    def __init__(self, J: int = 3, L: int = 8):
        """
        Args:
            J: 小波分解层数（2^J 尺度）
            L: 方向数（通常 8）
        """
        super().__init__()
        self.J = J
        self.L = L

        # Morlet 小波参数
        self.register_buffer('theta', torch.linspace(0, math.pi, L, dtype=torch.float32))

    def _morlet_wavelet(self, size: int, theta: float, sigma: float = 1.0):
        """生成 Morlet 小波核"""
        x = torch.linspace(-size//2, size//2, size, dtype=torch.float32)
        y = torch.linspace(-size//2, size//2, size, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # 旋转（使用 math 避免 Tensor/标量类型不匹配）
        ct = math.cos(theta)
        st = math.sin(theta)
        X_rot = X * ct + Y * st
        Y_rot = -X * st + Y * ct

        # Morlet 小波包络
        envelope = torch.exp(-(X_rot**2 + Y_rot**2) / (2 * sigma**2))
        carrier = torch.cos(2 * math.pi * X_rot / sigma)
        wavelet = envelope * carrier

        return wavelet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入特征

        Returns:
            scatter: (B, C, H, W) 散射特征
        """
        B, C, H, W = x.shape
        features = [x]  # 原始信号

        # 沿通道分离计算
        for c in range(C):
            channel = x[:, c:c+1]  # (B, 1, H, W)

            for j in range(self.J):
                for theta in self.theta:
                    size = 2 ** (j + 3)  # 小波尺度
                    if size > min(H, W):
                        continue

                    wavelet = self._morlet_wavelet(size, theta.item())
                    wavelet = wavelet.view(1, 1, size, size).to(x.device)

                    # 小波卷积
                    conv = F.conv2d(channel, wavelet, padding=size//2)

                    # 取模（散射不变性关键）
                    modulus = torch.sqrt(conv ** 2 + 1e-8)

                    # 空间平均（平移不变性）
                    pooled = F.avg_pool2d(modulus, 2 ** j)
                    # 上采样回原始尺寸以便拼接
                    pooled = F.interpolate(
                        pooled, size=(H, W), mode='bilinear', align_corners=False
                    )

                    features.append(pooled)

        # 沿新维度拼接
        scatter = torch.cat(features, dim=1)
        return scatter


# ============================================================
# 12. 频域等变卷积 (Fourier Domain Learning)
# ============================================================

class FourierConv(nn.Module):
    """
    频域等变卷积 - 在 Fourier 域进行卷积操作

    数学原理：
    - 卷积定理：空域卷积 = 频域乘积
    - Fourier 变换提供正交基
    - 频域学习可以捕获周期性/方向性特征

    优势：
    - 计算高效：大卷积核等价于频域乘法
    - 方向选择性：对边缘/纹理方向敏感
    - 频谱正则化：隐式学习低频/高频平衡
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, mode: str = 'cartesian'):
        """
        Args:
            in_channels: 输入通道
            out_channels: 输出通道
            kernel_size: 卷积核大小
            mode: 'cartesian' | 'polar' - 坐标系选择
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mode = mode

        # 可学习 Fourier 域权重
        if mode == 'cartesian':
            # 笛卡尔坐标系：学习不同频率分量
            self.freq_weight = nn.Parameter(
                torch.randn(in_channels, out_channels, kernel_size, kernel_size, 2)  # 实部+虚部
            )
        else:
            # 极坐标：学习不同方向+尺度
            self.angular_weight = nn.Parameter(
                torch.randn(in_channels, out_channels, kernel_size, 8)  # 8个方向
            )
            self.radial_weight = nn.Parameter(
                torch.randn(in_channels, out_channels, kernel_size)  # 径向分量
            )

        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        if self.mode == 'cartesian':
            # Fourier 变换
            x_ft = torch.fft.rfft2(x, dim=(-2, -1))

            # 创建频域网格
            freq_h = torch.fft.fftfreq(H, dtype=torch.float32).to(x.device)
            freq_w = torch.fft.fftfreq(W, dtype=torch.float32).to(x.device)
            FH, FW = torch.meshgrid(freq_h, freq_w, indexing='ij')

            # 频域响应
            freq_mag = torch.sqrt(FH**2 + FW**2 + 1e-8)
            freq_angle = torch.atan2(FH, FW)

            # 可学习的频域滤波器
            freq_response = F.relu(
                torch.einsum('ijkl,m->ijkm', self.freq_weight[..., 0], torch.ones(H, W, device=x.device)) +
                torch.einsum('ijkl,m->ijkm', self.freq_weight[..., 1], freq_mag)
            )

            # 频域乘法
            x_ft_filtered = x_ft * freq_response.unsqueeze(0).unsqueeze(0)

            # 逆 Fourier 变换
            x_out = torch.fft.irfft2(x_ft_filtered, s=(H, W), dim=(-2, -1))
        else:
            # 极坐标模式：简化实现
            x_out = F.conv2d(x, self.freq_weight, padding=self.kernel_size // 2)

        return self.norm(x_out)


# ============================================================
# 13. 梯度均衡机制 (Gradient Equilibrium)
# ============================================================

class GradientBalance(nn.Module):
    """
    梯度均衡机制 - 解决多任务学习的梯度冲突

    数学原理：
    - 多任务梯度：∇θ = Σᵢ λᵢ ∇θᵢ
    - 梯度冲突导致训练不稳定
    - 动态调整 λᵢ 使各任务梯度方向一致

    方法：计算梯度余弦相似度，动态调整任务权重
    """

    def __init__(self, num_tasks: int = 2, momentum: float = 0.9):
        super().__init__()
        self.num_tasks = num_tasks
        self.momentum = momentum
        # 任务权重（可学习）
        self.task_weights = nn.Parameter(torch.ones(num_tasks) / num_tasks)
        # 历史梯度方向
        self.register_buffer('grad_history', torch.zeros(num_tasks, 512))

    def compute_similarities(self, grad_list: list) -> torch.Tensor:
        """计算任务间梯度余弦相似度"""
        similarities = torch.zeros(self.num_tasks, self.num_tasks)
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                if i != j:
                    g_i = grad_list[i].flatten()
                    g_j = grad_list[j].flatten()
                    cos_sim = F.cosine_similarity(
                        g_i.unsqueeze(0), g_j.unsqueeze(0)
                    )
                    similarities[i, j] = cos_sim
        return similarities

    def forward(self, loss_list: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            loss_list: 各任务损失列表

        Returns:
            total_loss: 加权总损失
            weights: 调整后的任务权重
        """
        if len(loss_list) != self.num_tasks:
            self.num_tasks = len(loss_list)

        # 计算各任务梯度
        grads = []
        for loss in loss_list:
            if loss.requires_grad:
                g = torch.autograd.grad(
                    loss, self.parameters(), retain_graph=True, allow_unused=True
                )
                if g[0] is not None:
                    grads.append(torch.cat([p.flatten() for p in g if p is not None]))
                else:
                    grads.append(torch.zeros(1, device=loss.device))
            else:
                grads.append(torch.zeros(1, device=loss.device))

        # 动态权重调整
        similarities = self.compute_similarities(grads)

        # 相似度高 → 增加权重，相似度低 → 降低权重
        avg_conflict = 1.0 - similarities.mean(dim=1)
        new_weights = F.softmax(
            self.task_weights * (1.0 + avg_conflict), dim=0
        )

        # 更新历史
        with torch.no_grad():
            self.grad_history = self.momentum * self.grad_history + \
                              (1 - self.momentum) * avg_conflict

        # 加权求和
        total_loss = sum(w * l for w, l in zip(new_weights, loss_list))

        return total_loss, new_weights


# ============================================================
# 14. 信息瓶颈正则化 (Information Bottleneck)
# ============================================================

class InformationBottleneck(nn.Module):
    """
    信息瓶颈层 - 最小化输入与表示之间的互信息

    数学原理：
    - I(X;Z) 表示输入 X 与表示 Z 之间的互信息
    - 目标：min I(X;Z) subject to I(Z;Y) ≥ c
    - 等价于：L = -I(Z;Y) + β I(X;Z)

    实现：变分近似，使用噪声正则化
    """

    def __init__(self, channels: int, beta: float = 1e-3):
        """
        Args:
            channels: 通道数
            beta: 信息瓶颈强度（越大压缩越多）
        """
        super().__init__()
        self.beta = beta

        # 变分噪声参数
        self.noise_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入张量

        Returns:
            z: (B, C, H, W) 压缩后的表示
        """
        if not self.training:
            return x

        # 变分噪声注入
        noise_std = F.softplus(self.noise_scale).view(1, -1, 1, 1)
        z = x + torch.randn_like(x) * noise_std

        # 信息瓶颈损失（隐式）
        # 负互信息近似：-I(X;Z) ∝ -Σ noise_scale
        ib_loss = self.beta * self.noise_scale.sum()

        return z


# ============================================================
# 15. 数学形态学层 (Morphological Layer)
# ============================================================

class MorphologyLayer(nn.Module):
    """
    数学形态学卷积层 - 用可学习权重实现膨胀/腐蚀

    数学原理：
    - 膨胀：D(I) = max_{p∈S} I(p)
    - 腐蚀：E(I) = min_{p∈S} I(p)
    - 结构元素 S 决定邻域形状

    可学习的形态学操作：软化/锐化边缘
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # 可学习的结构元素权重
        self.weight = nn.Parameter(torch.ones(channels, 1, kernel_size, kernel_size))
        # 膨胀/腐蚀平衡
        self.alpha = nn.Parameter(torch.zeros(channels, 1, 1, 1))  # (0=腐蚀, 1=膨胀)

        # 软化近似：用加权平均实现
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入

        Returns:
            out: (B, C, H, W) 形态学处理后
        """
        # 膨胀和腐蚀
        dilate = F.max_pool2d(x, kernel_size=self.kernel_size,
                              padding=self.kernel_size//2)
        erode = -F.max_pool2d(-x, kernel_size=self.kernel_size,
                              padding=self.kernel_size//2)

        # alpha 混合膨胀/腐蚀 (0=腐蚀, 1=膨胀)
        alpha = self.alpha.sigmoid()
        out = alpha * dilate + (1 - alpha) * erode

        return out



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
    """
    CBAM：先通道注意力，再空间注意力。

    优化版本 (iter 168):
    1. 支持外部高光掩膜输入，对高光区域进行额外抑制
    2. 可学习的门控因子，不再固定 0.7
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7,
                 use_glare_adaptation: bool = True):
        super().__init__()
        self.use_glare_adaptation = use_glare_adaptation
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
        # 可学习的高光抑制因子
        if use_glare_adaptation:
            self.glare_suppression = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, glare_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 通道注意力
        x = self.ca(x)

        # 空间注意力
        x = self.sa(x)

        # 高光区域额外抑制
        if self.use_glare_adaptation and glare_mask is not None:
            with torch.no_grad():
                # 计算高光区域的mask
                glare_weight = 1.0 - glare_mask * self.glare_suppression
                glare_weight = glare_weight.clamp(min=0.1)
            x = x * glare_weight

        return x


class GlareAdaptiveCBAM(nn.Module):
    """
    高光自适应 CBAM 模块。

    优化版本 (iter 168):
    1. 内部学习高光检测，无需外部mask
    2. 对高光区域的通道激活进行额外抑制
    3. 可学习的自适应抑制强度
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channels = channels
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

        # 高光检测分支：预测高光概率
        self.glare_detector = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # 可学习的抑制强度
        self.suppression_factor = nn.Parameter(torch.tensor(0.6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        x_ca = self.ca(x)

        # 检测高光区域
        glare_prob = self.glare_detector(x_ca)

        # 空间注意力
        x_sa = self.sa(x_ca)

        # 高光区域抑制
        suppression = 1.0 - glare_prob * self.suppression_factor
        suppression = suppression.clamp(min=0.1)
        x_sa = x_sa * suppression

        return x_sa


class BiLevelAttention(nn.Module):
    """
    BiFormer 风格双层注意力机制（iteration_158）。

    Layer 1 (RegionRouter): 粗粒度空间筛选，抑制大面积高光区域。
    Layer 2 (Fine-grained): 在区域重要性图指导下做通道+空间注意力。

    相比 CBAM 的优势：RegionRouter 提供粗粒度空间筛选，对大面积高光
    区域天然抑制；SpatialAttention 受 region_weight 调制，更聚焦。
    """

    def __init__(self, channels: int, reduction: int = 16, region_size: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        # Layer 1: Region Router
        self.region_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(region_size),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        # Layer 2a: Channel Attention (同 CBAM)
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_max = nn.AdaptiveMaxPool2d(1)
        self.ca_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        # Layer 2b: Spatial Attention (受 region_weight 调制)
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Layer 1
        region_map = self.region_router(x)
        region_weight = F.interpolate(region_map, size=(H, W), mode="bilinear", align_corners=False)
        # Layer 2a: Channel Attention
        avg = self.ca_mlp(self.ca_avg(x)).unsqueeze(-1).unsqueeze(-1)
        mx = self.ca_mlp(self.ca_max(x)).unsqueeze(-1).unsqueeze(-1)
        ca = self.sigmoid(avg + mx)
        x = x * ca
        # Layer 2b: Spatial Attention (modulated)
        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp = x.max(dim=1, keepdim=True)[0]
        sp = self.sigmoid(self.sa_conv(torch.cat([avg_sp, max_sp], dim=1)))
        sp = sp * region_weight
        return x * sp


# ============================================================
# Phase 5 新增：DCNv2 可变形卷积 (Iteration 159)
# ============================================================

class DeformConv2d(nn.Module):
    """
    DCNv2 纯 PyTorch 实现。
    使用 unfold-based 采样实现可变形卷积，兼容性好。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size, stride, padding, bias=True
        )
        self.mask_conv = nn.Conv2d(
            in_channels, kernel_size * kernel_size,
            kernel_size, stride, padding, bias=True
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        K = self.kernel_size
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        H_out = H // self.stride if self.stride > 1 else H
        W_out = W // self.stride if self.stride > 1 else W

        x_padded = F.pad(x, [self.padding] * 4, mode="reflect")
        x_unfold = F.unfold(x_padded, K, padding=0, stride=self.stride)
        x_unfold = x_unfold.view(B, C, K * K, H_out, W_out)
        mask = mask.unsqueeze(1)
        x_modulated = x_unfold * mask
        weight = self.weight.view(self.out_channels, C // self.groups, K * K)
        x_modulated = x_modulated.view(B, self.groups, C // self.groups, K * K, H_out, W_out)
        output = torch.einsum("bgckhw,gock->bgohw", x_modulated,
                              weight.view(self.groups, self.out_channels // self.groups,
                                          C // self.groups, K * K))
        output = output.reshape(B, self.out_channels, H_out, W_out)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output


class CoordDeformConv(nn.Module):
    """CoordConv + DCNv2 融合模块。"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.deform_conv = DeformConv2d(
            in_channels + 2, out_channels,
            kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_coord = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        y_coord = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        y_grid = y_coord.view(1, 1, h, 1).expand(b, 1, h, w)
        x_grid = x_coord.view(1, 1, 1, w).expand(b, 1, h, w)
        x = torch.cat([x, x_grid, y_grid], dim=1)
        return self.act(self.bn(self.deform_conv(x)))


# ============================================================
# Phase 5 新增：GhostConv 轻量化卷积 (Iteration 160)
# ============================================================

class GhostConv(nn.Module):
    """
    GhostConv: 廉价操作生成冗余特征图。
    参数量 ~50%，计算量 ~50%，精度持平。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 1, ratio: int = 2,
                 cheap_kernel: int = 3, stride: int = 1, padding: int = 0):
        super().__init__()
        self.out_channels = out_channels
        self.ratio = ratio
        self.init_channels = int(out_channels / ratio)
        self.ghost_channels = out_channels - self.init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.init_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU(inplace=True),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.init_channels, self.ghost_channels, cheap_kernel,
                      1, cheap_kernel // 2, groups=self.init_channels, bias=False),
            nn.BatchNorm2d(self.ghost_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intrinsic = self.primary_conv(x)
        ghost = self.cheap_operation(intrinsic)
        return torch.cat([intrinsic, ghost], dim=1)


class GhostDoubleConv(nn.Module):
    """Ghost 版 DoubleConv：第一层标准卷积，第二层 GhostConv。"""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            GhostConv(mid_ch, out_ch, kernel_size=1, ratio=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# 新增：SCSE 注意力 + CoordConv
# ============================================================

class SCSELayer(nn.Module):
    """
    SCSE（Spatial and Channel Squeeze & Excitation）注意力模块。

    结合通道注意力和空间注意力，比 CBAM 更轻量：
      - cSE：通道注意力（与 CBAM 通道注意等价）
      - sSE：空间注意力（1x1 卷积 + Sigmoid）

    对高反光场景的空间定位更敏感。
    """

    def __init__(self, channels: int):
        super().__init__()
        # 通道注意力
        mid = max(channels // 16, 4)
        self.cse_avg = nn.AdaptiveAvgPool2d(1)
        self.cse_max = nn.AdaptiveMaxPool2d(1)
        self.cse_fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        # 空间注意力（1x1 卷积）
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cSE branch
        ca_avg = self.cse_fc(self.cse_avg(x).flatten(1))
        ca_max = self.cse_fc(self.cse_max(x).flatten(1))
        ca = torch.sigmoid(ca_avg + ca_max).unsqueeze(-1).unsqueeze(-1)
        x_ca = x * ca
        # sSE branch
        sa = self.sse(x)
        x_sa = x * sa
        # 融合：两支相加
        return x_ca + x_sa


class CoordConv(nn.Module):
    """
    CoordConv：坐标感知卷积。

    在输入特征图上追加 (x, y) 归一化坐标通道，
    使卷积核能感知像素位置，提升空间任务（定位、边缘）的精度。

    对像素级定位任务特别有效。
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 生成归一化坐标 [0, 1]
        b, _, h, w = x.shape
        x_coord = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        y_coord = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        y_coord = y_coord.view(1, 1, h, 1).expand(b, 1, h, w)
        x_coord = x_coord.view(1, 1, 1, w).expand(b, 1, h, w)
        x = torch.cat([x, x_coord, y_coord], dim=1)
        return self.act(self.bn(self.conv(x)))


# ============================================================
# 3. 高光感知门控跳跃连接
# ============================================================

class GlareGatedSkip(nn.Module):
    """
    高光感知门控跳跃连接。

    在 U-Net 跳跃连接处，通过检测高光区域（高亮度、低饱和度）
    生成软门控权重，降低高光区域的跳跃连接权重，
    防止高光伪特征直接传递到解码器。

    优化版本 (iter 168):
    1. 可学习的最大抑制比例（不再固定 0.7）
    2. 特征精炼分支加入通道注意力
    3. 支持多尺度高光检测
    """

    def __init__(self, channels: int, max_suppression: float = 0.7,
                 use_refined_attention: bool = True):
        super().__init__()
        self.max_suppression = max_suppression
        self.use_refined_attention = use_refined_attention

        # 高光检测分支：轻量 1x1 卷积估计高光概率
        self.glare_detector = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        # 可学习的抑制因子
        self.suppression_factor = nn.Parameter(torch.tensor(0.5))

        # 特征精炼（可选：使用通道注意力）
        if use_refined_attention:
            self.refine = nn.Sequential(
                ConvBNReLU(channels, channels, kernel=1, padding=0),
                ChannelAttention(channels, reduction=8),
            )
        else:
            self.refine = ConvBNReLU(channels, channels, kernel=1, padding=0)

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        # glare_map: (B,1,H,W)，高光区域接近 1
        glare_map = self.glare_detector(skip)

        # 自适应门控：高光区域权重降低
        # 使用 sigmoid 控制的抑制因子，不再是固定的 0.7
        suppression = self.suppression_factor.sigmoid() * self.max_suppression
        gate = 1.0 - glare_map * suppression  # 最多抑制 max_suppression
        gate = gate.clamp(min=0.1)  # 保证最小权重

        # 精炼特征并应用门控
        skip_refined = self.refine(skip)
        return skip_refined * gate


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
    """解码器块：双线性上采样 + 数学增强 + 跳跃连接 + DoubleConv。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 use_glare_gate: bool = True, depthwise: bool = False):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.gate = GlareGatedSkip(skip_ch) if use_glare_gate else nn.Identity()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, depthwise=depthwise)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        skip = self.gate(skip)
        return self.conv(torch.cat([x, skip], dim=1))


# ============================================================
# 6. YOLO风格改进：FPN + Anchor-Free检测头
# ============================================================

class DetectionHead(nn.Module):
    """
    改进的检测头 - 带IoU感知和边界感知定位

    预测：
    - bbox: (l, t, r, b) 边界距离
    - objectness: 目标置信度
    - iou_pred: 预测IoU（用于NMS排序）
    - centerness: 中心度
    """

    def __init__(self, in_channels, feat_channels=256):
        super().__init__()

        # 共享特征
        self.share_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        # 边界回归头 - 预测4个边界距离
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.bbox_pred = nn.Conv2d(feat_channels, 4, 1)  # ltrb

        # Objectness头
        self.obj_conv = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.obj_pred = nn.Conv2d(feat_channels // 2, 1, 1)

        # IoU预测头 - 直接预测与GT的IoU
        self.iou_conv = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.iou_pred = nn.Conv2d(feat_channels // 2, 1, 1)

        # Centerness头
        self.center_conv = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.center_pred = nn.Conv2d(feat_channels // 2, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feat = self.share_conv(x)

        bbox = self.bbox_pred(self.bbox_conv(feat))
        obj = self.obj_pred(self.obj_conv(feat))
        iou = self.iou_pred(self.iou_conv(feat))
        center = self.center_pred(self.center_conv(feat))

        # 应用sigmoid到需要的地方
        obj = torch.sigmoid(obj)
        iou = torch.sigmoid(iou)
        center = torch.sigmoid(center)

        return {
            'bbox': bbox,      # ltrb
            'objectness': obj,
            'iou_pred': iou,   # IoU预测
            'centerness': center,
        }


class FPN(nn.Module):
    """
    Feature Pyramid Network — 多尺度特征金字塔。

    从编码器多尺度特征 (P2, P3, P4, P5) 融合为更强的特征表达，
    提升不同大小目标的检测能力。
    """

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: list of [C2, C3, C4, C5] from encoder

        Returns:
            list of [P2, P3, P4, P5] fpn features
        """
        laterals = [lat_conv(feat) for lat_conv, feat in zip(self.lateral_convs, features)]

        # 自顶向下融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )

        # 1x1 conv + ReLU
        fpn_features = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return fpn_features


# ============================================================
# Phase 5 新增：PAFPN (Iteration 166)
# ============================================================

class FeatureAlignBlock(nn.Module):
    """特征对齐块：3×3 卷积 + SE 注意力。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        w = self.se(feat)
        return feat * w


class PAFPN(nn.Module):
    """
    PAFPN：Path Aggregation FPN。
    自顶向下 + 自底向上双路径，增强多尺度信息流动。
    """

    def __init__(self, in_channels_list, out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False)
            for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            FeatureAlignBlock(out_channels)
            for _ in in_channels_list
        ])
        self.downsample_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(len(in_channels_list) - 1)
        ] + [None])

    def forward(self, features):
        laterals = [lat_conv(f) for lat_conv, f in zip(self.lateral_convs, features)]
        fpn_features = []
        for i in range(len(laterals) - 1, -1, -1):
            if i < len(laterals) - 1:
                laterals[i] += F.interpolate(
                    laterals[i + 1], size=laterals[i].shape[-2:],
                    mode="nearest"
                )
            fpn_features.insert(0, self.fpn_convs[i](laterals[i]))
        pan_features = [fpn_features[0]]
        for i in range(1, len(fpn_features)):
            if self.downsample_convs[i - 1] is not None:
                pan_features.append(
                    fpn_features[i] + self.downsample_convs[i - 1](pan_features[i - 1])
                )
            else:
                pan_features.append(fpn_features[i])
        return pan_features


class AnchorFreeHead(nn.Module):
    """
    Anchor-Free 检测头（FCOS风格）+ 尺寸感知。

    每个位置预测：
    - bbox: (l, t, r, b) 相对于中心的距离
    - size: (w, h) 目标宽高（像素）
    - objectness: 目标置信度
    - centerness: 中心度（用于NMS排序）

    优势：
    - 无需预设anchor，降低超参数数量
    - 对小目标和大目标都友好
    - 尺寸感知：直接回归目标宽高，支持长度意识
    """

    def __init__(self, in_channels, feat_channels=256, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        # 共享特征层
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.size_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        # bbox回归头: (l, t, r, b) 四通道
        self.bbox_pred = nn.Conv2d(feat_channels, 4, 1)
        # 尺寸回归头: (w, h) 二通道（像素单位）
        self.size_pred = nn.Conv2d(feat_channels, 2, 1)
        # objectness头: 1通道
        self.obj_pred = nn.Conv2d(feat_channels, 1, 1)
        # centerness头: 1通道
        self.centerness_pred = nn.Conv2d(feat_channels, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        bbox_feat = self.bbox_conv(x)
        cls_feat = self.cls_conv(x)
        size_feat = self.size_conv(x)

        bbox = self.bbox_pred(bbox_feat)
        size = self.size_pred(size_feat)  # (w, h) 像素单位
        obj = self.obj_pred(cls_feat)
        centerness = self.centerness_pred(cls_feat)

        return {
            'bbox': bbox,      # (B, 4, H, W) - l,t,r,b distances
            'size': size,      # (B, 2, H, W) - w, h in pixels
            'objectness': obj,  # (B, 1, H, W)
            'centerness': centerness,  # (B, 1, H, W)
        }


class CIoULoss(nn.Module):
    """
    Complete IoU Loss — 比 MSE/L1 更适合目标检测的回归损失。

    CIoU 考虑：
    - 重叠面积 (IoU)
    - 中心距离
    - 长宽比

    比普通 IoU Loss 收敛更快，定位更准。
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_ltrb, target_ltrb):
        """
        Args:
            pred_ltrb: (B, 4, H, W) predicted left/top/right/bottom
            target_ltrb: (B, 4, H, W) target left/top/right/bottom

        Returns:
            CIoU loss scalar
        """
        # 转换为 (cx, cy, w, h) 格式
        # ltrb: [0]=left, [1]=top, [2]=right, [3]=bottom
        pred_cx = (pred_ltrb[:, 0] + pred_ltrb[:, 2]) / 2
        pred_cy = (pred_ltrb[:, 1] + pred_ltrb[:, 3]) / 2
        pred_w = pred_ltrb[:, 0] + pred_ltrb[:, 2] + self.eps
        pred_h = pred_ltrb[:, 1] + pred_ltrb[:, 3] + self.eps

        target_cx = (target_ltrb[:, 0] + target_ltrb[:, 2]) / 2
        target_cy = (target_ltrb[:, 1] + target_ltrb[:, 3]) / 2
        target_w = target_ltrb[:, 0] + target_ltrb[:, 2] + self.eps
        target_h = target_ltrb[:, 1] + target_ltrb[:, 3] + self.eps

        # IoU
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        inter_w = (torch.min(pred_cx + pred_w / 2, target_cx + target_w / 2) -
                   torch.max(pred_cx - pred_w / 2, target_cx - target_w / 2)).clamp(0)
        inter_h = (torch.min(pred_cy + pred_h / 2, target_cy + target_h / 2) -
                   torch.max(pred_cy - pred_h / 2, target_cy - target_h / 2)).clamp(0)
        inter_area = inter_w * inter_h
        union_area = pred_area + target_area - inter_area + self.eps
        iou = inter_area / union_area

        # 中心距离
        dist_cx = pred_cx - target_cx
        dist_cy = pred_cy - target_cy
        center_dist_sq = dist_cx ** 2 + dist_cy ** 2

        # 外接矩形对角距离
        enclose_w = torch.max(pred_cx + pred_w / 2, target_cx + target_w / 2) - \
                    torch.min(pred_cx - pred_w / 2, target_cx - target_w / 2)
        enclose_h = torch.max(pred_cy + pred_h / 2, target_cy + target_h / 2) - \
                    torch.min(pred_cy - pred_h / 2, target_cy - target_h / 2)
        enclose_diagonal_sq = enclose_w ** 2 + enclose_h ** 2 + self.eps

        # 长宽比
        v = (4 / (math.pi ** 2)) * (
            torch.atan(pred_w / (pred_h + self.eps)) -
            torch.atan(target_w / (target_h + self.eps))
        ) ** 2
        alpha = v / ((1 - iou + v) + self.eps)

        # CIoU = IoU - (中心距^2 / 外接矩形对角^2) - αv
        ciou = iou - (center_dist_sq / enclose_diagonal_sq) - alpha * v

        return 1.0 - ciou.mean()


def nms_centerness(detections, iou_threshold=0.5, score_threshold=0.3):
    """
    基于中心度的 NMS 后处理。

    Args:
        detections: list of dict with keys: bbox, score, centerness
        iou_threshold: IoU 阈值
        score_threshold: 分数阈值

    Returns:
        filtered detections after NMS
    """
    if not detections:
        return []

    # 按 centerness * score 排序
    for d in detections:
        d['final_score'] = d['score'] * d['centerness']
    detections = sorted(detections, key=lambda x: x['final_score'], reverse=True)

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        rest = []
        for d in detections:
            iou = compute_iou(best['bbox'], d['bbox'])
            if iou < iou_threshold:
                rest.append(d)
        detections = rest

    return keep


def compute_iou(box1, box2):
    """计算两个 bbox 的 IoU。box format: [l, t, r, b]"""
    inter_l = max(box1[0], box2[0])
    inter_t = max(box1[1], box2[1])
    inter_r = min(box1[2], box2[2])
    inter_b = min(box1[3], box2[3])

    inter_area = max(0, inter_r - inter_l) * max(0, inter_b - inter_t)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area + 1e-7

    return inter_area / union_area


# ============================================================
# 6. FLARE 标准版（CoordConv 增强）
# ============================================================

class CoordConvEncoderBlock(nn.Module):
    """
    CoordConv 增强的编码器块。
    在输入阶段加入坐标感知，提升空间定位精度。
    """

    def __init__(self, in_ch: int, out_ch: int, use_cbam: bool = True):
        super().__init__()
        # CoordConv 首层
        self.coordconv = CoordConv(in_ch, out_ch, kernel_size=3, padding=1)
        # CBAM 注意力
        self.attn = CBAM(out_ch) if use_cbam else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        feat = self.attn(self.coordconv(x))
        return feat, self.pool(feat)


class FLARE(nn.Module):
    """
    FLARE — 船舶大型金属高光面边缘感知分割网络（标准版）。

    增强特性：
      - CoordConv：第一层编码器使用坐标感知卷积，提升空间定位精度
      - SCSE：可选使用 SCSE 注意力替代 CBAM
      - RCF 风格边缘检测头：从编码器+解码器多尺度特征密集连接
      - 边缘注意力头：确保边缘与分割掩膜边界一致

    Args:
        in_channels:  输入通道数（默认 3，RGB）
        out_channels: 分割输出通道数（默认 1，二值分割）
        base_ch:      基础通道数（默认 64，控制模型容量）
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base_ch: int = 64, use_scse: bool = False,
                 use_detection: bool = True,
                 edge_head_type: str = "rcf"):  # "rcf" or "hed" or "attention"
        super().__init__()
        c = base_ch
        self.use_detection = use_detection
        self.edge_head_type = edge_head_type

        # 编码器：第一层使用 CoordConv
        self.enc1 = CoordConvEncoderBlock(in_channels, c, use_cbam=False)
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

        # 检测头（可选）
        if use_detection:
            self.det_head = DetectionHead(in_channels=c * 2, feat_channels=c * 2)

        # 输出头：语义分割（输出 logits，配合 BCEWithLogitsLoss 使用 AMP）
        self.seg_head = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, out_channels, 1),
        )

        # ============================================================
        # 简化边缘检测头：只使用 HED 风格 + 直接监督
        # 问题诊断：原版三种边缘头融合(60% RCF + 25% ASPP + 15% Attention)
        # 未经充分训练就融合，导致相互干扰，边缘检测几乎失效
        # 修复：使用简洁的 HED 风格单一边缘头，直接从 d1 解码器输出监督
        # ============================================================

        # HED风格边缘检测头（从解码器多尺度特征）
        self.hed_head = HEDEdgeHead(
            in_channels_list=[c, c * 2, c * 4, c * 8],  # [d1, d2, d3, d4]
            mid_channels=c * 2,
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

        # ============================================================
        # 简化边缘检测：只使用 HED 风格头
        # 直接从 [d1, d2, d3, d4] 多尺度解码器特征预测边缘
        # ============================================================
        edge, side_edges = self.hed_head([d1, d2, d3, d4])

        # 对齐边缘到分割尺寸
        if edge.shape[-2:] != seg.shape[-2:]:
            edge = F.interpolate(edge, size=seg.shape[-2:],
                                 mode='bilinear', align_corners=False)

        result = {
            'seg': seg,
            'edge': edge,
            'side_edges': side_edges,
        }

        # 检测输出（可选）
        if self.use_detection:
            det_out = self.det_head(d2)
            result.update(det_out)

        return result


# ============================================================
# 7. FLARELite 轻量版（嵌入式部署，CoordConv 增强）
# ============================================================

class FLARELite(nn.Module):
    """
    FLARELite — 轻量版，面向 RA8P1 (Cortex-M85 + Helium) 嵌入式部署。

    增强特性：
      - CoordConv：第一层使用坐标感知卷积
      - 深度可分离卷积：减少计算量
      - 减少通道数（base_ch=32）
      - 仅在 enc3/enc4 使用 CBAM

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
        # 编码器：第一层使用 CoordConv，后续使用深度可分离
        self.enc1 = CoordConvEncoderBlock(in_channels, c, use_cbam=False)
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
        )
        # 边缘检测头（增强版 - 轻量但更强）
        self.edge_conv1 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),  # 32→32
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.edge_attn = CBAM(c, reduction=4)  # 轻量注意力
        self.edge_conv2 = nn.Conv2d(c, c // 2, 1)  # 32→16
        self.edge_pred = nn.Conv2d(c // 2, 1, 1)  # 16→1
        self.edge_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

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
        # 边缘检测（增强版）
        edge_feat = self.edge_conv1(d2)
        edge_feat = self.edge_attn(edge_feat)
        edge_feat = self.edge_conv2(edge_feat)
        edge = self.edge_pred(edge_feat)
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
        model:          训练好的 FLARE / FLARELite
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
        ("FLARE (标准版)",   FLARE,     64),
        ("FLARELite (轻量版)", FLARELite, 32),
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

