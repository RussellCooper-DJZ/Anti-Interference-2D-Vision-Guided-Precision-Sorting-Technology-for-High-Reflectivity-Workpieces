"""
test_feature_extraction_extended.py — Phase 5 新增模块测试
补充 DeformConv2d, GhostConv, PAFPN, EdgeRefinementHead, WaveletScattering 等
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytest
import numpy as np

from vision.feature_extraction import (
    DeformConv2d, CoordDeformConv,
    GhostConv, GhostDoubleConv,
    PAFPN, FeatureAlignBlock,
    EdgeRefinementHead,
    WaveletScattering,
    BiLevelAttention, CBAM,
)


# ============================================================
# DeformConv2d / CoordDeformConv
# ============================================================

class TestDeformConv2d:
    """DCNv2 可变形卷积测试"""

    def test_forward_shape(self):
        dcn = DeformConv2d(16, 32, 3, padding=1)
        x = torch.randn(2, 16, 32, 32)
        out = dcn(x)
        assert out.shape == (2, 32, 32, 32)

    def test_forward_stride2(self):
        dcn = DeformConv2d(16, 32, 3, stride=2, padding=1)
        x = torch.randn(1, 16, 32, 32)
        out = dcn(x)
        assert out.shape == (1, 32, 16, 16)

    def test_forward_different_sizes(self):
        dcn = DeformConv2d(8, 16, 3, padding=1)
        for h, w in [(16, 16), (32, 64), (128, 128)]:
            x = torch.randn(1, 8, h, w)
            out = dcn(x)
            assert out.shape[:2] == (1, 16)

    def test_gradient_flow(self):
        dcn = DeformConv2d(8, 16, 3, padding=1)
        x = torch.randn(1, 8, 16, 16, requires_grad=True)
        out = dcn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_parameter_count(self):
        """DCNv2 参数量应大于标准卷积"""
        import torch.nn as nn
        std = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        dcn = DeformConv2d(16, 32, 3, padding=1, bias=False)
        std_p = sum(p.numel() for p in std.parameters())
        dcn_p = sum(p.numel() for p in dcn.parameters())
        assert dcn_p > std_p, "DCNv2 should have more params than standard conv"


class TestCoordDeformConv:
    """CoordConv + DCNv2 融合测试"""

    def test_forward_shape(self):
        cdc = CoordDeformConv(3, 64, 3, padding=1)
        x = torch.randn(1, 3, 64, 64)
        out = cdc(x)
        assert out.shape == (1, 64, 64, 64)

    def test_coord_channels(self):
        """验证坐标通道被正确追加"""
        cdc = CoordDeformConv(3, 16, 3, padding=1)
        x = torch.randn(1, 3, 32, 32)
        out = cdc(x)
        assert out.shape == (1, 16, 32, 32)


# ============================================================
# GhostConv / GhostDoubleConv
# ============================================================

class TestGhostConv:
    """GhostConv 轻量化卷积测试"""

    def test_forward_shape(self):
        gc = GhostConv(32, 64, kernel_size=1, ratio=2)
        x = torch.randn(2, 32, 16, 16)
        out = gc(x)
        assert out.shape == (2, 64, 16, 16)

    def test_parameter_reduction(self):
        """GhostConv 参数量应小于标准卷积"""
        import torch.nn as nn
        std = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        ghost = GhostConv(64, 128, kernel_size=3, ratio=2, cheap_kernel=3, padding=1)
        std_p = sum(p.numel() for p in std.parameters())
        ghost_p = sum(p.numel() for p in ghost.parameters())
        assert ghost_p < std_p, "GhostConv should reduce params"

    def test_different_ratios(self):
        """测试不同 ratio"""
        for ratio in [2, 4]:
            gc = GhostConv(16, 32, kernel_size=1, ratio=ratio)
            x = torch.randn(1, 16, 8, 8)
            out = gc(x)
            assert out.shape == (1, 32, 8, 8)

    def test_gradient_flow(self):
        gc = GhostConv(8, 16, kernel_size=1, ratio=2)
        x = torch.randn(1, 8, 8, 8, requires_grad=True)
        out = gc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestGhostDoubleConv:
    """GhostDoubleConv 测试"""

    def test_forward_shape(self):
        gdc = GhostDoubleConv(32, 64)
        x = torch.randn(1, 32, 16, 16)
        out = gdc(x)
        assert out.shape == (1, 64, 16, 16)


# ============================================================
# PAFPN / FeatureAlignBlock
# ============================================================

class TestFeatureAlignBlock:
    """特征对齐块测试"""

    def test_forward_shape_preserved(self):
        fab = FeatureAlignBlock(64)
        x = torch.randn(2, 64, 32, 32)
        out = fab(x)
        assert out.shape == x.shape

    def test_se_attention_range(self):
        """SE注意力输出应在合理范围"""
        fab = FeatureAlignBlock(32)
        x = torch.randn(1, 32, 16, 16)
        out = fab(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestPAFPN:
    """PAFPN 测试"""

    @pytest.fixture
    def encoder_features(self):
        """模拟编码器多尺度特征"""
        return [
            torch.randn(1, 64, 128, 128),
            torch.randn(1, 128, 64, 64),
            torch.randn(1, 256, 32, 32),
            torch.randn(1, 512, 16, 16),
        ]

    def test_forward_output_count(self, encoder_features):
        pafpn = PAFPN([64, 128, 256, 512], out_channels=256)
        outputs = pafpn(encoder_features)
        assert len(outputs) == 4

    def test_forward_output_channels(self, encoder_features):
        pafpn = PAFPN([64, 128, 256, 512], out_channels=128)
        outputs = pafpn(encoder_features)
        for out in outputs:
            assert out.shape[1] == 128

    def test_forward_spatial_sizes(self, encoder_features):
        pafpn = PAFPN([64, 128, 256, 512], out_channels=256)
        outputs = pafpn(encoder_features)
        expected_sizes = [(128, 128), (64, 64), (32, 32), (16, 16)]
        for out, (h, w) in zip(outputs, expected_sizes):
            assert out.shape[-2:] == (h, w)

    def test_single_level(self):
        """测试单层级输入"""
        pafpn = PAFPN([64], out_channels=32)
        feats = [torch.randn(1, 64, 16, 16)]
        outputs = pafpn(feats)
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 32, 16, 16)

    def test_gradient_flow(self, encoder_features):
        pafpn = PAFPN([64, 128, 256, 512], out_channels=128)
        feats = [f.clone().requires_grad_(True) for f in encoder_features]
        outputs = pafpn(feats)
        loss = sum(o.sum() for o in outputs)
        loss.backward()
        assert all(f.grad is not None for f in feats)


# ============================================================
# EdgeRefinementHead
# ============================================================

class TestEdgeRefinementHead:
    """边缘细化头测试"""

    def test_forward_shape(self):
        er = EdgeRefinementHead(edge_channels=1, seg_channels=1, mid_channels=32)
        edge = torch.randn(1, 1, 64, 64)
        seg = torch.randn(1, 1, 64, 64)
        out = er(edge, seg)
        assert out.shape == (1, 1, 64, 64)

    def test_output_range(self):
        """输出应在 [0, 1]（Sigmoid）"""
        er = EdgeRefinementHead()
        edge = torch.randn(1, 1, 32, 32)
        seg = torch.randn(1, 1, 32, 32)
        out = er(edge, seg)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_different_sizes(self):
        er = EdgeRefinementHead()
        for h, w in [(16, 16), (32, 64), (128, 128)]:
            edge = torch.randn(1, 1, h, w)
            seg = torch.randn(1, 1, h, w)
            out = er(edge, seg)
            assert out.shape == (1, 1, h, w)

    def test_size_mismatch(self):
        """测试 edge 和 seg 尺寸不同时的插值"""
        er = EdgeRefinementHead()
        edge = torch.randn(1, 1, 32, 32)
        seg = torch.randn(1, 1, 64, 64)
        out = er(edge, seg)
        assert out.shape == (1, 1, 64, 64)

    def test_batch_size(self):
        er = EdgeRefinementHead()
        for b in [1, 2, 4]:
            edge = torch.randn(b, 1, 32, 32)
            seg = torch.randn(b, 1, 32, 32)
            out = er(edge, seg)
            assert out.shape[0] == b

    def test_gradient_flow(self):
        er = EdgeRefinementHead()
        edge = torch.randn(1, 1, 16, 16, requires_grad=True)
        seg = torch.randn(1, 1, 16, 16, requires_grad=True)
        out = er(edge, seg)
        loss = out.sum()
        loss.backward()
        assert edge.grad is not None
        assert seg.grad is not None


# ============================================================
# WaveletScattering
# ============================================================

class TestWaveletScattering:
    """小波散射变换测试"""

    def test_forward_shape(self):
        ws = WaveletScattering(J=2, L=4)
        x = torch.randn(1, 3, 64, 64)
        out = ws(x)
        # 原始 3 + 3*2*4 = 27 通道
        assert out.shape == (1, 27, 64, 64)

    def test_different_jl(self):
        for J, L in [(1, 4), (2, 4), (2, 8)]:
            ws = WaveletScattering(J=J, L=L)
            x = torch.randn(1, 3, 64, 64)
            out = ws(x)
            expected_c = 3 + 3 * J * L
            assert out.shape == (1, expected_c, 64, 64)

    def test_spatial_size_preserved(self):
        ws = WaveletScattering(J=2, L=4)
        for h, w in [(32, 32), (64, 128), (128, 128)]:
            x = torch.randn(1, 3, h, w)
            out = ws(x)
            assert out.shape[-2:] == (h, w)

    def test_gradient_flow(self):
        ws = WaveletScattering(J=2, L=4)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        out = ws(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_nan_inf(self):
        ws = WaveletScattering(J=3, L=8)
        x = torch.randn(2, 3, 64, 64)
        out = ws(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ============================================================
# BiLevelAttention / CBAM
# ============================================================

class TestBiLevelAttention:
    """BiLevelAttention 测试"""

    def test_forward_shape(self):
        bla = BiLevelAttention(channels=64, reduction=8)
        x = torch.randn(2, 64, 32, 32)
        out = bla(x)
        assert out.shape == x.shape

    def test_forward_different_sizes(self):
        bla = BiLevelAttention(channels=32, reduction=4)
        for h, w in [(16, 16), (32, 64), (128, 128)]:
            x = torch.randn(1, 32, h, w)
            out = bla(x)
            assert out.shape == (1, 32, h, w)

    def test_parameter_count(self):
        """BiLevelAttention 参数量应接近 CBAM"""
        cbam = CBAM(channels=64, reduction=8)
        bla = BiLevelAttention(channels=64, reduction=8)
        cbam_p = sum(p.numel() for p in cbam.parameters())
        bla_p = sum(p.numel() for p in bla.parameters())
        # 允许 ±20% 差异
        assert 0.8 <= bla_p / cbam_p <= 1.5

    def test_gradient_flow(self):
        bla = BiLevelAttention(channels=16, reduction=4)
        x = torch.randn(1, 16, 16, 16, requires_grad=True)
        out = bla(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
