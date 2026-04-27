"""
test_feature_extraction.py — FLARE 模型测试
验证模型结构、输出形状、梯度流动
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytest
import numpy as np

from vision.feature_extraction import (
    FLARE, FLARELite, get_model_info,
    CoordConvEncoderBlock, SCSELayer,
)


class TestFLARE:
    """FLARE 标准版测试"""

    @pytest.fixture
    def model(self):
        return FLARE(in_channels=3, base_ch=32)  # 小模型加速测试

    @pytest.fixture
    def batch_input(self):
        return torch.randn(2, 3, 256, 256)

    def test_output_shape(self, model, batch_input):
        """测试输出形状"""
        model.eval()
        with torch.no_grad():
            outputs = model(batch_input)

        assert 'seg' in outputs
        assert 'edge' in outputs
        assert outputs['seg'].shape == (2, 1, 256, 256)
        assert outputs['edge'].shape == (2, 1, 256, 256)

    def test_output_range(self, model, batch_input):
        """测试输出值范围 [0, 1]"""
        model.eval()
        with torch.no_grad():
            outputs = model(batch_input)

        seg = torch.sigmoid(outputs['seg'])
        edge = torch.sigmoid(outputs['edge'])

        assert seg.min() >= 0.0
        assert seg.max() <= 1.0
        assert edge.min() >= 0.0
        assert edge.max() <= 1.0

    def test_gradient_flow(self, model, batch_input):
        """测试梯度流动"""
        model.train()
        outputs = model(batch_input)
        loss = outputs['seg'].mean() + outputs['edge'].mean()
        loss.backward()

        # 检查编码器参数有梯度
        for name, param in model.named_parameters():
            if 'enc1' in name and param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"

    def test_model_info(self, model):
        """测试模型信息"""
        info = get_model_info(model)
        assert 'total_params' in info
        assert 'trainable_params' in info
        assert info['total_params'] > 0


class TestFLARELite:
    """FLARELite 轻量版测试"""

    @pytest.fixture
    def model(self):
        return FLARELite(in_channels=3, base_ch=16)

    @pytest.fixture
    def batch_input(self):
        return torch.randn(1, 3, 256, 256)

    def test_output_shape(self, model, batch_input):
        """测试输出形状"""
        model.eval()
        with torch.no_grad():
            outputs = model(batch_input)

        assert outputs['seg'].shape == (1, 1, 256, 256)
        assert outputs['edge'].shape == (1, 1, 256, 256)

    def test_lightweight(self):
        """测试轻量版参数更少"""
        standard = FLARE(in_channels=3, base_ch=32)
        lite = FLARELite(in_channels=3, base_ch=16)

        standard_info = get_model_info(standard)
        lite_info = get_model_info(lite)

        assert lite_info['total_params'] < standard_info['total_params']


class TestCoordConv:
    """CoordConv 测试"""

    def test_coordconv_output(self):
        """测试 CoordConv 输出形状"""
        layer = CoordConvEncoderBlock(in_ch=3, out_ch=32)
        x = torch.randn(1, 3, 64, 64)
        feat, pooled = layer(x)

        assert feat.shape[1] == 32
        assert pooled.shape[1] == 32
        assert feat.shape[2:] == x.shape[2:]  # Spatial size preserved


class TestSCSE:
    """SCSE 注意力测试"""

    def test_scse_output_shape(self):
        """测试 SCSE 输出形状不变"""
        layer = SCSELayer(channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = layer(x)

        assert out.shape == x.shape

    def test_scse_attention(self):
        """测试 SCSE 注意力机制"""
        layer = SCSELayer(channels=64)
        x = torch.randn(1, 64, 32, 32)

        # 无输入时也应有输出
        out = layer(x)
        assert not torch.isnan(out).any()


class TestCoordConvStandalone:
    """独立 CoordConv 层测试"""

    def test_coordconv_standalone(self):
        """测试独立 CoordConv 层"""
        from vision.feature_extraction import CoordConv

        layer = CoordConv(in_channels=3, out_channels=16)
        x = torch.randn(1, 3, 32, 32)
        out = layer(x)

        assert out.shape[1] == 16
        assert out.shape[2:] == x.shape[2:]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
