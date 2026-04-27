"""
pytest 共享 fixtures
为所有测试模块提供统一的测试数据和配置
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
import cv2


# ============================================================
# 图像 fixtures
# ============================================================

@pytest.fixture
def small_image():
    """小型测试图像 (100, 100, 3) uint8"""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def medium_image():
    """中型测试图像 (256, 256, 3) uint8"""
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    # 添加一个矩形工件
    cv2.rectangle(img, (80, 80), (180, 180), (200, 200, 200), -1)
    return img


@pytest.fixture
def highlight_image():
    """带高光区域的测试图像"""
    img = np.full((256, 256, 3), 100, dtype=np.uint8)
    # 添加高光
    cv2.circle(img, (128, 128), 40, (255, 255, 255), -1)
    return img


@pytest.fixture
def edge_mask():
    """二值边缘掩膜"""
    mask = np.zeros((128, 128), dtype=np.uint8)
    # 画一个矩形边缘
    cv2.rectangle(mask, (30, 30), (100, 100), 255, 2)
    return mask


@pytest.fixture
def seg_mask():
    """二值分割掩膜"""
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(mask, (30, 30), (100, 100), 255, -1)
    return mask


# ============================================================
# PyTorch fixtures
# ============================================================

@pytest.fixture
def device():
    """测试设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_tensor():
    """小型输入张量 (1, 3, 64, 64)"""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def medium_tensor():
    """中型输入张量 (2, 3, 128, 128)"""
    return torch.randn(2, 3, 128, 128)


@pytest.fixture
def batch_tensor():
    """批次输入张量 (2, 3, 256, 256)"""
    return torch.randn(2, 3, 256, 256)


# ============================================================
# 模型 fixtures
# ============================================================

@pytest.fixture
def flare_small():
    """小型 FLARE 模型（用于快速测试）"""
    from vision.feature_extraction import FLARE
    return FLARE(in_channels=3, base_ch=16)


@pytest.fixture
def flarelite_small():
    """小型 FLARELite 模型"""
    from vision.feature_extraction import FLARELite
    return FLARELite(in_channels=3, base_ch=16)
