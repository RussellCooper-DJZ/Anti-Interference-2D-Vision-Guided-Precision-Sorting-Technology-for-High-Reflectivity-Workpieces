"""
test_data_augmentation.py — 数据增强模块测试
基于实际 API 重写
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import cv2
import torch

from data.data_augmentation import (
    ShipHullAugPipeline,
    random_brightness_contrast,
    random_gamma,
    random_sun_glare,
    random_gaussian_noise,
    random_motion_blur,
    random_hsv_jitter,
    random_erasing,
    generate_edge_from_mask,
    cutmix,
    mixup,
)


# ============================================================
# ShipHullAugPipeline
# ============================================================

class TestShipHullAugPipeline:
    """船舶专项增强管线测试"""

    @pytest.fixture
    def sample_image_mask_edge(self):
        """创建测试图像、掩膜和边缘"""
        image = np.full((256, 256, 3), 128, dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        edge = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(image, (80, 80), (180, 180), (200, 200, 200), -1)
        cv2.rectangle(mask, (80, 80), (180, 180), 255, -1)
        cv2.rectangle(edge, (80, 80), (180, 180), 255, 2)
        return image, mask, edge

    def test_pipeline_output_shapes(self, sample_image_mask_edge):
        image, mask, edge = sample_image_mask_edge
        aug = ShipHullAugPipeline(p=1.0)
        img_aug, mask_aug, edge_aug = aug(image, mask, edge)
        assert img_aug.shape == image.shape
        assert mask_aug.shape == mask.shape
        assert edge_aug.shape == edge.shape

    def test_pipeline_preserves_dtype(self, sample_image_mask_edge):
        image, mask, edge = sample_image_mask_edge
        aug = ShipHullAugPipeline(p=1.0)
        img_aug, mask_aug, edge_aug = aug(image, mask, edge)
        assert img_aug.dtype == np.uint8
        assert mask_aug.dtype == np.uint8
        assert edge_aug.dtype == np.uint8

    def test_mask_binary(self, sample_image_mask_edge):
        """增强后的 mask 应保持二值"""
        image, mask, edge = sample_image_mask_edge
        aug = ShipHullAugPipeline(p=1.0)
        _, mask_aug, _ = aug(image, mask, edge)
        unique = set(np.unique(mask_aug))
        assert unique.issubset({0, 255})

    def test_empty_mask(self):
        """空掩膜测试"""
        image = np.full((128, 128, 3), 128, dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        edge = np.zeros((128, 128), dtype=np.uint8)
        aug = ShipHullAugPipeline(p=1.0)
        img_aug, mask_aug, edge_aug = aug(image, mask, edge)
        assert mask_aug.max() == 0

    def test_probability_zero(self, sample_image_mask_edge):
        """p=0 时不应增强"""
        image, mask, edge = sample_image_mask_edge
        aug = ShipHullAugPipeline(p=0.0)
        img_aug, mask_aug, edge_aug = aug(image, mask, edge)
        np.testing.assert_array_equal(img_aug, image)
        np.testing.assert_array_equal(mask_aug, mask)


# ============================================================
# 单项增强函数测试
# ============================================================

class TestBrightnessContrast:
    """亮度/对比度测试"""

    def test_output_shape(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = random_brightness_contrast(img)
        assert result.shape == img.shape


class TestGamma:
    """Gamma 校正测试"""

    def test_output_shape(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = random_gamma(img)
        assert result.shape == img.shape

    def test_gamma_with_range(self):
        """测试 gamma 范围参数"""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = random_gamma(img, gamma_range=(1.0, 2.0))
        assert result.shape == img.shape


class TestSunGlare:
    """太阳光模拟测试"""

    def test_output_shape(self):
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        result = random_sun_glare(img, n_glares=3)
        assert result.shape == img.shape


class TestGaussianNoise:
    """高斯噪声测试"""

    def test_output_shape(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = random_gaussian_noise(img)
        assert result.shape == img.shape

    def test_noise_changes_image(self):
        """噪声应改变图像"""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = random_gaussian_noise(img, std_range=(25, 25), p=1.0)
        assert not np.array_equal(result, img)


class TestMotionBlur:
    """运动模糊测试"""

    def test_output_shape(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = random_motion_blur(img)
        assert result.shape == img.shape


class TestHSVJitter:
    """HSV 扰动测试"""

    def test_output_shape(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = random_hsv_jitter(img)
        assert result.shape == img.shape

    def test_output_range(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = random_hsv_jitter(img)
        assert result.min() >= 0
        assert result.max() <= 255


class TestRandomErasing:
    """Random Erasing 测试"""

    def test_output_shape(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = random_erasing(img)
        assert result.shape == img.shape

    def test_erasing_changes_image(self):
        """擦除应改变图像"""
        img = np.full((256, 256, 3), 200, dtype=np.uint8)
        result = random_erasing(img, n_patches=3, p=1.0)
        assert not np.array_equal(result, img)


# ============================================================
# 边缘生成测试
# ============================================================

class TestGenerateEdgeFromMask:
    """从掩膜生成边缘测试"""

    def test_basic(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (30, 30), (70, 70), 255, -1)
        edge = generate_edge_from_mask(mask)
        assert edge.shape == mask.shape
        assert edge.dtype == np.uint8
        assert edge.max() > 0

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        edge = generate_edge_from_mask(mask)
        assert edge.max() == 0

    def test_full_mask(self):
        mask = np.full((100, 100), 255, dtype=np.uint8)
        edge = generate_edge_from_mask(mask, edge_width=3)
        # 全满掩膜可能没有内部边缘（取决于实现）
        assert edge.shape == mask.shape
        assert edge.dtype == np.uint8


# ============================================================
# CutMix / MixUp 测试
# ============================================================

class TestCutMix:
    """CutMix 测试"""

    def test_output_shapes(self):
        imgs = torch.randn(4, 3, 64, 64)
        masks = torch.randint(0, 3, (4, 1, 64, 64)).float()
        edges = torch.zeros_like(masks)
        out_imgs, out_masks, out_edges = cutmix(imgs.clone(), masks.clone(), edges.clone(), imgs.clone(), masks.clone(), edges.clone(), alpha=1.0)
        assert out_imgs.shape == imgs.shape
        assert out_masks.shape == masks.shape
        assert out_edges.shape == edges.shape


class TestMixUp:
    """MixUp 测试"""

    def test_output_shapes(self):
        imgs = torch.randn(4, 3, 64, 64)
        masks = torch.randint(0, 3, (4, 1, 64, 64)).float()
        edges = torch.zeros_like(masks)
        out_imgs, out_masks, out_edges = mixup(imgs.clone(), masks.clone(), edges.clone(), imgs.clone(), masks.clone(), edges.clone(), alpha=0.4)
        assert out_imgs.shape == imgs.shape
        assert out_masks.shape == masks.shape
        assert out_edges.shape == edges.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
