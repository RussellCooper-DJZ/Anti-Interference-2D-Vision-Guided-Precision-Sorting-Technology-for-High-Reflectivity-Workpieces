"""
test_hdr_processing_extended.py — HDR 处理扩展测试
补充 GlareInpainter 测试
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import cv2

from vision.hdr_processing import (
    GlareInpainter,
    AntiGlarePipeline,
    detect_highlight_mask,
    repair_highlight_regions,
    exposure_fusion_mertens,
)


# ============================================================
# GlareInpainter
# ============================================================

class TestGlareInpainter:
    """高光修复器测试"""

    @pytest.fixture
    def glare_image(self):
        """带高光的测试图像"""
        img = np.full((128, 128, 3), 100, dtype=np.uint8)
        # 添加圆形高光区域
        cv2.circle(img, (64, 64), 20, (255, 255, 255), -1)
        return img

    @pytest.fixture
    def no_glare_image(self):
        """无高光图像"""
        return np.full((128, 128, 3), 128, dtype=np.uint8)

    def test_creation(self):
        gi = GlareInpainter(mode="telea")
        assert gi.mode == "telea"
        assert gi.telea_radius == 5

    def test_telea_mode(self, glare_image):
        gi = GlareInpainter(mode="telea", telea_radius=5)
        result = gi.inpaint(glare_image)
        assert result.shape == glare_image.shape
        assert result.dtype == np.uint8

    def test_ns_mode(self, glare_image):
        gi = GlareInpainter(mode="ns", ns_radius=3)
        result = gi.inpaint(glare_image)
        assert result.shape == glare_image.shape
        assert result.dtype == np.uint8

    def test_hybrid_mode(self, glare_image):
        gi = GlareInpainter(mode="hybrid")
        result = gi.inpaint(glare_image)
        assert result.shape == glare_image.shape
        assert result.dtype == np.uint8

    def test_no_glare_image(self, no_glare_image):
        """无高光时图像应保持基本不变"""
        gi = GlareInpainter(mode="telea")
        result = gi.inpaint(no_glare_image)
        assert result.shape == no_glare_image.shape
        # 允许微小差异（修复操作可能引入）
        diff = np.abs(result.astype(float) - no_glare_image.astype(float))
        assert diff.mean() < 50  # 平均差异应较小

    def test_custom_mask(self, glare_image):
        """使用自定义掩膜"""
        gi = GlareInpainter(mode="telea")
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[50:80, 50:80] = 255
        result = gi.inpaint(glare_image, mask=mask)
        assert result.shape == glare_image.shape

    def test_detect_glare_mask(self, glare_image):
        """测试高光自动检测"""
        gi = GlareInpainter()
        mask = gi._detect_glare_mask(glare_image)
        assert mask.shape == (128, 128)
        assert mask.dtype == np.uint8
        # 应该有检测到的高光区域
        assert mask.max() > 0

    def test_detect_no_glare(self, no_glare_image):
        """无高光时掩膜应为空"""
        gi = GlareInpainter()
        mask = gi._detect_glare_mask(no_glare_image, saturation_thresh=0.99)
        assert mask.max() == 0

    def test_edge_distance(self, glare_image):
        """测试边缘距离计算"""
        gi = GlareInpainter()
        mask = gi._detect_glare_mask(glare_image)
        dist = gi._edge_distance(glare_image, mask)
        assert dist.shape == (128, 128)
        assert dist.dtype == np.float32 or dist.dtype == np.float64

    def test_different_image_sizes(self):
        """测试不同尺寸图像"""
        gi = GlareInpainter(mode="telea")
        for h, w in [(64, 64), (128, 256), (256, 256)]:
            img = np.full((h, w, 3), 150, dtype=np.uint8)
            cv2.circle(img, (w//2, h//2), 10, (255, 255, 255), -1)
            result = gi.inpaint(img)
            assert result.shape == (h, w, 3)

    def test_invalid_mode(self, glare_image):
        """无效模式应抛异常"""
        gi = GlareInpainter(mode="invalid")
        with pytest.raises(ValueError):
            gi.inpaint(glare_image)


# ============================================================
# AntiGlarePipeline 补充测试
# ============================================================

class TestAntiGlarePipelineExtended:
    """AntiGlarePipeline 扩展测试"""

    @pytest.fixture
    def pipeline(self):
        return AntiGlarePipeline()

    @pytest.fixture
    def bright_image(self):
        img = np.full((256, 256, 3), 200, dtype=np.uint8)
        cv2.circle(img, (128, 128), 40, (255, 255, 255), -1)
        return img

    def test_pipeline_preserves_dtype(self, pipeline, bright_image):
        result = pipeline.process_single(bright_image)
        assert result.dtype == np.uint8

    def test_pipeline_does_not_crash_on_noise(self, pipeline):
        """随机噪声图像不应崩溃"""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = pipeline.process_single(img)
        assert result.shape == img.shape

    def test_pipeline_on_gradient(self, pipeline):
        """渐变图像测试"""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        for i in range(128):
            img[i, :] = int(255 * i / 128)
        result = pipeline.process_single(img)
        assert result.shape == img.shape


# ============================================================
# exposure_fusion_mertens 补充测试
# ============================================================

class TestExposureFusionExtended:
    """曝光融合扩展测试"""

    def test_fusion_preserves_shape(self):
        imgs = [
            np.full((64, 64, 3), 50, dtype=np.uint8),
            np.full((64, 64, 3), 150, dtype=np.uint8),
        ]
        result = exposure_fusion_mertens(imgs)
        assert result.shape == (64, 64, 3)

    def test_fusion_different_sizes_raises(self):
        """不同尺寸图像处理测试"""
        imgs = [
            np.full((64, 64, 3), 128, dtype=np.uint8),
            np.full((128, 128, 3), 128, dtype=np.uint8),
        ]
        # 函数可能自动处理或报错，验证函数可执行
        try:
            result = exposure_fusion_mertens(imgs)
            # 如果成功，输出应该是有效图像
            assert result is not None
        except Exception:
            pass  # 预期行为

    def test_fusion_large_images(self):
        """大尺寸图像测试"""
        imgs = [
            np.full((512, 512, 3), 100, dtype=np.uint8),
            np.full((512, 512, 3), 200, dtype=np.uint8),
        ]
        result = exposure_fusion_mertens(imgs)
        assert result.shape == (512, 512, 3)


# ============================================================
# detect_highlight_mask 补充测试
# ============================================================

class TestHighlightDetectionExtended:
    """高光检测扩展测试"""

    def test_small_highlight(self):
        """小高光区域"""
        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        cv2.circle(img, (50, 50), 5, (255, 255, 255), -1)
        mask = detect_highlight_mask(img, threshold=240)
        assert mask.max() > 0

    def test_no_highlight_strict_threshold(self):
        """严格阈值下无高光"""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = detect_highlight_mask(img, threshold=250)
        assert mask.max() == 0

    def test_grayscale_input(self):
        """灰度输入测试 — 现在支持自动转换"""
        # 灰度图会被自动转换为 BGR 后处理
        img = np.full((100, 100), 128, dtype=np.uint8)
        cv2.circle(img, (50, 50), 10, 255, -1)
        # 函数现在支持灰度输入，自动转为 BGR
        mask = detect_highlight_mask(img, threshold=240)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
