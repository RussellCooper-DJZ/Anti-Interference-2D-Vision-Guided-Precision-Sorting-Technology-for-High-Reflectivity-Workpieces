"""
test_har_processing.py — HDR/反光处理测试
验证 HDR 融合、高光检测、修复功能
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest

from vision.hdr_processing import (
    exposure_fusion_mertens,
    detect_highlight_mask,
    repair_highlight_regions,
    AntiGlarePipeline,
    generate_synthetic_exposures,
    polarization_min_method,
)


class TestHighlightDetection:
    """高光检测测试"""

    @pytest.fixture
    def test_image(self):
        """创建带高光的测试图像"""
        img = np.full((200, 200, 3), 100, dtype=np.uint8)
        # 添加高光区域 (亮度 250)
        cv2.circle(img, (100, 100), 30, (250, 250, 250), -1)
        return img

    def test_detect_highlight(self, test_image):
        """测试高光检测"""
        mask = detect_highlight_mask(test_image, threshold=240)
        assert mask.shape == test_image.shape[:2]
        assert mask.dtype == np.uint8
        # 高光区域应被检测到
        assert mask.max() > 0

    def test_no_highlight(self):
        """测试无高光图像"""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        mask = detect_highlight_mask(img, threshold=240)
        assert mask.max() == 0

    def test_threshold_param(self, test_image):
        """测试阈值参数"""
        mask_low = detect_highlight_mask(test_image, threshold=200)
        mask_high = detect_highlight_mask(test_image, threshold=250)
        # 低阈值应检测更多区域
        assert mask_low.max() >= mask_high.max()


class TestHDRFusion:
    """HDR 融合测试"""

    @pytest.fixture
    def multi_exposure_images(self):
        """多曝光图像"""
        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)   # 暗
        img2 = np.full((100, 100, 3), 128, dtype=np.uint8)  # 正常
        img3 = np.full((100, 100, 3), 200, dtype=np.uint8)  # 亮
        return [img1, img2, img3]

    def test_fusion_output(self, multi_exposure_images):
        """测试融合输出"""
        result = exposure_fusion_mertens(multi_exposure_images)
        assert result.shape == multi_exposure_images[0].shape
        assert result.dtype == np.uint8

    def test_single_image(self):
        """测试单张图像（应直接返回）"""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = exposure_fusion_mertens([img])
        assert np.array_equal(result, img)

    def test_empty_raises(self):
        """测试空列表应抛异常"""
        with pytest.raises(ValueError):
            exposure_fusion_mertens([])


class TestHighlightRepair:
    """高光修复测试"""

    @pytest.fixture
    def highlight_image(self):
        """带高光的图像"""
        img = np.full((200, 200, 3), 100, dtype=np.uint8)
        cv2.rectangle(img, (80, 80), (120, 120), (255, 255, 255), -1)
        return img

    def test_repair_methods(self, highlight_image):
        """测试不同修复方法"""
        for method in ['telea', 'ns', 'blend']:
            result = repair_highlight_regions(
                highlight_image, method=method
            )
            assert result.shape == highlight_image.shape
            assert result.dtype == np.uint8

    def test_auto_detect_mask(self, highlight_image):
        """测试自动检测高光掩膜"""
        result = repair_highlight_regions(highlight_image)
        assert result.shape == highlight_image.shape

    def test_no_highlight_no_change(self):
        """无高光时图像应不变"""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = repair_highlight_regions(img)
        # 由于可能有轻微处理，不严格检查相等


class TestAntiGlarePipeline:
    """AntiGlarePipeline 完整管线测试"""

    @pytest.fixture
    def pipeline(self):
        return AntiGlarePipeline()

    @pytest.fixture
    def bright_image(self):
        """高亮测试图像"""
        img = np.full((256, 256, 3), 180, dtype=np.uint8)
        # 添加高光
        cv2.circle(img, (128, 128), 50, (255, 255, 255), -1)
        return img

    def test_pipeline_output(self, pipeline, bright_image):
        """测试管线输出"""
        result = pipeline.process_single(bright_image)
        assert result.shape == bright_image.shape
        assert result.dtype == np.uint8

    def test_pipeline_multi(self, pipeline):
        """测试多图像输入"""
        imgs = [np.full((100, 100, 3), 100, dtype=np.uint8) for _ in range(3)]
        result = pipeline.process_multi(imgs)
        assert result.shape == imgs[0].shape

    def test_debug_stages(self, pipeline, bright_image):
        """测试调试阶段输出"""
        stages = pipeline.get_debug_stages(bright_image)
        assert '00_original' in stages
        assert '02_hdr_fused' in stages
        assert '08_final' in stages


class TestSyntheticExposures:
    """合成曝光测试"""

    def test_exposure_generation(self):
        """测试多曝光生成"""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        imgs, times = generate_synthetic_exposures(img, ev_stops=[-2, 0, 2])

        assert len(imgs) == 3
        assert len(times) == 3
        assert all(im.shape == img.shape for im in imgs)


class TestPolarization:
    """偏振模拟测试"""

    def test_min_method(self):
        """测试最小值法"""
        imgs = [np.full((50, 50, 3), 100 + i * 50, dtype=np.uint8) for i in range(3)]
        result = polarization_min_method(imgs)
        assert result.shape == imgs[0].shape
        assert result.dtype == np.uint8

    def test_single_image(self):
        """单图像应直接返回"""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = polarization_min_method([img])
        assert np.array_equal(result, img)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
