"""
test_measurement.py — 精密测量模块测试
基于实际 API 重写
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import cv2

from vision.measurement import CaliperMeasurement, GapMeasurement


# ============================================================
# CaliperMeasurement
# ============================================================

class TestCaliperMeasurement:
    """卡尺测量测试"""

    @pytest.fixture
    def parallel_edges_image(self):
        """双平行边缘图像"""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (50, 50), (50, 150), 255, 3)
        cv2.line(img, (150, 50), (150, 150), 255, 3)
        return img

    def test_creation(self):
        caliper = CaliperMeasurement()
        assert caliper is not None

    def test_measure_parallel_edges(self, parallel_edges_image):
        """测试平行边缘距离测量"""
        caliper = CaliperMeasurement()
        result = caliper.measure(parallel_edges_image, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)
        assert 'distance' in result
        assert 'valid' in result

    def test_empty_image(self):
        """空白图像测试"""
        img = np.zeros((100, 100), dtype=np.uint8)
        caliper = CaliperMeasurement()
        result = caliper.measure(img, roi=(0, 0, 100, 100))
        assert isinstance(result, dict)
        # 空白图像可能无效
        assert 'valid' in result


# ============================================================
# GapMeasurement
# ============================================================

class TestGapMeasurement:
    """间隙测量测试"""

    @pytest.fixture
    def multi_edge_image(self):
        """多边缘图像"""
        img = np.zeros((200, 400), dtype=np.uint8)
        cv2.line(img, (80, 20), (80, 180), 255, 2)
        cv2.line(img, (200, 20), (200, 180), 255, 2)
        cv2.line(img, (320, 20), (320, 180), 255, 2)
        return img

    def test_creation(self):
        gm = GapMeasurement()
        assert gm is not None

    def test_measure_multi_edge(self, multi_edge_image):
        """测试多边缘测量"""
        gm = GapMeasurement()
        result = gm.measure(multi_edge_image, roi=(0, 0, 400, 200))
        assert isinstance(result, dict)
        assert 'pitches' in result or 'widths' in result

    def test_empty_image(self):
        """空白图像"""
        img = np.zeros((100, 100), dtype=np.uint8)
        gm = GapMeasurement()
        result = gm.measure(img, roi=(0, 0, 100, 100))
        assert isinstance(result, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
