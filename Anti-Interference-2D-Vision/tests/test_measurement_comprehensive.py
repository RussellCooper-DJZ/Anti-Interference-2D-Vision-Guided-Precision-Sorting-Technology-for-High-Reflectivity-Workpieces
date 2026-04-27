"""
test_measurement_comprehensive.py — 测量模块综合测试
使用pytest parametrize覆盖多场景
覆盖: CaliperMeasurement, GapMeasurement
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest

from vision.measurement import (
    CaliperMeasurement,
    GapMeasurement,
)


# ============================================================
# CaliperMeasurement - 扩展参数化测试
# ============================================================

class TestCaliperMeasurementParametrized:
    """CaliperMeasurement扩展测试"""

    @pytest.fixture
    def parallel_edges_image(self):
        """平行边缘图像"""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (50, 30), (50, 170), 255, 3)
        cv2.line(img, (150, 30), (150, 170), 255, 3)
        return img

    @pytest.mark.parametrize("search_direction", [
        'top_to_bottom', 'bottom_to_top',
        'left_to_right', 'right_to_left'
    ])
    def test_all_directions(self, parallel_edges_image, search_direction):
        """所有搜索方向"""
        caliper = CaliperMeasurement(search_direction=search_direction)
        result = caliper.measure(parallel_edges_image, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)
        assert 'distance' in result

    @pytest.mark.parametrize("polarity", ['black_to_white', 'white_to_black', 'all'])
    def test_polarities(self, parallel_edges_image, polarity):
        """所有极性"""
        caliper = CaliperMeasurement(polarity=polarity)
        result = caliper.measure(parallel_edges_image, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("edge_intensity", [10, 20, 30, 50, 100])
    def test_edge_intensity(self, parallel_edges_image, edge_intensity):
        """边缘强度阈值"""
        caliper = CaliperMeasurement(edge_intensity=edge_intensity)
        result = caliper.measure(parallel_edges_image, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("search_line_count", [5, 10, 20, 50])
    def test_search_line_count(self, parallel_edges_image, search_line_count):
        """搜索线数量"""
        caliper = CaliperMeasurement(search_line_count=search_line_count)
        result = caliper.measure(parallel_edges_image, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("edge_width", [3, 5, 7, 10])
    def test_edge_width(self, parallel_edges_image, edge_width):
        """边缘宽度"""
        caliper = CaliperMeasurement(edge_width=edge_width)
        result = caliper.measure(parallel_edges_image, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("projection_width", [3, 5, 7, 10])
    def test_projection_width(self, parallel_edges_image, projection_width):
        """投影宽度"""
        caliper = CaliperMeasurement(projection_width=projection_width)
        result = caliper.measure(parallel_edges_image, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)

    def test_single_roi(self):
        """不同ROI"""
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(img, (20, 10), (20, 90), 255, 2)
        cv2.line(img, (80, 10), (80, 90), 255, 2)
        caliper = CaliperMeasurement()
        result = caliper.measure(img, roi=(0, 0, 100, 100))
        assert isinstance(result, dict)

    def test_empty_image(self):
        """空白图像"""
        img = np.zeros((100, 100), dtype=np.uint8)
        caliper = CaliperMeasurement()
        result = caliper.measure(img, roi=(0, 0, 100, 100))
        assert isinstance(result, dict)
        assert 'valid' in result

    def test_wide_spacing(self):
        """宽间距"""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (30, 50), (30, 150), 255, 3)
        cv2.line(img, (170, 50), (170, 150), 255, 3)
        caliper = CaliperMeasurement()
        result = caliper.measure(img, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)

    def test_narrow_spacing(self):
        """窄间距"""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (95, 50), (95, 150), 255, 3)
        cv2.line(img, (105, 50), (105, 150), 255, 3)
        caliper = CaliperMeasurement()
        result = caliper.measure(img, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)


# ============================================================
# GapMeasurement - 扩展参数化测试
# ============================================================

class TestGapMeasurementParametrized:
    """GapMeasurement扩展测试"""

    @pytest.fixture
    def multi_gap_image(self):
        """多间隙图像"""
        img = np.zeros((200, 400), dtype=np.uint8)
        cv2.line(img, (60, 20), (60, 180), 255, 2)
        cv2.line(img, (140, 20), (140, 180), 255, 2)
        cv2.line(img, (220, 20), (220, 180), 255, 2)
        cv2.line(img, (300, 20), (300, 180), 255, 2)
        return img

    @pytest.mark.parametrize("search_direction", [
        'left_to_right', 'right_to_left'
    ])
    def test_horizontal_search(self, multi_gap_image, search_direction):
        """水平搜索方向"""
        gm = GapMeasurement(search_direction=search_direction)
        result = gm.measure(multi_gap_image, roi=(0, 0, 400, 200))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("polarity", ['dark_to_bright', 'bright_to_dark', 'all'])
    def test_polarity(self, multi_gap_image, polarity):
        """边缘极性"""
        gm = GapMeasurement(polarity=polarity)
        result = gm.measure(multi_gap_image, roi=(0, 0, 400, 200))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("edge_threshold", [10, 20, 30, 50])
    def test_edge_threshold(self, multi_gap_image, edge_threshold):
        """边缘阈值"""
        gm = GapMeasurement(edge_intensity=edge_threshold)
        result = gm.measure(multi_gap_image, roi=(0, 0, 400, 200))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("gap_width", [5, 10, 20, 30])
    def test_gap_width_param(self, gap_width):
        """间隙宽度参数"""
        img = np.zeros((200, 300), dtype=np.uint8)
        cv2.line(img, (50, 20), (50, 180), 255, 2)
        cv2.line(img, (50 + gap_width, 20), (50 + gap_width, 180), 255, 2)
        gm = GapMeasurement()
        result = gm.measure(img, roi=(0, 0, 300, 200))
        assert isinstance(result, dict)

    def test_empty_image_gap(self):
        """空白图像"""
        img = np.zeros((100, 100), dtype=np.uint8)
        gm = GapMeasurement()
        result = gm.measure(img, roi=(0, 0, 100, 100))
        assert isinstance(result, dict)

    def test_single_edge(self):
        """单边缘"""
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (50, 20), (50, 180), 255, 2)
        gm = GapMeasurement()
        result = gm.measure(img, roi=(0, 0, 200, 200))
        assert isinstance(result, dict)

    def test_many_gaps(self):
        """多间隙"""
        img = np.zeros((200, 500), dtype=np.uint8)
        for x in range(50, 450, 40):
            cv2.line(img, (x, 20), (x, 180), 255, 2)
        gm = GapMeasurement()
        result = gm.measure(img, roi=(0, 0, 500, 200))
        assert isinstance(result, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
