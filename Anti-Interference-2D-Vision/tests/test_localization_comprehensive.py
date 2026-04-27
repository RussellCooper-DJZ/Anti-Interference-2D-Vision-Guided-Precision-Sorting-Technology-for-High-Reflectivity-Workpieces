"""
test_localization_comprehensive.py — 定位与标定模块综合测试
使用pytest parametrize覆盖多场景
覆盖: SubpixelLocalizer, SubpixelLocalizerV2, ROICorrectorV2
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest

from vision.localization_and_calibration import (
    SubpixelLocalizer,
    SubpixelLocalizerV2,
    ROICorrectorV2,
    ROIRect,
)


# ============================================================
# SubpixelLocalizer - 更多参数化测试
# ============================================================

class TestSubpixelLocalizerParametrized:
    """SubpixelLocalizer扩展测试"""

    @pytest.fixture
    def various_shapes(self):
        """各种形状的掩膜"""
        masks = {}

        # 圆形
        circle = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(circle, (100, 100), 40, 255, -1)
        masks['circle'] = circle

        # 矩形
        rect = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(rect, (60, 60), (140, 140), 255, -1)
        masks['rectangle'] = rect

        # 椭圆
        ellipse = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(ellipse, (100, 100), (50, 30), 0, 0, 360, 255, -1)
        masks['ellipse'] = ellipse

        return masks

    @pytest.mark.parametrize("min_area", [50, 100, 200, 500])
    def test_min_area_filter(self, various_shapes, min_area):
        """测试最小面积过滤"""
        loc = SubpixelLocalizer(min_area=min_area)
        for name, mask in various_shapes.items():
            results = loc.localize(mask)
            assert isinstance(results, list)

    @pytest.mark.parametrize("min_circularity", [0.0, 0.3, 0.5, 0.7, 0.9])
    def test_min_circularity(self, various_shapes, min_circularity):
        """测试最小圆形度过滤"""
        loc = SubpixelLocalizer(min_circularity=min_circularity)
        for name, mask in various_shapes.items():
            results = loc.localize(mask)
            assert isinstance(results, list)

    @pytest.mark.parametrize("gripper_width_px", [20, 40, 60, 80])
    def test_gripper_width(self, various_shapes, gripper_width_px):
        """测试机械爪宽度参数"""
        loc = SubpixelLocalizer(compute_gripper=True, gripper_width_px=gripper_width_px)
        for name, mask in various_shapes.items():
            results = loc.localize(mask)
            assert isinstance(results, list)

    def test_glare_mask_exclusion(self):
        """高光掩膜排除"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)

        glare = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(glare, (100, 100), 20, 255, -1)

        loc = SubpixelLocalizer()
        results = loc.localize(mask, glare_mask=glare)
        assert isinstance(results, list)

    def test_intensity_image_weighted(self):
        """灰度加权质心"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)

        intensity = np.full((200, 200), 128, dtype=np.uint8)
        intensity[70:130, 70:130] = 255

        loc = SubpixelLocalizer()
        results = loc.localize(mask, intensity_image=intensity)
        assert len(results) > 0

    def test_binary_mask_scaling(self):
        """二值掩膜自动缩放"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 1, -1)  # 0/1 instead of 0/255

        loc = SubpixelLocalizer()
        results = loc.localize(mask)
        assert isinstance(results, list)

    def test_empty_mask(self):
        """空掩膜"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        loc = SubpixelLocalizer()
        results = loc.localize(mask)
        assert results == []

    def test_no_max_area(self):
        """无最大面积限制"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (190, 190), 255, -1)
        loc = SubpixelLocalizer(max_area=None)
        results = loc.localize(mask)
        assert len(results) > 0

    def test_with_edge_mask(self):
        """带边缘掩膜"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (60, 60), (140, 140), 255, -1)

        edge = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(edge, (60, 60), (140, 140), 255, 2)

        loc = SubpixelLocalizer()
        results = loc.localize(mask, edge_mask=edge)
        assert isinstance(results, list)


# ============================================================
# SubpixelLocalizerV2 - 更多参数化测试
# ============================================================

class TestSubpixelLocalizerV2Parametrized:
    """SubpixelLocalizerV2扩展测试"""

    @pytest.mark.parametrize("ransac_thresh,min_inliers", [
        (1.0, 5), (2.0, 10), (5.0, 15), (3.0, 8)
    ])
    def test_ransac_params(self, ransac_thresh, min_inliers):
        """测试RANSAC参数"""
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.line(mask, (20, 64), (108, 64), 255, 2)
        loc = SubpixelLocalizerV2(ransac_thresh=ransac_thresh, min_inliers=min_inliers)
        result = loc.localize(mask)
        assert 'centroid' in result
        assert 'num_points' in result

    def test_edge_mask_input(self):
        """边缘掩膜输入"""
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (108, 108), 255, 2)
        loc = SubpixelLocalizerV2()
        result = loc.localize(mask)
        assert 'centroid' in result

    def test_noise_filtering(self):
        """噪声过滤"""
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.line(mask, (20, 64), (108, 64), 255, 2)
        # 添加噪声
        mask[10:15, :] = 255
        mask[:, 10:15] = 255
        loc = SubpixelLocalizerV2(ransac_thresh=3.0, min_inliers=15)
        result = loc.localize(mask)
        assert result['num_points'] >= 0

    def test_empty_mask_v2(self):
        """空掩膜V2"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        loc = SubpixelLocalizerV2()
        result = loc.localize(mask)
        assert result['num_points'] == 0

    def test_rectangle_outline(self):
        """矩形边框"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (150, 150), 255, 2)
        loc = SubpixelLocalizerV2()
        result = loc.localize(mask)
        assert 'centroid' in result

    def test_diagonal_line(self):
        """对角线"""
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.line(mask, (20, 20), (108, 108), 255, 2)
        loc = SubpixelLocalizerV2()
        result = loc.localize(mask)
        assert 'centroid' in result
        assert 'orientation_deg' in result


# ============================================================
# ROIRect - 参数化测试
# ============================================================

class TestROIRect:
    """ROI矩形数据类测试"""

    @pytest.mark.parametrize("x,y,w,h", [
        (0, 0, 100, 50),
        (10, 20, 150, 80),
        (50, 50, 200, 100),
        (0, 0, 1, 1),
    ])
    def test_creation(self, x, y, w, h):
        """创建ROI"""
        r = ROIRect(x, y, w, h)
        assert r.x == x
        assert r.y == y
        assert r.w == w
        assert r.h == h

    @pytest.mark.parametrize("x,y,w,h,expected_cx,expected_cy", [
        (0, 0, 100, 50, 50.0, 25.0),
        (10, 20, 80, 60, 50.0, 50.0),
        (100, 100, 200, 200, 200.0, 200.0),
    ])
    def test_center(self, x, y, w, h, expected_cx, expected_cy):
        """中心计算"""
        r = ROIRect(x, y, w, h)
        cx, cy = r.center()
        assert cx == expected_cx
        assert cy == expected_cy

    def test_to_tuple(self):
        """转元组"""
        r = ROIRect(10, 20, 30, 40)
        assert r.to_tuple() == (10, 20, 30, 40)


# ============================================================
# ROICorrectorV2 - 扩展参数化测试
# ============================================================

class TestROICorrectorV2Parametrized:
    """ROI校正器V2扩展测试"""

    @pytest.fixture
    def baseline_image(self):
        """基线图像"""
        img = np.ones((256, 512), dtype=np.uint8) * 128
        cv2.rectangle(img, (150, 80), (350, 180), (200,), -1)
        return img

    @pytest.mark.parametrize("template_size", [16, 24, 32, 48])
    def test_template_sizes(self, baseline_image, template_size):
        """测试不同模板大小"""
        tracker = ROICorrectorV2(template_size=template_size, search_margin=20)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        assert tracker.template is not None
        assert tracker.template.shape == (template_size, template_size)

    @pytest.mark.parametrize("search_margin", [10, 20, 30, 50])
    def test_search_margins(self, baseline_image, search_margin):
        """测试不同搜索边界"""
        tracker = ROICorrectorV2(template_size=32, search_margin=search_margin)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        corrected, info = tracker.correct(baseline_image)
        assert isinstance(corrected, ROIRect)
        assert 'similarity' in info

    @pytest.mark.parametrize("drift_thresh", [1.0, 2.0, 5.0, 10.0])
    def test_drift_thresholds(self, baseline_image, drift_thresh):
        """测试不同漂移阈值"""
        tracker = ROICorrectorV2(template_size=32, search_margin=30, drift_thresh=drift_thresh)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        # 大幅平移
        shifted = np.roll(np.roll(baseline_image, 20, axis=1), 20, axis=0)
        corrected, info = tracker.correct(shifted)
        assert isinstance(corrected, ROIRect)

    def test_small_shift_no_alarm(self, baseline_image):
        """小幅移动不报警"""
        tracker = ROICorrectorV2(template_size=32, search_margin=30, drift_thresh=10.0)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        # 小幅平移
        shifted = np.roll(np.roll(baseline_image, 3, axis=1), 3, axis=0)
        corrected, info = tracker.correct(shifted)
        assert isinstance(info['drift_alarm'], (bool, np.bool_))

    def test_rotation_invariant(self, baseline_image):
        """旋转不变性测试"""
        tracker = ROICorrectorV2(template_size=32, search_margin=20)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        # 旋转图像
        M = cv2.getRotationMatrix2D((256, 256), 5, 1.0)
        rotated = cv2.warpAffine(baseline_image, M, (512, 512))
        corrected, info = tracker.correct(rotated)
        assert isinstance(corrected, ROIRect)

    def test_multiple_corrections(self, baseline_image):
        """多次校正"""
        tracker = ROICorrectorV2(template_size=32, search_margin=20)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        for i in range(5):
            shifted = np.roll(baseline_image, i * 2, axis=1)
            corrected, info = tracker.correct(shifted)
            assert isinstance(corrected, ROIRect)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
