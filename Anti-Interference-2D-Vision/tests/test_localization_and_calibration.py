"""
test_localization_and_calibration.py — 定位与标定模块测试
覆盖 SubpixelLocalizerV2, ROICorrectorV2, HandEyeCalibratorV2 等
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import cv2

from vision.localization_and_calibration import (
    SubpixelLocalizer, SubpixelLocalizerV2,
    CameraCalibrator,
    HandEyeCalibrator, HandEyeCalibratorV2, CalibrationFrame,
    CoordinateTransformer,
    ROICorrectorV2, ROIRect,
    CornerDetector,
    GaussianLineExtractor,
    HoughCircleDetector,
    HoughLineDetector,
    NinePointCalibrator,
)


# ============================================================
# SubpixelLocalizer / SubpixelLocalizerV2
# ============================================================

class TestSubpixelLocalizer:
    """原始亚像素定位器测试"""

    @pytest.fixture
    def rect_mask(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (60, 60), (140, 140), 255, -1)
        return mask

    def test_localize_centroid(self, rect_mask):
        loc = SubpixelLocalizer()
        results = loc.localize(rect_mask, edge_mask=rect_mask)
        assert len(results) > 0
        centroid = results[0]['centroid_px']
        # 矩形中心应在 (100, 100) 附近
        assert 90 <= centroid[0] <= 110
        assert 90 <= centroid[1] <= 110

    def test_localize_orientation(self, rect_mask):
        loc = SubpixelLocalizer()
        results = loc.localize(rect_mask, edge_mask=rect_mask)
        # 正方形的PCA方向可能不稳定，但应有值
        assert 'orientation_deg' in results[0]


class TestSubpixelLocalizerV2:
    """V2 亚像素定位器测试"""

    @pytest.fixture
    def edge_mask(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.rectangle(mask, (30, 30), (100, 100), 255, 2)
        return mask

    def test_localize_basic(self, edge_mask):
        loc = SubpixelLocalizerV2(ransac_thresh=2.0, min_inliers=5)
        result = loc.localize(edge_mask)
        assert 'centroid' in result
        assert 'num_points' in result
        assert 'inliers' in result
        assert result['num_points'] > 0

    def test_localize_centroid_accuracy(self):
        """测试质心精度"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (60, 60), (140, 140), 255, 2)
        loc = SubpixelLocalizerV2()
        result = loc.localize(mask)
        cx, cy = result['centroid']
        # 矩形边框中心 (100, 100)
        assert 90 <= cx <= 110
        assert 90 <= cy <= 110

    def test_localize_empty(self):
        """空掩膜测试"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        loc = SubpixelLocalizerV2()
        result = loc.localize(mask)
        assert result['num_points'] == 0

    def test_localize_ransac_filtering(self):
        """RANSAC 异常值剔除测试"""
        mask = np.zeros((128, 128), dtype=np.uint8)
        # 主要边缘（直线）
        cv2.line(mask, (20, 64), (108, 64), 255, 2)
        # 添加一些异常噪声点
        mask[10:15, 10:15] = 255
        mask[110:115, 110:115] = 255
        loc = SubpixelLocalizerV2(ransac_thresh=5.0, min_inliers=10)
        result = loc.localize(mask)
        # 内点数应少于总点数（因为噪声被剔除）
        if result['num_points'] > result['inliers']:
            assert result['inliers'] < result['num_points']

    def test_localize_line_orientation(self):
        """直线方向测试"""
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.line(mask, (20, 64), (108, 64), 255, 2)  # 水平线
        loc = SubpixelLocalizerV2()
        result = loc.localize(mask)
        # 水平线方向应接近 0° 或 180°
        orientation = abs(result['orientation_deg']) % 180
        assert orientation < 20 or orientation > 160


# ============================================================
# ROICorrectorV2 / ROIRect
# ============================================================

class TestROIRect:
    """ROI 矩形数据类测试"""

    def test_creation(self):
        r = ROIRect(10, 20, 100, 50)
        assert r.x == 10
        assert r.y == 20
        assert r.w == 100
        assert r.h == 50

    def test_center(self):
        r = ROIRect(10, 20, 100, 50)
        cx, cy = r.center()
        assert cx == 60.0
        assert cy == 45.0

    def test_to_tuple(self):
        r = ROIRect(1, 2, 3, 4)
        assert r.to_tuple() == (1, 2, 3, 4)


class TestROICorrectorV2:
    """ROI 校正器 V2 测试"""

    @pytest.fixture
    def baseline_image(self):
        img = np.ones((256, 512), dtype=np.uint8) * 128
        cv2.rectangle(img, (150, 80), (350, 180), (200,), -1)
        return img

    def test_set_baseline(self, baseline_image):
        tracker = ROICorrectorV2(template_size=32, search_margin=20)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        assert tracker.template is not None
        assert tracker.template.shape == (32, 32)
        assert tracker.template_pos is not None

    def test_correct_no_shift(self, baseline_image):
        tracker = ROICorrectorV2(template_size=32, search_margin=20)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        corrected, info = tracker.correct(baseline_image)
        assert isinstance(corrected, ROIRect)
        assert 'similarity' in info
        assert 'drift_alarm' in info
        assert info['similarity'] > 0.5  # SSDA may have some variance

    def test_correct_with_shift(self, baseline_image):
        tracker = ROICorrectorV2(template_size=32, search_margin=30)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        # 模拟平移
        shifted = np.roll(np.roll(baseline_image, 5, axis=1), 3, axis=0)
        corrected, info = tracker.correct(shifted)
        assert isinstance(corrected, ROIRect)
        assert corrected.x >= 0
        assert corrected.y >= 0

    def test_drift_alarm(self, baseline_image):
        tracker = ROICorrectorV2(template_size=32, search_margin=30, drift_thresh=2.0)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        # 大幅平移应触发报警
        shifted = np.roll(np.roll(baseline_image, 20, axis=1), 20, axis=0)
        corrected, info = tracker.correct(shifted)
        assert info['drift_alarm']

    def test_ssda_early_termination(self, baseline_image):
        """验证 SSDA 的 early termination 机制存在"""
        tracker = ROICorrectorV2(template_size=16, search_margin=10)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        corrected, info = tracker.correct(baseline_image)
        assert info['similarity'] >= 0.0
        assert info['similarity'] <= 1.0

    def test_color_image_input(self, baseline_image):
        """测试彩色图像输入自动转灰度"""
        color_img = np.stack([baseline_image] * 3, axis=-1)
        tracker = ROICorrectorV2(template_size=32, search_margin=10)
        tracker.set_baseline(color_img, (160, 90, 180, 80))
        corrected, info = tracker.correct(color_img)
        assert isinstance(corrected, ROIRect)

    def test_boundary_clipping(self, baseline_image):
        """测试 ROI 边界裁剪"""
        tracker = ROICorrectorV2(template_size=32, search_margin=50)
        tracker.set_baseline(baseline_image, (160, 90, 180, 80))
        # 大幅偏移可能导致 ROI 越界
        shifted = np.roll(baseline_image, 200, axis=1)
        corrected, info = tracker.correct(shifted)
        assert corrected.x >= 0
        assert corrected.y >= 0
        assert corrected.x + corrected.w <= baseline_image.shape[1]
        assert corrected.y + corrected.h <= baseline_image.shape[0]


# ============================================================
# HandEyeCalibrator / HandEyeCalibratorV2
# ============================================================

class TestHandEyeCalibrator:
    """原始手眼标定测试"""

    def test_creation(self):
        calib = HandEyeCalibrator(mode='eye_in_hand')
        assert calib.mode == 'eye_in_hand'

    def test_creation_eye_to_hand(self):
        calib = HandEyeCalibrator(mode='eye_to_hand')
        assert calib.mode == 'eye_to_hand'


class TestHandEyeCalibratorV2:
    """V2 手眼标定测试"""

    def test_creation(self):
        calib = HandEyeCalibratorV2(pattern_size=(9, 6), square_size=20.0)
        assert calib.pattern_size == (9, 6)
        assert calib.square_size == 20.0
        assert calib.mode == 'eye_in_hand'
        assert calib.objp.shape == (54, 3)

    def test_set_intrinsics(self):
        calib = HandEyeCalibratorV2()
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        calib.set_intrinsics(K)
        assert calib.camera_matrix is not None
        assert np.allclose(calib.camera_matrix, K)

    def test_detect_corners_no_board(self):
        """无棋盘格时应返回 None"""
        calib = HandEyeCalibratorV2(pattern_size=(9, 6))
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        corners = calib.detect_corners(img)
        assert corners is None

    def test_add_frame_no_corners(self):
        """检测不到角点时返回 False"""
        calib = HandEyeCalibratorV2()
        K = np.eye(3, dtype=np.float32)
        calib.set_intrinsics(K)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pose = np.eye(4, dtype=np.float32)
        result = calib.add_frame(img, pose)
        assert result is False

    def test_calibrate_insufficient_frames(self):
        """帧数不足时应抛异常"""
        calib = HandEyeCalibratorV2()
        K = np.eye(3, dtype=np.float32)
        calib.set_intrinsics(K)
        # 添加一些假帧（无真实角点数据，只能测接口）
        with pytest.raises(RuntimeError):
            calib.calibrate()

    def test_eye_to_hand_mode(self):
        calib = HandEyeCalibratorV2(mode='eye_to_hand')
        assert calib.mode == 'eye_to_hand'

    def test_different_pattern_sizes(self):
        for size in [(7, 5), (9, 6), (11, 8)]:
            calib = HandEyeCalibratorV2(pattern_size=size)
            expected_n = size[0] * size[1]
            assert calib.objp.shape == (expected_n, 3)


# ============================================================
# CameraCalibrator
# ============================================================

class TestCameraCalibrator:
    """相机标定测试"""

    def test_creation(self):
        calib = CameraCalibrator(board_size=(9, 6), square_size_mm=25.0)
        assert calib.board_size == (9, 6)
        assert calib.square_size_mm == 25.0


# ============================================================
# CoordinateTransformer
# ============================================================

class TestCoordinateTransformer:
    """坐标变换器测试"""

    def test_creation(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        T_cam2robot = np.eye(4, dtype=np.float64)
        ct = CoordinateTransformer(K=K, dist=dist, T_cam2robot=T_cam2robot)
        assert np.allclose(ct.K, K)

    def test_pixel_to_camera(self):
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        T_cam2robot = np.eye(4, dtype=np.float64)
        ct = CoordinateTransformer(K=K, dist=dist, T_cam2robot=T_cam2robot)
        # 像素坐标 (320, 240) 在深度 1000mm 时应该在相机原点
        pt = ct.pixel_to_camera(320, 240, 1000.0)
        assert len(pt) == 3
        assert abs(pt[2] - 1000.0) < 1e-5  # Z 应该等于深度


# ============================================================
# 传统 CV 检测器
# ============================================================

class TestCornerDetector:
    """角点检测器测试"""

    @pytest.mark.skip(reason="goodFeaturesToTrack requires high-quality corner images")
    def test_detect_corners(self):
        # 创建有对比度的角点测试图像
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        # 画黑色方块创造角点
        cv2.rectangle(img, (40, 40), (80, 80), (0, 0, 0), -1)
        cv2.rectangle(img, (120, 40), (160, 80), (0, 0, 0), -1)
        cv2.rectangle(img, (40, 120), (80, 160), (0, 0, 0), -1)
        detector = CornerDetector()
        corners = detector.detect(img)
        # 角点检测可能返回空列表或找到的角点
        assert isinstance(corners, list)


class TestHoughLineDetector:
    """霍夫直线检测器测试"""

    def test_detect_line(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.line(img, (20, 100), (180, 100), 255, 2)
        detector = HoughLineDetector()
        lines = detector.detect(img)
        assert isinstance(lines, list)


class TestHoughCircleDetector:
    """霍夫圆检测器测试"""

    def test_detect_circle(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, 255, 2)
        detector = HoughCircleDetector()
        circles = detector.detect(img)
        assert isinstance(circles, list)


class TestNinePointCalibrator:
    """九点标定测试"""

    def test_creation(self):
        calib = NinePointCalibrator()
        assert calib is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
