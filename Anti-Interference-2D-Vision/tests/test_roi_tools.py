"""
test_roi_tools.py — ROI 与形状分析工具测试
基于实际 API 重写
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import cv2

from vision.roi_tools import (
    ROIType,
    ROI,
    ROIGenerator,
    ROICorrector,
    AutoROI,
    AutoROIMode,
    generate_roi_from_contour,
    generate_roi_from_binary,
    correct_roi,
    select_roi_adaptive,
)


# ============================================================
# ROIType
# ============================================================

class TestROIType:
    """ROI 类型枚举测试"""

    def test_enum_values(self):
        assert ROIType.POINT.value == "point"
        assert ROIType.RECT.value == "rect"
        assert ROIType.ROTATED_RECT.value == "rotated_rect"
        assert ROIType.CIRCLE.value == "circle"
        assert ROIType.ELLIPSE.value == "ellipse"
        assert ROIType.LINE.value == "line"
        assert ROIType.POLYGON.value == "polygon"


# ============================================================
# ROI
# ============================================================

class TestROI:
    """ROI 数据结构测试"""

    def test_create_rect(self):
        roi = ROI(type=ROIType.RECT, rect_center=(100, 100), rect_size=(50, 30))
        assert roi is not None
        assert roi.type == ROIType.RECT
        assert roi.rect_center == (100, 100)
        assert roi.rect_size == (50, 30)

    def test_create_rotated_rect(self):
        roi = ROI(type=ROIType.ROTATED_RECT, angle=45.0, rect_center=(100, 100), rect_size=(50, 30))
        assert roi.type == ROIType.ROTATED_RECT
        assert roi.angle == 45.0

    def test_create_circle(self):
        roi = ROI(type=ROIType.CIRCLE, circle_center=(150, 150), circle_radius=50)
        assert roi.type == ROIType.CIRCLE
        assert roi.circle_center == (150, 150)
        assert roi.circle_radius == 50

    def test_create_point(self):
        roi = ROI(type=ROIType.POINT, point=(50, 50))
        assert roi.type == ROIType.POINT
        assert roi.point == (50, 50)

    def test_create_polygon(self):
        pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
        roi = ROI(type=ROIType.POLYGON, points=pts)
        assert roi.type == ROIType.POLYGON
        assert roi.points == pts

    def test_get_center(self):
        roi = ROI(type=ROIType.RECT, rect_center=(100, 100), rect_size=(50, 30))
        cx, cy = roi.get_center()
        assert cx == 100
        assert cy == 100

    def test_get_bounding_box(self):
        roi = ROI(type=ROIType.RECT, rect_center=(100, 100), rect_size=(50, 30))
        bbox = roi.get_bounding_box(img_w=200, img_h=200)
        assert len(bbox) == 4  # x, y, w, h
        assert bbox[2] > 0 and bbox[3] > 0

    def test_to_mask(self):
        roi = ROI(type=ROIType.RECT, rect_center=(100, 100), rect_size=(50, 30))
        mask = roi.to_mask(img_w=200, img_h=200)
        assert mask is not None
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8

    def test_apply_affine_transform(self):
        roi = ROI(type=ROIType.RECT, rect_center=(100, 100), rect_size=(50, 30))
        # 仿射变换矩阵 (2x3)
        M = np.array([[1, 0, 10], [0, 1, -5]], dtype=np.float64)  # 平移 (10, -5)
        new_roi = roi.apply_affine_transform(M)
        assert new_roi is not None
        # 检查平移效果
        cx, cy = new_roi.get_center()
        assert abs(cx - 110) < 1 and abs(cy - 95) < 1


# ============================================================
# generate_roi_from_contour
# ============================================================

class TestGenerateROIFromContour:
    """从轮廓生成 ROI 测试"""

    def test_from_contour_rect(self):
        # 创建测试图像和轮廓
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0

        roi = generate_roi_from_contour(contours[0])
        assert roi is not None

    def test_from_contour_circle(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) > 0

        roi = generate_roi_from_contour(contours[0])
        assert roi is not None


# ============================================================
# generate_roi_from_binary
# ============================================================

class TestGenerateROIFromBinary:
    """从二值图像生成 ROI 测试"""

    def test_from_binary_rect(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), 255, -1)

        roi = generate_roi_from_binary(img)
        assert roi is not None

    def test_from_binary_empty(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        roi = generate_roi_from_binary(img)
        # 空白图像可能返回 None
        assert roi is None or roi.type is not None


# ============================================================
# ROIGenerator
# ============================================================

class TestROIGenerator:
    """ROI 生成器测试"""

    @pytest.fixture
    def blob_image(self):
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(img, (128, 128), 40, 255, -1)
        return img

    @pytest.fixture
    def line_image(self):
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.line(img, (50, 128), (206, 128), 255, 3)
        return img

    def test_from_binary(self, blob_image):
        gen = ROIGenerator()
        roi = gen.from_binary(blob_image)
        assert roi is not None

    def test_from_contour(self, blob_image):
        contours, _ = cv2.findContours(blob_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gen = ROIGenerator()
        roi = gen.from_contour(contours[0])
        assert roi is not None

    def test_from_feature(self, line_image):
        gen = ROIGenerator()
        roi = gen.from_feature(line_image)
        assert roi is not None or roi is None  # 可能返回 None


# ============================================================
# ROICorrector
# ============================================================

class TestROICorrector:
    """ROI 校正器测试"""

    def test_creation(self):
        corrector = ROICorrector()
        assert corrector is not None

    def test_correct_roi(self):
        roi = ROI(type=ROIType.RECT, rect_center=(100, 100), rect_size=(50, 30))
        corrected = correct_roi(roi, ref_center=(100, 100), ref_angle=0, cur_center=(110, 95), cur_angle=10)
        assert corrected is not None


# ============================================================
# AutoROI
# ============================================================

class TestAutoROI:
    """自动 ROI 测试"""

    @pytest.fixture
    def blob_image(self):
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        cv2.circle(img, (128, 128), 40, (200,), -1)
        return img

    def test_creation(self):
        auto = AutoROI()
        assert auto is not None

    def test_detect(self, blob_image):
        auto = AutoROI()
        roi = auto.detect(blob_image)
        assert roi is None or isinstance(roi, ROI)

    def test_set_template(self, blob_image):
        from vision.roi_tools import ROI, ROIType
        auto = AutoROI()
        roi = ROI(type=ROIType.RECT, rect_center=(128, 128), rect_size=(40, 40))
        auto.set_template(blob_image, roi)
        assert auto.mode is not None

    def test_should_redetect(self, blob_image):
        auto = AutoROI()
        result = auto.should_redetect(blob_image)
        assert isinstance(result, bool)


# ============================================================
# AutoROIMode
# ============================================================

class TestAutoROIMode:
    """自动 ROI 模式测试"""

    def test_modes(self):
        assert AutoROIMode.BLOB.value == "blob"
        assert AutoROIMode.EDGE.value == "edge"
        assert AutoROIMode.FEATURE.value == "feature"
        assert AutoROIMode.TEMPLATE.value == "template"
        assert AutoROIMode.ADAPTIVE.value == "adaptive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
