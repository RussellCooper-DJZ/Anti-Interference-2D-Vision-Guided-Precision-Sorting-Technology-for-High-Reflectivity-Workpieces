"""
roi_tools.py — 完整ROI工具箱：多类型ROI、ROI校正、自动检测与切换
:Author: RussellCooper

基于Smart3第9.4-9.5章及第7.1章ROI功能实现：

1. ROI类型支持：
   - 点(Point)、线(Line)、旋转矩形(RotatedRect)
   - 圆(Circle)、椭圆(Ellipse)、圆环(Annulus)
   - 多边形(Polygon)、折线(Polyline)、不规则形(Irregular)
   - 矩阵ROI(Array)、二值图像ROI(BinaryMask)

2. ROI生成方式：
   - 手动绘制（坐标参数）
   - 引用已有特征（Blob/匹配/几何关系）
   - 轮廓线生成（沿轮廓等分/间隔）
   - 二值图像生成（白色区域=ROI）

3. ROI校正（基准跟随）：
   - 水平/垂直/角度补偿
   - 基于参考点和参考角度的仿射变换
   - 支持ROI和Mask独立校正

4. 自动检测与切换：
   - 基于图像内容自动检测最佳ROI
   - 场景变化时自动切换ROI模式
   - 无需人工干预的自适应ROI

用法::

    from vision.roi_tools import ROI, ROIGenerator, ROICorrector, AutoROI

    # 创建ROI
    roi = ROI.create('rotated_rect', center=(100,100), size=(50,30), angle=45)

    # 自动检测ROI
    auto_roi = AutoROI()
    best_roi = auto_roi.detect(image, mode='blob')  # 基于Blob/边缘/匹配

    # ROI校正
    corrector = ROICorrector(mode='horizontal_vertical_angle')
    corrector.set_reference(center=(100,100), angle=0)
    corrected_roi = corrector.apply(roi, current_center=(120,90), current_angle=15)
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = [
    # ROI类型枚举
    "ROIType",
    # ROI数据结构
    "ROI",
    # ROI生成器
    "ROIGenerator",
    "generate_roi_from_contour",
    "generate_roi_from_binary",
    "generate_array_roi",
    # ROI校正器
    "ROICorrector",
    "correct_roi",
    # 自动ROI检测与切换
    "AutoROI",
    "AutoROIMode",
    "select_roi_adaptive",
]


# ============================================================
# 1. ROI类型枚举
# ============================================================

class ROIType(Enum):
    """ROI类型枚举。"""
    POINT = "point"
    LINE = "line"
    RECT = "rect"                 # 正矩形
    ROTATED_RECT = "rotated_rect"  # 旋转矩形
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    ANNULUS = "annulus"            # 圆环
    POLYGON = "polygon"            # 多边形
    POLYLINE = "polyline"         # 折线
    IRREGULAR = "irregular"        # 不规则形
    ARRAY = "array"                # 矩阵ROI
    BINARY_MASK = "binary_mask"    # 二值图像ROI


# ============================================================
# 2. ROI数据结构
# ============================================================

@dataclass
class ROI:
    """
    ROI数据结构 — 支持多种几何类型。

    用法::

        roi = ROI.create('circle', center=(100, 200), radius=30)
        mask = roi.to_mask(512, 512)  # 生成二值掩膜
    """

    type: ROIType
    # 共用字段
    angle: float = 0.0           # 旋转角度（度）
    # 点
    point: Optional[Tuple[float, float]] = None
    # 线
    line_start: Optional[Tuple[float, float]] = None
    line_end: Optional[Tuple[float, float]] = None
    # 矩形/旋转矩形
    rect_center: Optional[Tuple[float, float]] = None
    rect_size: Optional[Tuple[float, float]] = None  # (w, h)
    # 圆
    circle_center: Optional[Tuple[float, float]] = None
    circle_radius: Optional[float] = None
    # 椭圆
    ellipse_center: Optional[Tuple[float, float]] = None
    ellipse_axes: Optional[Tuple[float, float]] = None  # (半长轴, 半短轴)
    # 圆环
    annulus_center: Optional[Tuple[float, float]] = None
    annulus_inner_r: Optional[float] = None
    annulus_outer_r: Optional[float] = None
    annulus_start_angle: Optional[float] = None  # 度
    annulus_end_angle: Optional[float] = None    # 度
    # 多边形/折线/不规则形
    points: Optional[List[Tuple[float, float]]] = None
    # 矩阵ROI参数
    array_shape: Optional[str] = None  # 'circle', 'rect', 'point'
    array_count: Optional[Tuple[int, int]] = None  # (rows, cols)
    array_spacing: Optional[Tuple[float, float]] = None  # (dx, dy)
    array_center_start: Optional[Tuple[float, float]] = None
    array_inner_r: Optional[float] = None
    array_outer_r: Optional[float] = None
    # 二值图像
    binary_mask: Optional[np.ndarray] = None

    @staticmethod
    def create(
        roi_type: str,
        **kwargs,
    ) -> 'ROI':
        """
        工厂方法：创建ROI。

        用法::

            roi = ROI.create('circle', center=(100, 200), radius=30)
            roi = ROI.create('rotated_rect', center=(100,100), size=(50,30), angle=45)
            roi = ROI.create('polygon', points=[(0,0), (100,0), (100,100), (0,100)])
        """
        type_map = {
            'point': ROIType.POINT,
            'line': ROIType.LINE,
            'rect': ROIType.RECT,
            'rotated_rect': ROIType.ROTATED_RECT,
            'circle': ROIType.CIRCLE,
            'ellipse': ROIType.ELLIPSE,
            'annulus': ROIType.ANNULUS,
            'polygon': ROIType.POLYGON,
            'polyline': ROIType.POLYLINE,
            'irregular': ROIType.IRREGULAR,
            'array': ROIType.ARRAY,
            'binary_mask': ROIType.BINARY_MASK,
        }

        t = type_map.get(roi_type.lower(), ROIType.RECT)
        roi = ROI(type=t)

        if t == ROIType.POINT:
            roi.point = kwargs.get('point') or kwargs.get('center')
        elif t == ROIType.LINE:
            roi.line_start = kwargs.get('start') or kwargs.get('line_start')
            roi.line_end = kwargs.get('end') or kwargs.get('line_end')
        elif t == ROIType.RECT:
            x = kwargs.get('x', 0)
            y = kwargs.get('y', 0)
            w = kwargs.get('w') or kwargs.get('width') or kwargs.get('size', (0, 0))[0]
            h = kwargs.get('h') or kwargs.get('height') or kwargs.get('size', (0, 0))[1]
            roi.rect_center = (float(x + w/2), float(y + h/2))
            roi.rect_size = (float(w), float(h))
        elif t == ROIType.ROTATED_RECT:
            roi.rect_center = kwargs.get('center')
            roi.rect_size = kwargs.get('size')
            roi.angle = kwargs.get('angle', 0.0)
        elif t == ROIType.CIRCLE:
            roi.circle_center = kwargs.get('center')
            roi.circle_radius = kwargs.get('radius')
        elif t == ROIType.ELLIPSE:
            roi.ellipse_center = kwargs.get('center')
            roi.ellipse_axes = kwargs.get('axes')
            roi.angle = kwargs.get('angle', 0.0)
        elif t == ROIType.ANNULUS:
            roi.annulus_center = kwargs.get('center')
            roi.annulus_inner_r = kwargs.get('inner_radius') or kwargs.get('inner_r')
            roi.annulus_outer_r = kwargs.get('outer_radius') or kwargs.get('outer_r')
            roi.annulus_start_angle = kwargs.get('start_angle', 0.0)
            roi.annulus_end_angle = kwargs.get('end_angle', 360.0)
        elif t in (ROIType.POLYGON, ROIType.POLYLINE, ROIType.IRREGULAR):
            roi.points = kwargs.get('points')
        elif t == ROIType.ARRAY:
            roi.array_shape = kwargs.get('shape', 'circle')
            roi.array_count = kwargs.get('count') or kwargs.get('array_count')
            roi.array_spacing = kwargs.get('spacing') or kwargs.get('array_spacing')
            roi.array_center_start = kwargs.get('center_start')
            roi.array_inner_r = kwargs.get('inner_radius') or kwargs.get('inner_r')
            roi.array_outer_r = kwargs.get('outer_radius') or kwargs.get('outer_r')
        elif t == ROIType.BINARY_MASK:
            roi.binary_mask = kwargs.get('mask')

        return roi

    def get_center(self) -> Tuple[float, float]:
        """返回ROI几何中心。"""
        if self.type == ROIType.POINT and self.point:
            return self.point
        elif self.type == ROIType.LINE and self.line_start and self.line_end:
            return ((self.line_start[0] + self.line_end[0]) / 2,
                    (self.line_start[1] + self.line_end[1]) / 2)
        elif self.type in (ROIType.RECT, ROIType.ROTATED_RECT) and self.rect_center:
            return self.rect_center
        elif self.type == ROIType.CIRCLE and self.circle_center:
            return self.circle_center
        elif self.type == ROIType.ELLIPSE and self.ellipse_center:
            return self.ellipse_center
        elif self.type == ROIType.ANNULUS and self.annulus_center:
            return self.annulus_center
        elif self.type in (ROIType.POLYGON, ROIType.POLYLINE, ROIType.IRREGULAR) and self.points:
            pts = np.array(self.points)
            return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
        return (0.0, 0.0)

    def get_bounding_box(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """
        返回外接正矩形 (x, y, w, h)。

        用于快速裁剪图像区域。
        """
        if self.type == ROIType.POINT and self.point:
            x, y = int(self.point[0]), int(self.point[1])
            return (max(0, x-5), max(0, y-5), 10, 10)
        elif self.type == ROIType.LINE and self.line_start and self.line_end:
            x1, y1 = int(self.line_start[0]), int(self.line_start[1])
            x2, y2 = int(self.line_end[0]), int(self.line_end[1])
            x = min(x1, x2); y = min(y1, y2)
            return (max(0, x-5), max(0, y-5), x2-x1+10, y2-y1+10)
        elif self.type in (ROIType.RECT, ROIType.ROTATED_RECT) and self.rect_center and self.rect_size:
            cx, cy = self.rect_center
            w, h = self.rect_size
            # 旋转矩形需要考虑角度
            if self.type == ROIType.ROTATED_RECT and abs(self.angle) > 0.1:
                # 计算旋转后的包围盒
                angle_rad = math.radians(-self.angle)
                cos_a, sin_a = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
                rw = w * cos_a + h * sin_a
                rh = w * sin_a + h * cos_a
                w, h = rw, rh
            x, y = int(cx - w/2), int(cy - h/2)
            return (max(0, x), max(0, y), int(w), int(h))
        elif self.type == ROIType.CIRCLE and self.circle_center and self.circle_radius:
            cx, cy = self.circle_center
            r = self.circle_radius
            x, y = int(cx - r), int(cy - r)
            return (max(0, x), max(0, y), int(2*r), int(2*r))
        elif self.type == ROIType.ELLIPSE and self.ellipse_center and self.ellipse_axes:
            cx, cy = self.ellipse_center
            ra, rb = self.ellipse_axes
            x, y = int(cx - ra), int(cy - rb)
            return (max(0, x), max(0, y), int(2*ra), int(2*rb))
        elif self.type == ROIType.ANNULUS and self.annulus_center:
            cx, cy = self.annulus_center
            r = self.annulus_outer_r
            x, y = int(cx - r), int(cy - r)
            return (max(0, x), max(0, y), int(2*r), int(2*r))
        elif self.type in (ROIType.POLYGON, ROIType.POLYLINE, ROIType.IRREGULAR) and self.points:
            pts = np.array(self.points)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            x, y = int(x_min), int(y_min)
            return (max(0, x-5), max(0, y-5), int(x_max-x_min)+10, int(y_max-y_min)+10)
        return (0, 0, img_w, img_h)

    def to_mask(
        self,
        img_h: int,
        img_w: int,
        fill_value: int = 255,
    ) -> np.ndarray:
        """
        将ROI转换为二值掩膜。

        Args:
            img_h, img_w: 图像尺寸
            fill_value: ROI填充值

        Returns:
            (img_h, img_w) uint8掩膜
        """
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if self.type == ROIType.POINT and self.point:
            x, y = int(self.point[0]), int(self.point[1])
            if 0 <= x < img_w and 0 <= y < img_h:
                mask[y, x] = fill_value

        elif self.type == ROIType.LINE and self.line_start and self.line_end:
            x1, y1 = int(self.line_start[0]), int(self.line_start[1])
            x2, y2 = int(self.line_end[0]), int(self.line_end[1])
            cv2.line(mask, (x1, y1), (x2, y2), fill_value, 1)

        elif self.type == ROIType.RECT and self.rect_center and self.rect_size:
            cx, cy = self.rect_center
            w, h = self.rect_size
            x, y = int(cx - w/2), int(cy - h/2)
            x, y = max(0, x), max(0, y)
            w, h = min(w, img_w - x), min(h, img_h - y)
            mask[y:y+h, x:x+w] = fill_value

        elif self.type == ROIType.ROTATED_RECT and self.rect_center and self.rect_size:
            cx, cy = self.rect_center
            w, h = self.rect_size
            rect = ((cx, cy), (w, h), self.angle)
            pts = cv2.boxPoints(rect)
            pts = np.int0(pts)
            cv2.fillPoly(mask, [pts], fill_value)

        elif self.type == ROIType.CIRCLE and self.circle_center and self.circle_radius:
            cx, cy = self.circle_center
            r = self.circle_radius
            cv2.circle(mask, (int(cx), int(cy)), int(r), fill_value, -1)

        elif self.type == ROIType.ELLIPSE and self.ellipse_center and self.ellipse_axes:
            cx, cy = self.ellipse_center
            ra, rb = self.ellipse_axes
            box = (int(cx), int(cy)), (int(2*ra), int(2*rb)), self.angle
            cv2.ellipse(mask, box, fill_value, -1)

        elif self.type == ROIType.ANNULUS and self.annulus_center:
            cx, cy = self.annulus_center
            inner_r = int(self.annulus_inner_r)
            outer_r = int(self.annulus_outer_r)
            start_a = int(self.annulus_start_angle or 0)
            end_a = int(self.annulus_end_angle or 360)
            cv2.ellipse(mask, (int(cx), int(cy)), (outer_r, outer_r),
                        0, start_a, end_a, fill_value, -1)
            if inner_r > 0:
                cv2.ellipse(mask, (int(cx), int(cy)), (inner_r, inner_r),
                            0, start_a, end_a, 0, -1)
                # 取差集 = 外圆 - 内圆
                inner_mask = np.zeros_like(mask)
                cv2.ellipse(inner_mask, (int(cx), int(cy)), (inner_r, inner_r),
                            0, start_a, end_a, 255, -1)
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(inner_mask))

        elif self.type in (ROIType.POLYGON, ROIType.POLYLINE, ROIType.IRREGULAR) and self.points:
            pts = np.array(self.points, dtype=np.int32)
            if self.type == ROIType.POLYGON:
                cv2.fillPoly(mask, [pts], fill_value)
            else:
                cv2.polylines(mask, [pts], False, fill_value, 1)

        elif self.type == ROIType.BINARY_MASK and self.binary_mask is not None:
            binary = self.binary_mask
            if binary.shape[:2] != (img_h, img_w):
                binary = cv2.resize(binary, (img_w, img_h))
            _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
            mask = binary

        return mask

    def apply_affine_transform(self, M: np.ndarray) -> 'ROI':
        """
        对ROI应用仿射变换矩阵。

        Args:
            M: (2, 3) 仿射变换矩阵

        Returns:
            变换后的新ROI
        """
        def transform_point(pt):
            if pt is None:
                return None
            x, y = pt
            x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            return (float(x_new), float(y_new))

        def transform_points(pts):
            if pts is None:
                return None
            arr = np.array(pts, dtype=np.float64)
            ones = np.ones((len(arr), 1))
            homogeneous = np.hstack([arr, ones])
            transformed = (M @ homogeneous.T).T
            return [(float(p[0]), float(p[1])) for p in transformed]

        new_roi = ROI(type=self.type, angle=self.angle)

        if self.type == ROIType.POINT:
            new_roi.point = transform_point(self.point)
        elif self.type == ROIType.LINE:
            new_roi.line_start = transform_point(self.line_start)
            new_roi.line_end = transform_point(self.line_end)
        elif self.type in (ROIType.RECT, ROIType.ROTATED_RECT):
            new_roi.rect_center = transform_point(self.rect_center)
            new_roi.rect_size = self.rect_size
            new_roi.angle = self.angle
        elif self.type == ROIType.CIRCLE:
            new_roi.circle_center = transform_point(self.circle_center)
            new_roi.circle_radius = self.circle_radius
        elif self.type == ROIType.ELLIPSE:
            new_roi.ellipse_center = transform_point(self.ellipse_center)
            new_roi.ellipse_axes = self.ellipse_axes
            new_roi.angle = self.angle
        elif self.type == ROIType.ANNULUS:
            new_roi.annulus_center = transform_point(self.annulus_center)
            new_roi.annulus_inner_r = self.annulus_inner_r
            new_roi.annulus_outer_r = self.annulus_outer_r
            new_roi.annulus_start_angle = self.annulus_start_angle
            new_roi.annulus_end_angle = self.annulus_end_angle
        elif self.type in (ROIType.POLYGON, ROIType.POLYLINE, ROIType.IRREGULAR):
            new_roi.points = transform_points(self.points)
        elif self.type == ROIType.ARRAY:
            new_roi.array_shape = self.array_shape
            new_roi.array_count = self.array_count
            new_roi.array_spacing = self.array_spacing
            new_roi.array_center_start = transform_point(self.array_center_start)
            new_roi.array_inner_r = self.array_inner_r
            new_roi.array_outer_r = self.array_outer_r

        return new_roi

    def __repr__(self):
        return f"ROI(type={self.type.value}, center={self.get_center()})"


# ============================================================
# 3. ROI生成器
# ============================================================

class ROIGenerator:
    """
    ROI生成器 — 从多种数据源生成ROI。

    支持：
      - 从轮廓生成（Blob边缘、霍夫圆/直线）
      - 从二值图像生成（白色区域=ROI）
      - 从矩阵参数生成（圆形/矩形点阵）
      - 从参考特征生成（Blob质心、匹配中心）

    用法::

        gen = ROIGenerator()
        roi = gen.from_contour(contour, type='rotated_rect')
        roi = gen.from_binary(binary_mask)
        roi = gen.array(count=(3, 3), spacing=(20, 20), shape='circle')
    """

    @staticmethod
    def from_contour(
        contour: np.ndarray,
        roi_type: str = 'rotated_rect',
        padding: int = 10,
    ) -> ROI:
        """
        从轮廓生成ROI。

        Args:
            contour: 轮廓点 (N, 1, 2)
            roi_type: 'rect', 'rotated_rect', 'circle', 'ellipse'
            padding: 扩展像素

        Returns:
            ROI对象
        """
        cnt = contour.reshape(-1, 2)

        # 外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w/2, y + h/2

        if roi_type == 'rect':
            return ROI.create('rect', x=x-padding, y=y-padding,
                            w=w+2*padding, h=h+2*padding)

        elif roi_type == 'rotated_rect':
            rect = cv2.minAreaRect(cnt)
            ((cx, cy), (w, h), angle) = rect
            return ROI.create('rotated_rect',
                            center=(float(cx), float(cy)),
                            size=(float(w) + 2*padding, float(h) + 2*padding),
                            angle=float(angle))

        elif roi_type == 'circle':
            # 最小外接圆
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            return ROI.create('circle',
                            center=(float(cx), float(cy)),
                            radius=float(radius) + padding)

        elif roi_type == 'ellipse':
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                ((cx, cy), (ma, MA), angle) = ellipse
                return ROI.create('ellipse',
                                center=(float(cx), float(cy)),
                                axes=(float(ma/2) + padding, float(MA/2) + padding),
                                angle=float(angle))
            else:
                return ROI.create('rect', x=x-padding, y=y-padding,
                                w=w+2*padding, h=h+2*padding)

        return ROI.create('rect', x=x-padding, y=y-padding,
                        w=w+2*padding, h=h+2*padding)

    @staticmethod
    def from_binary(
        binary_mask: np.ndarray,
        min_area: int = 100,
    ) -> Optional[ROI]:
        """
        从二值图像生成ROI（白色区域）。

        Args:
            binary_mask: (H, W) uint8 二值图像
            min_area: 最小区域面积

        Returns:
            ROI对象（包含二值掩膜）
        """
        if binary_mask.max() <= 1:
            binary_mask = (binary_mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # 取最大轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < min_area:
            return None

        # 生成ROI
        roi = ROIGenerator.from_contour(largest, roi_type='rotated_rect')
        roi.type = ROIType.BINARY_MASK
        roi.binary_mask = binary_mask
        return roi

    @staticmethod
    def array(
        count: Tuple[int, int],
        spacing: Tuple[float, float],
        shape: str = 'circle',
        center_start: Tuple[float, float] = (0, 0),
        inner_radius: float = 5,
        outer_radius: float = 10,
    ) -> ROI:
        """
        生成矩阵ROI（圆形/矩形点阵）。

        用法::

            roi = ROIGenerator.array(count=(3, 4), spacing=(30, 30),
                                    shape='circle', inner_radius=5, outer_radius=10)
        """
        return ROI.create('array',
                         shape=shape,
                         count=count,
                         spacing=spacing,
                         center_start=center_start,
                         inner_radius=inner_radius,
                         outer_radius=outer_radius)

    @staticmethod
    def from_feature(
        feature: Dict,
        roi_type: str = 'circle',
        default_radius: float = 30,
    ) -> ROI:
        """
        从检测到的特征生成ROI。

        Args:
            feature: 特征字典（包含centroid/center等）
            roi_type: ROI类型

        Returns:
            ROI对象
        """
        # 提取中心点
        center = None
        if 'centroid_px' in feature:
            center = feature['centroid_px']
        elif 'centroid' in feature:
            center = feature['centroid']
        elif 'center' in feature:
            center = feature['center']
        elif 'position' in feature:
            center = feature['position'][:2] if len(feature['position']) >= 2 else (0, 0)

        if center is None:
            center = (0, 0)

        # 确定大小
        radius = default_radius
        if 'area' in feature:
            radius = max(5, float(math.sqrt(feature['area'] / math.pi)) * 0.6)
        if 'radius' in feature:
            radius = float(feature['radius'])
        if 'bbox_w' in feature and 'bbox_h' in feature:
            radius = max(feature['bbox_w'], feature['bbox_h']) / 2

        if roi_type == 'circle':
            return ROI.create('circle', center=center, radius=radius)
        elif roi_type == 'rotated_rect':
            angle = feature.get('orientation_deg', feature.get('angle_deg', 0.0))
            return ROI.create('rotated_rect', center=center,
                            size=(radius*2, radius*2), angle=angle)
        else:
            return ROI.create('circle', center=center, radius=radius)


def generate_roi_from_contour(contour: np.ndarray, **kwargs) -> ROI:
    """便捷函数：从轮廓生成ROI。"""
    return ROIGenerator.from_contour(contour, **kwargs)


def generate_roi_from_binary(binary_mask: np.ndarray, **kwargs) -> Optional[ROI]:
    """便捷函数：从二值图像生成ROI。"""
    return ROIGenerator.from_binary(binary_mask, **kwargs)


def generate_array_roi(count, spacing, **kwargs) -> ROI:
    """便捷函数：生成矩阵ROI。"""
    return ROIGenerator.array(count, spacing, **kwargs)


# ============================================================
# 4. ROI校正器（基准跟随）
# ============================================================

class CorrectionMode(Enum):
    """ROI校正模式。"""
    HORIZONTAL = "horizontal"           # 仅水平
    VERTICAL = "vertical"               # 仅垂直
    HORIZONTAL_VERTICAL = "hv"         # 水平+垂直
    HORIZONTAL_VERTICAL_ANGLE = "hva"  # 水平+垂直+角度（完整）


class ROICorrector:
    """
    ROI校正器 — 基于参考点的仿射变换实现ROI基准跟随。

    Smart3第9.4章ROI校正基准设置实现：

    工作原理：
      1. 记录第一帧中检测到的参考特征（参考点、参考角度）
      2. 后续帧中跟踪同一特征获取当前位姿
      3. 计算变换矩阵（平移+旋转）
      4. 将该变换应用到所有ROI

    用法::

        corrector = ROICorrector(mode='horizontal_vertical_angle')
        corrector.set_reference(center=(100, 100), angle=0)

        # 后续帧：基于检测到的当前位姿校正ROI
        corrected = corrector.apply(roi,
                                   current_center=(120, 90),
                                   current_angle=15)
    """

    def __init__(
        self,
        mode: str = 'hv',
    ):
        """
        Args:
            mode: 'horizontal' / 'vertical' / 'hv' / 'hva'
        """
        mode_map = {
            'horizontal': CorrectionMode.HORIZONTAL,
            'vertical': CorrectionMode.VERTICAL,
            'hv': CorrectionMode.HORIZONTAL_VERTICAL,
            'hva': CorrectionMode.HORIZONTAL_VERTICAL_ANGLE,
        }
        self.mode = mode_map.get(mode, CorrectionMode.HORIZONTAL_VERTICAL)

        # 参考状态
        self.ref_center: Optional[Tuple[float, float]] = None
        self.ref_angle: float = 0.0

        # 当前状态
        self.cur_center: Optional[Tuple[float, float]] = None
        self.cur_angle: float = 0.0

    def set_reference(
        self,
        center: Optional[Tuple[float, float]] = None,
        angle: float = 0.0,
    ) -> None:
        """
        设置参考基准（第一帧中建立）。

        Args:
            center: 参考中心点 (x, y)
            angle: 参考角度（度）
        """
        self.ref_center = center
        self.ref_angle = angle

    def update_current(
        self,
        center: Optional[Tuple[float, float]] = None,
        angle: float = 0.0,
    ) -> None:
        """
        更新当前位姿（后续帧中检测到的特征位置）。

        Args:
            center: 当前中心点
            angle: 当前角度
        """
        self.cur_center = center
        self.cur_angle = angle

    def compute_transform(self) -> Optional[np.ndarray]:
        """
        计算从参考位姿到当前位姿的仿射变换矩阵。

        Returns:
            (2, 3) 仿射矩阵；如果未设置基准返回None
        """
        if self.ref_center is None or self.cur_center is None:
            return None

        rx, ry = self.ref_center
        cx, cy = self.cur_center

        dx = cx - rx
        dy = cy - ry
        d_angle = self.cur_angle - self.ref_angle

        if self.mode == CorrectionMode.HORIZONTAL:
            # 仅水平校正：只平移x
            M = np.array([[1, 0, dx],
                          [0, 1, 0]], dtype=np.float64)
        elif self.mode == CorrectionMode.VERTICAL:
            # 仅垂直校正
            M = np.array([[1, 0, 0],
                          [0, 1, dy]], dtype=np.float64)
        elif self.mode == CorrectionMode.HORIZONTAL_VERTICAL:
            # 水平+垂直校正
            M = np.array([[1, 0, dx],
                          [0, 1, dy]], dtype=np.float64)
        else:  # HORIZONTAL_VERTICAL_ANGLE
            # 完整仿射变换（平移+旋转）
            angle_rad = math.radians(d_angle)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            M = np.array([[cos_a, -sin_a, dx],
                          [sin_a,  cos_a, dy]], dtype=np.float64)

        return M

    def apply(self, roi: ROI) -> ROI:
        """
        对ROI应用校正变换。

        Args:
            roi: 输入ROI

        Returns:
            校正后的ROI
        """
        M = self.compute_transform()
        if M is None:
            return roi
        return roi.apply_affine_transform(M)


def correct_roi(
    roi: ROI,
    ref_center: Tuple[float, float],
    ref_angle: float,
    cur_center: Tuple[float, float],
    cur_angle: float,
    mode: str = 'hva',
) -> ROI:
    """
    便捷函数：ROI校正。

    用法::

        corrected = correct_roi(roi,
                                ref_center=(100, 100), ref_angle=0,
                                cur_center=(120, 90), cur_angle=15)
    """
    corrector = ROICorrector(mode=mode)
    corrector.set_reference(ref_center, ref_angle)
    corrector.update_current(cur_center, cur_angle)
    return corrector.apply(roi)


# ============================================================
# 5. 自动ROI检测与切换
# ============================================================

class AutoROIMode(Enum):
    """自动ROI检测模式。"""
    BLOB = "blob"               # 基于Blob分析
    EDGE = "edge"               # 基于边缘密度
    FEATURE = "feature"         # 基于特征匹配
    TEMPLATE = "template"        # 基于模板匹配
    ADAPTIVE = "adaptive"        # 自适应切换


class AutoROI:
    """
    自动ROI检测与切换 — 无需人工干预的自适应ROI系统。

    核心功能：
      1. 自动检测：基于图像内容自动找到最佳ROI位置和类型
      2. 自动切换：根据场景变化自动切换ROI模式
      3. 跟踪保持：持续跟踪目标并调整ROI

    用法::

        auto_roi = AutoROI(mode='adaptive')

        # 自动检测（第一帧或场景变化时）
        best_roi = auto_roi.detect(image,
                                   candidates=[roi1, roi2, roi3],
                                   method='edge_density')

        # 更新跟踪（后续帧）
        tracked = auto_roi.track(image, previous_roi)

        # 检查是否需要重新检测
        if auto_roi.should_redetect(image):
            best_roi = auto_roi.detect(image, ...)
    """

    def __init__(
        self,
        mode: str = 'adaptive',
        blob_threshold: float = 0.3,
        edge_threshold: float = 0.5,
        redetect_interval: int = 30,
        match_threshold: float = 0.6,
    ):
        """
        Args:
            mode: 'blob' / 'edge' / 'feature' / 'template' / 'adaptive'
            blob_threshold: Blob置信度阈值
            edge_threshold: 边缘密度阈值
            redetect_interval: 强制重新检测的帧间隔
            match_threshold: 模板匹配阈值
        """
        mode_map = {
            'blob': AutoROIMode.BLOB,
            'edge': AutoROIMode.EDGE,
            'feature': AutoROIMode.FEATURE,
            'template': AutoROIMode.TEMPLATE,
            'adaptive': AutoROIMode.ADAPTIVE,
        }
        self.mode = mode_map.get(mode, AutoROIMode.ADAPTIVE)

        self.blob_threshold = blob_threshold
        self.edge_threshold = edge_threshold
        self.redetect_interval = redetect_interval
        self.match_threshold = match_threshold

        # 状态
        self._frame_count = 0
        self._last_roi: Optional[ROI] = None
        self._last_center: Optional[Tuple[float, float]] = None
        self._last_image_hash: Optional[int] = None
        self._template_image: Optional[np.ndarray] = None

    def detect(
        self,
        image: np.ndarray,
        candidates: Optional[List[ROI]] = None,
        method: str = 'edge_density',
    ) -> Optional[ROI]:
        """
        自动检测最佳ROI。

        Args:
            image: 输入图像 BGR
            candidates: 候选ROI列表（可选）
            method: 'blob' / 'edge_density' / 'variance' / 'entropy'

        Returns:
            检测到的最佳ROI
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape

        if candidates is None:
            # 自动生成候选ROI网格
            candidates = self._generate_candidate_grid(w, h, step=50)

        scores = []
        for roi in candidates:
            score = self._evaluate_roi(gray, roi, method)
            scores.append((score, roi))

        if not scores:
            return None

        # 选择得分最高的ROI
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_roi = scores[0]

        self._last_roi = best_roi
        self._last_center = best_roi.get_center()
        self._frame_count = 1

        return best_roi

    def _generate_candidate_grid(
        self,
        w: int,
        h: int,
        step: int = 50,
    ) -> List[ROI]:
        """生成候选ROI网格。"""
        candidates = []
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                # 添加多种尺寸的ROI
                for size in [(30, 30), (50, 50), (80, 80), (100, 50), (50, 100)]:
                    roi = ROI.create('rect',
                                   x=x - size[0]//2,
                                   y=y - size[1]//2,
                                   w=size[0],
                                   h=size[1])
                    candidates.append(roi)
        return candidates

    def _evaluate_roi(
        self,
        gray: np.ndarray,
        roi: ROI,
        method: str,
    ) -> float:
        """
        评估ROI质量得分。

        Returns:
            得分 [0, 1]，越高越好
        """
        x, y, bw, bh = roi.get_bounding_box(gray.shape[1], gray.shape[0])
        if bw <= 0 or bh <= 0:
            return 0.0

        roi_img = gray[y:y+bh, x:x+bw]
        if roi_img.size == 0:
            return 0.0

        if method == 'edge_density':
            # 边缘密度评估
            edges = cv2.Canny(roi_img, 50, 150)
            density = edges.sum() / (255 * roi_img.size)
            return float(density)

        elif method == 'variance':
            # 灰度方差评估（高方差=特征丰富）
            var = np.var(roi_img.astype(float))
            return min(1.0, var / 1000.0)

        elif method == 'entropy':
            # 灰度熵评估
            hist = cv2.calcHist([roi_img], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return min(1.0, entropy / 8.0)

        elif method == 'blob':
            # Blob分析评估（高对比度区域）
            _, binary = cv2.threshold(roi_img, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ratio = binary.sum() / (255 * roi_img.size)
            return abs(0.5 - ratio)  # 偏离0.5越多越好

        return 0.0

    def track(
        self,
        image: np.ndarray,
        previous_roi: Optional[ROI] = None,
    ) -> Optional[ROI]:
        """
        跟踪ROI位置。

        Args:
            image: 当前帧
            previous_roi: 上一帧的ROI

        Returns:
            跟踪到的ROI
        """
        if previous_roi is None:
            previous_roi = self._last_roi

        if previous_roi is None:
            return self.detect(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape

        # 获取上一帧ROI的中心
        prev_center = previous_roi.get_center()

        # 在小范围内搜索最佳匹配（基于模板匹配）
        search_radius = 30
        cx, cy = int(prev_center[0]), int(prev_center[1])

        x1 = max(0, cx - search_radius)
        y1 = max(0, cy - search_radius)
        x2 = min(w, cx + search_radius)
        y2 = min(h, cy + search_radius)

        if x2 <= x1 or y2 <= y1:
            return previous_roi

        # 提取上一帧ROI区域作为模板
        prev_mask = previous_roi.to_mask(h, w)
        prev_gray = gray.copy()
        prev_gray[prev_mask == 0] = 0

        # 在搜索区域内找最佳匹配
        search_region = gray[y1:y2, x1:x2]
        if search_region.size == 0:
            return previous_roi

        # 简化的匹配：找灰度最相似的位置
        best_match_x, best_match_y = cx, cy
        best_score = float('inf')

        step = 5
        for sy in range(y1, y2, step):
            for sx in range(x1, x2, step):
                # 提取候选区域
                pw, ph = previous_roi.rect_size if previous_roi.rect_size else (30, 30)
                ex1, ey1 = max(0, sx - pw//2), max(0, sy - ph//2)
                ex2, ey2 = min(w, sx + pw//2), min(h, sy + ph//2)

                if ex2 <= ex1 or ey2 <= ey1:
                    continue

                candidate = gray[ey1:ey2, ex1:ex2]
                if candidate.size != prev_gray[prev_mask > 0].reshape(-1).shape[0]:
                    continue

                # 计算差异
                diff = np.mean(np.abs(candidate.astype(float) -
                                   prev_gray[prev_mask > 0].reshape(candidate.shape).astype(float)))
                if diff < best_score:
                    best_score = diff
                    best_match_x, best_match_y = sx, sy

        # 创建跟踪到的ROI
        tracked = ROI.create(
            previous_roi.type.value if hasattr(previous_roi.type, 'value') else 'rect',
            center=(float(best_match_x), float(best_match_y)),
            size=previous_roi.rect_size if previous_roi.rect_size else (30, 30),
            angle=previous_roi.angle
        )

        self._last_roi = tracked
        self._last_center = (best_match_x, best_match_y)
        self._frame_count += 1

        return tracked

    def should_redetect(self, image: np.ndarray) -> bool:
        """
        判断是否需要重新检测（场景变化检测）。

        Args:
            image: 当前帧

        Returns:
            True=需要重新检测
        """
        self._frame_count += 1

        # 周期性强制重新检测
        if self._frame_count >= self.redetect_interval:
            return True

        # 计算图像直方图变化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        img_hash = hash(gray.tobytes()[:1000])

        if self._last_image_hash is not None:
            # 图像内容变化检测
            change_ratio = bin(img_hash ^ self._last_image_hash).count('1') / 32.0
            if change_ratio > 0.3:  # 变化超过30%
                return True

        self._last_image_hash = img_hash
        return False

    def update_reference(
        self,
        roi: ROI,
        center: Tuple[float, float],
        angle: float = 0.0,
    ) -> None:
        """
        更新参考基准（用于ROI校正器联动）。

        用法::

            auto_roi = AutoROI()
            corrector = ROICorrector()
            corrector.set_reference(...)
            # 每帧更新
            corrector.update_current(center, angle)
            corrected = corrector.apply(roi)
        """
        self._last_roi = roi
        self._last_center = center

    def set_template(self, image: np.ndarray, roi: ROI) -> None:
        """
        设置模板图像（用于模板匹配模式）。

        Args:
            image: 模板图像
            roi: 模板ROI区域
        """
        h, w = image.shape[:2]
        mask = roi.to_mask(h, w)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        self._template_image = gray.copy()
        self._template_mask = mask
        self._template_roi = roi


def select_roi_adaptive(
    image: np.ndarray,
    candidates: List[ROI],
    method: str = 'edge_density',
) -> Optional[ROI]:
    """
    便捷函数：自适应选择最佳ROI。

    用法::

        best = select_roi_adaptive(image, [roi1, roi2, roi3], method='variance')
    """
    auto_roi = AutoROI(mode='adaptive')
    return auto_roi.detect(image, candidates, method)


# ============================================================
# 命令行验证
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ROI工具箱验证")
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "correction", "auto"])
    args = parser.parse_args()

    if args.mode == "basic":
        print("=== ROI基本功能测试 ===")
        # 创建多种ROI
        roi1 = ROI.create('circle', center=(100, 100), radius=30)
        roi2 = ROI.create('rotated_rect', center=(200, 150), size=(80, 40), angle=45)
        roi3 = ROI.create('polygon', points=[(50,50), (100,50), (100,100), (50,100)])

        print(f"ROI1: {roi1}")
        print(f"ROI2: {roi2}")
        print(f"ROI3: {roi3}")

        # 生成掩膜
        mask1 = roi1.to_mask(300, 300)
        mask2 = roi2.to_mask(300, 300)
        mask3 = roi3.to_mask(300, 300)

        print(f"掩膜1 (圆形): {mask1.sum()//255} 像素")
        print(f"掩膜2 (旋转矩形): {mask2.sum()//255} 像素")
        print(f"掩膜3 (多边形): {mask3.sum()//255} 像素")

    elif args.mode == "correction":
        print("=== ROI校正测试 ===")
        # 原始ROI
        roi = ROI.create('circle', center=(100, 100), radius=20)

        # 校正：参考点(100,100)->当前点(120,90)，角度0->15
        corrected = correct_roi(roi,
                              ref_center=(100, 100), ref_angle=0,
                              cur_center=(120, 90), cur_angle=15)
        print(f"原始ROI中心: {roi.get_center()}")
        print(f"校正后ROI中心: {corrected.get_center()}")

    elif args.mode == "auto":
        print("=== 自动ROI检测测试 ===")
        # 创建测试图像
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (150, 150), (200, 200, 200), -1)
        cv2.circle(img, (250, 200), 40, (150, 150, 150), -1)

        # 自动检测
        auto_roi = AutoROI()
        candidates = [
            ROI.create('rect', x=90, y=90, w=70, h=70),
            ROI.create('circle', center=(250, 200), radius=50),
            ROI.create('rotated_rect', center=(125, 125), size=(60, 60), angle=30),
        ]

        best = auto_roi.detect(img, candidates, method='edge_density')
        print(f"最佳ROI: {best}")
        print(f"得分ROI类型: {best.type.value if best else None}")
