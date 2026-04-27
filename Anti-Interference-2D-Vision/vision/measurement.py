"""
measurement.py — 测量与匹配算法模块：卡尺、间隙测量、几何关系、轮廓操作
:Author: RussellCooper

基于Smart3智能视觉系统用户手册算法实现：

1. 卡尺测量 (Caliper) — Smart3第10.2章
   - 在ROI区域内寻找两条平行边缘，计算距离
   - 支持多方向搜索、边缘极性检测

2. 间隙测量 (Gap Measurement) — Smart3第10.5章
   - 多边缘间隙测量
   - 支持节距、宽度、间距数据输出

3. 几何关系 (Geometric Relations) — Smart3第10.1章
   - 点、线、圆、椭圆、矩形之间的几何关系计算
   - 拟合功能

4. 轮廓操作 (Contour Operations) — Smart3第8.3章
   - 筛选、分割、连接、平滑轮廓

5. 图像锐度 (Image Sharpness) — Smart3第10.6章
   - 聚焦清晰度评分

6. Blob分析增强 — Smart3第7.1章
   - 连通域分析

依赖: opencv-contrib-python>=4.5, numpy>=1.21
"""

import argparse
import itertools
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

__all__ = [
    # 卡尺测量
    "CaliperMeasurement",
    "measure_caliper",
    # 间隙测量
    "GapMeasurement",
    "measure_gap",
    # 几何关系
    "GeometricRelations",
    "compute_geometric_relations",
    # 轮廓操作
    "ContourOperations",
    "filter_contours",
    "split_contour",
    "connect_collinear_contours",
    "smooth_contour",
    # 图像锐度
    "compute_sharpness",
    "ImageSharpness",
    # Blob分析
    "BlobAnalyzer",
    "analyze_blobs",
]



# ---------------------------------------------------------------------------
# 1. 卡尺测量 (Caliper)
# ---------------------------------------------------------------------------

class CaliperMeasurement:
    """
    卡尺测量 - 在灰度图像上寻找两条平行边缘并计算距离。

    算法流程：
      1. 在ROI区域内设置多条搜索线
      2. 沿搜索方向检测边缘点（根据极性和强度）
      3. 拟合平行的两条直线
      4. 计算两条直线之间的距离

    用于测量物体宽度。
    """

    def __init__(
        self,
        search_direction: str = 'left_to_right',
        polarity: str = 'black_to_white',
        edge_intensity: int = 20,
        search_line_count: int = 20,
        edge_width: int = 5,
        projection_width: int = 5,
        rejection_ratio: float = 0.1,
        rejection_distance: float = 5.0,
        max_angle: float = 10.0,
    ):
        """
        Args:
            search_direction: 'top_to_bottom', 'bottom_to_top', 'left_to_right', 'right_to_left'
            polarity: 'black_to_white', 'white_to_black', 'all'
            edge_intensity: 边缘强度阈值 (2-255)
            search_line_count: 搜索线数量
            edge_width: 边缘宽度（考虑渐变）
            projection_width: 投影宽度（降噪）
            rejection_ratio: 剔除比例
            rejection_distance: 剔除距离阈值
            max_angle: 最大角度限制
        """
        self.search_direction = search_direction
        self.polarity = polarity
        self.edge_intensity = edge_intensity
        self.search_line_count = search_line_count
        self.edge_width = edge_width
        self.projection_width = projection_width
        self.rejection_ratio = rejection_ratio
        self.rejection_distance = rejection_distance
        self.max_angle = max_angle

    def measure(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> Dict[str, any]:
        """
        执行卡尺测量。

        Args:
            image: BGR uint8 图像
            roi: (x, y, w, h) 感兴趣区域

        Returns:
            Dict with keys:
              - 'distance': float, 两条边缘的距离（像素）
              - 'edge1_line': (point, angle), 第一条边缘线
              - 'edge2_line': (point, angle), 第二条边缘线
              - 'edge_points1': list of (x, y), 第一条边缘点
              - 'edge_points2': list of (x, y), 第二条边缘点
              - 'valid': bool, 测量是否有效
        """
        x, y, w, h = roi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        roi_gray = gray[y:y+h, x:x+w]

        # 搜索方向向量
        dx, dy = self._get_search_direction()
        # 垂直方向（边缘方向）
        px, py = -dy, dx

        # 生成搜索线
        edge_points1, edge_points2 = [], []
        step = h // self.search_line_count if abs(dy) > 0 else w // self.search_line_count

        for i in range(self.search_line_count):
            if abs(dy) > 0:  # 垂直方向搜索
                start_y = i * step
                if start_y >= h:
                    break
                start_point = (x, y + start_y)
            else:  # 水平方向搜索
                start_x = i * step
                if start_x >= w:
                    break
                start_point = (x + start_x, y)

            # 沿搜索线找边缘
            pt1, pt2 = self._find_edge_pair_along_line(roi_gray, start_point, dx, dy)

            if pt1 is not None:
                edge_points1.append((pt1[0] + x, pt1[1] + y))
            if pt2 is not None:
                edge_points2.append((pt2[0] + x, pt2[1] + y))

        # 拟合直线
        line1 = self._fit_line(edge_points1)
        line2 = self._fit_line(edge_points2)

        # 计算距离
        distance = 0.0
        if line1 is not None and line2 is not None:
            distance = self._line_distance(line1, line2)

        return {
            'distance': distance,
            'edge1_line': line1,
            'edge2_line': line2,
            'edge_points1': edge_points1,
            'edge_points2': edge_points2,
            'valid': len(edge_points1) > 2 and len(edge_points2) > 2,
        }

    def _get_search_direction(self) -> Tuple[float, float]:
        """获取搜索方向向量。"""
        if self.search_direction == 'top_to_bottom':
            return (0, 1)
        elif self.search_direction == 'bottom_to_top':
            return (0, -1)
        elif self.search_direction == 'left_to_right':
            return (1, 0)
        elif self.search_direction == 'right_to_left':
            return (-1, 0)
        return (1, 0)

    def _find_edge_pair_along_line(
        self,
        gray: np.ndarray,
        start: Tuple[int, int],
        dx: float, dy: float,
    ) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """沿搜索线找一对边缘点。"""
        h, w = gray.shape
        x, y = start

        # 投影降噪
        profile = []
        for i in range(max(w, h)):
            px = int(x + dx * i)
            py = int(y + dy * i)
            if 0 <= px < w and 0 <= py < h:
                # 取投影宽度内的均值
                values = []
                for j in range(-self.projection_width, self.projection_width + 1):
                    if abs(dx) > 0:  # 水平搜索
                        ny = py + j
                        nx = px
                    else:  # 垂直搜索
                        nx = px + j
                        ny = py
                    if 0 <= nx < w and 0 <= ny < h:
                        values.append(gray[ny, nx])
                    else:
                        values.append(0)
                profile.append((np.mean(values), px, py))

        # 计算梯度
        gradients = []
        for i in range(len(profile) - 1):
            diff = profile[i+1][0] - profile[i][0]
            gradients.append((diff, profile[i][1], profile[i][2]))

        # 根据极性找边缘（找前两个匹配点）
        edge1, edge2 = None, None
        found_edge1 = False
        found_edge2 = False

        for grad, ex, ey in gradients:
            if self.polarity == 'black_to_white' and grad > self.edge_intensity:
                if not found_edge1:
                    edge1 = (ex, ey)
                    found_edge1 = True
                elif not found_edge2:
                    edge2 = (ex, ey)
                    found_edge2 = True
                    break
            elif self.polarity == 'white_to_black' and grad < -self.edge_intensity:
                if not found_edge1:
                    edge1 = (ex, ey)
                    found_edge1 = True
                elif not found_edge2:
                    edge2 = (ex, ey)
                    found_edge2 = True
                    break
            elif self.polarity == 'all':
                if abs(grad) > self.edge_intensity:
                    if not found_edge1:
                        edge1 = (ex, ey)
                        found_edge1 = True
                    elif not found_edge2:
                        edge2 = (ex, ey)
                        found_edge2 = True
                        break

        return edge1, edge2

    def _fit_line(self, points: List[Tuple[float, float]]) -> Optional[Tuple]:
        """拟合直线。"""
        if len(points) < 2:
            return None

        points = np.array(points, dtype=np.float64)
        if len(points) == 2:
            # 两点确定一条直线
            x1, y1 = points[0]
            x2, y2 = points[1]
            angle = math.atan2(y2 - y1, x2 - x1)
            cx, cy = points.mean(axis=0)
            return ((cx, cy), math.degrees(angle))

        # 使用SVD拟合直线
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, Vt = np.linalg.svd(centered)
        direction = Vt[0]
        angle = math.atan2(direction[1], direction[0])

        return (tuple(centroid), math.degrees(angle))

    def _line_distance(self, line1: Tuple, line2: Tuple) -> float:
        """
        计算两条平行线之间的距离。

        Args:
            line1: (centroid_pt, angle_deg) 直线1
            line2: (centroid_pt, angle_deg) 直线2

        Returns:
            两条直线之间的垂直距离（像素）
        """
        (x1, y1), angle1 = line1
        (x2, y2), angle2 = line2

        # 使用平均角度作为共同方向
        angle_rad = math.radians((angle1 + angle2) / 2)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        # 法向量（垂直于直线方向）
        normal_x, normal_y = -sin_a, cos_a

        # 点p2到直线1的垂直距离
        # d = |(p2 - p1) · n|
        dx = x2 - x1
        dy = y2 - y1
        distance = abs(dx * normal_x + dy * normal_y)

        return distance


def measure_caliper(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    **kwargs,
) -> Dict:
    """便捷函数：卡尺测量。"""
    caliper = CaliperMeasurement(**kwargs)
    return caliper.measure(image, roi)


# ---------------------------------------------------------------------------
# 2. 间隙测量 (Gap Measurement)
# ---------------------------------------------------------------------------

class GapMeasurement:
    """
    间隙测量 - 多边缘间隙测量。

    支持测量：
      - 节距 (pitch): 相邻边缘之间的距离
      - 宽度 (width): 两条边缘之间的宽度
      - 间距 (spacing): 边缘之间的间距
    """

    def __init__(
        self,
        search_direction: str = 'left_to_right',
        polarity: str = 'black_to_white',
        edge_intensity: int = 20,
        search_line_count: int = 10,
        edge_spacing: int = 5,
        rejection_ratio: float = 0.1,
        max_edge_count: int = 20,
        edge_width: int = 5,
    ):
        """
        Args:
            search_direction: 搜索方向
            polarity: 边缘极性
            edge_intensity: 边缘强度
            search_line_count: 搜索线数量
            edge_spacing: 边缘线间隔（合并相邻边缘）
            rejection_ratio: 抗干扰率
            max_edge_count: 最大边缘数目
            edge_width: 边缘宽度
        """
        self.search_direction = search_direction
        self.polarity = polarity
        self.edge_intensity = edge_intensity
        self.search_line_count = search_line_count
        self.edge_spacing = edge_spacing
        self.rejection_ratio = rejection_ratio
        self.max_edge_count = max_edge_count
        self.edge_width = edge_width

    def measure(
        self,
        image: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> Dict[str, any]:
        """
        执行间隙测量。

        Returns:
            Dict with keys:
              - 'pitches': list of float, 节距列表
              - 'widths': list of float, 宽度列表
              - 'spacings': list of float, 间距列表
              - 'edge_points': list of (x, y), 所有边缘点
        """
        x, y, w, h = roi
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        roi_gray = gray[y:y+h, x:x+w]

        # 收集所有搜索线的边缘点
        all_edges = []

        for i in range(self.search_line_count):
            if 'left_to_right' in self.search_direction or 'right_to_left' in self.search_direction:
                start_x = 0
                step = w // self.search_line_count
                start_point = (i * step, 0) if 'left_to_right' in self.search_direction else (w - 1 - i * step, 0)
                dx = 1 if 'left_to_right' in self.search_direction else -1
                dy = 0
            else:
                start_y = 0
                step = h // self.search_line_count
                start_point = (0, i * step) if 'top_to_bottom' in self.search_direction else (0, h - 1 - i * step)
                dx = 0
                dy = 1 if 'top_to_bottom' in self.search_direction else -1

            edges = self._find_edges_along_line(roi_gray, start_point, dx, dy)
            all_edges.extend([(e[0] + x, e[1] + y) for e in edges])

        # 计算间隙数据
        if len(all_edges) < 2:
            return {'pitches': [], 'widths': [], 'spacings': [], 'edge_points': all_edges}

        # 按位置排序
        if abs(dx) > 0:
            all_edges.sort(key=lambda p: p[0])
        else:
            all_edges.sort(key=lambda p: p[1])

        # 合并相邻过近的边缘
        merged = self._merge_close_edges(all_edges)

        # 计算节距、宽度、间距
        widths = []
        spacings = []
        for i in range(len(merged) - 1):
            gap = abs(merged[i+1][0] - merged[i][0]) if abs(dx) > 0 else abs(merged[i+1][1] - merged[i][1])
            if i % 2 == 0:
                widths.append(gap)
            else:
                spacings.append(gap)

        pitches = [w + s for w, s in itertools.zip_longest(widths, spacings, fillvalue=0)] if widths and spacings else []

        return {
            'pitches': pitches,
            'widths': widths,
            'spacings': spacings,
            'edge_points': merged,
        }

    def _find_edges_along_line(
        self,
        gray: np.ndarray,
        start: Tuple[int, int],
        dx: float, dy: float,
    ) -> List[Tuple[int, int]]:
        """沿搜索线找所有边缘点。"""
        h, w = gray.shape
        x, y = start

        # 采样边缘强度
        profile = []
        for i in range(max(w, h)):
            px = int(x + dx * i)
            py = int(y + dy * i)
            if 0 <= px < w and 0 <= py < h:
                profile.append((gray[py, px], px, py))

        if len(profile) < 2:
            return []

        # 计算梯度
        edges = []
        for i in range(len(profile) - 1):
            diff = profile[i+1][0] - profile[i][0]

            if self.polarity == 'black_to_white' and diff > self.edge_intensity:
                edges.append((profile[i][1], profile[i][2]))
            elif self.polarity == 'white_to_black' and diff < -self.edge_intensity:
                edges.append((profile[i][1], profile[i][2]))
            elif self.polarity == 'all' and abs(diff) > self.edge_intensity:
                edges.append((profile[i][1], profile[i][2]))

        return edges

    def _merge_close_edges(self, edges: List[Tuple]) -> List[Tuple]:
        """合并相邻过近的边缘点。"""
        if not edges:
            return []

        merged = [edges[0]]
        threshold = self.edge_spacing

        for edge in edges[1:]:
            last = merged[-1]
            dist = abs(edge[0] - last[0]) + abs(edge[1] - last[1])
            if dist > threshold:
                merged.append(edge)
            else:
                # 取平均位置
                merged[-1] = ((last[0] + edge[0]) / 2, (last[1] + edge[1]) / 2)

        return merged[:self.max_edge_count]


def measure_gap(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    **kwargs,
) -> Dict:
    """便捷函数：间隙测量。"""
    gap = GapMeasurement(**kwargs)
    return gap.measure(image, roi)


# ---------------------------------------------------------------------------
# 3. 几何关系 (Geometric Relations)
# ---------------------------------------------------------------------------

class GeometricRelations:
    """
    几何关系计算器。

    支持计算：
      - 点与点：距离
      - 点与线：距离、垂点
      - 线与线：距离、夹角、平行度、垂直度
      - 圆与圆：距离、夹角、相切
      - 点与圆：距离、垂点
    """

    @staticmethod
    def point_to_point(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算两点之间的距离。"""
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def point_to_line(
        point: Tuple[float, float],
        line_point: Tuple[float, float],
        line_angle: float,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        计算点到直线的距离和垂点。

        Returns:
            (distance, foot_point)
        """
        angle_rad = math.radians(line_angle)
        dx, dy = math.cos(angle_rad), math.sin(angle_rad)

        # 向量从线上点到目标点
        vx, vy = point[0] - line_point[0], point[1] - line_point[1]

        # 点乘得到投影长度
        t = vx * dx + vy * dy

        # 垂点
        foot = (line_point[0] + t * dx, line_point[1] + t * dy)

        # 距离
        dist = math.hypot(point[0] - foot[0], point[1] - foot[1])

        return dist, foot

    @staticmethod
    def line_to_line(
        line1_point: Tuple[float, float],
        line1_angle: float,
        line2_point: Tuple[float, float],
        line2_angle: float,
    ) -> Dict[str, float]:
        """
        计算两直线的关系。

        Returns:
            dict with 'angle', 'distance', 'is_parallel', 'is_perpendicular'
        """
        angle1_rad = math.radians(line1_angle)
        angle2_rad = math.radians(line2_angle)

        # 夹角差
        angle_diff = abs(line1_angle - line2_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        # 距离（点到线的距离）
        distance, _ = GeometricRelations.point_to_line(
            line2_point, line1_point, line1_angle
        )

        return {
            'angle_deg': angle_diff,
            'distance': distance,
            'is_parallel': angle_diff < 5.0,
            'is_perpendicular': abs(angle_diff - 90) < 5.0,
        }

    @staticmethod
    def circle_to_circle(
        center1: Tuple[float, float], radius1: float,
        center2: Tuple[float, float], radius2: float,
    ) -> Dict[str, float]:
        """
        计算两圆的关系。

        Returns:
            dict with 'center_distance', 'radius_diff', 'is_tangent', 'is_intersecting'
        """
        center_dist = GeometricRelations.point_to_point(center1, center2)
        radius_diff = abs(radius1 - radius2)

        # 判断相切和相交
        sum_radius = radius1 + radius2
        is_tangent = abs(center_dist - sum_radius) < 5.0 or \
                     abs(center_dist - radius_diff) < 5.0
        is_intersecting = radius_diff < center_dist < sum_radius

        return {
            'center_distance': center_dist,
            'radius_diff': radius_diff,
            'is_tangent': is_tangent,
            'is_intersecting': is_intersecting,
        }

    @staticmethod
    def fit_line(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], float]:
        """
        拟合直线。

        Returns:
            (center_point, angle_deg)
        """
        if len(points) < 2:
            return (0, 0), 0.0

        pts = np.array(points, dtype=np.float64)
        centroid = pts.mean(axis=0)
        centered = pts - centroid

        _, _, Vt = np.linalg.svd(centered)
        direction = Vt[0]
        angle = math.degrees(math.atan2(direction[1], direction[0]))

        return tuple(centroid), angle

    @staticmethod
    def fit_circle(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], float]:
        """
        拟合圆。

        Returns:
            (center, radius)
        """
        if len(points) < 3:
            # 估算
            pts = np.array(points)
            center = pts.mean(axis=0)
            radius = np.mean(np.linalg.norm(pts - center, axis=1))
            return tuple(center), float(radius)

        # 最小二乘拟合
        pts = np.array(points)
        x = pts[:, 0]
        y = pts[:, 1]

        # 构建方程 Ax + By + C = x^2 + y^2
        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        b = x**2 + y**2

        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = result[0], result[1]
            radius = np.sqrt(cx**2 + cy**2 + result[2])
            return (float(cx), float(cy)), float(radius)
        except:
            center = pts.mean(axis=0)
            radius = np.mean(np.linalg.norm(pts - center, axis=1))
            return tuple(center), float(radius)


def compute_geometric_relations(
    element1_type: str,
    element1_params: tuple,
    element2_type: str,
    element2_params: tuple,
) -> Dict[str, float]:
    """
    便捷函数：计算几何关系。

    Args:
        element1_type: 'point', 'line', 'circle'
        element1_params: 对应参数
        element2_type: 同上
        element2_params: 同上
    """
    geo = GeometricRelations()

    if element1_type == 'point' and element2_type == 'point':
        dist = geo.point_to_point(element1_params, element2_params)
        return {'distance': dist}

    elif element1_type == 'point' and element2_type == 'line':
        dist, foot = geo.point_to_line(element1_params, element2_params[0], element2_params[1])
        return {'distance': dist, 'foot_point': foot}

    elif element1_type == 'line' and element2_type == 'line':
        return geo.line_to_line(element1_params[0], element1_params[1],
                               element2_params[0], element2_params[1])

    elif element1_type == 'circle' and element2_type == 'circle':
        return geo.circle_to_circle(element1_params[0], element1_params[1],
                                   element2_params[0], element2_params[1])

    return {}


# ---------------------------------------------------------------------------
# 4. 轮廓操作 (Contour Operations)
# ---------------------------------------------------------------------------

class ContourOperations:
    """
    轮廓操作工具集。

    支持：
      - 筛选轮廓（按面积、周长、圆形度等）
      - 分割轮廓（分割为线段或圆弧）
      - 连接共线/共圆轮廓
      - 平滑轮廓
      - 轮廓内缩/外扩
    """

    @staticmethod
    def filter_contours(
        contours: List[np.ndarray],
        min_area: float = 0,
        max_area: float = float('inf'),
        min_perimeter: float = 0,
        max_perimeter: float = float('inf'),
        min_circularity: float = 0,
        max_circularity: float = 1.0,
        min_aspect_ratio: float = 0,
        max_aspect_ratio: float = float('inf'),
    ) -> List[np.ndarray]:
        """
        根据属性筛选轮廓。

        Args:
            contours: 轮廓列表
            min_area, max_area: 面积范围
            min_perimeter, max_perimeter: 周长范围
            min_circularity, max_circularity: 圆形度范围
            min_aspect_ratio, max_aspect_ratio: 长宽比范围

        Returns:
            筛选后的轮廓列表
        """
        filtered = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < min_perimeter or perimeter > max_perimeter:
                continue

            circularity = 4 * math.pi * area / (perimeter ** 2 + 1e-8)
            if circularity < min_circularity or circularity > max_circularity:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / (h + 1e-8)
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue

            filtered.append(cnt)

        return filtered

    @staticmethod
    def split_contour(
        contour: np.ndarray,
        epsilon_factor: float = 0.01,
    ) -> List[np.ndarray]:
        """
        将轮廓分割为线段。

        Args:
            contour: 输入轮廓
            epsilon_factor: 近似因子

        Returns:
            分割后的轮廓列表
        """
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter

        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 转换为轮廓列表
        segments = []
        pts = approx.reshape(-1, 2)
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            segment = np.array([[p1], [p2]], dtype=np.int32)
            segments.append(segment)

        return segments

    @staticmethod
    def connect_collinear_contours(
        contours: List[np.ndarray],
        min_distance: float = 5.0,
        max_distance: float = 50.0,
        max_angle: float = 15.0,
    ) -> List[np.ndarray]:
        """
        连接共线轮廓。

        Args:
            contours: 输入轮廓列表
            min_distance: 两轮廓端点之间的最小距离
            max_distance: 最大连接距离
            max_angle: 最大夹角（度）

        Returns:
            连接后的轮廓列表
        """
        if len(contours) < 2:
            return contours

        # 提取每个轮廓的端点和方向
        contour_info = []
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            if len(pts) < 2:
                continue

            # 找到主方向
            x, y, w, h = cv2.boundingRect(cnt)
            if w > h:
                direction = 0  # 水平
            else:
                direction = 90  # 垂直

            contour_info.append({
                'contour': cnt,
                'start': tuple(pts[0]),
                'end': tuple(pts[-1]),
                'direction': direction,
                'points': pts,
            })

        # 连接满足条件的轮廓
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(contour_info):
                j = i + 1
                while j < len(contour_info):
                    info1 = contour_info[i]
                    info2 = contour_info[j]

                    # 检查方向相似度
                    angle_diff = abs(info1['direction'] - info2['direction'])
                    if angle_diff > 90:
                        angle_diff = 180 - angle_diff

                    if angle_diff > max_angle:
                        j += 1
                        continue

                    # 检查端点距离
                    for p1 in [info1['start'], info1['end']]:
                        for p2 in [info2['start'], info2['end']]:
                            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                            if min_distance <= dist <= max_distance:
                                # 合并
                                all_pts = np.vstack([info1['points'], info2['points']])
                                new_cnt = all_pts.reshape(-1, 1, 2)
                                contour_info[i] = {
                                    'contour': new_cnt,
                                    'start': tuple(all_pts[0]),
                                    'end': tuple(all_pts[-1]),
                                    'direction': info1['direction'],
                                    'points': all_pts,
                                }
                                contour_info.pop(j)
                                merged = True
                                break
                        if merged:
                            break
                i += 1

        return [info['contour'] for info in contour_info]

    @staticmethod
    def smooth_contour(
        contour: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """
        平滑轮廓。

        Args:
            contour: 输入轮廓
            window_size: 平滑窗口大小（奇数）

        Returns:
            平滑后的轮廓
        """
        pts = contour.reshape(-1, 2).astype(np.float64)

        # 移动平均平滑
        smoothed = pts.copy()
        half_win = window_size // 2

        for i in range(len(pts)):
            start = max(0, i - half_win)
            end = min(len(pts), i + half_win + 1)
            smoothed[i] = pts[start:end].mean(axis=0)

        return smoothed.reshape(-1, 1, 2).astype(np.int32)

    @staticmethod
    def offset_contour(
        contour: np.ndarray,
        offset: float,
    ) -> np.ndarray:
        """
        轮廓内缩/外扩（基于多边形近似）。

        Args:
            contour: 输入轮廓 (N, 1, 2)
            offset: 偏移量（正=外扩，负=内缩），单位为像素

        Returns:
            偏移后的轮廓
        """
        if len(contour) < 3:
            return contour.copy()

        # 简化轮廓
        perimeter = cv2.arcLength(contour, True)
        epsilon = abs(offset) * 0.1 / max(perimeter, 1e-8)
        epsilon = max(0.001, min(epsilon, 0.1))  # 防止过小或过大
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)

        if len(approx) < 3:
            approx = contour

        # 找到轮廓中心
        M = cv2.moments(approx)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = 0, 0

        # 围绕中心缩放
        pts = approx.reshape(-1, 2).astype(np.float32)
        centered = pts - np.array([cx, cy], dtype=np.float32)
        scale = (perimeter + offset) / (perimeter + 1e-8)
        scale = max(0.1, min(scale, 10.0))  # 防止极端缩放
        scaled = centered * scale
        result = (scaled + np.array([cx, cy], dtype=np.float32)).reshape(-1, 1, 2)

        return result.astype(np.int32)


def filter_contours(contours: List, **kwargs) -> List:
    """便捷函数：筛选轮廓。"""
    return ContourOperations.filter_contours(contours, **kwargs)


def split_contour(contour: np.ndarray, **kwargs) -> List:
    """便捷函数：分割轮廓。"""
    return ContourOperations.split_contour(contour, **kwargs)


def connect_collinear_contours(contours: List, **kwargs) -> List:
    """便捷函数：连接共线轮廓。"""
    return ContourOperations.connect_collinear_contours(contours, **kwargs)


def smooth_contour(contour: np.ndarray, **kwargs) -> np.ndarray:
    """便捷函数：平滑轮廓。"""
    return ContourOperations.smooth_contour(contour, **kwargs)


# ---------------------------------------------------------------------------
# 5. 图像锐度 (Image Sharpness)
# ---------------------------------------------------------------------------

class ImageSharpness:
    """
    图像锐度/清晰度评估。

    用于相机对焦评分，帮助找到最佳对焦位置。

    评估方法：
      - Laplacian方差：边缘越清晰，方差越大
      - Brenner梯度：计算像素与相隔两个像素的梯度平方和
      - FFT高频能量比：高频分量越多越清晰
    """

    @staticmethod
    def compute_sharpness(
        image: np.ndarray,
        method: str = 'laplacian',
    ) -> Tuple[float, float]:
        """
        计算图像锐度。

        Args:
            image: BGR uint8 图像
            method: 'laplacian', 'brenner', 'fft'

        Returns:
            (score, focus_grade)
            score: 锐度分数（越高越清晰）
            focus_grade: 'excellent', 'good', 'fair', 'poor'
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        if method == 'laplacian':
            # Laplacian方差法
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            score = laplacian.var()

        elif method == 'brenner':
            # Brenner梯度法
            shifted = gray[:, 2:]
            original = gray[:, :-2]
            diff = (shifted.astype(np.float64) - original.astype(np.float64)) ** 2
            score = np.sum(diff) / (gray.shape[0] * (gray.shape[1] - 2))

        elif method == 'fft':
            # FFT高频能量比
            f = np.fft.fft2(gray.astype(np.float64))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)

            h, w = gray.shape
            cy, cx = h // 2, w // 2

            # 低频区域半径
            low_r = min(h, w) // 8
            y, x = np.ogrid[:h, :w]
            low_mask = (x - cx)**2 + (y - cy)**2 <= low_r**2

            total_energy = np.sum(magnitude**2)
            low_energy = np.sum(magnitude**2 * low_mask)
            high_energy = total_energy - low_energy

            score = high_energy / (total_energy + 1e-8)

        else:
            score = 0.0

        # 评级
        if score > 1000:
            grade = 'excellent'
        elif score > 500:
            grade = 'good'
        elif score > 200:
            grade = 'fair'
        else:
            grade = 'poor'

        return float(score), grade

    @staticmethod
    def find_best_focus(
        images: List[np.ndarray],
        method: str = 'laplacian',
    ) -> Tuple[int, float]:
        """
        在多张不同焦距的图像中找到最佳对焦位置。

        Args:
            images: 不同对焦位置的图像列表
            method: 锐度计算方法

        Returns:
            (best_index, best_score)
        """
        scores = []
        for img in images:
            score, _ = ImageSharpness.compute_sharpness(img, method)
            scores.append(score)

        best_idx = np.argmax(scores)
        return best_idx, float(scores[best_idx])


def compute_sharpness(image: np.ndarray, method: str = 'laplacian') -> Tuple[float, str]:
    """便捷函数：计算图像锐度。"""
    return ImageSharpness.compute_sharpness(image, method)


# ---------------------------------------------------------------------------
# 6. Blob分析 (Enhanced)
# ---------------------------------------------------------------------------

class BlobAnalyzer:
    """
    Blob分析器 - 连通域分析。

    基于Smart3第7.1章Blob分析，支持：
      - 多种二值化方法
      - 形态学预处理
      - 多种筛选条件
      - Hu矩、圆形度、矩形度等特征计算
    """

    def __init__(
        self,
        threshold_method: str = 'otsu',
        threshold_low: int = 127,
        threshold_high: int = 255,
        morphology_method: str = 'none',
        morphology_kernel: int = 5,
        morphology_iterations: int = 1,
    ):
        """
        Args:
            threshold_method: 'manual', 'otsu', 'adaptive'
            threshold_low, threshold_high: 手动阈值范围
            morphology_method: 'none', 'erode', 'dilate', 'open', 'close'
            morphology_kernel: 形态学核大小
            morphology_iterations: 迭代次数
        """
        self.threshold_method = threshold_method
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.morphology_method = morphology_method
        self.morphology_kernel = morphology_kernel
        self.morphology_iterations = morphology_iterations

    def analyze(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_area: int = 0,
        max_area: int = 1000000,
    ) -> List[Dict]:
        """
        分析图像中的Blob。

        Args:
            image: BGR uint8 图像
            roi: (x, y, w, h) 感兴趣区域
            min_area, max_area: 面积筛选范围

        Returns:
            Blob列表，每个Blob包含：
              - centroid: (cx, cy) 质心
              - area: 面积
              - perimeter: 周长
              - bbox: (x, y, w, h) 外接矩形
              - circularity: 圆形度
              - aspect_ratio: 长宽比
              - hu_moments: 7个Hu矩
              - contour: 轮廓点
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        # ROI处理
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        # 二值化
        if self.threshold_method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        else:  # manual
            _, binary = cv2.threshold(gray, self.threshold_low, self.threshold_high,
                                      cv2.THRESH_BINARY)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morphology_kernel, self.morphology_kernel))
        if self.morphology_method == 'erode':
            binary = cv2.erode(binary, kernel, iterations=self.morphology_iterations)
        elif self.morphology_method == 'dilate':
            binary = cv2.dilate(binary, kernel, iterations=self.morphology_iterations)
        elif self.morphology_method == 'open':
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=self.morphology_iterations)
        elif self.morphology_method == 'close':
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=self.morphology_iterations)

        # 轮廓提取
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)

            # 质心
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
            else:
                cx, cy = 0, 0

            # 外接矩形
            x, y, w, h = cv2.boundingRect(cnt)

            # 圆形度
            circularity = 4 * math.pi * area / (perimeter ** 2 + 1e-8)

            # 长宽比
            aspect_ratio = float(w) / (h + 1e-8)

            # Hu矩
            moments = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(moments).flatten()

            # 对数变换使数值更稳定
            hu_log = []
            for m in hu_moments:
                if m != 0:
                    hu_log.append(-math.log10(abs(m) + 1e-10))
                else:
                    hu_log.append(0)

            results.append({
                'centroid': (float(cx), float(cy)),
                'area': float(area),
                'perimeter': float(perimeter),
                'bbox': (int(x), int(y), int(w), int(h)),
                'circularity': float(circularity),
                'aspect_ratio': float(aspect_ratio),
                'hu_moments': hu_log,
                'contour': cnt,
            })

        return results


def analyze_blobs(image: np.ndarray, **kwargs) -> List[Dict]:
    """便捷函数：Blob分析。"""
    analyzer = BlobAnalyzer(**kwargs)
    return analyzer.analyze(image)


# ---------------------------------------------------------------------------
# 7. 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测量与匹配算法模块")
    parser.add_argument("--mode", type=str, default="sharpness",
                        choices=["sharpness", "caliper", "blob", "contour"])
    parser.add_argument("--output_dir", type=str, default="./measurement_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "sharpness":
        print("=== 图像锐度测试 ===")
        # 创建测试图像（清晰 vs 模糊）
        sharp_img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.circle(sharp_img, (128, 128), 50, (255, 255, 255), -1)

        blur_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)

        score_sharp, grade_sharp = ImageSharpness.compute_sharpness(sharp_img)
        score_blur, grade_blur = ImageSharpness.compute_sharpness(blur_img)

        print(f"清晰图像: score={score_sharp:.1f}, grade={grade_sharp}")
        print(f"模糊图像: score={score_blur:.1f}, grade={grade_blur}")

    elif args.mode == "blob":
        print("=== Blob分析测试 ===")
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(test_img, (100, 100), 30, (255, 255, 255), -1)
        cv2.circle(test_img, (250, 200), 50, (255, 255, 255), -1)
        cv2.rectangle(test_img, (300, 50), (380, 150), (255, 255, 255), -1)

        blobs = analyze_blobs(test_img, threshold_method='otsu', min_area=100)
        print(f"检测到 {len(blobs)} 个Blob")
        for i, blob in enumerate(blobs):
            print(f"  Blob[{i}]: area={blob['area']:.1f}, circularity={blob['circularity']:.3f}")

    elif args.mode == "contour":
        print("=== 轮廓操作测试 ===")
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.drawContours(test_img, [np.array([[50, 50], [100, 50], [100, 100]])], -1, (255, 255, 255), 2)
        cv2.drawContours(test_img, [np.array([[100, 55], [150, 55], [150, 105]])], -1, (255, 255, 255), 2)

        contours, _ = cv2.findContours(test_img[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(f"找到 {len(contours)} 个轮廓")

        filtered = filter_contours(contours, min_area=100)
        print(f"筛选后剩余 {len(filtered)} 个轮廓")

    print(f"\n结果已保存到: {args.output_dir}")
    print("完成。")
