"""
gripper_simulation.py — 机械抓取边缘位置模拟模块
:Author: RussellCooper

功能：
  1. GripperEdgePlanner — 从检测到的工件轮廓计算最优抓取点
  2. 抓取质量评估
  3. 接近角度计算

算法：
  - 基于轮廓几何计算 antipodal points（对向点）
  - 平行爪抓取配置计算
  - 抓取稳定性评分

用法::

    planner = GripperEdgePlanner(gripper_width_px=40)
    grasp = planner.plan_grasp(contour, approach_angle_deg=0)
    print(grasp['left_contact_px'], grasp['right_contact_px'])
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


__all__ = ["GripperEdgePlanner", "GraspQuality"]


# ============================================================
# 抓取质量评分
# ============================================================

class GraspQuality:
    """抓取质量评估指标"""

    @staticmethod
    def compute_score(
        gripper_width_px: float,
        object_width_px: float,
        object_area_px: float,
        contour_circularity: float,
        target_width_px: float = 40.0,
    ) -> float:
        """
        计算抓取质量评分 [0, 1]。

        Args:
            gripper_width_px:    抓取宽度（像素）
            object_width_px:     物体宽度（像素）
            object_area_px:      物体面积（像素）
            contour_circularity: 轮廓圆形度 [0,1]
            target_width_px:     目标抓取宽度（像素）

        Returns:
            抓取质量分数，越接近 1 越好
        """
        # 1. 宽度适配度：抓取宽度是否在物体尺寸的 0.3-0.8 倍之间
        if object_width_px <= 0:
            width_score = 0.0
        else:
            ratio = gripper_width_px / object_width_px
            if 0.3 <= ratio <= 0.8:
                width_score = 1.0 - abs(ratio - 0.55) / 0.55
            elif ratio < 0.3:
                width_score = ratio / 0.3 * 0.5  # 太窄，部分抓取
            else:
                width_score = max(0.0, 1.0 - (ratio - 0.8) / 0.4)

        # 2. 面积适配度：物体面积与抓取框面积比
        expected_area = gripper_width_px * object_width_px * 0.5
        if expected_area > 0:
            area_ratio = min(object_area_px / expected_area, 1.0)
        else:
            area_ratio = 0.0

        # 3. 形状适配度：非圆形物体更容易抓取
        shape_score = 1.0 - contour_circularity * 0.5

        # 综合评分
        score = width_score * 0.5 + area_ratio * 0.3 + shape_score * 0.2
        return max(0.0, min(1.0, score))


# ============================================================
# 机械抓取边缘规划器
# ============================================================

class GripperEdgePlanner:
    """
    从工件轮廓计算机械抓取边缘位置。

    支持：
      - 平行双爪抓取
      - 多角度评估最优接近角
      - 抓取质量评分

    适用于高反光钢板的工业分拣场景。
    """

    def __init__(
        self,
        gripper_width_px: int = 40,
        num_approach_angles: int = 8,
        num_samples: int = 180,
    ):
        """
        Args:
            gripper_width_px:     机械爪宽度（像素），用于抓取计算
            num_approach_angles:  评估的接近角数量
            num_samples:          轮廓采样点数（用于计算）
        """
        self.gripper_width_px = gripper_width_px
        self.num_approach_angles = num_approach_angles
        self.num_samples = num_samples

    def plan_grasp(
        self,
        contour: np.ndarray,
        approach_angle_deg: float = 0.0,
        gripper_width_px: Optional[int] = None,
    ) -> Dict:
        """
        计算单个工件的最优抓取配置。

        Args:
            contour:             工件轮廓点 (N, 1, 2)
            approach_angle_deg:  预设接近角（度），-90~90
            gripper_width_px:    抓取宽度（像素），None 则使用默认值

        Returns:
            grasp_config = {
                'left_contact_px':  (x, y) 左爪接触点
                'right_contact_px': (x, y) 右爪接触点
                'center_px':         (x, y) 抓取中心点
                'approach_angle_deg': float 接近角度（度）
                'gripper_width_px': float 实际抓取宽度
                'grip_quality':      float 抓取质量 [0,1]
                'jaw_trajectory':    list 爪尖轨迹（用于动画）
            }
        """
        width = gripper_width_px or self.gripper_width_px

        # 简化轮廓为边界框
        x, y, w, h = cv2.boundingRect(contour)
        object_width_px = max(w, h)
        object_area_px = cv2.contourArea(contour)

        # 计算轮廓圆形度
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * math.pi * object_area_px /
                      (perimeter ** 2 + 1e-8)) if perimeter > 0 else 0

        # 获取轮廓上的点
        pts = contour.reshape(-1, 2).astype(np.float32)

        # 采样点
        if len(pts) > self.num_samples:
            indices = np.linspace(0, len(pts) - 1, self.num_samples, dtype=int)
            sampled_pts = pts[indices]
        else:
            sampled_pts = pts

        # 计算最优抓取点
        left_pt, right_pt, best_angle = self._find_antipodal_points(
            sampled_pts, approach_angle_deg, width
        )

        # 抓取中心
        center_x = (left_pt[0] + right_pt[0]) / 2
        center_y = (left_pt[1] + right_pt[1]) / 2
        center = (float(center_x), float(center_y))

        # 实际抓取宽度
        actual_width = math.hypot(
            right_pt[0] - left_pt[0], right_pt[1] - left_pt[1]
        )

        # 抓取质量
        quality = GraspQuality.compute_score(
            gripper_width_px=actual_width,
            object_width_px=object_width_px,
            object_area_px=object_area_px,
            contour_circularity=circularity,
            target_width_px=width,
        )

        # 生成爪尖轨迹（用于动画）
        trajectory = self._generate_jaw_trajectory(
            center, left_pt, right_pt, best_angle, n_steps=5
        )

        return {
            'left_contact_px': (float(left_pt[0]), float(left_pt[1])),
            'right_contact_px': (float(right_pt[0]), float(right_pt[1])),
            'center_px': center,
            'approach_angle_deg': float(best_angle),
            'gripper_width_px': float(actual_width),
            'grip_quality': float(quality),
            'jaw_trajectory': trajectory,
        }

    def plan_multi_angle_grasp(
        self,
        contour: np.ndarray,
        gripper_width_px: Optional[int] = None,
    ) -> List[Dict]:
        """
        评估多个接近角，返回最优抓取配置。

        Args:
            contour:  工件轮廓
            gripper_width_px: 抓取宽度

        Returns:
            List[grasp_config]，按质量排序
        """
        width = gripper_width_px or self.gripper_width_px

        results = []
        for angle in range(-90, 90, 180 // self.num_approach_angles):
            grasp = self.plan_grasp(
                contour, approach_angle_deg=float(angle), gripper_width_px=width
            )
            results.append(grasp)

        # 按抓取质量排序
        results.sort(key=lambda g: g['grip_quality'], reverse=True)
        return results

    def _find_antipodal_points(
        self,
        pts: np.ndarray,
        approach_angle_deg: float,
        gripper_width: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        找到一对抓取接触点（antipodal points）。

        对于给定接近角，计算与轮廓相交的一对平行线的交点。
        """
        angle_rad = math.radians(approach_angle_deg)
        direction = np.array([math.cos(angle_rad), math.sin(angle_rad)])
        perp_direction = np.array([-direction[1], direction[0]])

        best_left = None
        best_right = None
        best_score = -1.0

        # 计算轮廓边界
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

        # 沿接近方向扫描（投影到垂直方向上确定扫描范围）
        scan_min = pts @ perp_direction
        scan_max = scan_min.max()

        # 沿垂直于接近方向扫描
        for perp_offset in np.linspace(
            scan_min.min() + gripper_width / 2,
            scan_max.max() - gripper_width / 2,
            30
        ):
            # 平行线上的两个远处点
            center = perp_offset * perp_direction
            p1 = center + direction * 1000
            p2 = center - direction * 1000

            # 找到与轮廓的交点
            intersections = self._line_contour_intersections(
                pts, (p1[0], p1[1]), (p2[0], p2[1])
            )

            if len(intersections) >= 2:
                # 选择最远的两个交点
                for i in range(len(intersections)):
                    for j in range(i + 1, len(intersections)):
                        pt1 = intersections[i]
                        pt2 = intersections[j]
                        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

                        # 宽度适配度：越接近目标宽度越好
                        width_diff = abs(dist - gripper_width)
                        score = 1.0 / (1.0 + width_diff)

                        if score > best_score:
                            best_score = width_diff
                            best_left = pt1
                            best_right = pt2

        # 如果没找到合适的，使用默认方法：取 bounding box 的边缘中点
        if best_left is None:
            x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
            center_x, center_y = x + w / 2, y + h / 2

            # 根据接近角计算边缘点
            angle_rad = math.radians(approach_angle_deg)
            half_w = w / 2
            half_h = h / 2

            # 沿接近角方向的偏移量
            dx = math.cos(angle_rad) * half_w
            dy = math.sin(angle_rad) * half_h

            # 抓取方向的两侧中点
            if w >= h:
                best_left = (float(center_x - dx), float(center_y - dy))
                best_right = (float(center_x + dx), float(center_y + dy))
            else:
                best_left = (float(center_x - dx), float(center_y - dy))
                best_right = (float(center_x + dx), float(center_y + dy))
            best_score = gripper_width

        return best_left, best_right, approach_angle_deg

    def _line_contour_intersections(
        self,
        pts: np.ndarray,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> List[np.ndarray]:
        """计算线段与点集的最短连接点"""
        intersections = []

        # 计算每个点到线段的最近点
        for pt in pts:
            closest = self._closest_point_on_segment(
                pt, np.array(p1), np.array(p2)
            )
            dist = math.hypot(pt[0] - closest[0], pt[1] - closest[1])
            if dist < 5.0:  # 阈值内的点认为是交点
                intersections.append(closest)

        return intersections

    @staticmethod
    def _closest_point_on_segment(
        pt: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
    ) -> np.ndarray:
        """计算点到线段的最近点"""
        diff = seg_end - seg_start
        length_sq = np.dot(diff, diff)

        if length_sq < 1e-8:
            return seg_start

        t = max(0.0, min(1.0, np.dot(pt - seg_start, diff) / length_sq))
        return seg_start + t * diff

    def _generate_jaw_trajectory(
        self,
        center: Tuple[float, float],
        left_contact: np.ndarray,
        right_contact: np.ndarray,
        approach_angle_deg: float,
        n_steps: int = 5,
    ) -> List[Dict]:
        """生成爪尖运动轨迹（用于动画）"""
        trajectory = []
        angle_rad = math.radians(approach_angle_deg)

        for i in range(n_steps):
            # 从远处接近到接触
            t = i / (n_steps - 1) if n_steps > 1 else 1.0
            approach_offset = 50 * (1 - t)  # 从50像素外接近

            # 左爪
            left_offset = np.array([
                -math.cos(angle_rad) * approach_offset,
                -math.sin(angle_rad) * approach_offset,
            ])
            left_pos = left_contact + left_offset * (1 - t)

            # 右爪
            right_offset = np.array([
                -math.cos(angle_rad) * approach_offset,
                -math.sin(angle_rad) * approach_offset,
            ])
            right_pos = right_contact + right_offset * (1 - t)

            trajectory.append({
                'step': i,
                'left_px': (float(left_pos[0]), float(left_pos[1])),
                'right_px': (float(right_pos[0]), float(right_pos[1])),
                'jaw_open_mm': 30 * (1 - t),  # 逐渐闭合
            })

        return trajectory


# ============================================================
# 可视化工具
# ============================================================

def draw_grasp_on_image(
    image: np.ndarray,
    grasp_config: Dict,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制抓取配置。

    Args:
        image:         BGR 图像
        grasp_config: GripperEdgePlanner.plan_grasp() 返回的配置
        color:         绘制颜色 (B, G, R)
        thickness:     线条粗细

    Returns:
        绘制后的图像副本
    """
    result = image.copy()

    left = grasp_config['left_contact_px']
    right = grasp_config['right_contact_px']
    center = grasp_config['center_px']
    angle_deg = grasp_config['approach_angle_deg']

    # 1. 接触点
    cv2.circle(result, (int(left[0]), int(left[1])), 6, (0, 0, 255), -1)   # 左爪 - 红色
    cv2.circle(result, (int(right[0]), int(right[1])), 6, (0, 255, 0), -1) # 右爪 - 绿色

    # 2. 抓取中心
    cv2.circle(result, (int(center[0]), int(center[1])), 4, (255, 255, 0), -1)  # 青色

    # 3. 抓取宽度线
    cv2.line(result,
             (int(left[0]), int(left[1])),
             (int(right[0]), int(right[1])),
             (255, 255, 0), thickness)

    # 4. 接近方向箭头
    angle_rad = math.radians(angle_deg)
    arrow_length = 40
    end_x = center[0] - arrow_length * math.cos(angle_rad)
    end_y = center[1] - arrow_length * math.sin(angle_rad)
    cv2.arrowedLine(result,
                    (int(center[0]), int(center[1])),
                    (int(end_x), int(end_y)),
                    (255, 0, 255), thickness, tipLength=0.3)

    # 5. 抓取框（虚线矩形包围）
    x_min = min(left[0], right[0]) - 20
    x_max = max(left[0], right[0]) + 20
    y_min = center[1] - 30
    y_max = center[1] + 30
    cv2.rectangle(result,
                  (int(x_min), int(y_min)),
                  (int(x_max), int(y_max)),
                  (0, 255, 255), 1, lineType=cv2.LINE_4)

    return result


def draw_all_grasps(
    image: np.ndarray,
    detections: List[Dict],
    show_quality: bool = True,
) -> np.ndarray:
    """
    在图像上绘制所有检测到的抓取配置。

    Args:
        image:         BGR 图像
        detections:    包含 gripper_config 的检测列表
        show_quality:  是否显示质量分数

    Returns:
        绘制后的图像
    """
    result = image.copy()

    colors = [
        (0, 255, 255),   # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 0),    # 绿色
        (255, 255, 0),  # 青色
    ]

    for i, det in enumerate(detections):
        gc = det.get('gripper_config')
        if gc is None:
            continue

        color = colors[i % len(colors)]
        result = draw_grasp_on_image(result, gc, color=color, thickness=2)

        if show_quality:
            cx, cy = gc['center_px']
            label = f"Q:{gc['grip_quality']:.2f}"
            cv2.putText(result, label,
                        (int(cx) + 15, int(cy) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return result


# ============================================================
# 命令行验证
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="机械抓取模拟验证")
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "quality"])
    args = parser.parse_args()

    if args.mode == "demo":
        print("=== 机械抓取模拟验证 ===")

        # 创建测试轮廓
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        rect_contour = np.array([[[50, 50]], [[250, 50]], [[250, 200]], [[50, 200]]], dtype=np.int32)
        cv2.drawContours(test_img, [rect_contour], -1, (200, 200, 200), -1)

        # 规划抓取
        planner = GripperEdgePlanner(gripper_width_px=50)
        grasp = planner.plan_grasp(rect_contour, approach_angle_deg=0)

        print(f"左爪接触点: {grasp['left_contact_px']}")
        print(f"右爪接触点: {grasp['right_contact_px']}")
        print(f"抓取中心: {grasp['center_px']}")
        print(f"接近角度: {grasp['approach_angle_deg']}°")
        print(f"抓取宽度: {grasp['gripper_width_px']:.1f} px")
        print(f"抓取质量: {grasp['grip_quality']:.3f}")

        # 可视化
        vis = draw_grasp_on_image(test_img, grasp)
        cv2.imwrite("/tmp/grasp_demo.png", vis)
        print("\n可视化已保存: /tmp/grasp_demo.png")

    elif args.mode == "quality":
        print("=== 抓取质量评估 ===")
        q = GraspQuality()

        # 测试不同场景
        test_cases = [
            (50, 100, 5000, 0.3, "宽物体"),
            (30, 100, 5000, 0.3, "窄物体"),
            (80, 100, 5000, 0.3, "宽抓取"),
            (50, 80, 3000, 0.8, "接近圆形"),
        ]

        for width, obj_w, area, circ, desc in test_cases:
            score = q.compute_score(width, obj_w, area, circ)
            print(f"{desc}: 质量={score:.3f} (w={width}, obj_w={obj_w})")
