"""
test_gripper_simulation.py — 机械抓取模拟测试
验证抓取点计算、可视化功能
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest

from vision.gripper_simulation import (
    GripperEdgePlanner,
    GraspQuality,
    draw_grasp_on_image,
    draw_all_grasps,
)


class TestGraspQuality:
    """抓取质量评估测试"""

    def test_perfect_grasp(self):
        """理想抓取场景"""
        score = GraspQuality.compute_score(
            gripper_width_px=40,
            object_width_px=80,
            object_area_px=4000,
            contour_circularity=0.3,
            target_width_px=40,
        )
        assert 0.7 <= score <= 1.0

    def test_too_narrow(self):
        """抓取宽度太窄"""
        score = GraspQuality.compute_score(
            gripper_width_px=10,
            object_width_px=100,
            object_area_px=5000,
            contour_circularity=0.3,
            target_width_px=40,
        )
        assert score < 0.7  # 窄抓取质量较低但仍有部分分数

    def test_too_wide(self):
        """抓取宽度太宽"""
        score = GraspQuality.compute_score(
            gripper_width_px=100,
            object_width_px=80,
            object_area_px=5000,
            contour_circularity=0.3,
            target_width_px=40,
        )
        assert score < 0.5


class TestGripperEdgePlanner:
    """抓取规划器测试"""

    @pytest.fixture
    def rect_contour(self):
        """矩形轮廓"""
        return np.array([[[50, 50]], [[250, 50]], [[250, 200]], [[50, 200]]], dtype=np.int32)

    @pytest.fixture
    def circle_contour(self):
        """圆形轮廓"""
        circle = np.zeros((1, 100, 2), dtype=np.int32)
        for i in range(100):
            angle = i * 2 * np.pi / 100
            circle[0, i, 0] = int(100 + 50 * np.cos(angle))
            circle[0, i, 1] = int(100 + 50 * np.sin(angle))
        return circle

    def test_plan_grasp_rect(self, rect_contour):
        """测试矩形工件抓取"""
        planner = GripperEdgePlanner(gripper_width_px=50)
        grasp = planner.plan_grasp(rect_contour, approach_angle_deg=0)

        assert 'left_contact_px' in grasp
        assert 'right_contact_px' in grasp
        assert 'center_px' in grasp
        assert 'approach_angle_deg' in grasp
        assert 'gripper_width_px' in grasp
        assert 'grip_quality' in grasp

        # 检查接触点有效
        assert grasp['gripper_width_px'] > 0
        assert 0 <= grasp['grip_quality'] <= 1

    def test_plan_grasp_circle(self, circle_contour):
        """测试圆形工件抓取"""
        planner = GripperEdgePlanner(gripper_width_px=40)
        grasp = planner.plan_grasp(circle_contour, approach_angle_deg=45)

        assert grasp['gripper_width_px'] > 0
        assert 0 <= grasp['grip_quality'] <= 1

    def test_multi_angle(self, rect_contour):
        """测试多角度抓取"""
        planner = GripperEdgePlanner(gripper_width_px=50, num_approach_angles=4)
        results = planner.plan_multi_angle_grasp(rect_contour)

        assert len(results) > 0
        # 结果应该按质量排序
        qualities = [r['grip_quality'] for r in results]
        assert qualities == sorted(qualities, reverse=True)

    def test_jaw_trajectory(self, rect_contour):
        """测试爪尖轨迹生成"""
        planner = GripperEdgePlanner(gripper_width_px=50)
        grasp = planner.plan_grasp(rect_contour)

        assert 'jaw_trajectory' in grasp
        assert len(grasp['jaw_trajectory']) > 0
        for step in grasp['jaw_trajectory']:
            assert 'step' in step
            assert 'left_px' in step
            assert 'right_px' in step
            assert 'jaw_open_mm' in step


class TestVisualization:
    """可视化测试"""

    @pytest.fixture
    def test_image(self):
        """测试图像"""
        return np.zeros((400, 400, 3), dtype=np.uint8)

    @pytest.fixture
    def grasp_config(self):
        """测试抓取配置"""
        return {
            'left_contact_px': (100.0, 200.0),
            'right_contact_px': (200.0, 200.0),
            'center_px': (150.0, 200.0),
            'approach_angle_deg': 0.0,
            'gripper_width_px': 100.0,
            'grip_quality': 0.85,
            'jaw_trajectory': [],
        }

    def test_draw_grasp(self, test_image, grasp_config):
        """测试单个抓取绘制"""
        result = draw_grasp_on_image(test_image, grasp_config)
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_draw_all_grasps(self, test_image, grasp_config):
        """测试多抓取绘制"""
        detections = [
            {'gripper_config': grasp_config},
            {'gripper_config': {**grasp_config,
                               'left_contact_px': (300.0, 100.0),
                               'right_contact_px': (350.0, 100.0),
                               'center_px': (325.0, 100.0)}},
        ]
        result = draw_all_grasps(test_image, detections, show_quality=True)
        assert result.shape == test_image.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
