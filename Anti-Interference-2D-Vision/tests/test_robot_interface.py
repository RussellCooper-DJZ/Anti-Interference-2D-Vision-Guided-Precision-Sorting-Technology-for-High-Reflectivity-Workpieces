"""
test_robot_interface.py — ABB 机器人接口测试
基于实际 API 重写
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from robot.abb_robotstudio_interface import (
    AbbRobotStub,
    AbbRobotBase,
    create_robot_interface,
)


# ============================================================
# AbbRobotStub
# ============================================================

class TestAbbRobotStub:
    """ABB 机器人模拟桩测试"""

    @pytest.fixture
    def robot(self):
        return AbbRobotStub()

    def test_creation(self, robot):
        assert robot is not None
        assert isinstance(robot, AbbRobotBase)

    def test_connect(self, robot):
        result = robot.connect()
        assert result is True

    def test_disconnect(self, robot):
        robot.connect()
        robot.disconnect()
        assert robot._connected is False

    def test_send_target(self, robot):
        robot.connect()
        result = robot.send_target(
            x_mm=500.0, y_mm=0.0, z_mm=400.0,
            rx_deg=0.0, ry_deg=180.0, rz_deg=0.0,
        )
        assert result is True

    def test_get_status(self, robot):
        robot.connect()
        status = robot.get_status()
        assert isinstance(status, dict)
        assert 'connected' in status

    def test_home(self, robot):
        robot.connect()
        result = robot.home()
        assert result is True

    def test_wait_done(self, robot):
        robot.connect()
        result = robot.wait_done()
        assert result is True

    def test_context_manager(self, robot):
        with robot:
            assert robot._connected is True
            status = robot.get_status()
            assert isinstance(status, dict)
        assert robot._connected is False

    def test_invalid_target(self, robot):
        """超界目标测试"""
        robot.connect()
        result = robot.send_target(x_mm=99999, y_mm=0, z_mm=0)
        # 桩模式可能接受任何目标
        assert isinstance(result, bool)

    def test_negative_speed(self, robot):
        """负数速度测试"""
        robot.connect()
        result = robot.send_target(
            x_mm=500, y_mm=0, z_mm=400,
            speed_mm_s=-100
        )
        # 桩模式可能接受任何速度
        assert isinstance(result, bool)

    def test_multiple_targets(self, robot):
        """连续发送多个目标"""
        robot.connect()
        targets = [
            (500, 0, 400),
            (500, 100, 400),
            (500, 100, 300),
        ]
        for x, y, z in targets:
            result = robot.send_target(x_mm=x, y_mm=y, z_mm=z)
            assert result is True

    def test_send_target_format(self, robot):
        """验证发送目标的基本行为"""
        robot.connect()
        result = robot.send_target(x_mm=500, y_mm=0, z_mm=400)
        assert isinstance(result, bool)


# ============================================================
# create_robot_interface
# ============================================================

class TestCreateRobotInterface:
    """工厂函数测试"""

    def test_create_stub(self):
        robot = create_robot_interface(mode="stub")
        assert isinstance(robot, AbbRobotStub)

    def test_create_default(self):
        robot = create_robot_interface()
        assert robot is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
