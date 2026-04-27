"""
gantry_system.py — 衍架悬挂系统
:Author: RussellCooper

功能：
  - 衍架式悬挂机器人系统
  - 最大负载 50KG
  - 多种工具配置（吸盘、夹爪、磁力吸盘）
  - X/Y/Z 三轴运动控制

规格：
  - 最大负载: 50KG
  - X轴行程: 2000mm
  - Y轴行程: 1500mm
  - Z轴行程: 800mm
  - 重复精度: ±0.3mm
  - 最大速度: 1000mm/s

工具类型：
  - SUCTION: 真空吸盘（适用于平整表面）
  - GRIPPER: 夹爪（适用于不规则形状）
  - MAGNET: 磁力吸盘（适用于金属件）

用法::

    from robot.cells import GantrySystem

    gantry = GantrySystem(host="192.168.1.102", port=6512)
    gantry.connect()
    gantry.set_tool("SUCTION")
    gantry.move_to(x=500, y=300, z=100)
    gantry.pick()
    gantry.move_to(x=800, y=400, z=200)
    gantry.place()
    gantry.disconnect()
"""

import time
import math
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass


class ToolType(Enum):
    """工具类型"""
    SUCTION = "suction"    # 真空吸盘
    GRIPPER = "gripper"  # 夹爪
    MAGNET = "magnet"     # 磁力吸盘


@dataclass
class GantrySpec:
    """衍架系统规格"""
    max_load_kg: float = 50.0
    x_travel_mm: float = 2000.0
    y_travel_mm: float = 1500.0
    z_travel_mm: float = 800.0
    repeatability_mm: float = 0.3
    max_speed_mm_s: float = 1000.0


class GantrySystem:
    """
    衍架悬挂系统控制器

    支持三轴运动和多种工具切换
    """

    def __init__(
        self,
        host: str = "192.168.1.102",
        port: int = 6512,
        spec: Optional[GantrySpec] = None,
        verbose: bool = True,
    ):
        self.host = host
        self.port = port
        self.spec = spec or GantrySpec()
        self.verbose = verbose

        # 当前位置
        self.current_pos = [0.0, 0.0, 0.0]  # x, y, z mm
        self.current_tool: ToolType = ToolType.SUCTION
        self.tool_attached: bool = False
        self._connected: bool = False

        self._logger(f"衍架系统初始化: 负载{self.spec.max_load_kg}KG, "
                     f"行程({self.spec.x_travel_mm}x{self.spec.y_travel_mm}x{self.spec.z_travel_mm})mm")

    def _logger(self, msg: str):
        if self.verbose:
            print(f"[衍架系统] {msg}")

    def connect(self) -> bool:
        """连接衍架控制器"""
        self._connected = True
        self._logger(f"已连接到 {self.host}:{self.port}")
        return True

    def disconnect(self):
        """断开连接"""
        self._connected = False
        self._logger("已断开连接")

    def set_tool(self, tool: ToolType) -> bool:
        """
        切换工具

        Args:
            tool: 工具类型

        Returns:
            True=成功
        """
        if not self._connected:
            self._logger("错误：未连接")
            return False

        self._logger(f"切换工具: {tool.value}")
        time.sleep(0.3)  # 工具切换时间
        self.current_tool = tool
        self.tool_attached = True
        return True

    def move_to(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        speed_mm_s: Optional[float] = None,
        wait: bool = True,
    ) -> bool:
        """
        移动到目标位置

        Args:
            x_mm, y_mm, z_mm: 目标位置
            speed_mm_s: 移动速度（默认最大速度）
            wait: 是否等待到位

        Returns:
            True=成功
        """
        if not self._connected:
            self._logger("错误：未连接")
            return False

        # 行程检查
        if not (0 <= x_mm <= self.spec.x_travel_mm):
            self._logger(f"错误：X坐标 {x_mm} 超出范围")
            return False
        if not (0 <= y_mm <= self.spec.y_travel_mm):
            self._logger(f"错误：Y坐标 {y_mm} 超出范围")
            return False
        if not (0 <= z_mm <= self.spec.z_travel_mm):
            self._logger(f"错误：Z坐标 {z_mm} 超出范围")
            return False

        speed = speed_mm_s or self.spec.max_speed_mm_s

        # 计算移动时间
        dist = math.sqrt(
            (x_mm - self.current_pos[0]) ** 2 +
            (y_mm - self.current_pos[1]) ** 2 +
            (z_mm - self.current_pos[2]) ** 2
        )
        move_time = dist / speed

        self._logger(f"移动到 ({x_mm:.0f}, {y_mm:.0f}, {z_mm:.0f})mm, "
                     f"速度={speed:.0f}mm/s, 预计{move_time:.2f}s")

        self.current_pos = [x_mm, y_mm, z_mm]

        if wait:
            time.sleep(move_time * 0.1)  # 模拟移动

        return True

    def move_relative(
        self,
        dx_mm: float,
        dy_mm: float,
        dz_mm: float,
        speed_mm_s: Optional[float] = None,
    ) -> bool:
        """
        相对移动

        Args:
            dx_mm, dy_mm, dz_mm: 相对位移

        Returns:
            True=成功
        """
        target = [
            self.current_pos[0] + dx_mm,
            self.current_pos[1] + dy_mm,
            self.current_pos[2] + dz_mm,
        ]
        return self.move_to(target[0], target[1], target[2], speed_mm_s)

    def pick(self, object_z_mm: float = 0.0) -> bool:
        """
        吸取/抓取物品

        Args:
            object_z_mm: 物品表面Z坐标

        Returns:
            True=成功
        """
        if not self._connected:
            self._logger("错误：未连接")
            return False

        if not self.tool_attached:
            self._logger("错误：没有安装工具")
            return False

        # 移动到物品上方
        pick_pos = self.current_pos.copy()
        self.move_to(pick_pos[0], pick_pos[1], object_z_mm + 50)

        # 下降
        self.move_to(pick_pos[0], pick_pos[1], object_z_mm)

        # 执行吸取/抓取
        if self.current_tool == ToolType.SUCTION:
            self._logger("执行真空吸取")
        elif self.current_tool == ToolType.GRIPPER:
            self._logger("执行夹爪抓取")
        elif self.current_tool == ToolType.MAGNET:
            self._logger("执行磁力吸附")

        time.sleep(0.2)
        self._logger("物品已抓取")

        return True

    def place(self, target_z_mm: float = 0.0) -> bool:
        """
        放置物品

        Args:
            target_z_mm: 放置位置Z坐标

        Returns:
            True=成功
        """
        if not self._connected:
            self._logger("错误：未连接")
            return False

        # 移动到放置位置上方
        place_pos = self.current_pos.copy()
        self.move_to(place_pos[0], place_pos[1], target_z_mm + 50)

        # 下降
        self.move_to(place_pos[0], place_pos[1], target_z_mm)

        # 执行释放
        if self.current_tool == ToolType.SUCTION:
            self._logger("释放真空")
        elif self.current_tool == ToolType.GRIPPER:
            self._logger("释放夹爪")
        elif self.current_tool == ToolType.MAGNET:
            self._logger("消磁")

        time.sleep(0.2)

        # 上升到安全高度
        self.move_to(place_pos[0], place_pos[1], target_z_mm + 100)

        self._logger("物品已放置")
        return True

    def get_status(self) -> Dict:
        """获取系统状态"""
        return {
            "connected": self._connected,
            "position_mm": self.current_pos.copy(),
            "current_tool": self.current_tool.value,
            "tool_attached": self.tool_attached,
            "spec": {
                "max_load_kg": self.spec.max_load_kg,
                "x_travel_mm": self.spec.x_travel_mm,
                "y_travel_mm": self.spec.y_travel_mm,
                "z_travel_mm": self.spec.z_travel_mm,
                "repeatability_mm": self.spec.repeatability_mm,
                "max_speed_mm_s": self.spec.max_speed_mm_s,
            },
        }

    def home(self) -> bool:
        """回原点"""
        self._logger("回原点...")
        return self.move_to(0, 0, 0, wait=True)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
