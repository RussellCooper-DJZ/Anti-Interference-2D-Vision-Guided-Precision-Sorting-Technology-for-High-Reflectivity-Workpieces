"""
robot — 机器人控制与仿真接口包

包含以下子模块：
  abb_robotstudio_interface: ABB RobotStudio TCP/IP 通信接口
  abb_rapid/               : ABB RAPID 控制器代码（abb_server.mod）
  cells/                   : 机器人分拣单元模块

cells模块包含：
  - PalletizingCell: 码垛单元（12托盘，1800mm堆叠）
  - GantrySystem: 衍架悬挂系统（50KG负载）
  - SortingCell: 机器人分拣单元
  - PrecisionSortingCell: 精密分拣单元（3T/6T机器人）
"""

from .cells import (
    PalletizingCell,
    GantrySystem,
    ToolType,
    SortingCell,
    PrecisionSortingCell,
)

__all__ = [
    "AbbRobotBase",
    "AbbRobotStub",
    "AbbRobotStudioSim",
    "create_robot_interface",
    # 分拣单元
    "PalletizingCell",
    "GantrySystem",
    "ToolType",
    "SortingCell",
    "PrecisionSortingCell",
]
