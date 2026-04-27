"""
robot/cells — 机器人分拣系统单元模块
:Author: RussellCooper

包含以下单元：
  - PalletizingCell: 码垛单元（12托盘堆叠）
  - GantrySystem: 衍架悬挂系统（50KG负载）
  - SortingCell: 机器人分拣单元
  - PrecisionSortingCell: 精密分拣单元（3T/6T机器人）

参考规格：
  - 码垛单元: 12托盘，自动供托，最大堆叠高度1800mm
  - 衍架系统: 最大负载50KG，多种工具配置
  - 分拣单元: 3-6秒循环周期，±0.5mm精度
"""

from .palletizing_cell import PalletizingCell
from .gantry_system import GantrySystem, ToolType
from .sorting_cell import SortingCell
from .precision_sorting_cell import PrecisionSortingCell

__all__ = [
    "PalletizingCell",
    "GantrySystem",
    "ToolType",
    "SortingCell",
    "PrecisionSortingCell",
]
