"""
palletizing_cell.py — 码垛单元
:Author: RussellCooper

功能：
  - 12托盘自动堆叠系统
  - 自动供托机构
  - 最大堆叠高度 1800mm
  - 适用于机器人分拣后的成品码垛

规格：
  - 托盘数量: 12个
  - 最大堆叠高度: 1800mm
  - 循环周期: 4-8秒/托盘
  - 定位精度: ±1mm

用法::

    from robot.cells import PalletizingCell

    cell = PalletizingCell(host="192.168.1.101", port=6511)
    cell.connect()

    # 码垛一个托盘
    cell.stack_pallet(pallet_id=1, layers=10)

    cell.disconnect()
"""

import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PalletSpec:
    """托盘规格"""
    id: int
    width_mm: float = 1200.0       # 托盘宽度
    depth_mm: float = 1200.0        # 托盘深度
    max_height_mm: float = 1800.0   # 最大堆叠高度
    max_weight_kg: float = 500.0    # 最大承重


@dataclass
class StackPosition:
    """堆叠位置"""
    x_mm: float
    y_mm: float
    z_mm: float
    layer: int
    row: int
    col: int


class PalletizingCell:
    """
    码垛单元控制器

    支持：
      - 12托盘自动管理
      - 最多12层堆叠
      - 自动供托机构控制
      - 托盘到位检测
    """

    def __init__(
        self,
        host: str = "192.168.1.101",
        port: int = 6511,
        pallet_count: int = 12,
        max_stack_height: float = 1800.0,
        verbose: bool = True,
    ):
        self.host = host
        self.port = port
        self.pallet_count = pallet_count
        self.max_stack_height = max_stack_height
        self.verbose = verbose

        # 托盘状态
        self.pallets: Dict[int, PalletSpec] = {}
        self.current_pallet_id: Optional[int] = None
        self.stack_height_mm: float = 0.0
        self._connected: bool = False

        # 初始化托盘
        for i in range(1, pallet_count + 1):
            self.pallets[i] = PalletSpec(id=i)

        self._logger(f"码垛单元初始化: {pallet_count}托盘, 最大高度{max_stack_height}mm")

    def _logger(self, msg: str):
        if self.verbose:
            print(f"[码垛单元] {msg}")

    def connect(self) -> bool:
        """连接供托机构"""
        # 模拟连接
        self._connected = True
        self._logger(f"已连接到 {self.host}:{self.port}")
        return True

    def disconnect(self):
        """断开连接"""
        self._connected = False
        self._logger("已断开连接")

    def get_available_pallet(self) -> Optional[int]:
        """获取可用托盘ID"""
        for pid, spec in self.pallets.items():
            if pid != self.current_pallet_id:
                return pid
        return None

    def supply_pallet(self, pallet_id: int) -> bool:
        """
        供托：将托盘输送到堆叠位置

        Args:
            pallet_id: 托盘编号 (1-12)

        Returns:
            True=成功
        """
        if not self._connected:
            self._logger("错误：未连接")
            return False

        if pallet_id < 1 or pallet_id > self.pallet_count:
            self._logger(f"错误：无效托盘ID {pallet_id}")
            return False

        # 模拟供托动作
        self._logger(f"供托 #{pallet_id}...")
        time.sleep(0.5)  # 供托时间

        self.current_pallet_id = pallet_id
        self.stack_height_mm = 0.0
        self._logger(f"托盘 #{pallet_id} 已就位，当前高度{self.stack_height_mm}mm")
        return True

    def calculate_stack_position(
        self,
        layer: int,
        items_per_layer: int = 4,
    ) -> List[StackPosition]:
        """
        计算堆叠位置

        Args:
            layer: 层号 (0-indexed)
            items_per_layer: 每层物品数

        Returns:
            堆叠位置列表
        """
        pallet = self.pallets[self.current_pallet_id]
        positions = []

        # 计算每层高度 (假设每个物品高度约50mm)
        layer_height_mm = 50.0
        z_base = layer * layer_height_mm

        # 计算物品间距
        spacing_x = pallet.width_mm / (items_per_layer + 1)
        spacing_y = pallet.depth_mm / (items_per_layer + 1)

        for row in range(items_per_layer):
            for col in range(items_per_layer):
                x = spacing_x * (col + 1) - pallet.width_mm / 2
                y = spacing_y * (row + 1) - pallet.depth_mm / 2
                positions.append(StackPosition(
                    x_mm=x,
                    y_mm=y,
                    z_mm=z_base,
                    layer=layer,
                    row=row,
                    col=col,
                ))

        return positions

    def place_item(
        self,
        item_x_mm: float,
        item_y_mm: float,
        item_z_mm: float,
        item_weight_kg: float = 1.0,
    ) -> bool:
        """
        放置物品到当前托盘

        Args:
            item_x_mm: 物品X坐标
            item_y_mm: 物品Y坐标
            item_z_mm: 物品Z坐标（当前高度）
            item_weight_kg: 物品重量

        Returns:
            True=成功
        """
        if self.current_pallet_id is None:
            self._logger("错误：没有就位的托盘")
            return False

        pallet = self.pallets[self.current_pallet_id]

        # 检查高度限制
        if item_z_mm > self.max_stack_height:
            self._logger(f"错误：超过最大高度 {self.max_stack_height}mm")
            return False

        # 检查重量限制
        current_weight = (pallet.max_height_mm - self.stack_height_mm) * 0.5  # 简化计算
        if item_weight_kg > pallet.max_weight_kg:
            self._logger(f"错误：超过最大承重 {pallet.max_weight_kg}kg")
            return False

        # 更新堆叠高度
        self.stack_height_mm = max(self.stack_height_mm, item_z_mm + 50.0)

        self._logger(f"放置物品于 ({item_x_mm:.0f}, {item_y_mm:.0f}, {item_z_mm:.0f})mm")
        return True

    def stack_pallet(self, pallet_id: int, layers: int = 10) -> bool:
        """
        码垛一个托盘（完整流程）

        Args:
            pallet_id: 托盘编号
            layers: 堆叠层数

        Returns:
            True=成功
        """
        if not self._connected:
            self._logger("错误：未连接")
            return False

        # 供托
        if not self.supply_pallet(pallet_id):
            return False

        # 堆叠每层
        items_per_layer = 4
        for layer in range(layers):
            positions = self.calculate_stack_position(layer, items_per_layer)
            for pos in positions:
                if not self.place_item(pos.x_mm, pos.y_mm, pos.z_mm):
                    self._logger(f"警告：层{layer}放置失败")
                    return False
                time.sleep(0.1)  # 模拟放置时间

            self._logger(f"层 {layer + 1}/{layers} 完成，高度={self.stack_height_mm:.0f}mm")

        # 完成托盘
        self._logger(f"托盘 #{pallet_id} 码垛完成！")
        self.current_pallet_id = None
        return True

    def get_status(self) -> Dict:
        """获取单元状态"""
        return {
            "connected": self._connected,
            "current_pallet": self.current_pallet_id,
            "stack_height_mm": self.stack_height_mm,
            "max_height_mm": self.max_stack_height,
            "available_pallets": self.get_available_pallet(),
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()
