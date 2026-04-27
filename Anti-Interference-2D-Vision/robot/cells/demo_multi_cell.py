"""
demo_multi_cell.py — 多单元协同演示
:Author: RussellCooper

演示精密分拣系统多单元协同工作：
  1. 精密分拣单元 — 视觉引导抓取
  2. 码垛单元 — 成品堆叠
  3. 衍架系统 — 物料输送

流程：
  相机 → FLARE检测 → 分拣机器人抓取 → 码垛单元堆叠

用法::

    python3 robot/cells/demo_multi_cell.py
"""

import time
import argparse
from typing import Dict, List

from robot.cells import (
    PrecisionSortingCell,
    PalletizingCell,
    GantrySystem,
)


class MultiCellCoordinator:
    """
    多单元协调器

    管理精密分拣单元、码垛单元、衍架系统的协同工作
    """

    def __init__(
        self,
        sorting_cell: PrecisionSortingCell,
        palletizing_cell: PalletizingCell,
        gantry_system: GantrySystem = None,
        verbose: bool = True,
    ):
        self.sorting = sorting_cell
        self.palletizing = palletizing_cell
        self.gantry = gantry_system
        self.verbose = verbose

        # 当前托盘
        self.current_pallet_id: int = 1
        self.current_layer: int = 0
        self.items_on_pallet: int = 0
        self.items_per_layer: int = 4

        # 统计
        self.total_processed: int = 0
        self.total_failed: int = 0

        self._logger("多单元协调器初始化完成")

    def _logger(self, msg: str):
        if self.verbose:
            print(f"[Coordinator] {msg}")

    def connect_all(self) -> bool:
        """连接所有单元"""
        self._logger("连接所有单元...")

        self.sorting.connect()
        self.palletizing.connect()

        if self.gantry:
            self.gantry.connect()

        self._logger("所有单元已连接")
        return True

    def disconnect_all(self):
        """断开所有连接"""
        self._logger("断开所有连接...")

        self.sorting.disconnect()
        self.palletizing.disconnect()

        if self.gantry:
            self.gantry.disconnect()

    def run_sorting_cycle(self) -> bool:
        """执行一个完整的分拣+码垛周期"""
        self._logger("=" * 50)
        self._logger(f"开始分拣周期 #{self.total_processed + 1}")
        self._logger(f"当前托盘 #{self.current_pallet_id}, 层 {self.current_layer}, 件数 {self.items_on_pallet}")

        # Step 1: 分拣单元执行抓取
        self._logger("[1/4] 分拣单元执行抓取...")
        success_sort, cycle_time = self.sorting.run_cycle()

        if not success_sort:
            self._logger("分拣失败")
            self.total_failed += 1
            return False

        # Step 2: 计算放置位置
        self._logger("[2/4] 计算放置位置...")

        # 计算当前层的位置
        layer_positions = self.palletizing.calculate_stack_position(
            self.current_layer,
            self.items_per_layer
        )

        # 获取下一个放置位置
        place_idx = self.items_on_pallet % self.items_per_layer
        if place_idx < len(layer_positions):
            place_pos = layer_positions[place_idx]
            place_xyz = (place_pos.x_mm, place_pos.y_mm, place_pos.z_mm)
        else:
            # 放置到托盘边缘（备用）
            place_xyz = (300, 300, self.current_layer * 50)

        self._logger(f"   放置位置: ({place_xyz[0]:.0f}, {place_xyz[1]:.0f}, {place_xyz[2]:.0f})mm")

        # Step 3: 放置到码垛位置
        self._logger("[3/4] 放置到码垛位置...")
        self.palletizing.place_item(
            place_xyz[0], place_xyz[1], place_xyz[2],
            item_weight_kg=1.0
        )

        # Step 4: 更新计数
        self.items_on_pallet += 1
        self.total_processed += 1

        # 检查是否需要换层
        if self.items_on_pallet >= self.items_per_layer:
            self.current_layer += 1
            self.items_on_pallet = 0
            self._logger(f"   层 {self.current_layer} 完成!")

        # 检查是否需要换托盘
        if self.current_layer >= 10:  # 每托盘最多10层
            self._logger(f"托盘 #{self.current_pallet_id} 已满，切换托盘")
            self.current_pallet_id += 1
            if self.current_pallet_id > 12:
                self.current_pallet_id = 1
            self.current_layer = 0
            self.palletizing.stack_pallet(self.current_pallet_id, layers=10)

        self._logger(f"[4/4] 周期完成! 总计: {self.total_processed}件, 失败: {self.total_failed}件")

        return True

    def run(self, n_cycles: int = 20, cycle_delay_s: float = 0.5):
        """
        运行多周期

        Args:
            n_cycles: 运行周期数
            cycle_delay_s: 周期间隔
        """
        if not self.connect_all():
            return

        self._logger(f"开始运行 {n_cycles} 个分拣周期...")
        self._logger(f"规格:")
        self._logger(f"  - 分拣精度: ±0.5mm")
        self._logger(f"  - 目标周期: 3-6秒/件")
        self._logger(f"  - 码垛: 12托盘, 最大1800mm")
        self._logger(f"  - 衍架: 50KG负载 (可选)")
        self._logger("-" * 50)

        # 初始化第一个托盘
        self.palletizing.supply_pallet(self.current_pallet_id)

        for i in range(n_cycles):
            success = self.run_sorting_cycle()

            if success:
                self._logger(f"周期 {i+1}/{n_cycles} 完成")

            if cycle_delay_s > 0:
                time.sleep(cycle_delay_s)

        self._logger("=" * 50)
        self._logger("运行完成!")
        self._logger(f"总计处理: {self.total_processed}件")
        self._logger(f"失败: {self.total_failed}件")
        self._logger(f"成功率: {self.total_processed / max(1, self.total_processed + self.total_failed) * 100:.1f}%")

        # 打印统计
        self._logger("\n单元状态:")
        self._logger(f"  分拣: {self.sorting.get_status()}")
        self._logger(f"  码垛: {self.palletizing.get_status()}")
        if self.gantry:
            self._logger(f"  衍架: {self.gantry.get_status()}")

        self.disconnect_all()


def main():
    parser = argparse.ArgumentParser(description="多单元协同演示")
    parser.add_argument("--cycles", type=int, default=20, help="运行周期数")
    parser.add_argument("--robot-model", type=str, default="6T", choices=["3T", "6T"],
                       help="机器人型号")
    parser.add_argument("--use-gantry", action="store_true", help="启用衍架系统")
    args = parser.parse_args()

    print("=" * 60)
    print("  精密分拣系统 — 多单元协同演示")
    print("=" * 60)

    # 创建单元
    sorting_cell = PrecisionSortingCell(
        robot_model=args.robot_model,
        robot_host="192.168.1.100",
    )

    palletizing_cell = PalletizingCell(
        host="192.168.1.101",
        pallet_count=12,
        max_stack_height=1800.0,
    )

    gantry_system = None
    if args.use_gantry:
        gantry_system = GantrySystem(
            host="192.168.1.102",
        )

    # 创建协调器
    coordinator = MultiCellCoordinator(
        sorting_cell=sorting_cell,
        palletizing_cell=palletizing_cell,
        gantry_system=gantry_system,
    )

    # 运行
    coordinator.run(n_cycles=args.cycles)


if __name__ == "__main__":
    main()
