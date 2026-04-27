"""
圆桌交流系统示例
展示四位工程师的并行工作流程
"""

import time
import threading
import sys
import os

# 添加父目录到路径，以便能够 import round_table
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from round_table.coordinator import RoundTable
from round_table.config_engineer import ConfigEngineer
from round_table.logger_engineer import LoggerEngineer
from round_table.monitor_engineer import MonitorEngineer
from round_table.checkpoint_engineer import CheckpointEngineer
from round_table.events import EventType


def main():
    """主函数 - 展示圆桌交流系统的工作流程"""

    print("=" * 60)
    print("Round Table Communication System Demo")
    print("=" * 60)
    print()

    # 重置单例（确保干净状态）
    rt = RoundTable()
    rt.reset()

    print("[Main] Creating engineers...")
    print()

    # 创建四位工程师
    config_engineer = ConfigEngineer()
    logger_engineer = LoggerEngineer()
    monitor_engineer = MonitorEngineer()
    checkpoint_engineer = CheckpointEngineer()

    print("[Main] All engineers created, starting parallel work...")
    print()

    # 启动所有工程师的工作线程
    threads = []

    config_thread = threading.Thread(target=config_engineer.work, daemon=True)
    logger_thread = threading.Thread(target=logger_engineer.work, daemon=True)
    monitor_thread = threading.Thread(target=monitor_engineer.work, daemon=True)
    checkpoint_thread = threading.Thread(target=checkpoint_engineer.work, daemon=True)

    threads.extend([config_thread, logger_thread, monitor_thread, checkpoint_thread])

    # 启动所有线程
    for t in threads:
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join(timeout=15.0)

    print()
    print("=" * 60)
    print("Event History")
    print("=" * 60)

    # 显示事件历史
    history = rt.get_event_history()
    for event in history:
        print(f"  {event}")

    print()
    print(f"Total events: {len(history)}")
    print()
    print("Demo completed!")


if __name__ == "__main__":
    main()
