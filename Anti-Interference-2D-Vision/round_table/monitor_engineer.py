"""
UnifiedMonitor 工程师
负责监控管理工作
"""

import time
from round_table.engineer import Engineer
from round_table.events import EventType


class MonitorEngineer(Engineer):
    """
    UnifiedMonitor 工程师
    负责监控系统初始化，需要等待配置和日志就绪
    """

    def __init__(self, round_table=None):
        super().__init__("UnifiedMonitor", round_table)

    def work(self) -> None:
        """监控工程师的工作流程"""
        print(f"[{self.name}] Starting monitor system initialization...")

        # 等待配置和日志都就绪
        print(f"[{self.name}] Waiting for dependencies...")

        ready_events = self.wait_for_multiple(
            {EventType.UNIFIED_CONFIG_READY, EventType.UNIFIED_LOGGER_READY},
            timeout=5.0
        )

        if EventType.UNIFIED_CONFIG_READY in ready_events:
            print(f"[{self.name}] Configuration dependency satisfied")
        else:
            print(f"[{self.name}] Configuration dependency timeout")

        if EventType.UNIFIED_LOGGER_READY in ready_events:
            print(f"[{self.name}] Logger dependency satisfied")
        else:
            print(f"[{self.name}] Logger dependency timeout")

        # 模拟监控初始化
        time.sleep(0.5)
        print(f"[{self.name}] Initializing metrics collectors...")

        time.sleep(0.4)
        print(f"[{self.name}] Setting up alerting system...")

        time.sleep(0.3)
        print(f"[{self.name}] Monitor system initialization complete!")

        # 宣布监控就绪
        self.announce_done(EventType.UNIFIED_MONITOR_READY, {
            "metrics_count": 42,
            "alert_channels": ["email", "webhook"]
        })

        print(f"[{self.name}] Waiting for other modules to be ready...")
        # 等待所有模块就绪
        self.wait_for(EventType.ALL_MODULES_READY, timeout=10.0)
        print(f"[{self.name}] All modules ready, monitor work finalized")
