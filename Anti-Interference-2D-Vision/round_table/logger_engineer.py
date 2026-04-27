"""
UnifiedLogger 工程师
负责日志管理工作
"""

import time
from round_table.engineer import Engineer
from round_table.events import EventType


class LoggerEngineer(Engineer):
    """
    UnifiedLogger 工程师
    负责日志系统初始化，需要等待配置就绪
    """

    def __init__(self, round_table=None):
        super().__init__("UnifiedLogger", round_table)

    def work(self) -> None:
        """日志工程师的工作流程"""
        print(f"[{self.name}] Starting logger initialization...")

        # 等待配置就绪
        print(f"[{self.name}] Waiting for configuration to be ready...")
        if self.wait_for(EventType.UNIFIED_CONFIG_READY, timeout=5.0):
            print(f"[{self.name}] Configuration received, applying logger config...")
        else:
            print(f"[{self.name}] Configuration timeout, using default settings...")

        # 模拟日志系统初始化
        time.sleep(0.4)
        print(f"[{self.name}] Initializing log handlers...")

        time.sleep(0.4)
        print(f"[{self.name}] Setting up log rotation...")

        time.sleep(0.3)
        print(f"[{self.name}] Logger initialization complete!")

        # 宣布日志系统就绪
        self.announce_done(EventType.UNIFIED_LOGGER_READY, {
            "log_level": "INFO",
            "handlers": ["console", "file"]
        })

        print(f"[{self.name}] Waiting for other modules to be ready...")
        # 等待所有模块就绪
        self.wait_for(EventType.ALL_MODULES_READY, timeout=10.0)
        print(f"[{self.name}] All modules ready, logger work finalized")
