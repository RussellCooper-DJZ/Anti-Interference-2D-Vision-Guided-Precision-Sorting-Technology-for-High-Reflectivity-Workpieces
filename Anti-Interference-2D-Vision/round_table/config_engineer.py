"""
UnifiedConfig 工程师
负责配置管理工作
"""

import time
from round_table.engineer import Engineer
from round_table.events import EventType


class ConfigEngineer(Engineer):
    """
    UnifiedConfig 工程师
    负责管理系统配置，完成后通知其他工程师
    """

    def __init__(self, round_table=None):
        super().__init__("UnifiedConfig", round_table)

    def work(self) -> None:
        """配置工程师的工作流程"""
        print(f"[{self.name}] Starting configuration work...")

        # 模拟配置加载工作
        time.sleep(0.5)
        print(f"[{self.name}] Loading configurations...")

        time.sleep(0.5)
        print(f"[{self.name}] Validating configuration schema...")

        time.sleep(0.3)
        print(f"[{self.name}] Configuration complete!")

        # 宣布配置就绪
        self.announce_done(EventType.UNIFIED_CONFIG_READY, {
            "config_version": "1.0.0",
            "modules": ["database", "cache", "api"]
        })

        print(f"[{self.name}] Waiting for other modules to be ready...")
        # 等待所有模块就绪
        self.wait_for(EventType.ALL_MODULES_READY, timeout=10.0)
        print(f"[{self.name}] All modules ready, configuration work finalized")
