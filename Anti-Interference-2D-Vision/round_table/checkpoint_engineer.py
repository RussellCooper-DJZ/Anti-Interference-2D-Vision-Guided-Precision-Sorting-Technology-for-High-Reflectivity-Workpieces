"""
UnifiedCheckpoint 工程师
负责检查点/快照管理工作
"""

import time
from round_table.engineer import Engineer
from round_table.events import EventType


class CheckpointEngineer(Engineer):
    """
    UnifiedCheckpoint 工程师
    负责检查点系统初始化，需要等待所有其他模块就绪
    """

    def __init__(self, round_table=None):
        super().__init__("UnifiedCheckpoint", round_table)

    def work(self) -> None:
        """检查点工程师的工作流程"""
        print(f"[{self.name}] Starting checkpoint system initialization...")

        # 等待所有其他模块就绪
        print(f"[{self.name}] Waiting for all modules to be ready...")

        ready_events = self.wait_for_multiple(
            {
                EventType.UNIFIED_CONFIG_READY,
                EventType.UNIFIED_LOGGER_READY,
                EventType.UNIFIED_MONITOR_READY
            },
            timeout=10.0
        )

        dependencies_met = True
        for event_type in [EventType.UNIFIED_CONFIG_READY,
                           EventType.UNIFIED_LOGGER_READY,
                           EventType.UNIFIED_MONITOR_READY]:
            if event_type in ready_events:
                print(f"[{self.name}] {event_type.value} satisfied")
            else:
                print(f"[{self.name}] {event_type.value} NOT met (timeout)")
                dependencies_met = False

        # 模拟检查点系统初始化
        time.sleep(0.6)
        print(f"[{self.name}] Initializing checkpoint manager...")

        time.sleep(0.5)
        print(f"[{self.name}] Setting up storage backend...")

        time.sleep(0.4)
        print(f"[{self.name}] Configuring checkpoint policies...")

        time.sleep(0.3)
        print(f"[{self.name}] Checkpoint system initialization complete!")

        # 宣布检查点就绪
        self.announce_done(EventType.UNIFIED_CHECKPOINT_READY, {
            "checkpoint_interval": 300,
            "retention_days": 7
        })

        # 等待所有模块都就绪后，发布 ALL_MODULES_READY 事件
        # 这个事件应该由最后一个完成的工程师发布
        all_ready = self.wait_for_multiple(
            {
                EventType.UNIFIED_CONFIG_READY,
                EventType.UNIFIED_LOGGER_READY,
                EventType.UNIFIED_MONITOR_READY,
                EventType.UNIFIED_CHECKPOINT_READY
            },
            timeout=2.0
        )

        if len(all_ready) == 4:
            print(f"[{self.name}] All modules ready! Publishing ALL_MODULES_READY event...")
            self.announce_done(EventType.ALL_MODULES_READY, {
                "total_modules": 4,
                "system_ready": True
            })

        print(f"[{self.name}] Checkpoint work finalized")
