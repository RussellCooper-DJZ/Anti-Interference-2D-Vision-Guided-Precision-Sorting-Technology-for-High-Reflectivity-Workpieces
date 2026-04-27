"""
事件定义模块
定义系统中所有的事件类型和事件数据结构
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


class EventType(Enum):
    """事件类型枚举"""
    UNIFIED_CONFIG_READY = "unified_config_ready"
    UNIFIED_LOGGER_READY = "unified_logger_ready"
    UNIFIED_MONITOR_READY = "unified_monitor_ready"
    UNIFIED_CHECKPOINT_READY = "unified_checkpoint_ready"
    ALL_MODULES_READY = "all_modules_ready"


@dataclass
class Event:
    """事件数据结构"""
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Event({self.event_type.value}, from={self.source}, at={self.timestamp.strftime('%H:%M:%S.%f')[:-3]})"
