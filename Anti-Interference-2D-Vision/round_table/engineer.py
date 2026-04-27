"""
工程师基类
定义所有工程师的通用行为
"""

import threading
from datetime import datetime
from typing import Set, Optional

from round_table.events import Event, EventType
from round_table.coordinator import RoundTable


class Engineer:
    """
    工程师基类
    提供事件发布和等待的通用功能
    """

    def __init__(self, name: str, round_table: Optional[RoundTable] = None):
        """
        初始化工程师

        Args:
            name: 工程师名称
            round_table: 圆桌协调器实例，如果为None则使用单例
        """
        self.name = name
        self.rt = round_table if round_table else RoundTable()
        self.waiting_for: Set[EventType] = set()
        self._thread: Optional[threading.Thread] = None

    def announce_done(self, event_type: EventType, data: dict = None) -> None:
        """
        宣布自己完成任务，发布事件

        Args:
            event_type: 事件类型
            data: 附加数据
        """
        event = Event(
            event_type=event_type,
            timestamp=datetime.now(),
            source=self.name,
            data=data or {}
        )
        print(f"[{self.name}] Announcing done: {event_type.value}")
        self.rt.publish(event)

    def wait_for(self, event_type: EventType, timeout: Optional[float] = None) -> bool:
        """
        等待某个工程师完成任务

        Args:
            event_type: 要等待的事件类型
            timeout: 超时时间（秒）

        Returns:
            True if event occurred, False if timeout
        """
        self.waiting_for.add(event_type)
        print(f"[{self.name}] Waiting for {event_type.value}")
        return self.rt.wait_for(event_type, timeout)

    def wait_for_multiple(self, event_types: Set[EventType], timeout: Optional[float] = None) -> Set[EventType]:
        """
        等待多个事件

        Args:
            event_types: 要等待的事件类型集合
            timeout: 超时时间（秒）

        Returns:
            已发生的事件类型集合
        """
        self.waiting_for.update(event_types)
        print(f"[{self.name}] Waiting for multiple events: {[e.value for e in event_types]}")
        return self.rt.wait_for_multiple(event_types, timeout)

    def subscribe(self, event_type: EventType, callback) -> None:
        """
        订阅事件

        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        self.rt.subscribe(event_type, callback)

    def start_work(self) -> None:
        """启动工作线程"""
        self._thread = threading.Thread(target=self.work, daemon=True)
        self._thread.start()

    def work(self) -> None:
        """
        具体的工作逻辑，由子类实现
        """
        raise NotImplementedError("Subclasses must implement work()")

    def join(self, timeout: Optional[float] = None) -> None:
        """
        等待工作线程结束

        Args:
            timeout: 超时时间
        """
        if self._thread:
            self._thread.join(timeout=timeout)
