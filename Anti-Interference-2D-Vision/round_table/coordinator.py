"""
圆桌协调器
单例模式的事件调度中心
"""

import threading
from datetime import datetime
from typing import Dict, Set, Callable, Optional, List
from concurrent.futures import TimeoutError

from round_table.events import Event, EventType


class RoundTable:
    """
    圆桌协调器 - 单例
    负责事件的发布、订阅和分发
    """

    _instance: Optional["RoundTable"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RoundTable":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._subscribers: Dict[EventType, Set[Callable]] = {}
        self._event_history: List[Event] = []
        self._waiters: Dict[EventType, threading.Condition] = {}
        self._event_occurred: Dict[EventType, bool] = {}
        self._internal_lock = threading.Lock()
        self._initialized = True

        # 初始化所有事件类型的条件变量
        for event_type in EventType:
            self._waiters[event_type] = threading.Condition()
            self._event_occurred[event_type] = False

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        订阅事件

        Args:
            event_type: 事件类型
            callback: 回调函数，当事件发生时被调用
        """
        with self._internal_lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = set()
            self._subscribers[event_type].add(callback)
        print(f"[RoundTable] Subscriber registered for {event_type.value}")

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        取消订阅事件

        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        with self._internal_lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].discard(callback)

    def publish(self, event: Event) -> None:
        """
        发布事件，唤醒等待者

        Args:
            event: 要发布的事件
        """
        # 记录事件历史
        with self._internal_lock:
            self._event_history.append(event)

        print(f"[RoundTable] Publishing: {event}")

        # 标记事件已发生
        with self._internal_lock:
            self._event_occurred[event.event_type] = True

        # 唤醒所有等待该事件的线程
        with self._waiters[event.event_type]:
            self._waiters[event.event_type].notify_all()

        # 调用所有订阅者回调
        with self._internal_lock:
            subscribers = self._subscribers.get(event.event_type, set()).copy()

        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                print(f"[RoundTable] Error in subscriber callback: {e}")

    def wait_for(self, event_type: EventType, timeout: Optional[float] = None) -> bool:
        """
        等待事件发生

        Args:
            event_type: 要等待的事件类型
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            True if event occurred, False if timeout
        """
        with self._waiters[event_type]:
            # 如果事件已经发生，直接返回
            if self._event_occurred.get(event_type, False):
                print(f"[RoundTable] Event {event_type.value} already occurred, returning immediately")
                return True

            print(f"[RoundTable] Waiting for {event_type.value} (timeout={timeout}s)")
            try:
                self._waiters[event_type].wait(timeout=timeout)
                return self._event_occurred.get(event_type, False)
            except TimeoutError:
                return False

    def wait_for_multiple(self, event_types: Set[EventType], timeout: Optional[float] = None) -> Set[EventType]:
        """
        等待多个事件发生

        Args:
            event_types: 要等待的事件类型集合
            timeout: 超时时间（秒）

        Returns:
            已发生的事件类型集合
        """
        start_time = datetime.now()
        occurred = set()

        while occurred < event_types:
            remaining = event_types - occurred

            # 计算剩余时间
            if timeout is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                remaining_timeout = max(0.1, timeout - elapsed)
                if elapsed >= timeout:
                    break
            else:
                remaining_timeout = None

            # 找出最近要发生的事件
            for event_type in remaining:
                if self._event_occurred.get(event_type, False):
                    occurred.add(event_type)

            if occurred < event_types:
                # 等待任一事件
                with self._internal_lock:
                    available_events = [et for et in remaining if not self._event_occurred.get(et, False)]

                if not available_events:
                    break

                # 使用一个简单的轮询方式等待
                import time
                time.sleep(0.01)

                # 检查超时
                if timeout is not None and (datetime.now() - start_time).total_seconds() >= timeout:
                    break

        return occurred

    def get_event_history(self) -> List[Event]:
        """
        获取事件历史

        Returns:
            所有已发生事件的列表
        """
        with self._internal_lock:
            return self._event_history.copy()

    def is_event_occurred(self, event_type: EventType) -> bool:
        """
        检查事件是否已发生

        Args:
            event_type: 事件类型

        Returns:
            True if event has occurred
        """
        return self._event_occurred.get(event_type, False)

    def reset(self) -> None:
        """
        重置协调器状态（主要用于测试）
        """
        with self._internal_lock:
            self._event_history.clear()
            for event_type in EventType:
                self._event_occurred[event_type] = False
                self._subscribers[event_type] = set()
