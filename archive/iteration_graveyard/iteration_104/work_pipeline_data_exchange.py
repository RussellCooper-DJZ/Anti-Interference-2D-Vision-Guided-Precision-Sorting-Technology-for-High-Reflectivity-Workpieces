"""
流水线数据交换模块

DataExchanger — 数据交换器
DataPacket — 数据包
DataTransformer — 数据转换器
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DataPacket:
    packet_id: str
    source_step: str
    target_step: str
    data: Any
    data_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataExchanger:
    def __init__(self):
        self._packets: Dict[str, DataPacket] = {}
        self._lock = threading.RLock()
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)

    def send(self, packet: DataPacket) -> None:
        with self._lock:
            self._packets[packet.packet_id] = packet

        self._notify_handlers(packet)

    def receive(self, packet_id: str) -> Optional[DataPacket]:
        with self._lock:
            return self._packets.get(packet_id)

    def get_packets_by_target(self, target_step: str) -> List[DataPacket]:
        with self._lock:
            return [p for p in self._packets.values() if p.target_step == target_step]

    def register_handler(self, step_name: str, handler: Callable[[DataPacket], None]) -> None:
        self._handlers[step_name].append(handler)

    def _notify_handlers(self, packet: DataPacket) -> None:
        handlers = self._handlers.get(packet.target_step, [])
        for handler in handlers:
            try:
                handler(packet)
            except Exception:
                pass


class DataTransformer:
    def __init__(self):
        self._transforms: Dict[str, Callable] = {}

    def register(self, name: str, transform_func: Callable[[Any], Any]) -> None:
        self._transforms[name] = transform_func

    def transform(self, name: str, data: Any) -> Any:
        if name not in self._transforms:
            return data
        return self._transforms[name](data)

    def unregister(self, name: str) -> bool:
        if name in self._transforms:
            del self._transforms[name]
            return True
        return False
