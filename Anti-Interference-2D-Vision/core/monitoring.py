"""
core/monitoring.py — 统一指标监控

提取并合并自：
- results/auto_tuning/iteration_39/work_metrics_monitor.py
- results/auto_tuning/iteration_48/work_model_monitoring.py
- results/auto_tuning/iteration_62/work_monitoring_alerting.py
"""

import json
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "MetricType",
    "MetricRecord",
    "MetricsCollector",
    "AlertSeverity",
    "AlertStatus",
    "Alert",
    "AlertRule",
    "AlertManager",
]


class MetricType(Enum):
    """指标类型"""
    LOSS = "loss"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    IoU = "iou"
    LATENCY = "latency"
    FPS = "fps"
    MEMORY = "memory"
    CUSTOM = "custom"


@dataclass
class MetricRecord:
    """指标记录"""
    name: str
    value: float
    step: int
    timestamp: float
    phase: str = "train"


class MetricsCollector:
    """指标收集器：收集和统计训练/推理过程中的各项指标。"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._global_step = 0
        self._lock = threading.Lock()

    def log(self, name: str, value: float, step: Optional[int] = None, phase: str = "train"):
        with self._lock:
            step = step if step is not None else self._global_step
            record = MetricRecord(name=name, value=float(value), step=step, timestamp=time.time(), phase=phase)
            self._metrics[name].append(record)
            self._global_step = max(self._global_step, step)

    def log_dict(self, metrics: Dict[str, float], step: Optional[int] = None, phase: str = "train"):
        for name, value in metrics.items():
            self.log(name, value, step, phase)

    def get_history(self, name: str) -> List[MetricRecord]:
        return list(self._metrics.get(name, []))

    def get_latest(self, name: str) -> Optional[MetricRecord]:
        history = self._metrics.get(name)
        return history[-1] if history else None

    def get_average(self, name: str, last_n: Optional[int] = None) -> Optional[float]:
        history = self._metrics.get(name)
        if not history:
            return None
        items = list(history)[-last_n:] if last_n else list(history)
        return sum(r.value for r in items) / len(items)

    def summary(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for name, history in self._metrics.items():
            values = [r.value for r in history]
            result[name] = {
                "count": len(values),
                "latest": values[-1] if values else None,
                "mean": sum(values) / len(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        return result

    def reset(self):
        with self._lock:
            self._metrics.clear()
            self._global_step = 0


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """告警"""
    alert_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    metric_name: str
    current_value: float
    threshold: float
    fired_at: Optional[str] = None
    resolved_at: Optional[str] = None


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ge', 'le'
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    duration: int = 1  # 持续多少个点才触发

    def check(self, value: float) -> bool:
        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "ge": lambda v, t: v >= t,
            "le": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
        }
        return ops.get(self.condition, lambda v, t: False)(value, self.threshold)


class AlertManager:
    """告警管理器：基于规则监控指标并触发告警。"""

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.alerts: List[Alert] = []
        self._counters: Dict[str, int] = defaultdict(int)
        self._callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def add_callback(self, callback: Callable[[Alert], None]):
        self._callbacks.append(callback)

    def check(self, metrics: Dict[str, float]):
        with self._lock:
            for rule in self.rules:
                value = metrics.get(rule.metric_name)
                if value is None:
                    continue
                if rule.check(value):
                    self._counters[rule.name] += 1
                    if self._counters[rule.name] >= rule.duration:
                        self._fire(rule, value)
                else:
                    self._counters[rule.name] = 0

    def _fire(self, rule: AlertRule, value: float):
        alert_id = f"{rule.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        alert = Alert(
            alert_id=alert_id,
            name=rule.name,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            message=f"{rule.metric_name}={value:.4f} 触发 {rule.condition} {rule.threshold}",
            metric_name=rule.metric_name,
            current_value=value,
            threshold=rule.threshold,
            fired_at=datetime.now().isoformat(),
        )
        self.alerts.append(alert)
        for cb in self._callbacks:
            cb(alert)

    def resolve(self, alert_id: str):
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now().isoformat()

    def get_active_alerts(self) -> List[Alert]:
        return [a for a in self.alerts if a.status == AlertStatus.FIRING]

    def summary(self) -> Dict[str, Any]:
        return {
            "rules": len(self.rules),
            "total_alerts": len(self.alerts),
            "active_alerts": len(self.get_active_alerts()),
        }
