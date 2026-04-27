"""
UnifiedMonitor - 统一监控模块
支持指标采集、上报、告警，Prometheus格式输出
"""
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
from collections import defaultdict


@dataclass
class Metric:
    """指标数据"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, summary


class UnifiedMonitor:
    """统一监控管理器（单例）"""
    _instance: Optional['UnifiedMonitor'] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> 'UnifiedMonitor':
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()  # Use RLock to avoid deadlock when callbacks call getter methods
        self._callbacks: List[Callable[[List[Metric]], None]] = []
        self._alert_conditions: List[Callable[[], bool]] = []
        self._alert_messages: List[str] = []
        self._initialized = True

    def counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """增加计数器"""
        with self._lock:
            self._counters[name] += value
            metric = Metric(
                name=name,
                value=self._counters[name],
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type="counter"
            )
            self._metrics[name].append(metric)
            self._trigger_callbacks([metric])

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """设置仪表值"""
        with self._lock:
            self._gauges[name] = value
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type="gauge"
            )
            self._metrics[name].append(metric)
            self._trigger_callbacks([metric])

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """记录直方图值"""
        with self._lock:
            self._histograms[name].append(value)
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metric_type="histogram"
            )
            self._metrics[name].append(metric)
            self._trigger_callbacks([metric])

    def timing(self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None) -> None:
        """记录耗时"""
        self.histogram(name, duration_ms, labels)

    def _trigger_callbacks(self, metrics: List[Metric]) -> None:
        """触发指标回调"""
        for callback in self._callbacks:
            try:
                callback(metrics)
            except Exception:
                pass  # 避免回调异常影响主流程

    def report(self) -> Dict[str, Any]:
        """获取监控报告"""
        with self._lock:
            report: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {
                        "count": len(values),
                        "sum": sum(values),
                        "min": min(values) if values else None,
                        "max": max(values) if values else None,
                        "mean": sum(values) / len(values) if values else None,
                    }
                    for name, values in self._histograms.items()
                },
                "metric_count": sum(len(m) for m in self._metrics.values()),
            }
            return report

    def to_prometheus(self) -> str:
        """转换为Prometheus格式"""
        lines: List[str] = []
        timestamp_ms = int(datetime.now().timestamp() * 1000)

        with self._lock:
            # 输出 counters
            for name, value in self._counters.items():
                labels_str = ""
                if self._metrics[name]:
                    last_metric = self._metrics[name][-1]
                    if last_metric.labels:
                        label_parts = [f'{k}="{v}"' for k, v in last_metric.labels.items()]
                        labels_str = "{" + ",".join(label_parts) + "}"
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels_str} {value} {timestamp_ms}")

            # 输出 gauges
            for name, value in self._gauges.items():
                labels_str = ""
                if self._metrics[name]:
                    last_metric = self._metrics[name][-1]
                    if last_metric.labels:
                        label_parts = [f'{k}="{v}"' for k, v in last_metric.labels.items()]
                        labels_str = "{" + ",".join(label_parts) + "}"
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels_str} {value} {timestamp_ms}")

            # 输出 histograms
            for name, values in self._histograms.items():
                if not values:
                    continue
                sorted_values = sorted(values)
                count = len(sorted_values)
                sum_val = sum(sorted_values)

                # 计算分位数
                quantiles = [0.5, 0.9, 0.95, 0.99]
                quantile_lines = []
                for q in quantiles:
                    idx = int(count * q)
                    if idx >= count:
                        idx = count - 1
                    if idx < 0:
                        idx = 0
                    quantile_lines.append(f"{{quantile=\"{q}\"}} {sorted_values[idx]}")

                labels_str = ""
                if self._metrics[name]:
                    last_metric = self._metrics[name][-1]
                    if last_metric.labels:
                        label_parts = [f'{k}="{v}"' for k, v in last_metric.labels.items()]
                        labels_str = "{" + ",".join(label_parts) + "}"

                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count{labels_str} {count} {timestamp_ms}")
                lines.append(f"{name}_sum{labels_str} {sum_val} {timestamp_ms}")
                for q_line in quantile_lines:
                    lines.append(f"{name}{q_line} {timestamp_ms}")

        return "\n".join(lines) + "\n"

    def alert(self, condition: Callable[[], bool], message: str) -> None:
        """设置告警条件"""
        with self._lock:
            self._alert_conditions.append(condition)
            self._alert_messages.append(message)

    def check_alerts(self) -> List[str]:
        """检查告警条件并返回触发的告警消息"""
        triggered: List[str] = []
        with self._lock:
            for condition, msg in zip(self._alert_conditions, self._alert_messages):
                try:
                    if condition():
                        triggered.append(msg)
                except Exception:
                    pass
        return triggered

    def on_metrics(self, callback: Callable[[List[Metric]], None]) -> None:
        """注册指标回调"""
        with self._lock:
            self._callbacks.append(callback)

    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    def get_metrics(self, name: str) -> List[Metric]:
        """获取指定名称的指标历史"""
        with self._lock:
            return list(self._metrics.get(name, []))

    def get_counter(self, name: str) -> float:
        """获取计数器值"""
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """获取仪表值"""
        with self._lock:
            return self._gauges.get(name, 0)

    def get_histogram_stats(self, name: str) -> Optional[Dict[str, float]]:
        """获取直方图统计信息"""
        with self._lock:
            values = self._histograms.get(name)
            if not values:
                return None
            sorted_values = sorted(values)
            count = len(sorted_values)
            return {
                "count": count,
                "sum": sum(sorted_values),
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "mean": sum(sorted_values) / count,
                "p50": sorted_values[int(count * 0.5)],
                "p90": sorted_values[int(count * 0.9)] if count >= 10 else sorted_values[-1],
                "p95": sorted_values[int(count * 0.95)] if count >= 20 else sorted_values[-1],
                "p99": sorted_values[int(count * 0.99)] if count >= 100 else sorted_values[-1],
            }


class Timer:
    """耗时计时器"""

    def __init__(self, monitor: UnifiedMonitor, name: str, labels: Optional[Dict[str, str]] = None):
        self.monitor = monitor
        self.name = name
        self.labels = labels or {}
        self.start_time: float = 0

    def __enter__(self) -> 'Timer':
        self.start_time = time.time()
        return self

    def __exit__(self, *args: Any) -> None:
        duration_ms = (time.time() - self.start_time) * 1000
        self.monitor.timing(self.name, duration_ms, self.labels)


def get_monitor() -> UnifiedMonitor:
    """获取全局监控实例"""
    return UnifiedMonitor()
