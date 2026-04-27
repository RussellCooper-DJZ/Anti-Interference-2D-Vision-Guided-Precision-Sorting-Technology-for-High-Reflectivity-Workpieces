"""
tests/test_unified_monitor.py - UnifiedMonitor 模块测试
"""
import pytest
import time
from typing import List

from core.unified_monitor import (
    UnifiedMonitor,
    Metric,
    Timer,
    get_monitor,
)


class TestMetric:
    """Metric 数据类测试"""

    def test_metric_creation(self):
        """测试 Metric 创建"""
        metric = Metric(name="test", value=1.0)
        assert metric.name == "test"
        assert metric.value == 1.0
        assert metric.labels == {}
        assert metric.metric_type == "gauge"

    def test_metric_with_labels(self):
        """测试带标签的 Metric"""
        labels = {"env": "test", "region": "us"}
        metric = Metric(name="test", value=2.0, labels=labels)
        assert metric.labels == {"env": "test", "region": "us"}

    def test_metric_with_custom_type(self):
        """测试自定义类型的 Metric"""
        metric = Metric(name="counter_test", value=5.0, metric_type="counter")
        assert metric.metric_type == "counter"


class TestUnifiedMonitorSingleton:
    """UnifiedMonitor 单例模式测试"""

    def test_singleton_same_instance(self):
        """测试单例返回同一实例"""
        monitor1 = UnifiedMonitor()
        monitor2 = UnifiedMonitor()
        assert monitor1 is monitor2

    def test_get_monitor_same_as_instance(self):
        """测试 get_monitor 返回相同实例"""
        monitor = get_monitor()
        instance = UnifiedMonitor()
        assert monitor is instance


class TestUnifiedMonitorCounter:
    """UnifiedMonitor 计数器测试"""

    def test_counter_increment(self):
        """测试计数器递增"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.counter("requests")
        monitor.counter("requests")
        monitor.counter("requests")
        assert monitor.get_counter("requests") == 3

    def test_counter_with_value(self):
        """测试带值的计数器"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.counter("bytes", 1024)
        monitor.counter("bytes", 2048)
        assert monitor.get_counter("bytes") == 3072

    def test_counter_with_labels(self):
        """测试带标签的计数器"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.counter("requests", labels={"method": "GET"})
        assert monitor.get_counter("requests") == 1


class TestUnifiedMonitorGauge:
    """UnifiedMonitor 仪表测试"""

    def test_gauge_set_value(self):
        """测试设置仪表值"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.gauge("cpu_usage", 75.5)
        assert monitor.get_gauge("cpu_usage") == 75.5

    def test_gauge_overwrite(self):
        """测试仪表值覆盖"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.gauge("memory", 1000)
        monitor.gauge("memory", 800)
        assert monitor.get_gauge("memory") == 800


class TestUnifiedMonitorHistogram:
    """UnifiedMonitor 直方图测试"""

    def test_histogram_record(self):
        """测试直方图记录"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.histogram("latency", 10.5)
        monitor.histogram("latency", 20.5)
        monitor.histogram("latency", 30.5)
        stats = monitor.get_histogram_stats("latency")
        assert stats is not None
        assert stats["count"] == 3
        assert stats["sum"] == 61.5

    def test_histogram_stats(self):
        """测试直方图统计"""
        monitor = UnifiedMonitor()
        monitor.reset()
        for i in range(1, 101):
            monitor.histogram("response_time", float(i))
        stats = monitor.get_histogram_stats("response_time")
        assert stats is not None
        assert stats["count"] == 100
        assert stats["min"] == 1.0
        assert stats["max"] == 100.0
        assert stats["mean"] == 50.5


class TestUnifiedMonitorTiming:
    """UnifiedMonitor 计时测试"""

    def test_timing(self):
        """测试计时功能"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.timing("operation_duration", 50.0)
        stats = monitor.get_histogram_stats("operation_duration")
        assert stats is not None
        assert stats["count"] == 1
        assert stats["sum"] == 50.0

    def test_timer_context_manager(self):
        """测试 Timer 上下文管理器"""
        monitor = UnifiedMonitor()
        monitor.reset()
        with Timer(monitor, "task"):
            pass
        stats = monitor.get_histogram_stats("task")
        assert stats is not None
        assert stats["count"] == 1


class TestUnifiedMonitorReport:
    """UnifiedMonitor 报告测试"""

    def test_report_structure(self):
        """测试报告结构"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.counter("requests", 10)
        monitor.gauge("cpu", 50.0)
        monitor.histogram("latency", 25.0)

        report = monitor.report()
        assert "timestamp" in report
        assert "counters" in report
        assert "gauges" in report
        assert "histograms" in report
        assert report["counters"]["requests"] == 10
        assert report["gauges"]["cpu"] == 50.0

    def test_report_histogram_stats(self):
        """测试报告中的直方图统计"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.histogram("latency", 10.0)
        monitor.histogram("latency", 20.0)

        report = monitor.report()
        hist_stats = report["histograms"]["latency"]
        assert hist_stats["count"] == 2
        assert hist_stats["sum"] == 30.0
        assert hist_stats["min"] == 10.0
        assert hist_stats["max"] == 20.0
        assert hist_stats["mean"] == 15.0


class TestUnifiedMonitorPrometheus:
    """UnifiedMonitor Prometheus 格式输出测试"""

    def test_prometheus_counter_output(self):
        """测试 Prometheus 计数器输出"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.counter("http_requests_total", labels={"method": "GET", "status": "200"})

        output = monitor.to_prometheus()
        assert "# TYPE http_requests_total counter" in output
        assert 'method="GET"' in output
        assert 'status="200"' in output

    def test_prometheus_gauge_output(self):
        """测试 Prometheus 仪表输出"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.gauge("process_memory_bytes", 1024000, labels={"pid": "1234"})

        output = monitor.to_prometheus()
        assert "# TYPE process_memory_bytes gauge" in output
        assert 'pid="1234"' in output

    def test_prometheus_histogram_output(self):
        """测试 Prometheus 直方图输出"""
        monitor = UnifiedMonitor()
        monitor.reset()
        for _ in range(100):
            monitor.histogram("request_duration_seconds", 0.5)

        output = monitor.to_prometheus()
        assert "# TYPE request_duration_seconds histogram" in output
        assert "_count" in output
        assert "_sum" in output

    def test_prometheus_format_valid(self):
        """测试 Prometheus 格式有效性"""
        monitor = UnifiedMonitor()
        monitor.reset()
        monitor.counter("test_counter", 1)
        monitor.gauge("test_gauge", 42.0)
        monitor.histogram("test_histogram", 3.14)

        output = monitor.to_prometheus()
        lines = output.strip().split("\n")

        # 检查每行格式
        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.split()
            assert len(parts) == 3, f"Invalid line format: {line}"
            # 第三部分应该是数字（指标值或时间戳）
            assert parts[1].replace(".", "").replace("-", "").isdigit(), f"Invalid value: {line}"


class TestUnifiedMonitorAlerts:
    """UnifiedMonitor 告警测试"""

    def test_alert_condition(self):
        """测试告警条件设置"""
        monitor = UnifiedMonitor()
        monitor.reset()

        def condition() -> bool:
            return monitor.get_gauge("error_rate") > 0.5

        monitor.alert(condition, "High error rate detected")

        # 触发告警条件
        monitor.gauge("error_rate", 0.6)
        alerts = monitor.check_alerts()
        assert "High error rate detected" in alerts

    def test_check_alerts_no_trigger(self):
        """测试未触发告警"""
        monitor = UnifiedMonitor()
        monitor.reset()

        monitor.alert(lambda: monitor.get_gauge("temp") > 100, "Temperature too high")
        monitor.gauge("temp", 50)

        alerts = monitor.check_alerts()
        assert len(alerts) == 0


class TestUnifiedMonitorCallbacks:
    """UnifiedMonitor 回调测试"""

    def test_on_metrics_callback(self):
        """测试指标回调"""
        monitor = UnifiedMonitor()
        monitor.reset()

        received_metrics: List[Metric] = []

        def callback(metrics: List[Metric]) -> None:
            received_metrics.extend(metrics)

        monitor.on_metrics(callback)
        monitor.counter("test_counter", 1)

        assert len(received_metrics) == 1
        assert received_metrics[0].name == "test_counter"


class TestUnifiedMonitorReset:
    """UnifiedMonitor 重置测试"""

    def test_reset_all_metrics(self):
        """测试重置所有指标"""
        monitor = UnifiedMonitor()
        monitor.reset()

        monitor.counter("requests", 100)
        monitor.gauge("cpu", 80.0)
        monitor.histogram("latency", 25.0)

        monitor.reset()

        assert monitor.get_counter("requests") == 0
        assert monitor.get_gauge("cpu") == 0
        assert monitor.get_histogram_stats("latency") is None


class TestUnifiedMonitorGetMetrics:
    """UnifiedMonitor 获取指标测试"""

    def test_get_metrics_history(self):
        """测试获取指标历史"""
        monitor = UnifiedMonitor()
        monitor.reset()

        monitor.counter("requests", 1)
        time.sleep(0.001)
        monitor.counter("requests", 1)
        time.sleep(0.001)
        monitor.counter("requests", 1)

        metrics = monitor.get_metrics("requests")
        assert len(metrics) == 3


class TestTimerContextManager:
    """Timer 上下文管理器测试"""

    def test_timer_basic(self):
        """测试基本计时功能"""
        monitor = UnifiedMonitor()
        monitor.reset()

        with Timer(monitor, "operation"):
            pass

        stats = monitor.get_histogram_stats("operation")
        assert stats is not None
        assert stats["count"] == 1

    def test_timer_with_labels(self):
        """测试带标签的计时"""
        monitor = UnifiedMonitor()
        monitor.reset()

        with Timer(monitor, "db_query", {"table": "users"}):
            pass

        metrics = monitor.get_metrics("db_query")
        assert len(metrics) == 1
        assert metrics[0].labels == {"table": "users"}

    def test_timer_nested(self):
        """测试嵌套计时"""
        monitor = UnifiedMonitor()
        monitor.reset()

        with Timer(monitor, "outer"):
            with Timer(monitor, "inner"):
                pass

        outer_stats = monitor.get_histogram_stats("outer")
        inner_stats = monitor.get_histogram_stats("inner")
        assert outer_stats is not None
        assert inner_stats is not None
        assert outer_stats["count"] == 1
        assert inner_stats["count"] == 1
