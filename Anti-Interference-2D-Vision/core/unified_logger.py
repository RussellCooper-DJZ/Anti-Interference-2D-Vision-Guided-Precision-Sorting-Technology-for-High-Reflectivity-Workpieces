"""
UnifiedLogger - 统一日志模块
支持结构化日志、多级别、多输出、日志上下文
"""
import logging
import sys
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler


class LogContext:
    """日志上下文"""

    def __init__(self) -> None:
        self._context: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """设置上下文键值对"""
        self._context[key] = value

    def get(self, key: str) -> Any:
        """获取上下文值"""
        return self._context.get(key)

    def clear(self) -> None:
        """清空上下文"""
        self._context.clear()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._context.copy()


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": getattr(record, "context", {}),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """文本格式日志格式化器"""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class UnifiedLogger:
    """统一日志管理器（单例）"""

    _instance: Optional["UnifiedLogger"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "UnifiedLogger":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # 双重检查
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._logger = logging.getLogger("AGEANet")
        self._thread_local = threading.local()
        self._handlers: List[logging.Handler] = []
        self._level = logging.INFO
        self._output = "console"
        self._file_path: Optional[str] = None
        self._initialized = True

    def setup(
        self,
        level: str = "INFO",
        output: str = "console",
        file_path: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        format_type: str = "text",
    ) -> None:
        """配置日志系统

        Args:
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            output: 输出类型 (console, file, rotating)
            file_path: 日志文件路径
            max_bytes: 轮转文件最大字节数
            backup_count: 轮转文件备份数量
            format_type: 格式化类型 (text, json)
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        self._level = level_map.get(level.upper(), logging.INFO)
        self._output = output
        self._file_path = file_path

        self._logger.setLevel(self._level)
        self._logger.handlers.clear()
        self._handlers.clear()

        if format_type == "json":
            formatter = StructuredFormatter()
        else:
            formatter = TextFormatter()

        if output == "console":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._level)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
            self._handlers.append(console_handler)

        elif output in ("file", "rotating"):
            if not file_path:
                raise ValueError("file_path is required for file/rotating output")
            import os

            log_dir = os.path.dirname(file_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            if output == "rotating":
                handler: logging.Handler = RotatingFileHandler(
                    file_path, maxBytes=max_bytes, backupCount=backup_count
                )
            else:
                handler = logging.FileHandler(file_path)
            handler.setLevel(self._level)
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._handlers.append(handler)

    def _log(self, level: int, msg: str, **kwargs) -> None:
        """内部日志方法"""
        extra_context = kwargs.pop("context", {})
        thread_context = getattr(self._thread_local, 'context', None)
        context_dict = thread_context.to_dict() if thread_context else {}
        merged_context = {**context_dict, **extra_context}

        record = self._logger.makeRecord(
            self._logger.name, level, "(unknown)", 0, msg, (), None
        )
        record.context = merged_context
        self._logger.handle(record)

    def debug(self, msg: str, **kwargs) -> None:
        """调试级别日志"""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs) -> None:
        """信息级别日志"""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs) -> None:
        """警告级别日志"""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs) -> None:
        """错误级别日志"""
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs) -> None:
        """严重级别日志"""
        self._log(logging.CRITICAL, msg, **kwargs)

    def exception(self, msg: str, **kwargs) -> None:
        """异常日志（自动包含堆栈信息）"""
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, **kwargs)

    def with_context(self, **kwargs) -> LogContext:
        """设置日志上下文

        Returns:
            LogContext: 上下文管理器

        Example:
            logger.with_context(trace_id="abc", request_id="123")
            logger.info("message")  # 自动包含 trace_id 和 request_id
        """
        if not hasattr(self._thread_local, 'context'):
            self._thread_local.context = LogContext()
        for key, value in kwargs.items():
            self._thread_local.context.set(key, value)
        return self._thread_local.context

    def set_context(self, **kwargs) -> None:
        """设置日志上下文（一次性设置多个键值）"""
        if not hasattr(self._thread_local, 'context'):
            self._thread_local.context = LogContext()
        for key, value in kwargs.items():
            self._thread_local.context.set(key, value)

    def get_context(self) -> LogContext:
        """获取当前日志上下文"""
        if not hasattr(self._thread_local, 'context'):
            self._thread_local.context = LogContext()
        return self._thread_local.context

    def clear_context(self) -> None:
        """清空日志上下文"""
        if hasattr(self._thread_local, 'context'):
            self._thread_local.context.clear()

    def add_handler(self, handler: logging.Handler, format_type: str = "text") -> None:
        """添加自定义处理器

        Args:
            handler: 日志处理器
            format_type: 格式化类型 (text, json)
        """
        if format_type == "json":
            handler.setFormatter(StructuredFormatter())
        else:
            handler.setFormatter(TextFormatter())
        self._logger.addHandler(handler)
        self._handlers.append(handler)

    def remove_handler(self, handler: logging.Handler) -> None:
        """移除处理器"""
        if handler in self._handlers:
            self._logger.removeHandler(handler)
            self._handlers.remove(handler)

    def flush(self) -> None:
        """刷新所有处理器"""
        for handler in self._handlers:
            handler.flush()

    def close(self) -> None:
        """关闭所有处理器"""
        for handler in self._handlers:
            handler.close()
            self._logger.removeHandler(handler)
        self._handlers.clear()

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录指标（兼容遗留接口）

        Args:
            metrics: 指标字典，如 {"acc": 0.95, "loss": 0.1}
            step: 训练步数
        """
        step_str = f"[step={step}]" if step is not None else ""
        for name, value in metrics.items():
            self.info(f"{step_str} {name}={value}")


def get_logger(name: str = "global") -> UnifiedLogger:
    """获取全局日志实例

    Args:
        name: 日志器名称（兼容遗留接口，单例模式下忽略）

    Returns:
        UnifiedLogger: 全局日志实例
    """
    return UnifiedLogger()


def get_unified_logger() -> UnifiedLogger:
    """获取全局日志实例（统一接口）

    Returns:
        UnifiedLogger: 全局日志实例
    """
    return UnifiedLogger()
