"""
core/logging.py — 统一结构化日志

提取并合并自：
- results/auto_tuning/iteration_44/work_logging_system.py
"""

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "LogLevel",
    "LogFormat",
    "LogHandlerConfig",
    "LogHandler",
    "LogFormatter",
    "LogContext",
    "StructuredLogger",
    "LogManager",
    "get_logger",
]


class LogLevel(Enum):
    """日志级别"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogFormat(Enum):
    """日志格式"""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"


@dataclass
class LogHandlerConfig:
    """处理器配置"""
    name: str = "default"
    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.TEXT
    console: bool = True
    file_path: Optional[str] = None
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


class LogHandler:
    """日志处理器，管理输出目标。"""

    def __init__(self, config: Optional[LogHandlerConfig] = None):
        self.config = config or LogHandlerConfig()
        self.handlers: List[logging.Handler] = []
        self._setup_handlers()

    def _setup_handlers(self):
        if self.config.console:
            h = logging.StreamHandler(sys.stdout)
            h.setLevel(self.config.level.value)
            self.handlers.append(h)
        if self.config.file_path:
            h = self._create_file_handler()
            if h:
                self.handlers.append(h)

    def _create_file_handler(self) -> Optional[logging.Handler]:
        log_dir = os.path.dirname(self.config.file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        from logging.handlers import RotatingFileHandler
        h = RotatingFileHandler(
            self.config.file_path,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count,
        )
        h.setLevel(self.config.level.value)
        return h

    def get_handlers(self) -> List[logging.Handler]:
        return self.handlers


class LogFormatter(logging.Formatter):
    """支持 TEXT/JSON/CSV 的日志格式化器。"""

    def __init__(self, format_type: LogFormat = LogFormat.TEXT):
        self.format_type = format_type
        if format_type == LogFormat.TEXT:
            super().__init__(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        if self.format_type == LogFormat.TEXT:
            return super().format(record)
        elif self.format_type == LogFormat.JSON:
            return self._format_json(record)
        elif self.format_type == LogFormat.CSV:
            return self._format_csv(record)
        return super().format(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "extra_fields"):
            data.update(record.extra_fields)
        if record.exc_info:
            data["exception"] = traceback.format_exception(*record.exc_info)
        return json.dumps(data, ensure_ascii=False)

    def _format_csv(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        return f"{ts},{record.levelname},{record.name},{record.module},{record.funcName},{record.lineno},{record.getMessage()}"


@dataclass
class LogContext:
    """日志上下文"""
    run_id: Optional[str] = None
    experiment: Optional[str] = None
    phase: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    """结构化日志器，支持上下文和指标记录。"""

    def __init__(
        self,
        name: str,
        handlers: Optional[List[LogHandler]] = None,
        default_level: LogLevel = LogLevel.INFO,
    ):
        self.name = name
        self.context = LogContext()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(default_level.value)
        self.logger.handlers.clear()

        if handlers:
            for handler_config in handlers:
                for h in handler_config.get_handlers():
                    h.setFormatter(LogFormatter(handler_config.config.format))
                    self.logger.addHandler(h)
        else:
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(LogFormatter())
            self.logger.addHandler(h)

    def _log(self, level: int, message: str, *args, **kwargs):
        extra_fields = kwargs.pop("extra", {})
        log_context = {
            "run_id": self.context.run_id,
            "experiment": self.context.experiment,
            "phase": self.context.phase,
            **self.context.extra,
            **extra_fields,
        }
        log_context = {k: v for k, v in log_context.items() if v is not None}

        record = self.logger.makeRecord(
            self.logger.name, level, "(unknown)", 0, message, args, None
        )
        record.extra_fields = log_context
        self.logger.handle(record)

    def debug(self, message: str, *args, **kwargs):
        self._log(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        self._log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self._log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self._log(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        self._log(logging.CRITICAL, message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        kwargs["exc_info"] = True
        self._log(logging.ERROR, message, *args, **kwargs)

    def set_context(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.extra[key] = value

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        data = {"metrics": metrics}
        if step is not None:
            data["step"] = step
        self.info(f"Metrics: {json.dumps(metrics)}", extra=data)

    def log_config(self, config: Dict):
        self.info(f"Config: {json.dumps(config)}", extra={"config": config})


class LogManager:
    """日志管理器单例，统一管理所有日志器。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loggers: Dict[str, StructuredLogger] = {}
            cls._instance._default_config: Optional[LogHandlerConfig] = None
        return cls._instance

    def set_default_config(self, config: LogHandlerConfig):
        self._default_config = config

    def get_logger(
        self,
        name: str,
        config: Optional[LogHandlerConfig] = None,
        level: LogLevel = LogLevel.INFO,
    ) -> StructuredLogger:
        if name in self._loggers:
            return self._loggers[name]
        cfg = config or self._default_config
        if cfg:
            handler = LogHandler(cfg)
            logger = StructuredLogger(name, [handler], level)
        else:
            logger = StructuredLogger(name, default_level=level)
        self._loggers[name] = logger
        return logger

    def add_file_handler(
        self,
        name: str,
        file_path: str,
        level: LogLevel = LogLevel.INFO,
        fmt: LogFormat = LogFormat.JSON,
    ):
        if name not in self._loggers:
            self.get_logger(name)
        logger = self._loggers[name]
        cfg = LogHandlerConfig(
            name=f"{name}_file",
            level=level,
            format=fmt,
            console=False,
            file_path=file_path,
        )
        handler = LogHandler(cfg)
        for h in handler.get_handlers():
            h.setFormatter(LogFormatter(fmt))
            logger.logger.addHandler(h)

    def shutdown(self):
        for logger in self._loggers.values():
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)


_default_manager = LogManager()


def get_logger(
    name: str,
    config: Optional[LogHandlerConfig] = None,
    level: LogLevel = LogLevel.INFO,
) -> StructuredLogger:
    """获取日志器的便捷函数。"""
    return _default_manager.get_logger(name, config, level)
