"""
core — 公共基础设施包

提取自多个迭代中的重复实现，统一提供：
- 配置管理 (UnifiedConfig)
- 结构化日志 (UnifiedLogger)
- 检查点管理 (UnifiedCheckpoint)
- 指标监控 (UnifiedMonitor)

迁移指南（Legacy → Unified）:
- ConfigManager  → UnifiedConfig
- StructuredLogger / LogManager → UnifiedLogger
- CheckpointManager → UnifiedCheckpoint
- MetricsCollector / AlertManager → UnifiedMonitor

Legacy 接口将在 v0.7.0 移除，请尽快迁移。
"""

# Unified modules (primary interface) — 推荐优先使用
from .unified_config import UnifiedConfig, get_config
from .unified_logger import UnifiedLogger, get_logger, get_unified_logger
from .unified_monitor import UnifiedMonitor, get_monitor
from .unified_checkpoint import UnifiedCheckpoint, get_checkpoint

# Legacy modules (backward compatibility) — 已弃用，将在 v0.7.0 移除
from .config import ConfigManager, ConfigValidator, EnvInterpolator, ConfigMergeStrategy, ValidationRule
from .logging import get_logger as _legacy_get_logger, StructuredLogger, LogManager, LogLevel, LogFormat, LogHandler, LogHandlerConfig
from .checkpoint import CheckpointManager
from .monitoring import MetricsCollector, MetricType, MetricRecord, AlertManager, AlertRule, AlertSeverity

# Integration module
from .integrate import initialize_core, get_core_status

__all__ = [
    # Unified modules (primary interface) — 推荐
    "UnifiedConfig",
    "get_config",
    "UnifiedLogger",
    "get_logger",
    "get_unified_logger",
    "UnifiedMonitor",
    "get_monitor",
    "UnifiedCheckpoint",
    "get_checkpoint",
    # Legacy modules (deprecated, will be removed in v0.7.0)
    "ConfigManager",
    "ConfigValidator",
    "EnvInterpolator",
    "ConfigMergeStrategy",
    "ValidationRule",
    "StructuredLogger",
    "LogManager",
    "LogLevel",
    "LogFormat",
    "LogHandler",
    "LogHandlerConfig",
    "CheckpointManager",
    "MetricsCollector",
    "MetricType",
    "MetricRecord",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    # Integration module
    "initialize_core",
    "get_core_status",
]
