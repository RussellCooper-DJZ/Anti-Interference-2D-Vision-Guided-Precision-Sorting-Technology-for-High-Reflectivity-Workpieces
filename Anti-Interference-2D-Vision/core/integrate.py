"""
core/integrate.py — 统一抽象层集成初始化模块

提供一站式初始化接口，统一启动所有核心组件：
- UnifiedConfig   (配置管理)
- UnifiedLogger   (日志)
- UnifiedMonitor  (指标监控)
- UnifiedCheckpoint (检查点)

用法::

    from core.integrate import initialize_core

    # 初始化所有组件（使用默认配置）
    initialize_core()

    # 或者自定义配置
    initialize_core(
        config_path="config.yaml",
        log_level="INFO",
        checkpoint_dir="./checkpoints",
        monitor_port=8080,
    )

    # 获取已初始化的组件
    from core import get_config, get_logger, get_monitor, get_checkpoint
"""

from typing import Optional, Dict, Any
import os
import logging

from .unified_config import UnifiedConfig, get_config
from .unified_logger import UnifiedLogger, get_logger
from .unified_monitor import UnifiedMonitor, get_monitor
from .unified_checkpoint import UnifiedCheckpoint, get_checkpoint


def initialize_core(
    config_path: Optional[str] = None,
    log_level: str = "INFO",
    log_dir: str = "./logs",
    checkpoint_dir: str = "checkpoints",
    monitor_port: int = 0,
    project_name: str = "AGEANet",
    enable_monitor: bool = False,
) -> Dict[str, Any]:
    """
    初始化所有统一抽象层组件

    Args:
        config_path: 配置文件路径 (YAML/JSON)
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_dir: 日志目录
        checkpoint_dir: 检查点目录
        monitor_port: 监控服务端口 (0=禁用)
        project_name: 项目名称
        enable_monitor: 是否启用监控服务

    Returns:
        包含所有组件实例的字典
    """
    components = {}

    # 1. 初始化配置管理器
    config = get_config()
    if config_path and os.path.exists(config_path):
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config.load_yaml(config_path)
        elif config_path.endswith('.json'):
            config.load_json(config_path)
    components['config'] = config

    # 2. 初始化日志管理器
    logger = get_logger()
    logger.setup(
        level=log_level,
        output="console" if not log_dir else "rotating",
        file_path=os.path.join(log_dir, f"{project_name}.log") if log_dir else None,
    )
    components['logger'] = logger

    # 3. 初始化检查点管理器
    checkpoint = get_checkpoint(checkpoint_dir=checkpoint_dir)
    components['checkpoint'] = checkpoint

    # 4. 初始化监控管理器 (可选)
    if enable_monitor:
        monitor = get_monitor(port=monitor_port)
        components['monitor'] = monitor

    return components


def get_core_status() -> Dict[str, bool]:
    """
    获取各组件初始化状态

    Returns:
        各组件是否已初始化的状态字典
    """
    status = {}

    try:
        config = get_config()
        status['config'] = hasattr(config, '_initialized') and config._initialized
    except Exception:
        status['config'] = False

    try:
        logger = get_logger()
        status['logger'] = hasattr(logger, '_initialized') and logger._initialized
    except Exception:
        status['logger'] = False

    try:
        checkpoint = get_checkpoint()
        status['checkpoint'] = hasattr(checkpoint, '_initialized') and checkpoint._initialized
    except Exception:
        status['checkpoint'] = False

    try:
        monitor = get_monitor()
        status['monitor'] = hasattr(monitor, '_initialized') and monitor._initialized
    except Exception:
        status['monitor'] = False

    return status


__all__ = [
    "initialize_core",
    "get_core_status",
]
