# 集成方案

## 项目背景
AGEANet 项目需要统一视觉流水线的核心基础设施组件，将分散的配置文件、日志、监控和检查点管理整合到统一抽象层。

## 需要修改的文件
1. `main_pipeline.py` - 添加统一抽象层初始化
2. `core/integrate.py` - 新建集成初始化模块

## 替换清单

| 旧模块 | 新模块 | 说明 |
|--------|--------|------|
| ConfigManager | UnifiedConfig | 统一配置管理，支持 YAML/JSON/ENV/命令行参数 |
| LogManager | UnifiedLogger | 结构化日志，支持多级别、多输出、日志上下文 |
| CheckpointManager | UnifiedCheckpoint | 检查点管理，支持保存/加载模型状态 |
| MetricsCollector | UnifiedMonitor | 指标监控，支持指标收集和告警 |

## 架构说明

```
core/
├── __init__.py          # 统一导出
├── integrate.py         # [新建] 集成初始化模块
├── unified_config.py    # UnifiedConfig
├── unified_logger.py    # UnifiedLogger
├── unified_monitor.py   # UnifiedMonitor
├── unified_checkpoint.py# UnifiedCheckpoint
├── config.py            # ConfigManager (legacy, backward compat)
├── logging.py           # LogManager (legacy, backward compat)
├── checkpoint.py        # CheckpointManager (legacy, backward compat)
└── monitoring.py        # MetricsCollector (legacy, backward compat)
```

## 注意事项
- 保持向后兼容：legacy 模块保留在 core/ 中
- 单一数据源原则：统一模块使用单例模式
- vision/ 模块为纯算法模块，无需修改
- 测试文件不受影响

## 验证步骤
```bash
python -c "from core import UnifiedConfig, UnifiedLogger, UnifiedMonitor, UnifiedCheckpoint; print('OK')"
python main_pipeline.py --help 2>&1 | head -5
```
