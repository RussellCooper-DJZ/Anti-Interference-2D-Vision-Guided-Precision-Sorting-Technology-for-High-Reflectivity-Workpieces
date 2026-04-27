"""
training — 模型训练包

包含以下子模块：
  train: FLARE 端到端训练流程（损失函数、评估指标、学习率调度）
"""

__all__ = [
    "DiceLoss",
    "FocalLoss",
    "FLARELoss",
    "MetricTracker",
    "train_one_epoch",
    "validate",
]
