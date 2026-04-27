"""
test_training_modules.py — training/ 子模块快速冒烟测试
覆盖 train.py、evaluate.py 的公共 API
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


class TestTrainModule:
    """training/train.py 基础测试"""

    def test_import(self):
        from training.train import (
            TrainingPipeline,
            EMAModel,
            CosineAnnealingWarmup,
            LovaszLoss,
            get_loss_fn,
        )
        assert TrainingPipeline is not None
        assert EMAModel is not None
        assert CosineAnnealingWarmup is not None
        assert LovaszLoss is not None
        assert get_loss_fn is not None

    def test_ema_model_initial_state(self):
        import torch
        from training.train import EMAModel
        model = torch.nn.Linear(10, 2)
        ema = EMAModel(model, decay=0.999)
        assert ema.decay == 0.999

    def test_cosine_warmup_schedulers(self):
        import torch
        from training.train import CosineAnnealingWarmup
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = CosineAnnealingWarmup(optimizer, warmup_epochs=5, total_epochs=100)
        assert scheduler is not None

    def test_get_loss_fn_exists(self):
        from training.train import get_loss_fn
        # 验证函数存在且可调用
        assert callable(get_loss_fn)


class TestEvaluateModule:
    """training/evaluate.py 基础测试"""

    def test_import(self):
        from training.evaluate import (
            Evaluator,
            compute_iou,
            compute_dice,
        )
        assert Evaluator is not None
        assert compute_iou is not None
        assert compute_dice is not None

    def test_compute_iou_basic(self):
        import torch
        from training.evaluate import compute_iou
        pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        iou = compute_iou(pred, target)
        assert isinstance(iou, float)
        assert 0.0 <= iou <= 1.0

    def test_compute_dice_basic(self):
        import torch
        from training.evaluate import compute_dice
        pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dice = compute_dice(pred, target)
        assert isinstance(dice, float)
        assert 0.0 <= dice <= 1.0
