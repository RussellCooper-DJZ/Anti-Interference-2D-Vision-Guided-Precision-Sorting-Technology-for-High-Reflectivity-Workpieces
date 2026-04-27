"""
core/checkpoint.py — 统一检查点管理

提取并合并自：
- results/auto_tuning/iteration_195/work_checkpoint_manager.py
- results/auto_tuning/iteration_38/work_early_stopping_v2.py (ModelCheckpoint)
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["CheckpointManager"]


class CheckpointManager:
    """检查点管理器：保存最优模型、早停控制、自动清理、加载恢复。"""

    def __init__(
        self,
        save_dir: str,
        patience: int = 10,
        keep_last: int = 3,
        mode: str = "max",
    ):
        """
        Args:
            save_dir: 检查点保存目录
            patience: 早停耐心值（score 不提升的轮数）
            keep_last: 保留最近检查点数量（不含 best）
            mode: "max" 或 "min"，决定 score 优化方向
        """
        self.save_dir = Path(save_dir)
        self.patience = patience
        self.keep_last = keep_last
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.checkpoints: List[str] = []
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def step(self, model, score: float, epoch: int, **extra) -> tuple[bool, bool]:
        """
        执行一步检查点逻辑。

        Returns:
            (saved, should_stop): 是否保存了新的最优检查点，是否应该停止训练
        """
        improved = (self.mode == "max" and score > self.best_score) or (
            self.mode == "min" and score < self.best_score
        )

        if improved:
            self.best_score = score
            self.counter = 0
            path = self.save_dir / f"best_epoch{epoch}_score{score:.4f}.pt"
            self._save(model, path, epoch, score, **extra)
            self.checkpoints.append(str(path))
            return True, False
        else:
            self.counter += 1
            should_stop = self.counter >= self.patience
            return False, should_stop

    def save_checkpoint(self, model, epoch: int, **extra) -> str:
        """保存一个普通检查点（非最优）。"""
        path = self.save_dir / f"checkpoint_epoch{epoch}.pt"
        self._save(model, path, epoch, None, **extra)
        self.checkpoints.append(str(path))
        self._prune_old()
        return str(path)

    def _save(self, model, path: Path, epoch: int, score: Optional[float], **extra):
        import torch
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "timestamp": datetime.now().isoformat(),
            **extra,
        }
        if score is not None:
            state["score"] = score
        torch.save(state, path)

    def _prune_old(self):
        """清理旧检查点，仅保留 keep_last 个非 best 检查点。"""
        regular = [c for c in self.checkpoints if "best" not in c]
        if len(regular) > self.keep_last:
            to_remove = regular[: len(regular) - self.keep_last]
            for old in to_remove:
                if os.path.exists(old):
                    os.remove(old)
                if old in self.checkpoints:
                    self.checkpoints.remove(old)

    def load_best(self, model) -> Optional[Dict[str, Any]]:
        """加载最优检查点。"""
        import torch
        best = [c for c in self.checkpoints if "best" in c]
        if not best:
            return None
        ckpt = torch.load(best[-1], map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt

    def load_checkpoint(self, model, path: str) -> Dict[str, Any]:
        """加载指定检查点。"""
        import torch
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt
