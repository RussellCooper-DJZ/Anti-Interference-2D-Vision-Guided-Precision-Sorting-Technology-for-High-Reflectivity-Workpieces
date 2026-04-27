"""
UnifiedCheckpoint - 统一检查点模块
支持模型保存/恢复、配置快照、回滚、版本管理
"""
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os
import json
import shutil
import threading
from pathlib import Path
import torch


@dataclass
class Checkpoint:
    """检查点数据"""
    version: str
    path: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于序列化"""
        return {
            "version": self.version,
            "path": self.path,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "score": self.score,
            "is_best": self.is_best,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """从字典反序列化"""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class UnifiedCheckpoint:
    """统一检查点管理器（单例）"""
    _instance: Optional['UnifiedCheckpoint'] = None
    _lock = threading.Lock()

    def __new__(cls, checkpoint_dir: str = "checkpoints") -> 'UnifiedCheckpoint':
        if cls._instance is None:  # 外层检查（快速路径）
            with cls._lock:
                if cls._instance is None:  # 内层检查（加锁保护）
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        if self._initialized:
            return
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: List[Checkpoint] = []
        self._best_checkpoint: Optional[Checkpoint] = None
        self._internal_lock = threading.Lock()
        self._index_file = self.checkpoint_dir / "checkpoint_index.json"
        self._score_mode = "max"  # "max" or "min"
        self._initialized = True
        self._load_index()

    def _load_index(self) -> None:
        """从索引文件加载检查点列表"""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._checkpoints = [
                        Checkpoint.from_dict(c) for c in data.get("checkpoints", [])
                    ]
                    best_version = data.get("best_version")
                    if best_version:
                        self._best_checkpoint = next(
                            (c for c in self._checkpoints if c.version == best_version),
                            None
                        )
                    self._score_mode = data.get("score_mode", "max")
            except (json.JSONDecodeError, KeyError):
                self._checkpoints = []
                self._best_checkpoint = None

    def _save_index(self) -> None:
        """保存检查点索引到文件"""
        with self._internal_lock:
            data = {
                "checkpoints": [c.to_dict() for c in self._checkpoints],
                "best_version": self._best_checkpoint.version if self._best_checkpoint else None,
                "score_mode": self._score_mode,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self._index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def _generate_version(self, epoch: int, name: Optional[str] = None) -> str:
        """生成检查点版本号"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if name:
            return f"v{epoch}_{name}_{timestamp}"
        return f"v{epoch}_{timestamp}"

    def _update_best(self, checkpoint: Checkpoint, score: Optional[float]) -> None:
        """更新最佳检查点"""
        if score is None:
            return

        # 清除之前最佳标记
        if self._best_checkpoint:
            self._best_checkpoint.is_best = False

        # 判断是否需要更新最佳
        should_update = False
        if self._best_checkpoint is None:
            should_update = True
        elif self._score_mode == "max" and score > (self._best_checkpoint.score or float("-inf")):
            should_update = True
        elif self._score_mode == "min" and score < (self._best_checkpoint.score or float("inf")):
            should_update = True

        if should_update:
            checkpoint.is_best = True
            self._best_checkpoint = checkpoint

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> Checkpoint:
        """
        保存检查点

        Args:
            model: 要保存的模型
            optimizer: 优化器状态（可选）
            epoch: 当前训练轮次
            score: 评估分数（用于最佳检查点追踪）
            metadata: 额外元数据
            name: 检查点名称

        Returns:
            Checkpoint: 保存的检查点信息
        """
        with self._internal_lock:
            version = self._generate_version(epoch, name)
            checkpoint_path = self.checkpoint_dir / f"{version}.pt"

            # 构建保存状态
            state: Dict[str, Any] = {
                "version": version,
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(),
            }

            if metadata:
                state["metadata"] = metadata

            # 保存模型状态
            state["model_state_dict"] = model.state_dict()

            if optimizer is not None:
                state["optimizer_state_dict"] = optimizer.state_dict()

            if score is not None:
                state["score"] = score

            # 保存检查点文件
            torch.save(state, checkpoint_path)

            # 创建检查点记录
            checkpoint = Checkpoint(
                version=version,
                path=str(checkpoint_path),
                timestamp=datetime.now(),
                metadata=metadata or {},
                score=score,
                is_best=False,
            )

            # 更新最佳检查点
            self._update_best(checkpoint, score)

            # 添加到列表
            self._checkpoints.append(checkpoint)

            # 排序：按时间戳从新到旧
            self._checkpoints.sort(key=lambda c: c.timestamp, reverse=True)

            # 保存索引
            self._save_index()

            return checkpoint

    def load(self, version: str) -> Optional[Dict[str, Any]]:
        """
        加载指定版本的检查点

        Args:
            version: 检查点版本号

        Returns:
            Dict containing checkpoint state, or None if not found
        """
        with self._internal_lock:
            checkpoint = next((c for c in self._checkpoints if c.version == version), None)
            if checkpoint is None:
                return None

            if not os.path.exists(checkpoint.path):
                return None

            state = torch.load(checkpoint.path, map_location="cpu")
            return state

    def load_best(self) -> Optional[Dict[str, Any]]:
        """
        加载最佳检查点

        Returns:
            Dict containing best checkpoint state, or None if no best exists
        """
        with self._internal_lock:
            if self._best_checkpoint is None:
                return None

            if not os.path.exists(self._best_checkpoint.path):
                return None

            state = torch.load(self._best_checkpoint.path, map_location="cpu")
            return state

    def list_checkpoints(self) -> List[Checkpoint]:
        """
        列出所有检查点（按时间戳从新到旧排序）

        Returns:
            List of Checkpoint objects
        """
        with self._internal_lock:
            return sorted(self._checkpoints.copy(), key=lambda c: c.timestamp, reverse=True)

    def delete_oldest(self, keep_count: int = 5) -> None:
        """
        删除旧检查点，保留最近的keep_count个

        Args:
            keep_count: 保留的最近检查点数量
        """
        with self._internal_lock:
            if len(self._checkpoints) <= keep_count:
                return

            # 排序：按时间戳从旧到新
            sorted_checkpoints = sorted(self._checkpoints, key=lambda c: c.timestamp)

            # 保留最新的keep_count个和最佳检查点
            to_delete = []
            best_version = self._best_checkpoint.version if self._best_checkpoint else None

            for checkpoint in sorted_checkpoints:
                if checkpoint.version == best_version:
                    continue
                if len(self._checkpoints) - len(to_delete) <= keep_count:
                    break
                to_delete.append(checkpoint)

            # 删除文件并更新列表
            for checkpoint in to_delete:
                if os.path.exists(checkpoint.path):
                    os.remove(checkpoint.path)
                if checkpoint in self._checkpoints:
                    self._checkpoints.remove(checkpoint)

            self._save_index()

    def cleanup(self, keep_best: bool = True, keep_last: bool = True) -> None:
        """
        清理检查点

        Args:
            keep_best: 是否保留最佳检查点
            keep_last: 是否保留最新的检查点
        """
        with self._internal_lock:
            if len(self._checkpoints) == 0:
                return

            to_delete = []

            if keep_best and self._best_checkpoint:
                # 保留最佳检查点
                pass

            if keep_last:
                # 保留最新的一个
                latest = self._checkpoints[0] if self._checkpoints else None
                if latest and latest.version != (self._best_checkpoint.version if self._best_checkpoint else None):
                    to_delete = [c for c in self._checkpoints[1:] if c.version != (self._best_checkpoint.version if self._best_checkpoint else None)]
            else:
                to_delete = list(self._checkpoints)

            # 删除
            for checkpoint in to_delete:
                if keep_best and checkpoint.is_best:
                    continue
                if os.path.exists(checkpoint.path):
                    os.remove(checkpoint.path)
                if checkpoint in self._checkpoints:
                    self._checkpoints.remove(checkpoint)

            self._save_index()

    def get_latest(self) -> Optional[Checkpoint]:
        """
        获取最新检查点

        Returns:
            Latest Checkpoint or None if no checkpoints exist
        """
        with self._internal_lock:
            if not self._checkpoints:
                return None
            return max(self._checkpoints, key=lambda c: c.timestamp)

    def get_best(self) -> Optional[Checkpoint]:
        """
        获取最佳检查点

        Returns:
            Best Checkpoint or None if no best exists
        """
        with self._internal_lock:
            return self._best_checkpoint

    def rollback(self, version: str) -> bool:
        """
        回滚到指定版本（通过返回状态字典，由调用者执行模型加载）

        Args:
            version: 要回滚到的版本号

        Returns:
            bool: 是否成功获取该版本的检查点状态
        """
        state = self.load(version)
        return state is not None

    def set_score_mode(self, mode: str) -> None:
        """
        设置评分模式

        Args:
            mode: "max" 或 "min"
        """
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        with self._internal_lock:
            self._score_mode = mode
            # 重新评估最佳检查点
            self._best_checkpoint = None
            for checkpoint in self._checkpoints:
                checkpoint.is_best = False
                self._update_best(checkpoint, checkpoint.score)
            self._save_index()

    def get_checkpoint_by_version(self, version: str) -> Optional[Checkpoint]:
        """
        根据版本号获取检查点信息

        Args:
            version: 检查点版本号

        Returns:
            Checkpoint object or None
        """
        with self._internal_lock:
            return next((c for c in self._checkpoints if c.version == version), None)

    def restore_model(
        self,
        model: torch.nn.Module,
        version: Optional[str] = None,
        load_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> bool:
        """
        恢复模型状态

        Args:
            model: 要加载状态的模型
            version: 检查点版本号（None表示加载最佳）
            load_optimizer: 是否加载优化器状态
            optimizer: 优化器（如果load_optimizer为True）

        Returns:
            bool: 是否成功加载
        """
        if version:
            state = self.load(version)
        else:
            state = self.load_best()

        if state is None:
            return False

        model.load_state_dict(state.get("model_state_dict", {}))

        if load_optimizer and optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        return True


def get_checkpoint(checkpoint_dir: str = "checkpoints") -> UnifiedCheckpoint:
    """获取全局检查点实例"""
    return UnifiedCheckpoint(checkpoint_dir)
