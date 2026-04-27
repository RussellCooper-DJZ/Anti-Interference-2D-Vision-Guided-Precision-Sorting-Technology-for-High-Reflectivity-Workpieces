"""
测试 UnifiedCheckpoint 模块
"""
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# 需要先设置路径
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.unified_checkpoint import (
    Checkpoint,
    UnifiedCheckpoint,
    get_checkpoint,
)


class SimpleModel(nn.Module):
    """用于测试的简单模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestCheckpoint:
    """测试 Checkpoint 数据类"""

    def test_checkpoint_creation(self):
        """测试检查点创建"""
        checkpoint = Checkpoint(
            version="v1_test",
            path="/path/to/checkpoint.pt",
            score=0.95,
            metadata={"epoch": 1},
        )
        assert checkpoint.version == "v1_test"
        assert checkpoint.path == "/path/to/checkpoint.pt"
        assert checkpoint.score == 0.95
        assert checkpoint.is_best is False

    def test_checkpoint_to_dict(self):
        """测试检查点序列化"""
        checkpoint = Checkpoint(
            version="v1_test",
            path="/path/to/checkpoint.pt",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            score=0.95,
            metadata={"key": "value"},
        )
        data = checkpoint.to_dict()
        assert data["version"] == "v1_test"
        assert data["score"] == 0.95
        assert data["metadata"] == {"key": "value"}
        assert "2024-01-01" in data["timestamp"]

    def test_checkpoint_from_dict(self):
        """测试检查点反序列化"""
        data = {
            "version": "v1_test",
            "path": "/path/to/checkpoint.pt",
            "timestamp": "2024-01-01T12:00:00",
            "score": 0.95,
            "metadata": {"key": "value"},
            "is_best": True,
        }
        checkpoint = Checkpoint.from_dict(data)
        assert checkpoint.version == "v1_test"
        assert checkpoint.score == 0.95
        assert checkpoint.is_best is True
        assert checkpoint.timestamp == datetime(2024, 1, 1, 12, 0, 0)


class TestUnifiedCheckpoint:
    """测试 UnifiedCheckpoint 管理器"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """创建检查点管理器实例"""
        # 清除单例状态以便测试
        UnifiedCheckpoint._instance = None
        manager = UnifiedCheckpoint(temp_dir)
        yield manager
        # 清理
        UnifiedCheckpoint._instance = None

    @pytest.fixture
    def simple_model(self):
        """创建简单模型用于测试"""
        return SimpleModel()

    def test_singleton_pattern(self, temp_dir):
        """测试单例模式"""
        UnifiedCheckpoint._instance = None
        manager1 = UnifiedCheckpoint(temp_dir)
        manager2 = UnifiedCheckpoint(temp_dir)
        assert manager1 is manager2

    def test_singleton_initialized_flag(self, temp_dir):
        """测试初始化标志"""
        UnifiedCheckpoint._instance = None
        manager1 = UnifiedCheckpoint(temp_dir)
        manager2 = UnifiedCheckpoint(temp_dir)
        # 第二次初始化应该直接返回
        manager2.__init__(temp_dir + "_other")
        assert manager1.checkpoint_dir == manager2.checkpoint_dir

    def test_save_basic(self, checkpoint_manager, simple_model, temp_dir):
        """测试基本保存功能"""
        checkpoint = checkpoint_manager.save(simple_model, epoch=1)
        assert checkpoint.version is not None
        assert "v1_" in checkpoint.version
        assert os.path.exists(checkpoint.path)
        assert checkpoint.path.startswith(temp_dir)

    def test_save_with_optimizer(self, checkpoint_manager, simple_model):
        """测试保存带优化器"""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        checkpoint = checkpoint_manager.save(simple_model, optimizer, epoch=1)
        assert checkpoint is not None

        # 加载验证
        state = checkpoint_manager.load(checkpoint.version)
        assert state is not None
        assert "optimizer_state_dict" in state

    def test_save_with_score(self, checkpoint_manager, simple_model):
        """测试保存带分数"""
        checkpoint = checkpoint_manager.save(simple_model, epoch=1, score=0.85)
        assert checkpoint.score == 0.85

    def test_save_with_metadata(self, checkpoint_manager, simple_model):
        """测试保存带元数据"""
        metadata = {"learning_rate": 0.01, "batch_size": 32}
        checkpoint = checkpoint_manager.save(
            simple_model, epoch=1, metadata=metadata
        )
        assert checkpoint.metadata == metadata

        loaded = checkpoint_manager.load(checkpoint.version)
        assert loaded["metadata"] == metadata

    def test_save_with_name(self, checkpoint_manager, simple_model):
        """测试保存带名称"""
        checkpoint = checkpoint_manager.save(
            simple_model, epoch=1, name="baseline"
        )
        assert "baseline" in checkpoint.version

    def test_best_checkpoint_tracking_max(self, checkpoint_manager, simple_model):
        """测试最佳检查点追踪（最大化模式）"""
        checkpoint_manager.set_score_mode("max")

        # 保存多个检查点
        ckpt1 = checkpoint_manager.save(simple_model, epoch=1, score=0.5)
        ckpt2 = checkpoint_manager.save(simple_model, epoch=2, score=0.8)
        ckpt3 = checkpoint_manager.save(simple_model, epoch=3, score=0.7)

        assert checkpoint_manager.get_best() is not None
        assert checkpoint_manager.get_best().score == 0.8
        assert checkpoint_manager.get_best().version == ckpt2.version

    def test_best_checkpoint_tracking_min(self, checkpoint_manager, simple_model):
        """测试最佳检查点追踪（最小化模式）"""
        checkpoint_manager.set_score_mode("min")

        ckpt1 = checkpoint_manager.save(simple_model, epoch=1, score=0.5)
        ckpt2 = checkpoint_manager.save(simple_model, epoch=2, score=0.3)
        ckpt3 = checkpoint_manager.save(simple_model, epoch=3, score=0.4)

        assert checkpoint_manager.get_best().score == 0.3
        assert checkpoint_manager.get_best().version == ckpt2.version

    def test_load_existing(self, checkpoint_manager, simple_model):
        """测试加载存在的检查点"""
        checkpoint = checkpoint_manager.save(simple_model, epoch=1)
        state = checkpoint_manager.load(checkpoint.version)
        assert state is not None
        assert "model_state_dict" in state
        assert state["epoch"] == 1

    def test_load_nonexistent(self, checkpoint_manager):
        """测试加载不存在的检查点"""
        state = checkpoint_manager.load("nonexistent_version")
        assert state is None

    def test_load_best(self, checkpoint_manager, simple_model):
        """测试加载最佳检查点"""
        checkpoint_manager.save(simple_model, epoch=1, score=0.5)
        checkpoint_manager.save(simple_model, epoch=2, score=0.9)
        checkpoint_manager.save(simple_model, epoch=3, score=0.7)

        state = checkpoint_manager.load_best()
        assert state is not None
        assert state["score"] == 0.9

    def test_load_best_when_no_best(self, checkpoint_manager):
        """测试没有最佳检查点时加载"""
        state = checkpoint_manager.load_best()
        assert state is None

    def test_list_checkpoints(self, checkpoint_manager, simple_model):
        """测试列出检查点"""
        checkpoint_manager.save(simple_model, epoch=1)
        checkpoint_manager.save(simple_model, epoch=2)
        checkpoint_manager.save(simple_model, epoch=3)

        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 3
        # 验证按时间排序（最新的在前）
        assert checkpoints[0].timestamp >= checkpoints[1].timestamp

    def test_get_latest(self, checkpoint_manager, simple_model):
        """测试获取最新检查点"""
        assert checkpoint_manager.get_latest() is None

        ckpt1 = checkpoint_manager.save(simple_model, epoch=1)
        ckpt2 = checkpoint_manager.save(simple_model, epoch=2)

        latest = checkpoint_manager.get_latest()
        assert latest is not None
        assert latest.version == ckpt2.version

    def test_get_best(self, checkpoint_manager, simple_model):
        """测试获取最佳检查点"""
        assert checkpoint_manager.get_best() is None

        checkpoint_manager.save(simple_model, epoch=1, score=0.5)
        checkpoint_manager.save(simple_model, epoch=2, score=0.9)

        best = checkpoint_manager.get_best()
        assert best is not None
        assert best.score == 0.9

    def test_delete_oldest(self, checkpoint_manager, simple_model):
        """测试删除旧检查点"""
        for i in range(10):
            checkpoint_manager.save(simple_model, epoch=i, score=0.5 + i * 0.01)

        assert len(checkpoint_manager.list_checkpoints()) == 10

        checkpoint_manager.delete_oldest(keep_count=3)

        remaining = checkpoint_manager.list_checkpoints()
        assert len(remaining) == 3

    def test_delete_oldest_keeps_best(self, checkpoint_manager, simple_model):
        """测试删除旧检查点时保留最佳"""
        checkpoint_manager.set_score_mode("max")
        for i in range(10):
            checkpoint_manager.save(simple_model, epoch=i, score=0.5 + i * 0.05)

        # 最佳应该是 epoch=9
        best_version = checkpoint_manager.get_best().version

        checkpoint_manager.delete_oldest(keep_count=3)

        # 最佳检查点应该仍然存在
        remaining = checkpoint_manager.list_checkpoints()
        versions = [c.version for c in remaining]
        assert best_version in versions

    def test_cleanup_keep_best_and_last(self, checkpoint_manager, simple_model):
        """测试清理保留最佳和最新"""
        checkpoint_manager.set_score_mode("max")
        for i in range(5):
            checkpoint_manager.save(simple_model, epoch=i, score=0.5 + i * 0.1)

        checkpoint_manager.cleanup(keep_best=True, keep_last=True)

        remaining = checkpoint_manager.list_checkpoints()
        # 应该保留最佳和最新（可能重复）
        assert len(remaining) >= 1

    def test_cleanup_keep_none(self, checkpoint_manager, simple_model):
        """测试清理不保留任何"""
        for i in range(3):
            checkpoint_manager.save(simple_model, epoch=i, score=0.5)

        checkpoint_manager.cleanup(keep_best=False, keep_last=False)

        remaining = checkpoint_manager.list_checkpoints()
        assert len(remaining) == 0

    def test_rollback(self, checkpoint_manager, simple_model):
        """测试回滚功能"""
        checkpoint = checkpoint_manager.save(simple_model, epoch=1)
        checkpoint_manager.save(simple_model, epoch=2)

        # 回滚到 epoch=1
        success = checkpoint_manager.rollback(checkpoint.version)
        assert success is True

        # 回滚到不存在的版本
        success = checkpoint_manager.rollback("nonexistent")
        assert success is False

    def test_set_score_mode(self, checkpoint_manager, simple_model):
        """测试设置评分模式"""
        # 初始默认应该是 max
        checkpoint_manager.save(simple_model, epoch=1, score=0.5)
        checkpoint_manager.save(simple_model, epoch=2, score=0.8)

        # 验证是 max 模式
        assert checkpoint_manager._score_mode == "max"

        # 切换到 min 模式
        checkpoint_manager.set_score_mode("min")

        # 保存一个更低的分数
        checkpoint_manager.save(simple_model, epoch=3, score=0.3)

        # 最佳应该更新为 0.3
        assert checkpoint_manager.get_best().score == 0.3

    def test_set_score_mode_invalid(self, checkpoint_manager):
        """测试设置无效的评分模式"""
        with pytest.raises(ValueError):
            checkpoint_manager.set_score_mode("invalid")

    def test_get_checkpoint_by_version(self, checkpoint_manager, simple_model):
        """测试根据版本号获取检查点"""
        checkpoint = checkpoint_manager.save(simple_model, epoch=1)

        found = checkpoint_manager.get_checkpoint_by_version(checkpoint.version)
        assert found is not None
        assert found.version == checkpoint.version

        not_found = checkpoint_manager.get_checkpoint_by_version("nonexistent")
        assert not_found is None

    def test_restore_model(self, checkpoint_manager, simple_model):
        """测试恢复模型"""
        # 修改模型
        original_weight = simple_model.linear.weight.clone()
        with torch.no_grad():
            simple_model.linear.weight.fill_(999.0)

        # 保存
        checkpoint_manager.save(simple_model, epoch=1)

        # 重置模型
        with torch.no_grad():
            simple_model.linear.weight.fill_(0.0)

        # 恢复
        success = checkpoint_manager.restore_model(simple_model)
        assert success is True

        # 验证权重恢复
        assert torch.allclose(simple_model.linear.weight, original_weight)

    def test_restore_model_with_optimizer(self, checkpoint_manager, simple_model):
        """测试恢复模型和优化器"""
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)

        # 执行一步训练
        x = torch.randn(5, 10)
        y = simple_model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # 保存
        checkpoint_manager.save(simple_model, optimizer, epoch=1)

        # 创建新模型和优化器
        new_model = SimpleModel()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)

        # 恢复
        success = checkpoint_manager.restore_model(
            new_model, load_optimizer=True, optimizer=new_optimizer
        )
        assert success is True

    def test_restore_model_specific_version(self, checkpoint_manager, simple_model):
        """测试恢复指定版本模型"""
        checkpoint1 = checkpoint_manager.save(simple_model, epoch=1)

        # 修改模型
        with torch.no_grad():
            simple_model.linear.weight.fill_(123.0)

        checkpoint2 = checkpoint_manager.save(simple_model, epoch=2)

        # 重置
        with torch.no_grad():
            simple_model.linear.weight.fill_(0.0)

        # 恢复到版本1
        success = checkpoint_manager.restore_model(simple_model, version=checkpoint1.version)
        assert success is True

        # 恢复到版本2
        success = checkpoint_manager.restore_model(simple_model, version=checkpoint2.version)
        assert success is True

    def test_persistence(self, temp_dir, simple_model):
        """测试检查点索引持久化"""
        UnifiedCheckpoint._instance = None
        manager1 = UnifiedCheckpoint(temp_dir)
        manager1.save(simple_model, epoch=1, score=0.8)
        manager1.save(simple_model, epoch=2, score=0.9)

        # 创建新的管理器实例（模拟重启）
        UnifiedCheckpoint._instance = None
        manager2 = UnifiedCheckpoint(temp_dir)

        assert len(manager2.list_checkpoints()) == 2
        assert manager2.get_best() is not None
        assert manager2.get_best().score == 0.9

    def test_corrupted_index_file(self, temp_dir, simple_model):
        """测试损坏的索引文件"""
        # 创建检查点管理器并保存一些数据
        UnifiedCheckpoint._instance = None
        manager1 = UnifiedCheckpoint(temp_dir)
        manager1.save(simple_model, epoch=1)

        # 写入损坏的索引文件
        index_path = manager1._index_file
        with open(index_path, "w") as f:
            f.write("{ invalid json }")

        # 重新初始化应该能处理损坏的索引
        UnifiedCheckpoint._instance = None
        manager2 = UnifiedCheckpoint(temp_dir)

        # 应该从损坏的索引中恢复为空列表
        assert len(manager2.list_checkpoints()) == 0


class TestGetCheckpoint:
    """测试 get_checkpoint 工厂函数"""

    def test_get_checkpoint(self, temp_dir):
        """测试获取检查点实例"""
        UnifiedCheckpoint._instance = None
        checkpoint = get_checkpoint(temp_dir)
        assert isinstance(checkpoint, UnifiedCheckpoint)
        UnifiedCheckpoint._instance = None
