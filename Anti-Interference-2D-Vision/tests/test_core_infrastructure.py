"""
tests/test_core_infrastructure.py — 公共基础设施基类测试
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from core import (
    ConfigManager,
    ConfigValidator,
    EnvInterpolator,
    ValidationRule,
    CheckpointManager,
    MetricsCollector,
    AlertManager,
    AlertRule,
    AlertSeverity,
    get_logger,
)


class TestEnvInterpolator:
    def test_basic_interpolation(self):
        os.environ["TEST_VAR"] = "hello"
        interp = EnvInterpolator()
        assert interp.interpolate("${TEST_VAR}") == "hello"
        del os.environ["TEST_VAR"]

    def test_default_value(self):
        interp = EnvInterpolator()
        assert interp.interpolate("${NONEXISTENT:default}") == "default"

    def test_interpolate_dict(self):
        os.environ["TEST_NUM"] = "42"
        interp = EnvInterpolator()
        result = interp.interpolate_dict({"a": "${TEST_NUM}", "b": 1, "c": {"d": "${TEST_NUM}"}})
        assert result["a"] == "42"
        assert result["b"] == 1
        assert result["c"]["d"] == "42"
        del os.environ["TEST_NUM"]


class TestConfigValidator:
    def test_required(self):
        v = ConfigValidator([ValidationRule("model.backbone", "required")])
        assert not v.validate({})
        assert v.validate({"model": {"backbone": "resnet"}})

    def test_type_check(self):
        v = ConfigValidator([ValidationRule("batch_size", "type", int)])
        assert not v.validate({"batch_size": "32"})
        assert v.validate({"batch_size": 32})

    def test_range(self):
        v = ConfigValidator([ValidationRule("lr", "range", (1e-5, 1.0))])
        assert not v.validate({"lr": 2.0})
        assert v.validate({"lr": 0.01})

    def test_one_of(self):
        v = ConfigValidator([ValidationRule("optimizer", "one_of", ["adam", "sgd"])])
        assert not v.validate({"optimizer": "rmsprop"})
        assert v.validate({"optimizer": "adam"})


class TestConfigManager:
    def test_load_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            data = {"model": {"backbone": "resnet18"}, "lr": 0.001}
            with open(path, "w") as f:
                yaml.dump(data, f)
            mgr = ConfigManager()
            loaded = mgr.load(str(path), validate=False)
            assert loaded["model"]["backbone"] == "resnet18"
            assert loaded["lr"] == 0.001

    def test_load_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            data = {"batch_size": 16}
            with open(path, "w") as f:
                json.dump(data, f)
            mgr = ConfigManager()
            loaded = mgr.load(str(path), validate=False)
            assert loaded["batch_size"] == 16

    def test_env_override(self):
        os.environ["FLARE_TEST_KEY"] = "99"
        mgr = ConfigManager()
        loaded = mgr.load([], validate=False)
        assert loaded["test"]["key"] == 99
        del os.environ["FLARE_TEST_KEY"]

    def test_get_and_set(self):
        mgr = ConfigManager()
        mgr.load([], validate=False)
        mgr._config = {"a": {"b": 1}}
        assert mgr.get("a.b") == 1
        assert mgr.get("a.c", 2) == 2
        mgr.set("a.c", 3)
        assert mgr.get("a.c") == 3

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ConfigManager()
            mgr._config = {"x": 1}
            path = Path(tmpdir) / "out.json"
            mgr.save(str(path))
            with open(path) as f:
                assert json.load(f)["x"] == 1


class TestCheckpointManager:
    def test_step_improvement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, patience=2)
            model = torch.nn.Conv2d(3, 1, 3, padding=1)
            saved, stop = mgr.step(model, 0.5, 0)
            assert saved is True
            assert stop is False
            saved, stop = mgr.step(model, 0.6, 1)
            assert saved is True
            assert stop is False

    def test_early_stopping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, patience=2)
            model = torch.nn.Conv2d(3, 1, 3, padding=1)
            mgr.step(model, 0.5, 0)
            saved, stop = mgr.step(model, 0.4, 1)
            assert saved is False
            assert stop is False
            saved, stop = mgr.step(model, 0.3, 2)
            assert saved is False
            assert stop is True

    def test_load_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, patience=2)
            model = torch.nn.Conv2d(3, 1, 3, padding=1)
            mgr.step(model, 0.5, 0)
            ckpt = mgr.load_best(model)
            assert ckpt is not None
            assert ckpt["epoch"] == 0

    def test_min_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(tmpdir, patience=2, mode="min")
            model = torch.nn.Conv2d(3, 1, 3, padding=1)
            saved, stop = mgr.step(model, 0.5, 0)
            assert saved is True
            saved, stop = mgr.step(model, 0.4, 1)
            assert saved is True


class TestMetricsCollector:
    def test_log_and_get(self):
        collector = MetricsCollector()
        collector.log("loss", 0.5, step=0)
        collector.log("loss", 0.4, step=1)
        history = collector.get_history("loss")
        assert len(history) == 2
        assert history[0].value == 0.5
        assert collector.get_average("loss") == 0.45

    def test_log_dict(self):
        collector = MetricsCollector()
        collector.log_dict({"acc": 0.9, "loss": 0.1}, step=0)
        assert collector.get_latest("acc").value == 0.9
        assert collector.get_latest("loss").value == 0.1

    def test_summary(self):
        collector = MetricsCollector()
        collector.log("x", 1.0)
        collector.log("x", 3.0)
        s = collector.summary()["x"]
        assert s["mean"] == 2.0
        assert s["min"] == 1.0
        assert s["max"] == 3.0


class TestAlertManager:
    def test_rule_trigger(self):
        mgr = AlertManager()
        rule = AlertRule("high_loss", "loss", "gt", 1.0, severity=AlertSeverity.ERROR)
        mgr.add_rule(rule)
        mgr.check({"loss": 1.5})
        active = mgr.get_active_alerts()
        assert len(active) == 1
        assert active[0].severity == AlertSeverity.ERROR

    def test_rule_no_trigger(self):
        mgr = AlertManager()
        rule = AlertRule("high_loss", "loss", "gt", 1.0)
        mgr.add_rule(rule)
        mgr.check({"loss": 0.5})
        assert len(mgr.get_active_alerts()) == 0

    def test_duration(self):
        mgr = AlertManager()
        rule = AlertRule("high_loss", "loss", "gt", 1.0, duration=2)
        mgr.add_rule(rule)
        mgr.check({"loss": 1.5})
        assert len(mgr.get_active_alerts()) == 0
        mgr.check({"loss": 1.5})
        assert len(mgr.get_active_alerts()) == 1

    def test_callback(self):
        alerts = []
        mgr = AlertManager()
        mgr.add_callback(lambda a: alerts.append(a))
        rule = AlertRule("high_loss", "loss", "gt", 1.0)
        mgr.add_rule(rule)
        mgr.check({"loss": 1.5})
        assert len(alerts) == 1


class TestLogger:
    def test_get_logger(self):
        logger = get_logger("test_logger")
        logger.info("test message")

    def test_context(self):
        logger = get_logger("ctx_logger")
        logger.set_context(run_id="r1", experiment="exp1")
        logger.info("with context", extra={"custom": 1})

    def test_log_metrics(self):
        logger = get_logger("metrics_logger")
        logger.log_metrics({"acc": 0.95}, step=10)
