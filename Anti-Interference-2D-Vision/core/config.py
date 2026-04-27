"""
core/config.py — 统一配置管理

提取并合并自：
- results/auto_tuning/iteration_43/work_config_manager.py
"""

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

__all__ = [
    "ConfigMergeStrategy",
    "EnvInterpolator",
    "ValidationRule",
    "ConfigValidator",
    "ConfigManager",
]


class ConfigMergeStrategy(Enum):
    """配置合并策略"""
    OVERRIDE = "override"
    KEEP = "keep"
    MERGE = "merge"


class EnvInterpolator:
    """环境变量插值器，支持 ${VAR} 和 ${VAR:default} 格式。"""

    def __init__(self):
        self.env_pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

    def interpolate(self, value: str) -> str:
        if not isinstance(value, str):
            return value

        def replace(match):
            var_name = match.group(1)
            default_value = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            return match.group(0)

        return self.env_pattern.sub(replace, value)

    def interpolate_dict(self, config: Dict) -> Dict:
        result = {}
        for key, value in config.items():
            if isinstance(value, str):
                result[key] = self.interpolate(value)
            elif isinstance(value, dict):
                result[key] = self.interpolate_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.interpolate(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


@dataclass
class ValidationRule:
    """验证规则"""
    path: str
    rule_type: str
    expected: Any = None
    message: str = ""


class ConfigValidator:
    """配置验证器，支持 required / type / range / one_of 规则。"""

    def __init__(self, rules: Optional[List[ValidationRule]] = None):
        self.rules = rules or []
        self.errors: List[str] = []

    def add_rule(self, rule: ValidationRule):
        self.rules.append(rule)

    def validate(self, config: Dict) -> bool:
        self.errors.clear()
        for rule in self.rules:
            self._validate_rule(config, rule)
        return len(self.errors) == 0

    def _validate_rule(self, config: Dict, rule: ValidationRule):
        value = self._get_nested_value(config, rule.path)
        if rule.rule_type == "required" and value is None:
            self.errors.append(rule.message or f"必需配置项缺失: {rule.path}")
        elif rule.rule_type == "type" and value is not None and not isinstance(value, rule.expected):
            self.errors.append(
                rule.message
                or f"配置项类型错误: {rule.path}, 期望 {rule.expected.__name__}, 实际 {type(value).__name__}"
            )
        elif rule.rule_type == "range" and value is not None:
            min_val, max_val = rule.expected
            if not (min_val <= value <= max_val):
                self.errors.append(
                    rule.message or f"配置项超出范围: {rule.path}={value}, 期望 [{min_val}, {max_val}]"
                )
        elif rule.rule_type == "one_of" and value not in rule.expected:
            self.errors.append(
                rule.message or f"配置项值无效: {rule.path}={value}, 期望 one of {rule.expected}"
            )

    @staticmethod
    def _get_nested_value(config: Dict, path: str) -> Any:
        keys = path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def get_errors(self) -> List[str]:
        return self.errors


@dataclass
class ConfigManagerConfig:
    default_config_path: str = "./config"
    env_prefix: str = "FLARE_"
    allow_env_override: bool = True
    merge_strategy: ConfigMergeStrategy = ConfigMergeStrategy.OVERRIDE


class ConfigManager:
    """统一配置管理器，支持 YAML/JSON 和环境变量。"""

    def __init__(self, config: Optional[ConfigManagerConfig] = None):
        self.cfg = config or ConfigManagerConfig()
        self.interpolator = EnvInterpolator()
        self.validator = ConfigValidator()
        self._config: Dict = {}

    def load(self, config_path: Union[str, List[str]], validate: bool = True) -> Dict:
        configs = []
        paths = [config_path] if isinstance(config_path, str) else config_path

        for path in paths:
            if os.path.exists(path):
                configs.append(self._load_file(path))
            else:
                import logging
                logging.getLogger(__name__).warning("配置文件不存在: %s", path)

        self._config = self._merge_configs(configs)

        if self.cfg.allow_env_override:
            self._apply_env_overrides()

        self._config = self.interpolator.interpolate_dict(self._config)

        if validate and not self.validator.validate(self._config):
            raise ValueError(f"配置验证失败: {self.validator.get_errors()}")

        return self._config

    def _load_file(self, path: str) -> Dict:
        ext = Path(path).suffix.lower()
        with open(path, "r", encoding="utf-8") as f:
            if ext in (".yaml", ".yml"):
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML config files")
                return yaml.safe_load(f) or {}
            elif ext == ".json":
                return json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {ext}")

    def _merge_configs(self, configs: List[Dict]) -> Dict:
        if not configs:
            return {}
        result = configs[0].copy()
        for cfg in configs[1:]:
            result = self._merge_dict(result, cfg)
        return result

    def _merge_dict(self, base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dict(result[key], value)
            else:
                if self.cfg.merge_strategy == ConfigMergeStrategy.KEEP and key in result:
                    continue
                result[key] = value
        return result

    def _apply_env_overrides(self):
        prefix = self.cfg.env_prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self._set_nested_value(self._config, config_key, value)

    def _set_nested_value(self, config: Dict, path: str, value: str):
        keys = path.split("_")
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        final_key = keys[-1]
        current[final_key] = self._convert_type(value)

    @staticmethod
    def _convert_type(value: Any) -> Union[bool, int, float, str]:
        # Non-string values pass through unchanged
        if not isinstance(value, str):
            return value
        low = value.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if value.isdigit():
            return int(value)
        if value.replace(".", "", 1).isdigit():
            return float(value)
        return value

    def get(self, path: Optional[str] = None, default: Any = None) -> Any:
        if path is None:
            return self._config.copy()
        value = ConfigValidator._get_nested_value(self._config, path)
        return default if value is None else value

    def set(self, path: str, value: Any):
        self._set_nested_value(self._config, path.replace(".", "_"), value)

    def save(self, path: str, fmt: Optional[str] = None):
        fmt = fmt or ("yaml" if path.endswith((".yaml", ".yml")) else "json")
        with open(path, "w", encoding="utf-8") as f:
            if fmt == "yaml":
                if not HAS_YAML:
                    raise ImportError("PyYAML is required for YAML config files")
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            elif fmt == "json":
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的格式: {fmt}")

    def to_dict(self) -> Dict:
        return self._config.copy()
