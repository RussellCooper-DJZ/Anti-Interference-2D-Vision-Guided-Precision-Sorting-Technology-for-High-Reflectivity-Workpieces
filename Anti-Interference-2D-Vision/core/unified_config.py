"""
UnifiedConfig - 统一配置管理模块
支持 YAML/JSON/ENV/命令行参数，配置热加载和验证

Features:
    - 单例模式：全局唯一配置实例
    - 多格式支持：YAML、JSON、环境变量
    - 命令行参数：支持通过 argparse 解析命令行参数
    - 热加载：配置变更后无需重启即可生效
    - 配置验证：支持 schema 验证
    - 路径访问：支持点号分隔的嵌套路径访问如 'model.input_size'

Example:
    >>> config = get_config()
    >>> config.load_yaml("config.yaml")
    >>> config.load_env("AG_")
    >>> value = config.get("model.input_size", default=224)
    >>> config.set("model.input_size", 256)
    >>> config.validate({"model": {"input_size": int}})
"""
from typing import Any, Dict, Optional, Union, List, Callable, Type
import os
import sys
import json
import argparse
import threading
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

__all__ = ["UnifiedConfig", "get_config", "ValidationError"]


class ValidationError(Exception):
    """配置验证异常"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"配置验证失败: {'; '.join(errors)}")


class UnifiedConfig:
    """
    统一配置管理器（单例模式）

    支持从多种来源加载配置：
    - YAML 文件
    - JSON 文件
    - 环境变量
    - 命令行参数

    配置优先级（从低到高）：
    1. 默认值
    2. YAML 配置
    3. JSON 配置
    4. 环境变量
    5. 命令行参数

    Attributes:
        _instance: 单例实例
        _initialized: 是否已初始化
        _config: 配置数据字典
        _sources: 配置来源记录
        _argparse_parser: 命令行解析器
        _argparse_namespace: 命令行参数命名空间
    """

    _instance: Optional['UnifiedConfig'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'UnifiedConfig':
        """
        创建或返回单例实例（线程安全）

        Returns:
            UnifiedConfig: 单例实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        初始化配置管理器

        仅在首次创建时执行初始化逻辑
        """
        if self._initialized:
            return

        self._config: Dict[str, Any] = {}
        self._sources: Dict[str, Any] = {}  # 记录配置来源
        self._file_paths: List[str] = []  # 记录加载的文件路径
        self._argparse_parser: Optional[argparse.ArgumentParser] = None
        self._argparse_namespace: Optional[argparse.Namespace] = None
        self._initialized = True

    # ==================== 加载方法 ====================

    def load_yaml(self, path: str, override: bool = True) -> 'UnifiedConfig':
        """
        从YAML文件加载配置

        Args:
            path: YAML文件路径
            override: 是否覆盖已有配置，False时合并配置

        Returns:
            self: 返回自身以支持链式调用

        Raises:
            FileNotFoundError: 配置文件不存在
            ImportError: PyYAML未安装
            yaml.YAMLError: YAML解析错误

        Example:
            >>> config.load_yaml("config.yaml")
            >>> config.load_yaml("config.yaml", override=False)
        """
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML config files. Install it with: pip install pyyaml")

        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        self._file_paths.append(path)
        self._sources[path] = "yaml"

        if override:
            self._config = self._deep_merge(self._config, data)
        else:
            self._config = self._deep_merge(data, self._config)

        return self

    def load_json(self, path: str, override: bool = True) -> 'UnifiedConfig':
        """
        从JSON文件加载配置

        Args:
            path: JSON文件路径
            override: 是否覆盖已有配置，False时合并配置

        Returns:
            self: 返回自身以支持链式调用

        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON解析错误

        Example:
            >>> config.load_json("config.json")
            >>> config.load_json("config.json", override=False)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data is None:
            data = {}

        self._file_paths.append(path)
        self._sources[path] = "json"

        if override:
            self._config = self._deep_merge(self._config, data)
        else:
            self._config = self._deep_merge(data, self._config)

        return self

    def load_env(self, prefix: str = "AG_", override: bool = True) -> 'UnifiedConfig':
        """
        从环境变量加载配置

        环境变量名格式: {PREFIX}{SECTION}_{KEY} = value
        例如: AG_MODEL_INPUT_SIZE=224

        Args:
            prefix: 环境变量前缀
            override: 是否覆盖已有配置

        Returns:
            self: 返回自身以支持链式调用

        Example:
            >>> config.load_env("AG_")
            >>> # AG_MODEL_INPUT_SIZE=224 -> config["model"]["input_size"] = 224
        """
        env_config: Dict[str, Any] = {}

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # 去掉前缀，转换为小写
            config_key = key[len(prefix):].lower()

            # 只在第一个下划线处分割，分成 section 和 key
            # AG_MODEL_INPUT_SIZE -> section="model", key="input_size"
            # AG_DEBUG -> section="debug" (no nested)
            if "_" in config_key:
                section, rest = config_key.split("_", 1)
                keys = [section, rest]
            else:
                keys = [config_key]

            current = env_config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # 设置最终值，自动类型转换
            current[keys[-1]] = self._convert_type(value)

        self._sources["env_prefix"] = prefix

        if override:
            self._config = self._deep_merge(self._config, env_config)
        else:
            self._config = self._deep_merge(env_config, self._config)

        return self

    def load_args(self, args: Optional[List[str]] = None) -> 'UnifiedConfig':
        """
        从命令行参数加载配置

        支持两种格式：
        - 点号分隔: --model.input_size=224 (将被解析为嵌套结构 model.input_size)
        - 下划线分隔: --model_input_size=224 (将被解析为字面键名 model_input_size)

        Args:
            args: 命令行参数列表，默认为 sys.argv[1:]

        Returns:
            self: 返回自身以支持链式调用

        Example:
            >>> config.load_args()
            >>> # python script.py --model.input_size=224 --batch_size=32
        """
        if args is None:
            args = sys.argv[1:]

        # 手动解析命令行参数
        # 格式: --key=value 或 --key value
        for arg in args:
            if not arg.startswith("--"):
                continue

            # 提取等号后的值
            if "=" in arg:
                key_str, value = arg[2:].split("=", 1)
            else:
                # 格式为 --key value 的情况（value不能以--开头）
                parts = arg[2:].split(" ", 1)
                key_str = parts[0]
                value = parts[1] if len(parts) > 1 else "true"

            # _set_nested 使用点号作为嵌套分隔符
            # 所以如果原始键是 model.input_size，它会正确地设置为嵌套结构
            # 如果原始键是 model_input_size，它会设置为字面键名
            # 不需要做任何转换

            # 设置配置值
            self._set_nested(self._config, key_str, self._convert_type(value))

        return self

    # ==================== 访问方法 ====================

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔路径

        Args:
            key: 配置键，支持嵌套访问如 'model.input_size'
            default: 默认值，当键不存在时返回

        Returns:
            配置值或默认值

        Example:
            >>> config.get("model.input_size")
            >>> config.get("model.input_size", default=224)
            >>> config.get("nonexistent", default="fallback")
        """
        value = self._get_nested(self._config, key)
        return default if value is None else value

    def set(self, key: str, value: Any) -> None:
        """
        设置配置值，支持点号分隔路径

        Args:
            key: 配置键，支持嵌套访问如 'model.input_size'
            value: 配置值

        Example:
            >>> config.set("model.input_size", 256)
            >>> config.set("debug", True)
        """
        self._set_nested(self._config, key, value)

    def __getitem__(self, key: str) -> Any:
        """
        支持字典式访问 config["key"]

        Args:
            key: 配置键，支持点号分隔路径

        Returns:
            配置值

        Raises:
            KeyError: 键不存在
        """
        value = self.get(key)
        if value is None and "." not in key and key not in self._config:
            raise KeyError(f"配置项不存在: {key}")
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """
        支持字典式设置 config["key"] = value

        Args:
            key: 配置键，支持点号分隔路径
            value: 配置值
        """
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """
        支持 in 操作符检查配置项是否存在

        Args:
            key: 配置键，支持点号分隔路径

        Returns:
            bool: 键是否存在
        """
        return self.get(key) is not None

    # ==================== 验证方法 ====================

    def validate(self, schema: Dict[str, Any]) -> bool:
        """
        验证配置是否符合schema

        Schema格式:
            {
                "key": type or [type1, type2],  # 类型验证
                "nested.key": {                  # 嵌套验证
                    "sub_key": type
                }
            }

        Args:
            schema: 验证schema字典

        Returns:
            bool: 验证是否通过

        Raises:
            ValidationError: 验证失败时抛出，包含所有错误信息

        Example:
            >>> schema = {
            ...     "model.input_size": int,
            ...     "model.learning_rate": [int, float],
            ...     "debug": bool
            ... }
            >>> config.validate(schema)
        """
        errors: List[str] = []

        for path, expected_type in schema.items():
            value = self.get(path)

            # 检查必需项
            if value is None:
                errors.append(f"缺少必需配置项: {path}")
                continue

            # 类型验证
            if isinstance(expected_type, list):
                # 联合类型
                if not any(isinstance(value, t) for t in expected_type):
                    type_names = [t.__name__ if isinstance(t, type) else str(t) for t in expected_type]
                    errors.append(
                        f"配置项类型错误: {path}, 期望类型 {' | '.join(type_names)}, "
                        f"实际类型 {type(value).__name__}"
                    )
            elif isinstance(expected_type, type):
                if not isinstance(value, expected_type):
                    errors.append(
                        f"配置项类型错误: {path}, 期望类型 {expected_type.__name__}, "
                        f"实际类型 {type(value).__name__}"
                    )
            elif isinstance(expected_type, dict):
                # 嵌套schema验证
                nested_errors = self._validate_nested(value, expected_type, path)
                errors.extend(nested_errors)

        if errors:
            raise ValidationError(errors)

        return True

    def _validate_nested(self, value: Any, schema: Dict[str, Any], prefix: str) -> List[str]:
        """
        验证嵌套配置

        Args:
            value: 配置值
            schema: 嵌套schema
            prefix: 路径前缀

        Returns:
            错误列表
        """
        errors: List[str] = []

        if not isinstance(value, dict):
            errors.append(f"配置项类型错误: {prefix}, 期望类型 dict, 实际类型 {type(value).__name__}")
            return errors

        for key, expected_type in schema.items():
            nested_path = f"{prefix}.{key}"
            nested_value = value.get(key)

            if nested_value is None:
                errors.append(f"缺少必需配置项: {nested_path}")
                continue

            if isinstance(expected_type, type) and not isinstance(nested_value, expected_type):
                errors.append(
                    f"配置项类型错误: {nested_path}, 期望类型 {expected_type.__name__}, "
                    f"实际类型 {type(nested_value).__name__}"
                )

        return errors

    # ==================== 热加载方法 ====================

    def reload(self) -> 'UnifiedConfig':
        """
        重新加载所有配置（热加载）

        重新从所有已加载的文件和环境变量中读取配置

        Returns:
            self: 返回自身以支持链式调用

        Note:
            会保留文件路径记录，但重新读取文件内容
        """
        original_file_paths = self._file_paths.copy()
        original_sources = self._sources.copy()

        # 备份当前配置
        backup_config = deepcopy(self._config)

        try:
            # 清空当前配置
            self._config = {}
            self._file_paths = []
            self._sources = {}

            # 重新加载所有文件
            for path in original_file_paths:
                ext = Path(path).suffix.lower()
                if ext in (".yaml", ".yml"):
                    self.load_yaml(path)
                elif ext == ".json":
                    self.load_json(path)

            # 重新加载环境变量
            if "env_prefix" in original_sources:
                self.load_env(original_sources["env_prefix"])

        except Exception:
            # 如果重载失败，恢复原有配置
            self._config = backup_config
            self._file_paths = original_file_paths
            self._sources = original_sources
            raise

        return self

    def reload_file(self, path: str) -> 'UnifiedConfig':
        """
        重新加载指定配置文件

        Args:
            path: 文件路径

        Returns:
            self: 返回自身以支持链式调用

        Raises:
            FileNotFoundError: 文件不存在
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在: {path}")

        # 找到并移除该文件的旧配置
        ext = Path(path).suffix.lower()

        if ext in (".yaml", ".yml"):
            self.load_yaml(path)
        elif ext == ".json":
            self.load_json(path)

        return self

    # ==================== 工具方法 ====================

    def _get_nested(self, data: Dict[str, Any], key: str) -> Any:
        """
        获取嵌套配置值

        Args:
            data: 配置字典
            key: 点号分隔的键路径

        Returns:
            配置值，不存在返回None
        """
        keys = key.split(".")
        current = data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None

        return current

    def _set_nested(self, data: Dict[str, Any], key: str, value: Any) -> None:
        """
        设置嵌套配置值

        Args:
            data: 配置字典
            key: 点号分隔的键路径
            value: 要设置的值
        """
        keys = key.split(".")
        current = data

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典

        Args:
            base: 基础字典
            override: 覆盖字典

        Returns:
            合并后的新字典
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = UnifiedConfig._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    @staticmethod
    def _convert_type(value: str) -> Any:
        """
        将字符串转换为适当的类型

        Args:
            value: 字符串值

        Returns:
            转换后的值（bool/int/float/str）
        """
        # 布尔值
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

        # 整数
        if value.isdigit():
            return int(value)

        # 浮点数
        try:
            if "." in value and value.replace(".", "", 1).isdigit():
                return float(value)
        except ValueError:
            pass

        # 字符串
        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        获取配置字典副本

        Returns:
            配置字典的深拷贝
        """
        return deepcopy(self._config)

    def get_sources(self) -> Dict[str, Any]:
        """
        获取配置来源信息

        Returns:
            配置来源字典
        """
        return deepcopy(self._sources)

    def clear(self) -> 'UnifiedConfig':
        """
        清空所有配置

        Returns:
            self: 返回自身以支持链式调用
        """
        self._config = {}
        self._sources = {}
        self._file_paths = []
        return self

    def __repr__(self) -> str:
        """返回配置的字符串表示"""
        return f"UnifiedConfig(sources={list(self._sources.keys())}, config={self._config})"

    def __len__(self) -> int:
        """返回配置项数量"""
        return len(self._config)


# ==================== 快捷函数 ====================

def get_config() -> UnifiedConfig:
    """
    获取全局配置实例（单例）

    Returns:
        UnifiedConfig: 全局配置实例

    Example:
        >>> config = get_config()
        >>> config.load_yaml("config.yaml")
    """
    return UnifiedConfig()


def reset_config() -> None:
    """
    重置配置单例（主要用于测试）

    Example:
        >>> reset_config()
        >>> config = get_config()
    """
    UnifiedConfig._instance = None
