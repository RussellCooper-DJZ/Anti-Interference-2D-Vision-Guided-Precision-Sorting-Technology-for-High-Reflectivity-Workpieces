"""
test_unified_config.py - UnifiedConfig 模块测试

测试覆盖率目标: >80%
"""
import json
import os
import sys
import tempfile
import pytest
from pathlib import Path

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.unified_config import (
    UnifiedConfig,
    get_config,
    reset_config,
    ValidationError,
)


# ==================== Fixtures ====================

@pytest.fixture(autouse=True)
def reset_singleton():
    """每个测试前重置单例"""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def yaml_config_file(temp_config_dir):
    """创建临时YAML配置文件"""
    config = {
        "model": {
            "input_size": 224,
            "learning_rate": 0.001,
            "hidden_dims": [128, 256, 512]
        },
        "training": {
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "adam"
        },
        "debug": True
    }
    path = os.path.join(temp_config_dir, "config.yaml")
    try:
        import yaml
        HAS_YAML_AVAILABLE = True
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
    except ImportError:
        HAS_YAML_AVAILABLE = False
        # 如果yaml未安装，直接写文件
        with open(path, "w", encoding="utf-8") as f:
            f.write("model:\n  input_size: 224\n  learning_rate: 0.001\n  hidden_dims: [128, 256, 512]\n")
            f.write("training:\n  batch_size: 32\n  epochs: 100\n  optimizer: adam\n")
            f.write("debug: true\n")
    return path, HAS_YAML_AVAILABLE


@pytest.fixture
def yaml_config_available():
    """检查YAML是否可用"""
    try:
        import yaml
        return True
    except ImportError:
        return False


@pytest.fixture
def json_config_file(temp_config_dir):
    """创建临时JSON配置文件"""
    config = {
        "model": {
            "input_size": 224,
            "learning_rate": 0.001,
            "optimizer": "adam"
        },
        "data": {
            "root": "/data",
            "num_workers": 4
        }
    }
    path = os.path.join(temp_config_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f)
    return path


@pytest.fixture
def config_instance():
    """创建配置实例"""
    return UnifiedConfig()


# ==================== 单例模式测试 ====================

class TestSingleton:
    """测试单例模式"""

    def test_singleton_same_instance(self):
        """测试多次调用返回同一实例"""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_singleton_direct_init(self):
        """测试直接初始化返回同一实例"""
        config1 = UnifiedConfig()
        config2 = UnifiedConfig()
        assert config1 is config2

    def test_reset_config(self):
        """测试重置单例"""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2


# ==================== YAML加载测试 ====================

class TestLoadYAML:
    """测试YAML加载"""

    def test_load_yaml_basic(self, yaml_config_file, config_instance):
        """测试基本YAML加载"""
        path, yaml_available = yaml_config_file
        if not yaml_available:
            pytest.skip("PyYAML not installed")
        config_instance.load_yaml(path)
        assert config_instance.get("model.input_size") == 224
        assert config_instance.get("training.batch_size") == 32

    def test_load_yaml_override(self, yaml_config_file, config_instance):
        """测试YAML覆盖模式"""
        path, yaml_available = yaml_config_file
        if not yaml_available:
            pytest.skip("PyYAML not installed")
        config_instance.set("model.input_size", 100)
        config_instance.load_yaml(path)
        # override=True时新值覆盖旧值
        assert config_instance.get("model.input_size") == 224

    def test_load_yaml_merge(self, yaml_config_file, temp_config_dir, config_instance):
        """测试YAML合并模式"""
        path, yaml_available = yaml_config_file
        if not yaml_available:
            pytest.skip("PyYAML not installed")
        config_instance.set("extra.setting", 999)
        config_instance.load_yaml(path, override=False)
        assert config_instance.get("model.input_size") == 224
        assert config_instance.get("extra.setting") == 999

    def test_load_yaml_file_not_found(self, config_instance):
        """测试YAML文件不存在"""
        # 先检查yaml模块是否可用
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        with pytest.raises(FileNotFoundError):
            config_instance.load_yaml("/nonexistent/path.yaml")

    def test_load_yaml_chain(self, yaml_config_file, config_instance):
        """测试链式调用"""
        path, yaml_available = yaml_config_file
        if not yaml_available:
            pytest.skip("PyYAML not installed")
        result = config_instance.load_yaml(path)
        assert result is config_instance


# ==================== JSON加载测试 ====================

class TestLoadJSON:
    """测试JSON加载"""

    def test_load_json_basic(self, json_config_file, config_instance):
        """测试基本JSON加载"""
        config_instance.load_json(json_config_file)
        assert config_instance.get("model.input_size") == 224
        assert config_instance.get("data.num_workers") == 4

    def test_load_json_override(self, json_config_file, config_instance):
        """测试JSON覆盖模式"""
        config_instance.set("model.input_size", 100)
        config_instance.load_json(json_config_file)
        assert config_instance.get("model.input_size") == 224

    def test_load_json_merge(self, json_config_file, config_instance):
        """测试JSON合并模式"""
        config_instance.set("extra.setting", 999)
        config_instance.load_json(json_config_file, override=False)
        assert config_instance.get("model.input_size") == 224
        assert config_instance.get("extra.setting") == 999

    def test_load_json_file_not_found(self, config_instance):
        """测试JSON文件不存在"""
        with pytest.raises(FileNotFoundError):
            config_instance.load_json("/nonexistent/path.json")


# ==================== 环境变量加载测试 ====================

class TestLoadEnv:
    """测试环境变量加载"""

    def test_load_env_basic(self, config_instance, monkeypatch):
        """测试基本环境变量加载"""
        monkeypatch.setenv("AG_MODEL_INPUT_SIZE", "256")
        monkeypatch.setenv("AG_MODEL_LEARNING_RATE", "0.01")
        config_instance.load_env("AG_")
        assert config_instance.get("model.input_size") == 256
        assert config_instance.get("model.learning_rate") == 0.01

    def test_load_env_boolean_conversion(self, config_instance, monkeypatch):
        """测试布尔类型转换"""
        monkeypatch.setenv("AG_DEBUG", "true")
        monkeypatch.setenv("AG_VERBOSE", "false")
        config_instance.load_env("AG_")
        assert config_instance.get("debug") is True
        assert config_instance.get("verbose") is False

    def test_load_env_float_conversion(self, config_instance, monkeypatch):
        """测试浮点类型转换"""
        monkeypatch.setenv("AG_RATE", "3.14")
        config_instance.load_env("AG_")
        assert config_instance.get("rate") == 3.14

    def test_load_env_int_conversion(self, config_instance, monkeypatch):
        """测试整数类型转换"""
        monkeypatch.setenv("AG_COUNT", "42")
        config_instance.load_env("AG_")
        assert config_instance.get("count") == 42

    def test_load_env_string_preserved(self, config_instance, monkeypatch):
        """测试字符串保持"""
        monkeypatch.setenv("AG_NAME", "test_model")
        config_instance.load_env("AG_")
        assert config_instance.get("name") == "test_model"

    def test_load_env_prefix_filter(self, config_instance, monkeypatch):
        """测试前缀过滤"""
        monkeypatch.setenv("AG_TEST", "value1")
        monkeypatch.setenv("OTHER_TEST", "value2")
        config_instance.load_env("AG_")
        assert config_instance.get("test") == "value1"
        assert config_instance.get("other_test") is None

    def test_load_env_override(self, config_instance, monkeypatch):
        """测试环境变量覆盖"""
        config_instance.set("model.input_size", 100)
        monkeypatch.setenv("AG_MODEL_INPUT_SIZE", "200")
        config_instance.load_env("AG_")
        assert config_instance.get("model.input_size") == 200


# ==================== 命令行参数测试 ====================

class TestLoadArgs:
    """测试命令行参数加载"""

    def test_load_args_basic(self, config_instance):
        """测试基本命令行参数加载"""
        args = ["--model.input_size=512", "--batch_size=64"]
        config_instance.load_args(args)
        assert config_instance.get("model.input_size") == 512
        assert config_instance.get("batch_size") == 64

    def test_load_args_default(self, config_instance):
        """测试默认参数（无命令行参数）"""
        config_instance.load_args([])
        assert config_instance.get("model.input_size") is None

    def test_load_args_conversion(self, config_instance):
        """测试类型转换"""
        args = ["--debug=true", "--count=10", "--rate=3.14"]
        config_instance.load_args(args)
        assert config_instance.get("debug") is True
        assert config_instance.get("count") == 10
        assert config_instance.get("rate") == 3.14


# ==================== Get/Set测试 ====================

class TestGetSet:
    """测试get和set方法"""

    def test_get_simple_key(self, config_instance):
        """测试简单键获取"""
        config_instance.set("name", "test")
        assert config_instance.get("name") == "test"

    def test_get_nested_key(self, config_instance):
        """测试嵌套键获取"""
        config_instance.set("model.input_size", 224)
        assert config_instance.get("model.input_size") == 224

    def test_get_default_value(self, config_instance):
        """测试默认值"""
        assert config_instance.get("nonexistent") is None
        assert config_instance.get("nonexistent", default=42) == 42

    def test_set_simple_key(self, config_instance):
        """测试简单键设置"""
        config_instance.set("key", "value")
        assert config_instance.get("key") == "value"

    def test_set_nested_key(self, config_instance):
        """测试嵌套键设置"""
        config_instance.set("a.b.c", 123)
        assert config_instance.get("a.b.c") == 123

    def test_set_overwrite(self, config_instance):
        """测试覆盖已有值"""
        config_instance.set("key", "old")
        config_instance.set("key", "new")
        assert config_instance.get("key") == "new"

    def test_dict_access(self, config_instance):
        """测试字典式访问"""
        config_instance["name"] = "test"
        assert config_instance["name"] == "test"

    def test_dict_access_nested(self, config_instance):
        """测试字典式嵌套访问"""
        config_instance["model.input_size"] = 224
        assert config_instance["model.input_size"] == 224

    def test_dict_contains(self, config_instance):
        """测试in操作符"""
        config_instance.set("key", "value")
        assert "key" in config_instance
        assert "nonexistent" not in config_instance

    def test_dict_contains_nested(self, config_instance):
        """测试嵌套键in操作符"""
        config_instance.set("a.b.c", 123)
        assert "a.b.c" in config_instance

    def test_len(self, config_instance):
        """测试长度"""
        config_instance.set("a", 1)
        config_instance.set("b", 2)
        assert len(config_instance) >= 2


# ==================== 验证测试 ====================

class TestValidation:
    """测试配置验证"""

    def test_validate_success(self, config_instance):
        """测试验证成功"""
        config_instance.set("model.input_size", 224)
        config_instance.set("model.learning_rate", 0.001)
        config_instance.set("debug", True)

        schema = {
            "model.input_size": int,
            "model.learning_rate": float,
            "debug": bool
        }
        assert config_instance.validate(schema) is True

    def test_validate_missing_key(self, config_instance):
        """测试缺少必需键"""
        config_instance.set("model.input_size", 224)

        schema = {
            "model.input_size": int,
            "model.learning_rate": float  # 缺少此键
        }
        with pytest.raises(ValidationError) as exc_info:
            config_instance.validate(schema)
        assert "model.learning_rate" in str(exc_info.value)

    def test_validate_type_error(self, config_instance):
        """测试类型错误"""
        config_instance.set("model.input_size", "not_an_int")

        schema = {
            "model.input_size": int
        }
        with pytest.raises(ValidationError) as exc_info:
            config_instance.validate(schema)
        assert "类型错误" in str(exc_info.value)

    def test_validate_union_type(self, config_instance):
        """测试联合类型验证"""
        config_instance.set("value", 42)

        schema = {
            "value": [int, float]
        }
        assert config_instance.validate(schema) is True

    def test_validate_union_type_fail(self, config_instance):
        """测试联合类型验证失败"""
        config_instance.set("value", "string")

        schema = {
            "value": [int, float]
        }
        with pytest.raises(ValidationError):
            config_instance.validate(schema)


# ==================== 热加载测试 ====================

class TestReload:
    """测试热加载"""

    def test_reload(self, yaml_config_file, temp_config_dir, config_instance, monkeypatch):
        """测试重新加载配置"""
        path, yaml_available = yaml_config_file
        if not yaml_available:
            pytest.skip("PyYAML not installed")
        config_instance.load_yaml(path)
        original_value = config_instance.get("model.input_size")

        # 修改文件
        try:
            import yaml
            new_config = {
                "model": {
                    "input_size": 512,
                    "learning_rate": 0.001
                }
            }
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(new_config, f)

            # 重新加载
            config_instance.reload()

            assert config_instance.get("model.input_size") == 512

            # 恢复文件
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump({"model": {"input_size": original_value}}, f)
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_reload_file(self, yaml_config_file, temp_config_dir, config_instance):
        """测试重新加载指定文件"""
        path, yaml_available = yaml_config_file
        if not yaml_available:
            pytest.skip("PyYAML not installed")
        try:
            import yaml
            config_instance.load_yaml(path)

            # 修改文件
            new_config = {"model": {"input_size": 999}}
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(new_config, f)

            config_instance.reload_file(path)
            assert config_instance.get("model.input_size") == 999

            # 恢复文件
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump({"model": {"input_size": 224}}, f)
        except ImportError:
            pytest.skip("PyYAML not installed")


# ==================== 工具方法测试 ====================

class TestUtils:
    """测试工具方法"""

    def test_to_dict(self, config_instance):
        """测试转换为字典"""
        config_instance.set("a.b", 1)
        config_instance.set("c", 2)
        d = config_instance.to_dict()
        assert d["a"]["b"] == 1
        assert d["c"] == 2
        # 确保是深拷贝
        d["a"]["b"] = 999
        assert config_instance.get("a.b") == 1

    def test_get_sources(self, yaml_config_file, json_config_file, config_instance, monkeypatch):
        """测试获取配置来源"""
        path, yaml_available = yaml_config_file
        if yaml_available:
            config_instance.load_yaml(path)
        config_instance.load_json(json_config_file)
        monkeypatch.setenv("AG_TEST", "value")
        config_instance.load_env("AG_")

        sources = config_instance.get_sources()
        assert "env_prefix" in sources

    def test_clear(self, config_instance):
        """测试清空配置"""
        config_instance.set("a.b", 1)
        config_instance.clear()
        assert config_instance.get("a.b") is None
        assert len(config_instance) == 0

    def test_repr(self, config_instance):
        """测试字符串表示"""
        config_instance.set("key", "value")
        r = repr(config_instance)
        assert "UnifiedConfig" in r


class TestDeepMerge:
    """测试深度合并"""

    def test_deep_merge_simple(self):
        """测试简单深度合并"""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = UnifiedConfig._deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        """测试嵌套深度合并"""
        base = {"model": {"input_size": 224, "lr": 0.001}}
        override = {"model": {"lr": 0.01}, "extra": 1}
        result = UnifiedConfig._deep_merge(base, override)
        assert result == {"model": {"input_size": 224, "lr": 0.01}, "extra": 1}

    def test_deep_merge_list_override(self):
        """测试列表覆盖"""
        base = {"list": [1, 2, 3]}
        override = {"list": [4, 5]}
        result = UnifiedConfig._deep_merge(base, override)
        assert result == {"list": [4, 5]}


class TestConvertType:
    """测试类型转换"""

    def test_convert_bool_true(self):
        """测试布尔True转换"""
        assert UnifiedConfig._convert_type("true") is True
        assert UnifiedConfig._convert_type("True") is True
        assert UnifiedConfig._convert_type("TRUE") is True

    def test_convert_bool_false(self):
        """测试布尔False转换"""
        assert UnifiedConfig._convert_type("false") is False
        assert UnifiedConfig._convert_type("False") is False

    def test_convert_int(self):
        """测试整数转换"""
        assert UnifiedConfig._convert_type("42") == 42
        assert UnifiedConfig._convert_type("0") == 0

    def test_convert_float(self):
        """测试浮点数转换"""
        assert UnifiedConfig._convert_type("3.14") == 3.14
        assert UnifiedConfig._convert_type("0.001") == 0.001

    def test_convert_string(self):
        """测试字符串保持"""
        assert UnifiedConfig._convert_type("hello") == "hello"
        assert UnifiedConfig._convert_type("hello123") == "hello123"


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""

    def test_full_workflow(self, yaml_config_file, json_config_file, temp_config_dir, config_instance, monkeypatch):
        """测试完整工作流"""
        # 1. 加载YAML
        path, yaml_available = yaml_config_file
        if yaml_available:
            config_instance.load_yaml(path)

        # 2. 加载JSON
        config_instance.load_json(json_config_file)

        # 3. 从环境变量加载
        monkeypatch.setenv("AG_EXTRA_SETTING", "from_env")
        config_instance.load_env("AG_")

        # 4. 验证
        schema = {
            "model.input_size": int,
            "model.learning_rate": float,
            "data.num_workers": int
        }
        config_instance.validate(schema)

        # 5. 获取值
        assert config_instance.get("model.input_size") == 224
        assert config_instance.get("data.num_workers") == 4
        assert config_instance.get("extra.setting") == "from_env"

        # 6. 设置值
        config_instance.set("model.input_size", 512)

        # 7. 确认设置成功
        assert config_instance.get("model.input_size") == 512

    def test_chain_calls(self, yaml_config_file, json_config_file, config_instance):
        """测试链式调用"""
        path, yaml_available = yaml_config_file
        if yaml_available:
            config_instance.load_yaml(path).load_json(json_config_file)
        else:
            config_instance.load_json(json_config_file)
        assert config_instance.get("model.input_size") == 224


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
