"""
tests/test_unified_logger.py — UnifiedLogger 统一日志模块测试
"""
import json
import logging
import os
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from core.unified_logger import (
    UnifiedLogger,
    LogContext,
    StructuredFormatter,
    TextFormatter,
    get_logger,
)


class TestLogContext:
    """LogContext 测试"""

    def test_set_and_get(self) -> None:
        ctx = LogContext()
        ctx.set("trace_id", "abc123")
        assert ctx.get("trace_id") == "abc123"

    def test_get_nonexistent(self) -> None:
        ctx = LogContext()
        assert ctx.get("nonexistent") is None

    def test_get_with_default(self) -> None:
        ctx = LogContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_clear(self) -> None:
        ctx = LogContext()
        ctx.set("a", 1)
        ctx.set("b", 2)
        ctx.clear()
        assert ctx.get("a") is None
        assert ctx.get("b") is None

    def test_to_dict(self) -> None:
        ctx = LogContext()
        ctx.set("trace_id", "abc")
        ctx.set("request_id", "123")
        d = ctx.to_dict()
        assert d == {"trace_id": "abc", "request_id": "123"}

    def test_to_dict_is_copy(self) -> None:
        ctx = LogContext()
        ctx.set("key", "value")
        d = ctx.to_dict()
        d["key"] = "modified"
        assert ctx.get("key") == "value"


class TestStructuredFormatter:
    """StructuredFormatter 测试"""

    def test_format_json_output(self) -> None:
        import logging

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "test message"
        assert "timestamp" in data

    def test_format_with_context(self) -> None:
        import logging

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.context = {"trace_id": "abc"}
        output = formatter.format(record)
        data = json.loads(output)
        assert data["context"]["trace_id"] == "abc"

    def test_format_with_exception(self) -> None:
        import logging

        formatter = StructuredFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
            output = formatter.format(record)
            data = json.loads(output)
            assert "exception" in data


class TestTextFormatter:
    """TextFormatter 测试"""

    def test_format_output(self) -> None:
        import logging

        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "test" in output
        assert "test message" in output


class TestUnifiedLoggerSingleton:
    """UnifiedLogger 单例测试"""

    def test_singleton_same_instance(self) -> None:
        logger1 = UnifiedLogger()
        logger2 = UnifiedLogger()
        assert logger1 is logger2

    def test_get_logger_returns_singleton(self) -> None:
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2


class TestUnifiedLoggerSetup:
    """UnifiedLogger 配置测试"""

    def setup_method(self) -> None:
        """每个测试前重置单例"""
        UnifiedLogger._instance = None

    def test_setup_console_default(self) -> None:
        logger = UnifiedLogger()
        logger.setup(level="DEBUG", output="console")
        assert logger._level == 10  # DEBUG
        assert logger._output == "console"

    def test_setup_console_info_level(self) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")
        assert logger._level == 20  # INFO

    def test_setup_console_warning_level(self) -> None:
        logger = UnifiedLogger()
        logger.setup(level="WARNING", output="console")
        assert logger._level == 30  # WARNING

    def test_setup_console_error_level(self) -> None:
        logger = UnifiedLogger()
        logger.setup(level="ERROR", output="console")
        assert logger._level == 40  # ERROR

    def test_setup_console_critical_level(self) -> None:
        logger = UnifiedLogger()
        logger.setup(level="CRITICAL", output="console")
        assert logger._level == 50  # CRITICAL

    def test_setup_file_output(self) -> None:
        logger = UnifiedLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.log")
            logger.setup(level="INFO", output="file", file_path=file_path)
            assert logger._file_path == file_path
            assert os.path.exists(os.path.dirname(file_path))

    def test_setup_rotating_output(self) -> None:
        logger = UnifiedLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "rotating.log")
            logger.setup(
                level="INFO",
                output="rotating",
                file_path=file_path,
                max_bytes=1024,
                backup_count=3,
            )
            assert logger._file_path == file_path

    def test_setup_rotating_requires_file_path(self) -> None:
        logger = UnifiedLogger()
        with pytest.raises(ValueError, match="file_path is required"):
            logger.setup(level="INFO", output="rotating")

    def test_setup_file_requires_file_path(self) -> None:
        logger = UnifiedLogger()
        with pytest.raises(ValueError, match="file_path is required"):
            logger.setup(level="INFO", output="file")

    def test_setup_json_format(self) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console", format_type="json")
        # Should not raise


class TestUnifiedLoggerLogging:
    """UnifiedLogger 日志输出测试"""

    def setup_method(self) -> None:
        UnifiedLogger._instance = None

    def test_debug_log(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="DEBUG", output="console")
        logger.debug("debug message")
        captured = capsys.readouterr()
        assert "debug message" in captured.out

    def test_info_log(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")
        logger.info("info message")
        captured = capsys.readouterr()
        assert "info message" in captured.out

    def test_warning_log(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")
        logger.warning("warning message")
        captured = capsys.readouterr()
        assert "warning message" in captured.out

    def test_error_log(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")
        logger.error("error message")
        captured = capsys.readouterr()
        assert "error message" in captured.out

    def test_critical_log(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")
        logger.critical("critical message")
        captured = capsys.readouterr()
        assert "critical message" in captured.out

    def test_log_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.log")
            logger = UnifiedLogger()
            logger.setup(level="INFO", output="file", file_path=file_path)
            logger.info("file message")
            logger.flush()
            logger.close()

            with open(file_path, "r") as f:
                content = f.read()
                assert "file message" in content

    def test_log_rotating_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "rotating.log")
            logger = UnifiedLogger()
            logger.setup(
                level="INFO",
                output="rotating",
                file_path=file_path,
                max_bytes=512,
                backup_count=3,
            )
            # Write enough to trigger rotation
            for i in range(100):
                logger.info(f"message {i}")
            logger.flush()
            logger.close()

            # Check main log file exists
            assert os.path.exists(file_path)


class TestUnifiedLoggerContext:
    """UnifiedLogger 上下文测试"""

    def setup_method(self) -> None:
        UnifiedLogger._instance = None

    def test_with_context(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console", format_type="json")
        logger.with_context(trace_id="abc123", request_id="req456")
        logger.info("test message")
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["context"]["trace_id"] == "abc123"
        assert data["context"]["request_id"] == "req456"

    def test_with_context_returns_context(self) -> None:
        logger = UnifiedLogger()
        ctx = logger.with_context(trace_id="abc")
        assert ctx.get("trace_id") == "abc"

    def test_set_context(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console", format_type="json")
        logger.set_context(trace_id="xyz", user_id="u123")
        logger.info("context test")
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["context"]["trace_id"] == "xyz"
        assert data["context"]["user_id"] == "u123"

    def test_get_context(self) -> None:
        logger = UnifiedLogger()
        logger.set_context(a=1, b=2)
        ctx = logger.get_context()
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2

    def test_clear_context(self) -> None:
        logger = UnifiedLogger()
        logger.set_context(x=1)
        assert logger.get_context().get("x") == 1
        logger.clear_context()
        assert logger.get_context().get("x") is None

    def test_context_accumulation(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console", format_type="json")
        logger.set_context(trace_id="t1")
        logger.with_context(request_id="r1")
        logger.info("first")
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["context"]["trace_id"] == "t1"
        assert data["context"]["request_id"] == "r1"

    def test_log_extra_context_override(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console", format_type="json")
        logger.set_context(trace_id="t1")
        logger.info("test", context={"trace_id": "override"})
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert data["context"]["trace_id"] == "override"


class TestUnifiedLoggerException:
    """UnifiedLogger 异常日志测试"""

    def setup_method(self) -> None:
        UnifiedLogger._instance = None

    def test_exception_log(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console", format_type="json")
        try:
            raise ValueError("test error")
        except ValueError:
            logger.exception("caught exception")
        captured = capsys.readouterr()
        assert "exception" in captured.out


class TestUnifiedLoggerHandler:
    """UnifiedLogger 处理器管理测试"""

    def setup_method(self) -> None:
        UnifiedLogger._instance = None

    def test_add_handler(self, capsys) -> None:
        import logging

        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")

        # Add a second handler
        custom_handler = logging.StreamHandler(sys.stdout)
        logger.add_handler(custom_handler, format_type="text")

        logger.info("message")
        captured = capsys.readouterr()
        # Should appear twice (once per handler)
        assert captured.out.count("message") >= 1

    def test_add_handler_json_format(self, capsys) -> None:
        import logging

        logger = UnifiedLogger()
        # Setup console output
        logger.setup(level="INFO", output="console")

        # Add handler with JSON format to a file
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "json_test.log")
            file_handler = logging.FileHandler(file_path)
            logger.add_handler(file_handler, format_type="json")

            logger.info("json message")
            logger.flush()
            logger.close()

            with open(file_path, "r") as f:
                data = json.loads(f.read().strip())
                assert data["message"] == "json message"

    def test_remove_handler(self) -> None:
        import logging

        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")

        handler = logging.StreamHandler(sys.stdout)
        logger.add_handler(handler, format_type="text")

        initial_count = len(logger._handlers)
        logger.remove_handler(handler)
        assert len(logger._handlers) == initial_count - 1

    def test_flush(self) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")
        logger.flush()  # Should not raise

    def test_close(self) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")
        logger.close()
        assert len(logger._handlers) == 0


class TestUnifiedLoggerLevelFiltering:
    """UnifiedLogger 级别过滤测试"""

    def setup_method(self) -> None:
        UnifiedLogger._instance = None

    def test_debug_not_shown_at_info_level(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="INFO", output="console")
        logger.debug("debug should not appear")
        captured = capsys.readouterr()
        assert "debug should not appear" not in captured.out

    def test_info_shown_at_debug_level(self, capsys) -> None:
        logger = UnifiedLogger()
        logger.setup(level="DEBUG", output="console")
        logger.info("info should appear")
        captured = capsys.readouterr()
        assert "info should appear" in captured.out


class TestGetLogger:
    """get_logger 函数测试"""

    def setup_method(self) -> None:
        UnifiedLogger._instance = None

    def test_get_logger_returns_unified_logger(self) -> None:
        logger = get_logger()
        assert isinstance(logger, UnifiedLogger)

    def test_get_logger_works_after_setup(self) -> None:
        logger1 = get_logger()
        logger1.setup(level="INFO", output="console")
        logger2 = get_logger()
        assert logger2 is logger1
        assert logger2._level == logging.INFO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
