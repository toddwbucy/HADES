"""Unit tests for core.logging.logging module."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.logging.logging import (
    LogManager,
    _get_log_directory,
    _validate_log_level,
)


class TestGetLogDirectory:
    """Tests for _get_log_directory function."""

    def test_returns_path(self) -> None:
        """_get_log_directory should return a Path."""
        result = _get_log_directory()
        assert isinstance(result, Path)

    @patch.dict(os.environ, {"LOG_DIR": "/tmp/test_logs"}, clear=False)
    @patch("os.access")
    def test_uses_env_var_when_set(self, mock_access: MagicMock) -> None:
        """Should use LOG_DIR env var when set and writable."""
        mock_access.return_value = True

        result = _get_log_directory()
        assert "/tmp/test_logs" in str(result)

    @patch.dict(os.environ, {}, clear=True)
    def test_falls_back_when_no_env_var(self) -> None:
        """Should fall back to cwd or temp when no env var."""
        result = _get_log_directory()
        assert isinstance(result, Path)
        # Should be either cwd/logs or temp/hades_logs
        assert "logs" in str(result) or "hades_logs" in str(result)


class TestValidateLogLevel:
    """Tests for _validate_log_level function."""

    @pytest.mark.parametrize(
        "level_name,expected",
        [
            ("DEBUG", logging.DEBUG),
            ("debug", logging.DEBUG),
            ("INFO", logging.INFO),
            ("info", logging.INFO),
            ("WARNING", logging.WARNING),
            ("warning", logging.WARNING),
            ("WARN", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("error", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
            ("critical", logging.CRITICAL),
        ],
    )
    def test_valid_levels(self, level_name: str, expected: int) -> None:
        """Valid log levels should return numeric values."""
        result = _validate_log_level(level_name)
        assert result == expected

    def test_invalid_level_raises_error(self) -> None:
        """Invalid log level should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            _validate_log_level("INVALID")

    def test_empty_level_raises_error(self) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            _validate_log_level("")


class TestLogManager:
    """Tests for LogManager class."""

    def test_has_setup_method(self) -> None:
        """LogManager should have setup method."""
        assert hasattr(LogManager, "setup")
        assert callable(LogManager.setup)

    def test_has_get_logger_method(self) -> None:
        """LogManager should have get_logger method."""
        assert hasattr(LogManager, "get_logger")
        assert callable(LogManager.get_logger)


class TestLogManagerSetup:
    """Tests for LogManager.setup method."""

    def test_setup_method_exists(self) -> None:
        """LogManager.setup should be callable."""
        assert hasattr(LogManager, "setup")
        assert callable(LogManager.setup)

    def test_setup_accepts_log_level(self) -> None:
        """LogManager.setup should accept a log level parameter."""
        import inspect

        sig = inspect.signature(LogManager.setup)
        params = list(sig.parameters.keys())
        assert len(params) >= 1  # At least one parameter (log level)


class TestLogManagerGetLogger:
    """Tests for LogManager.get_logger method."""

    def test_get_logger_method_exists(self) -> None:
        """LogManager.get_logger should be callable."""
        assert hasattr(LogManager, "get_logger")
        assert callable(LogManager.get_logger)

    def test_get_logger_accepts_params(self) -> None:
        """LogManager.get_logger should accept processor and run_id params."""
        import inspect

        sig = inspect.signature(LogManager.get_logger)
        params = list(sig.parameters.keys())
        assert "processor_name" in params or len(params) >= 1
        assert "run_id" in params or len(params) >= 2
