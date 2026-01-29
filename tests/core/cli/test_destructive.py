"""Tests for destructive operation protection."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from core.cli.destructive import (
    ENV_VAR,
    ENABLED_VALUE,
    check_destructive_allowed,
    is_destructive_ops_enabled,
    is_interactive,
)
from core.cli.output import ErrorCode


class TestIsDestructiveOpsEnabled:
    """Tests for is_destructive_ops_enabled()."""

    def test_returns_false_when_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_VAR, None)
            assert is_destructive_ops_enabled() is False

    def test_returns_false_when_wrong_value(self) -> None:
        with patch.dict(os.environ, {ENV_VAR: "true"}):
            assert is_destructive_ops_enabled() is False

    def test_returns_true_when_enabled(self) -> None:
        with patch.dict(os.environ, {ENV_VAR: ENABLED_VALUE}):
            assert is_destructive_ops_enabled() is True

    def test_case_insensitive(self) -> None:
        with patch.dict(os.environ, {ENV_VAR: "ENABLED"}):
            assert is_destructive_ops_enabled() is True

        with patch.dict(os.environ, {ENV_VAR: "Enabled"}):
            assert is_destructive_ops_enabled() is True


class TestIsInteractive:
    """Tests for is_interactive()."""

    def test_returns_bool(self) -> None:
        # Just verify it returns a boolean
        result = is_interactive()
        assert isinstance(result, bool)


class TestCheckDestructiveAllowed:
    """Tests for check_destructive_allowed()."""

    def test_blocks_when_env_var_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_VAR, None)

            result = check_destructive_allowed(
                command="test.command",
                operation_desc="test operation",
                confirm_text="CONFIRM",
                start_time=0.0,
            )

            assert result is not None
            assert result.success is False
            assert result.error["code"] == ErrorCode.DESTRUCTIVE_OP_DISABLED.value
            assert ENV_VAR in result.error["message"]

    def test_blocks_when_not_interactive_and_no_force(self) -> None:
        with (
            patch.dict(os.environ, {ENV_VAR: ENABLED_VALUE}),
            patch("core.cli.destructive.is_interactive", return_value=False),
        ):
            result = check_destructive_allowed(
                command="test.command",
                operation_desc="test operation",
                confirm_text="CONFIRM",
                start_time=0.0,
                force=False,
            )

            assert result is not None
            assert result.success is False
            assert "Confirmation required" in result.error["message"]

    def test_allows_with_force_flag(self) -> None:
        with (
            patch.dict(os.environ, {ENV_VAR: ENABLED_VALUE}),
            patch("core.cli.destructive.is_interactive", return_value=False),
        ):
            result = check_destructive_allowed(
                command="test.command",
                operation_desc="test operation",
                confirm_text="CONFIRM",
                start_time=0.0,
                force=True,
            )

            # Should return None (allowed)
            assert result is None

    def test_error_includes_operation_details(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop(ENV_VAR, None)

            result = check_destructive_allowed(
                command="database.purge",
                operation_desc="purge paper 2409.04701",
                confirm_text="PURGE 2409.04701",
                start_time=0.0,
            )

            assert result.error["details"]["operation"] == "purge paper 2409.04701"
            assert "hint" in result.error["details"]
