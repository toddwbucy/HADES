"""Tests for CLI command decorators."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import typer

from core.cli.decorators import cli_command
from core.cli.output import CLIResponse, ErrorCode


class TestCliCommandDecorator:
    """Tests for cli_command decorator."""

    def test_injects_start_time(self) -> None:
        """Decorator should inject start_time into the function call."""
        received_start_time = None

        @cli_command("test.command", ErrorCode.UNKNOWN_ERROR)
        def test_func(start_time: float) -> CLIResponse:
            nonlocal received_start_time
            received_start_time = start_time
            return CLIResponse(success=True, command="test.command")

        with patch("core.cli.decorators.print_response"):
            test_func()

        assert received_start_time is not None
        assert received_start_time > 0

    def test_prints_successful_response(self) -> None:
        """Decorator should print successful responses."""

        @cli_command("test.command", ErrorCode.UNKNOWN_ERROR)
        def test_func(start_time: float) -> CLIResponse:
            return CLIResponse(success=True, command="test.command", data={"result": "ok"})

        with patch("core.cli.decorators.print_response") as mock_print:
            test_func()
            mock_print.assert_called_once()
            response = mock_print.call_args[0][0]
            assert response.success is True
            assert response.data == {"result": "ok"}

    def test_exits_with_code_1_on_failure_response(self) -> None:
        """Decorator should exit with code 1 when response.success is False."""

        @cli_command("test.command", ErrorCode.UNKNOWN_ERROR)
        def test_func(start_time: float) -> CLIResponse:
            return CLIResponse(success=False, command="test.command")

        with patch("core.cli.decorators.print_response"):
            with pytest.raises(typer.Exit) as exc_info:
                test_func()
            assert exc_info.value.exit_code == 1

    def test_wraps_exceptions_in_error_response(self) -> None:
        """Decorator should catch exceptions and return error response."""

        @cli_command("test.command", ErrorCode.PROCESSING_FAILED)
        def test_func(start_time: float) -> CLIResponse:
            raise ValueError("Something went wrong")

        with patch("core.cli.decorators.print_response") as mock_print:
            with pytest.raises(typer.Exit) as exc_info:
                test_func()

            assert exc_info.value.exit_code == 1
            mock_print.assert_called_once()
            response = mock_print.call_args[0][0]
            assert response.success is False
            assert response.error["code"] == ErrorCode.PROCESSING_FAILED.value
            assert "Something went wrong" in response.error["message"]

    def test_uses_specified_error_code(self) -> None:
        """Decorator should use the specified error code for exceptions."""

        @cli_command("test.command", ErrorCode.DATABASE_ERROR)
        def test_func(start_time: float) -> CLIResponse:
            raise RuntimeError("DB connection failed")

        with patch("core.cli.decorators.print_response") as mock_print:
            with pytest.raises(typer.Exit):
                test_func()

            response = mock_print.call_args[0][0]
            assert response.error["code"] == ErrorCode.DATABASE_ERROR.value

    def test_preserves_function_metadata(self) -> None:
        """Decorator should preserve function name and docstring."""

        @cli_command("test.command", ErrorCode.UNKNOWN_ERROR)
        def my_test_function(start_time: float) -> CLIResponse:
            """This is my docstring."""
            return CLIResponse(success=True, command="test.command")

        assert my_test_function.__name__ == "my_test_function"
        assert my_test_function.__doc__ == "This is my docstring."

    def test_passes_through_kwargs(self) -> None:
        """Decorator should pass kwargs to the wrapped function."""
        received_args = {}

        @cli_command("test.command", ErrorCode.UNKNOWN_ERROR)
        def test_func(name: str, count: int, start_time: float) -> CLIResponse:
            received_args["name"] = name
            received_args["count"] = count
            return CLIResponse(success=True, command="test.command")

        with patch("core.cli.decorators.print_response"):
            test_func(name="test", count=42)

        assert received_args["name"] == "test"
        assert received_args["count"] == 42
