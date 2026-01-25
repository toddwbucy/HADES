"""Tests for CLI output formatting."""

import json
import time

import pytest

from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    success_response,
)


class TestCLIResponse:
    """Tests for CLIResponse dataclass."""

    def test_success_response_to_json(self):
        """Test successful response JSON serialization."""
        response = CLIResponse(
            success=True,
            command="test.command",
            data={"key": "value"},
            metadata={"count": 1},
        )
        result = json.loads(response.to_json())

        assert result["success"] is True
        assert result["command"] == "test.command"
        assert result["data"] == {"key": "value"}
        assert result["metadata"]["count"] == 1

    def test_error_response_to_json(self):
        """Test error response JSON serialization."""
        response = CLIResponse(
            success=False,
            command="test.command",
            error={"code": "TEST_ERROR", "message": "Something went wrong"},
        )
        result = json.loads(response.to_json())

        assert result["success"] is False
        assert result["command"] == "test.command"
        assert result["error"]["code"] == "TEST_ERROR"
        assert "data" not in result


class TestSuccessResponse:
    """Tests for success_response helper."""

    def test_basic_success_response(self):
        """Test creating a basic success response."""
        response = success_response(
            command="test.cmd",
            data={"items": [1, 2, 3]},
        )

        assert response.success is True
        assert response.command == "test.cmd"
        assert response.data == {"items": [1, 2, 3]}

    def test_success_response_with_list_counts_items(self):
        """Test that list data automatically adds count to metadata."""
        response = success_response(
            command="test.cmd",
            data=[1, 2, 3, 4, 5],
        )

        assert response.metadata["count"] == 5

    def test_success_response_with_results_list_counts_items(self):
        """Test that data with results key counts correctly."""
        response = success_response(
            command="test.cmd",
            data={"results": ["a", "b", "c"]},
        )

        assert response.metadata["count"] == 3

    def test_success_response_with_duration(self):
        """Test that start_time produces duration_ms in metadata."""
        start = time.time() - 0.5  # 500ms ago
        response = success_response(
            command="test.cmd",
            data={},
            start_time=start,
        )

        assert "duration_ms" in response.metadata
        assert response.metadata["duration_ms"] >= 500


class TestErrorResponse:
    """Tests for error_response helper."""

    def test_basic_error_response(self):
        """Test creating a basic error response."""
        response = error_response(
            command="test.cmd",
            code=ErrorCode.DATABASE_ERROR,
            message="Connection failed",
        )

        assert response.success is False
        assert response.command == "test.cmd"
        assert response.error["code"] == "DATABASE_ERROR"
        assert response.error["message"] == "Connection failed"

    def test_error_response_with_details(self):
        """Test error response with additional details."""
        response = error_response(
            command="test.cmd",
            code=ErrorCode.PAPER_NOT_FOUND,
            message="Paper not found",
            details={"arxiv_id": "1234.5678"},
        )

        assert response.error["details"]["arxiv_id"] == "1234.5678"

    def test_error_response_with_duration(self):
        """Test that start_time produces duration_ms in metadata."""
        start = time.time() - 0.1  # 100ms ago
        response = error_response(
            command="test.cmd",
            code=ErrorCode.UNKNOWN_ERROR,
            message="Something went wrong",
            start_time=start,
        )

        assert "duration_ms" in response.metadata
        assert response.metadata["duration_ms"] >= 100


class TestErrorCodes:
    """Tests for ErrorCode enum."""

    def test_all_error_codes_are_strings(self):
        """Verify all error codes serialize to uppercase strings."""
        for code in ErrorCode:
            assert isinstance(code.value, str)
            assert code.value == code.value.upper()

    def test_common_error_codes_exist(self):
        """Verify expected error codes are defined."""
        expected = [
            "PAPER_NOT_FOUND",
            "INVALID_ARXIV_ID",
            "DATABASE_ERROR",
            "DOWNLOAD_FAILED",
            "PROCESSING_FAILED",
            "SEARCH_FAILED",
            "QUERY_FAILED",
            "CONFIG_ERROR",
        ]
        actual = [code.value for code in ErrorCode]
        for expected_code in expected:
            assert expected_code in actual
