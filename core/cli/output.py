"""JSON output formatting for HADES CLI.

All CLI commands return structured JSON for predictable AI parsing.
Progress messages go to stderr to avoid polluting JSON output.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standard error codes for CLI responses."""

    PAPER_NOT_FOUND = "PAPER_NOT_FOUND"
    INVALID_ARXIV_ID = "INVALID_ARXIV_ID"
    DATABASE_ERROR = "DATABASE_ERROR"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    SEARCH_FAILED = "SEARCH_FAILED"
    QUERY_FAILED = "QUERY_FAILED"
    CONFIG_ERROR = "CONFIG_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    SERVICE_ERROR = "SERVICE_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class CLIResponse:
    """Structured CLI response for JSON output."""

    success: bool
    command: str
    data: dict[str, Any] | list[Any] | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int | None = 2) -> str:
        """Convert response to JSON string."""
        result: dict[str, Any] = {
            "success": self.success,
            "command": self.command,
        }

        if self.data is not None:
            result["data"] = self.data

        if self.error is not None:
            result["error"] = self.error

        if self.metadata:
            result["metadata"] = self.metadata

        return json.dumps(result, indent=indent, default=str)


def success_response(
    command: str,
    data: dict[str, Any] | list[Any],
    start_time: float | None = None,
    **extra_metadata: Any,
) -> CLIResponse:
    """Create a successful CLI response."""
    metadata: dict[str, Any] = {}

    if start_time is not None:
        metadata["duration_ms"] = int((time.time() - start_time) * 1000)

    if isinstance(data, list):
        metadata["count"] = len(data)
    elif isinstance(data, dict) and "results" in data:
        metadata["count"] = len(data["results"])

    metadata.update(extra_metadata)

    return CLIResponse(
        success=True,
        command=command,
        data=data,
        metadata=metadata,
    )


def error_response(
    command: str,
    code: ErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
    start_time: float | None = None,
) -> CLIResponse:
    """Create an error CLI response."""
    metadata: dict[str, Any] = {}

    if start_time is not None:
        metadata["duration_ms"] = int((time.time() - start_time) * 1000)

    error_dict: dict[str, Any] = {
        "code": code.value,
        "message": message,
    }

    if details:
        error_dict["details"] = details

    return CLIResponse(
        success=False,
        command=command,
        error=error_dict,
        metadata=metadata,
    )


def print_response(response: CLIResponse) -> None:
    """Print response to stdout."""
    print(response.to_json())


def progress(message: str) -> None:
    """Print progress message to stderr (doesn't pollute JSON output)."""
    print(f"PROGRESS: {message}", file=sys.stderr, flush=True)


def warn(message: str) -> None:
    """Print warning message to stderr."""
    print(f"WARNING: {message}", file=sys.stderr, flush=True)
