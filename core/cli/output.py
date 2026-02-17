"""JSON output formatting for HADES CLI.

All CLI commands return structured JSON for predictable AI parsing.
Progress messages go to stderr to avoid polluting JSON output.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Console for stderr output (doesn't pollute JSON stdout)
_stderr_console = Console(stderr=True, force_terminal=None)


class ErrorCode(str, Enum):
    """Standard error codes for CLI responses."""

    PAPER_NOT_FOUND = "PAPER_NOT_FOUND"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_ARXIV_ID = "INVALID_ARXIV_ID"
    DATABASE_ERROR = "DATABASE_ERROR"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    SEARCH_FAILED = "SEARCH_FAILED"
    QUERY_FAILED = "QUERY_FAILED"
    CONFIG_ERROR = "CONFIG_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    SERVICE_ERROR = "SERVICE_ERROR"
    GRAPH_ERROR = "GRAPH_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DESTRUCTIVE_OP_DISABLED = "DESTRUCTIVE_OP_DISABLED"
    TASK_ERROR = "TASK_ERROR"
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


# =============================================================================
# Rich Progress Bar Support
# =============================================================================


def create_progress(
    *,
    transient: bool = True,
    show_speed: bool = True,
) -> Progress:
    """Create a rich Progress instance configured for CLI use.

    Args:
        transient: If True, progress bar disappears after completion
        show_speed: If True, shows items/second rate

    Returns:
        Rich Progress instance that outputs to stderr

    Usage:
        with create_progress() as progress:
            task = progress.add_task("Processing...", total=100)
            for item in items:
                process(item)
                progress.advance(task)
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    return Progress(
        *columns,
        console=_stderr_console,
        transient=transient,
        refresh_per_second=10,
    )


@contextmanager
def progress_bar(
    description: str,
    total: int | None = None,
    *,
    transient: bool = True,
) -> Generator[tuple[Progress, TaskID], None, None]:
    """Context manager for a single progress bar task.

    Args:
        description: Task description shown in progress bar
        total: Total number of items (None for indeterminate)
        transient: If True, progress bar disappears after completion

    Yields:
        Tuple of (Progress, TaskID) for updating progress

    Usage:
        with progress_bar("Processing papers", total=100) as (prog, task):
            for paper in papers:
                process(paper)
                prog.advance(task)
    """
    with create_progress(transient=transient) as prog:
        task_id = prog.add_task(description, total=total)
        yield prog, task_id


class BatchProgressTracker:
    """Progress tracker for batch operations with auto-updating stats.

    Provides a simple interface for tracking batch progress with
    automatic rate calculation and ETA.

    Usage:
        tracker = BatchProgressTracker("Syncing papers", total=1000)
        tracker.start()
        for batch in batches:
            process(batch)
            tracker.update(len(batch))
        tracker.finish()

    Or as context manager:
        with BatchProgressTracker("Syncing papers", total=1000) as tracker:
            for batch in batches:
                process(batch)
                tracker.update(len(batch))
    """

    def __init__(
        self,
        description: str,
        total: int | None = None,
        *,
        transient: bool = True,
    ):
        """Initialize batch progress tracker.

        Args:
            description: Task description
            total: Total items to process (None for indeterminate)
            transient: If True, progress bar disappears after completion
        """
        self.description = description
        self.total = total
        self.transient = transient
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._completed = 0
        self._start_time: float | None = None

    def start(self) -> BatchProgressTracker:
        """Start the progress bar."""
        self._progress = create_progress(transient=self.transient)
        self._progress.start()
        self._task_id = self._progress.add_task(self.description, total=self.total)
        self._start_time = time.time()
        self._completed = 0
        return self

    def update(self, advance: int = 1, *, description: str | None = None) -> None:
        """Update progress by advancing the counter.

        Args:
            advance: Number of items completed
            description: Optional new description
        """
        if self._progress is None or self._task_id is None:
            return

        self._completed += advance
        update_kwargs: dict[str, Any] = {"advance": advance}
        if description is not None:
            update_kwargs["description"] = description
        self._progress.update(self._task_id, **update_kwargs)

    def set_total(self, total: int) -> None:
        """Update the total count (useful when total is discovered during processing)."""
        if self._progress is None or self._task_id is None:
            return
        self.total = total
        self._progress.update(self._task_id, total=total)

    def finish(self, *, message: str | None = None) -> None:
        """Finish and clean up the progress bar.

        Args:
            message: Optional completion message to print
        """
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

        if message:
            progress(message)
        elif self._start_time is not None:
            elapsed = time.time() - self._start_time
            rate = self._completed / elapsed if elapsed > 0 else 0
            progress(f"Completed {self._completed:,} items in {elapsed:.1f}s ({rate:.1f}/s)")

    def __enter__(self) -> BatchProgressTracker:
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.finish()

    @property
    def completed(self) -> int:
        """Number of items completed."""
        return self._completed


def is_terminal() -> bool:
    """Check if stderr is a terminal (supports rich output)."""
    return _stderr_console.is_terminal
