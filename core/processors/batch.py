"""Batch processing with progress reporting, error isolation, and resume.

Handles large-scale ingestion with:
- JSON progress to stderr (stdout stays clean for final result)
- Per-document error isolation (one failure doesn't stop the batch)
- Resume via state file recording completed IDs

Usage:
    from core.processors.batch import BatchProcessor

    processor = BatchProcessor()
    results = processor.process_batch(
        items=["2501.12345", "2501.67890", "paper.pdf"],
        process_fn=lambda item: ingest_single(item),
    )
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.cli.output import progress


@dataclass
class BatchState:
    """Persistent state for resumable batch processing."""

    completed: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)  # id -> error
    started_at: str | None = None
    last_updated: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": list(self.completed),
            "failed": self.failed,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchState:
        return cls(
            completed=set(data.get("completed", [])),
            failed=data.get("failed", {}),
            started_at=data.get("started_at"),
            last_updated=data.get("last_updated"),
        )


@dataclass
class BatchResult:
    """Result of batch processing."""

    total: int
    completed: int
    failed: int
    skipped: int
    results: list[dict[str, Any]]
    errors: dict[str, str]
    duration_seconds: float


class BatchProcessor:
    """Batch processor with progress reporting and resume support.

    Attributes:
        state_file: Path to state file for resume functionality.
        state: Current batch state.
    """

    DEFAULT_STATE_FILE = ".hades-batch-state.json"

    def __init__(
        self,
        state_file: str | Path | None = None,
        progress_interval: float = 1.0,
    ) -> None:
        """Initialize batch processor.

        Args:
            state_file: Path to state file. None disables resume.
            progress_interval: Minimum seconds between progress updates.
        """
        self.state_file = Path(state_file) if state_file else None
        self.progress_interval = progress_interval
        self.state = BatchState()
        self._last_progress_time = 0.0

    def load_state(self) -> bool:
        """Load state from file if it exists.

        Returns:
            True if state was loaded, False otherwise.
        """
        if not self.state_file or not self.state_file.exists():
            return False

        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            self.state = BatchState.from_dict(data)
            return True
        except (json.JSONDecodeError, KeyError, OSError):
            # OSError covers permission issues, IO errors, etc.
            return False

    def save_state(self) -> None:
        """Save current state to file."""
        if not self.state_file:
            return

        from datetime import UTC, datetime

        self.state.last_updated = datetime.now(UTC).isoformat()
        self.state_file.write_text(json.dumps(self.state.to_dict(), indent=2), encoding="utf-8")

    def clear_state(self) -> None:
        """Clear state file."""
        if self.state_file and self.state_file.exists():
            self.state_file.unlink()
        self.state = BatchState()

    def _report_progress(
        self,
        current: int,
        total: int,
        item_id: str,
        status: str,
        force: bool = False,
    ) -> None:
        """Report progress to stderr as JSON.

        Args:
            current: Current item number (1-indexed).
            total: Total number of items.
            item_id: Current item identifier.
            status: Status string (processing, completed, failed, skipped).
            force: Force output even if interval hasn't elapsed.
        """
        now = time.time()
        if not force and (now - self._last_progress_time) < self.progress_interval:
            return

        self._last_progress_time = now
        progress_data = {
            "type": "progress",
            "current": current,
            "total": total,
            "percent": round(100 * current / total, 1) if total > 0 else 0,
            "item": item_id,
            "status": status,
        }
        print(json.dumps(progress_data), file=sys.stderr)

    def process_batch(
        self,
        items: list[str],
        process_fn: Callable[[str], dict[str, Any]],
        *,
        resume: bool = False,
    ) -> BatchResult:
        """Process a batch of items with error isolation.

        Args:
            items: List of item identifiers to process.
            process_fn: Function that processes a single item.
                Should return dict with at least {"success": bool}.
                On failure, should include {"error": str}.
            resume: If True, skip already-completed items from state file.

        Returns:
            BatchResult with summary and individual results.
        """
        from datetime import UTC, datetime

        start_time = time.time()

        # Load state if resuming
        if resume and self.state_file:
            if self.load_state():
                progress(f"Resuming batch: {len(self.state.completed)} already completed")
            else:
                progress("No previous state found, starting fresh")

        # Initialize state for new batch
        if not self.state.started_at:
            self.state.started_at = datetime.now(UTC).isoformat()

        results: list[dict[str, Any]] = []
        skipped = 0
        completed = 0
        failed = 0

        total = len(items)
        for i, item_id in enumerate(items, 1):
            # Skip if already completed
            if item_id in self.state.completed:
                self._report_progress(i, total, item_id, "skipped")
                skipped += 1
                results.append(
                    {
                        "id": item_id,
                        "success": True,
                        "skipped": True,
                        "message": "Already completed in previous run",
                    }
                )
                continue

            # Skip if previously failed (can retry with --force or clear state)
            if item_id in self.state.failed:
                self._report_progress(i, total, item_id, "skipped")
                skipped += 1
                results.append(
                    {
                        "id": item_id,
                        "success": False,
                        "skipped": True,
                        "error": self.state.failed[item_id],
                    }
                )
                continue

            # Process item with error isolation
            self._report_progress(i, total, item_id, "processing", force=True)

            try:
                result = process_fn(item_id)
                result["id"] = item_id

                if result.get("success"):
                    self.state.completed.add(item_id)
                    completed += 1
                    self._report_progress(i, total, item_id, "completed", force=True)
                else:
                    error_msg = result.get("error", "Unknown error")
                    self.state.failed[item_id] = error_msg
                    failed += 1
                    self._report_progress(i, total, item_id, "failed", force=True)

                results.append(result)

            except Exception as e:
                # Isolate the error — don't let it stop the batch
                error_msg = str(e)
                self.state.failed[item_id] = error_msg
                failed += 1
                self._report_progress(i, total, item_id, "failed", force=True)
                results.append(
                    {
                        "id": item_id,
                        "success": False,
                        "error": error_msg,
                    }
                )

            # Save state after each item for resume capability
            self.save_state()

        duration = time.time() - start_time

        # Clear state file if everything succeeded
        if failed == 0 and self.state_file:
            self.clear_state()

        return BatchResult(
            total=total,
            completed=completed,
            failed=failed,
            skipped=skipped,
            results=results,
            errors=dict(self.state.failed),
            duration_seconds=round(duration, 2),
        )
