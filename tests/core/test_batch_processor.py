"""Tests for core.processors.batch — batch processing with progress and resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.processors.batch import BatchProcessor, BatchResult, BatchState


class TestBatchState:
    def test_to_dict(self):
        state = BatchState(
            completed={"a", "b"},
            failed={"c": "error"},
            started_at="2025-01-01T00:00:00",
            last_updated="2025-01-01T00:01:00",
        )
        d = state.to_dict()
        assert set(d["completed"]) == {"a", "b"}
        assert d["failed"] == {"c": "error"}
        assert d["started_at"] == "2025-01-01T00:00:00"
        assert d["last_updated"] == "2025-01-01T00:01:00"

    def test_from_dict(self):
        data = {
            "completed": ["a", "b"],
            "failed": {"c": "error"},
            "started_at": "2025-01-01T00:00:00",
            "last_updated": "2025-01-01T00:01:00",
        }
        state = BatchState.from_dict(data)
        assert state.completed == {"a", "b"}
        assert state.failed == {"c": "error"}
        assert state.started_at == "2025-01-01T00:00:00"
        assert state.last_updated == "2025-01-01T00:01:00"

    def test_from_dict_empty(self):
        state = BatchState.from_dict({})
        assert state.completed == set()
        assert state.failed == {}
        assert state.started_at is None
        assert state.last_updated is None


class TestBatchProcessor:
    def test_process_batch_all_success(self, tmp_path: Path):
        """All items succeed."""
        state_file = tmp_path / "state.json"
        processor = BatchProcessor(state_file=state_file)

        def process_fn(item_id: str) -> dict[str, Any]:
            return {"success": True, "value": item_id}

        result = processor.process_batch(["a", "b", "c"], process_fn)

        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert result.completed == 3
        assert result.failed == 0
        assert result.skipped == 0
        assert len(result.results) == 3
        assert all(r["success"] for r in result.results)
        # State file should be cleared on success
        assert not state_file.exists()

    def test_process_batch_partial_failure(self, tmp_path: Path):
        """Some items fail."""
        state_file = tmp_path / "state.json"
        processor = BatchProcessor(state_file=state_file)

        def process_fn(item_id: str) -> dict[str, Any]:
            if item_id == "b":
                return {"success": False, "error": "test error"}
            return {"success": True}

        result = processor.process_batch(["a", "b", "c"], process_fn)

        assert result.total == 3
        assert result.completed == 2
        assert result.failed == 1
        assert result.skipped == 0
        assert "b" in result.errors
        assert result.errors["b"] == "test error"
        # State file should persist on failure
        assert state_file.exists()

    def test_process_batch_exception_isolation(self, tmp_path: Path):
        """Exceptions don't stop the batch."""
        state_file = tmp_path / "state.json"
        processor = BatchProcessor(state_file=state_file)

        def process_fn(item_id: str) -> dict[str, Any]:
            if item_id == "b":
                raise ValueError("test exception")
            return {"success": True}

        result = processor.process_batch(["a", "b", "c"], process_fn)

        assert result.total == 3
        assert result.completed == 2
        assert result.failed == 1
        assert result.skipped == 0
        assert "b" in result.errors
        assert "test exception" in result.errors["b"]

    def test_process_batch_resume(self, tmp_path: Path):
        """Resume skips completed items."""
        state_file = tmp_path / "state.json"

        # First run: complete a and b, fail c
        processor = BatchProcessor(state_file=state_file)

        def process_fn1(item_id: str) -> dict[str, Any]:
            if item_id == "c":
                return {"success": False, "error": "first failure"}
            return {"success": True}

        result1 = processor.process_batch(["a", "b", "c"], process_fn1)
        assert result1.completed == 2
        assert result1.failed == 1
        assert state_file.exists()

        # Second run with resume: a and b should be skipped
        processor2 = BatchProcessor(state_file=state_file)

        def process_fn2(item_id: str) -> dict[str, Any]:
            # This shouldn't be called for a or b
            if item_id in ("a", "b"):
                raise AssertionError(f"Should not process {item_id}")
            return {"success": True}

        # Note: c will also be skipped because it's in failed list
        result2 = processor2.process_batch(["a", "b", "c"], process_fn2, resume=True)
        assert result2.skipped == 3  # a, b completed; c failed
        assert result2.completed == 0  # Nothing new to process

    def test_process_batch_no_state_file(self):
        """Works without state file (no resume capability)."""
        processor = BatchProcessor(state_file=None)

        def process_fn(item_id: str) -> dict[str, Any]:
            return {"success": True}

        result = processor.process_batch(["a", "b"], process_fn)

        assert result.total == 2
        assert result.completed == 2

    def test_load_state_nonexistent(self, tmp_path: Path):
        """Load state returns False for missing file."""
        processor = BatchProcessor(state_file=tmp_path / "missing.json")
        assert processor.load_state() is False

    def test_load_state_invalid_json(self, tmp_path: Path):
        """Load state returns False for invalid JSON."""
        state_file = tmp_path / "state.json"
        state_file.write_text("not json")
        processor = BatchProcessor(state_file=state_file)
        assert processor.load_state() is False

    def test_save_state(self, tmp_path: Path):
        """Save state writes valid JSON."""
        state_file = tmp_path / "state.json"
        processor = BatchProcessor(state_file=state_file)
        processor.state.completed.add("test")
        processor.state.failed["bad"] = "error"
        processor.save_state()

        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert "test" in data["completed"]
        assert data["failed"]["bad"] == "error"
        assert "last_updated" in data

    def test_clear_state(self, tmp_path: Path):
        """Clear state removes file and resets state."""
        state_file = tmp_path / "state.json"
        state_file.write_text("{}")
        processor = BatchProcessor(state_file=state_file)
        processor.state.completed.add("test")

        processor.clear_state()

        assert not state_file.exists()
        assert processor.state.completed == set()

    def test_batch_result_duration(self, tmp_path: Path):
        """Batch result includes duration."""
        processor = BatchProcessor(state_file=None)

        def process_fn(item_id: str) -> dict[str, Any]:
            return {"success": True}

        result = processor.process_batch(["a"], process_fn)

        assert result.duration_seconds >= 0
        assert isinstance(result.duration_seconds, float)

    def test_empty_batch(self, tmp_path: Path):
        """Empty batch returns zeros."""
        processor = BatchProcessor(state_file=None)

        def process_fn(item_id: str) -> dict[str, Any]:
            return {"success": True}

        result = processor.process_batch([], process_fn)

        assert result.total == 0
        assert result.completed == 0
        assert result.failed == 0
        assert result.skipped == 0
