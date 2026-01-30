"""Unit tests for State Manager and Checkpoint Manager.

Tests for:
- StateManager: load, save, checkpoints, stats, metadata
- CheckpointManager: load, save, mark_processed, get_unprocessed
"""

import tempfile
from pathlib import Path


class TestStateManager:
    """Tests for StateManager class."""

    def test_creates_initial_state(self):
        """Should create initial state structure."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            manager = StateManager(state_file, "test_process")

            assert manager.state["process_name"] == "test_process"
            assert "created" in manager.state
            assert manager.state["checkpoints"] == {}
            assert manager.state["stats"] == {}
            assert manager.state["metadata"] == {}
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_save_and_load(self):
        """Should save and load state correctly."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            # Create and save state
            manager1 = StateManager(state_file, "test_process")
            manager1.set_checkpoint("step1", "completed")
            manager1.update_stats(items_processed=10)
            manager1.save()

            # Load in new instance
            manager2 = StateManager(state_file, "test_process")

            assert manager2.get_checkpoint("step1") == "completed"
            assert manager2.state["stats"]["items_processed"] == 10
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_atomic_save(self):
        """Should save atomically (temp file then rename)."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            manager = StateManager(state_file, "test_process")
            manager.set_checkpoint("test", "value")

            # Save should succeed
            result = manager.save()
            assert result is True

            # File should exist
            assert Path(state_file).exists()

            # Temp file should not exist
            temp_file = Path(state_file).with_suffix(".tmp")
            assert not temp_file.exists()
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_process_name_validation(self):
        """Should reject state files from different processes."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            # Create state for process A
            manager_a = StateManager(state_file, "process_a")
            manager_a.set_checkpoint("step", "done")
            manager_a.save()

            # Try to load for process B - should start fresh and backup old file
            manager_b = StateManager(state_file, "process_b")

            # Should have fresh state (not process_a's checkpoint)
            assert manager_b.get_checkpoint("step") is None
            assert manager_b.state["process_name"] == "process_b"
        finally:
            # Clean up main file and any backup files
            Path(state_file).unlink(missing_ok=True)
            for bak in Path(state_file).parent.glob("*.bak"):
                bak.unlink(missing_ok=True)

    def test_set_and_get_checkpoint(self):
        """Should set and get checkpoints with timestamps."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            manager = StateManager(state_file, "test_process")

            manager.set_checkpoint("current_page", 5)
            manager.set_checkpoint("last_id", "abc123")

            assert manager.get_checkpoint("current_page") == 5
            assert manager.get_checkpoint("last_id") == "abc123"
            assert manager.get_checkpoint("nonexistent") is None
            assert manager.get_checkpoint("nonexistent", "default") == "default"

            # Checkpoint should have timestamp
            checkpoints = manager.state["checkpoints"]
            assert "timestamp" in checkpoints["current_page"]
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_update_stats(self):
        """Should update statistics."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            manager = StateManager(state_file, "test_process")

            manager.update_stats(processed=10, failed=2)

            assert manager.state["stats"]["processed"] == 10
            assert manager.state["stats"]["failed"] == 2
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_increment_stat(self):
        """Should increment statistics counter."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            manager = StateManager(state_file, "test_process")

            manager.increment_stat("counter")
            assert manager.state["stats"]["counter"] == 1

            manager.increment_stat("counter")
            assert manager.state["stats"]["counter"] == 2

            manager.increment_stat("counter", 5)
            assert manager.state["stats"]["counter"] == 7
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_set_and_get_metadata(self):
        """Should set and get metadata."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            manager = StateManager(state_file, "test_process")

            manager.set_metadata("source", "arxiv")
            manager.set_metadata("config", {"batch_size": 32})

            assert manager.get_metadata("source") == "arxiv"
            assert manager.get_metadata("config") == {"batch_size": 32}
            assert manager.get_metadata("missing") is None
            assert manager.get_metadata("missing", "default") == "default"
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_clear(self):
        """Should clear state file and reset."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            manager = StateManager(state_file, "test_process")
            manager.set_checkpoint("step", "done")
            manager.save()

            assert Path(state_file).exists()

            manager.clear()

            assert not Path(state_file).exists()
            assert manager.state["checkpoints"] == {}
            assert manager.state["stats"] == {}
        finally:
            Path(state_file).unlink(missing_ok=True)

    def test_get_progress_summary(self):
        """Should return progress summary."""
        from core.workflows.state.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            manager = StateManager(state_file, "test_process")
            manager.set_checkpoint("step1", "done")
            manager.set_checkpoint("step2", "done")
            manager.update_stats(processed=100, failed=5)

            summary = manager.get_progress_summary()

            assert summary["process"] == "test_process"
            assert summary["checkpoints"] == 2
            assert summary["stats"]["processed"] == 100
            assert summary["stats"]["failed"] == 5
        finally:
            Path(state_file).unlink(missing_ok=True)


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    def test_creates_empty_checkpoint(self):
        """Should create empty checkpoint set."""
        from core.workflows.state.state_manager import CheckpointManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_file = f.name
        Path(checkpoint_file).unlink()  # Remove so it starts fresh

        try:
            manager = CheckpointManager(checkpoint_file)
            assert len(manager.processed) == 0
        finally:
            Path(checkpoint_file).unlink(missing_ok=True)

    def test_mark_and_check_processed(self):
        """Should mark items as processed and check status."""
        from core.workflows.state.state_manager import CheckpointManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_file = f.name
        Path(checkpoint_file).unlink()

        try:
            manager = CheckpointManager(checkpoint_file)

            assert manager.is_processed("item1") is False

            manager.mark_processed("item1")

            assert manager.is_processed("item1") is True
            assert manager.is_processed("item2") is False
        finally:
            Path(checkpoint_file).unlink(missing_ok=True)

    def test_mark_many_processed(self):
        """Should mark multiple items as processed."""
        from core.workflows.state.state_manager import CheckpointManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_file = f.name
        Path(checkpoint_file).unlink()

        try:
            manager = CheckpointManager(checkpoint_file)

            manager.mark_many_processed(["item1", "item2", "item3"])

            assert manager.is_processed("item1") is True
            assert manager.is_processed("item2") is True
            assert manager.is_processed("item3") is True
            assert manager.is_processed("item4") is False
        finally:
            Path(checkpoint_file).unlink(missing_ok=True)

    def test_save_and_load(self):
        """Should save and load checkpoints."""
        from core.workflows.state.state_manager import CheckpointManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_file = f.name
        Path(checkpoint_file).unlink()

        try:
            # Create and save
            manager1 = CheckpointManager(checkpoint_file)
            manager1.mark_many_processed(["a", "b", "c"])
            manager1.save()

            # Load in new instance
            manager2 = CheckpointManager(checkpoint_file)

            assert manager2.is_processed("a") is True
            assert manager2.is_processed("b") is True
            assert manager2.is_processed("c") is True
            assert len(manager2.processed) == 3
        finally:
            Path(checkpoint_file).unlink(missing_ok=True)

    def test_get_unprocessed(self):
        """Should filter to unprocessed items."""
        from core.workflows.state.state_manager import CheckpointManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_file = f.name
        Path(checkpoint_file).unlink()

        try:
            manager = CheckpointManager(checkpoint_file)
            manager.mark_many_processed(["item1", "item3"])

            all_items = ["item1", "item2", "item3", "item4"]
            unprocessed = manager.get_unprocessed(all_items)

            assert unprocessed == ["item2", "item4"]
        finally:
            Path(checkpoint_file).unlink(missing_ok=True)

    def test_clear(self):
        """Should clear all checkpoints."""
        from core.workflows.state.state_manager import CheckpointManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_file = f.name
        Path(checkpoint_file).unlink()

        try:
            manager = CheckpointManager(checkpoint_file)
            manager.mark_many_processed(["a", "b", "c"])
            manager.save()

            assert Path(checkpoint_file).exists()

            manager.clear()

            assert not Path(checkpoint_file).exists()
            assert len(manager.processed) == 0
        finally:
            Path(checkpoint_file).unlink(missing_ok=True)

    def test_get_stats(self):
        """Should return checkpoint statistics."""
        from core.workflows.state.state_manager import CheckpointManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_file = f.name
        Path(checkpoint_file).unlink()

        try:
            manager = CheckpointManager(checkpoint_file)
            manager.mark_many_processed(["a", "b", "c", "d", "e"])

            stats = manager.get_stats()

            assert stats["total_processed"] == 5
        finally:
            Path(checkpoint_file).unlink(missing_ok=True)

    def test_deterministic_save(self):
        """Should save sorted items when deterministic=True."""
        from core.workflows.state.state_manager import CheckpointManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_file = f.name
        Path(checkpoint_file).unlink()

        try:
            manager = CheckpointManager(checkpoint_file)
            # Add in non-sorted order
            manager.mark_many_processed(["z", "a", "m"])
            manager.save(deterministic=True)

            # Read raw file and verify sorted
            import orjson
            data = orjson.loads(Path(checkpoint_file).read_bytes())
            assert data == ["a", "m", "z"]
        finally:
            Path(checkpoint_file).unlink(missing_ok=True)

    def test_handles_string_conversion(self):
        """Should convert items to strings."""
        from core.workflows.state.state_manager import CheckpointManager

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            checkpoint_file = f.name
        Path(checkpoint_file).unlink()

        try:
            manager = CheckpointManager(checkpoint_file)

            # Mark with integer
            manager.mark_processed(123)

            # Check with string
            assert manager.is_processed("123") is True
            assert manager.is_processed(123) is True
        finally:
            Path(checkpoint_file).unlink(missing_ok=True)
