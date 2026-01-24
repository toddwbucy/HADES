"""Integration tests for StateManager."""

import json
from pathlib import Path

import pytest

from core.workflows.state.state_manager import CheckpointManager, StateManager


class TestStateManagerInit:
    """Tests for StateManager initialization."""

    def test_creates_new_state(self, tmp_path: Path) -> None:
        """StateManager should create new state when no file exists."""
        state_file = tmp_path / "state.json"
        manager = StateManager(str(state_file), "test_process")

        assert manager.process_name == "test_process"
        assert manager.state["process_name"] == "test_process"
        assert "created" in manager.state

    def test_loads_existing_state(self, tmp_path: Path) -> None:
        """StateManager should load existing state file."""
        state_file = tmp_path / "state.json"

        # Create initial state
        initial_state = {
            "process_name": "test_process",
            "created": "2024-01-01T00:00:00",
            "checkpoints": {"step1": {"value": True, "timestamp": "2024-01-01"}},
            "stats": {"count": 42},
            "metadata": {},
            "last_save": None,
        }
        with open(state_file, "w") as f:
            json.dump(initial_state, f)

        # Load existing state
        manager = StateManager(str(state_file), "test_process")

        assert manager.get_checkpoint("step1") is True
        assert manager.state["stats"]["count"] == 42

    def test_process_name_mismatch_creates_backup(self, tmp_path: Path) -> None:
        """StateManager should backup and start fresh on process name mismatch."""
        state_file = tmp_path / "state.json"

        # Create state for different process
        old_state = {
            "process_name": "old_process",
            "created": "2024-01-01T00:00:00",
        }
        with open(state_file, "w") as f:
            json.dump(old_state, f)

        # Load with different process name
        manager = StateManager(str(state_file), "new_process")

        # Should start fresh
        assert manager.state["process_name"] == "new_process"

        # Backup file should exist
        backup_files = list(tmp_path.glob("*.bak"))
        assert len(backup_files) == 1


class TestStateManagerOperations:
    """Tests for StateManager operations."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> StateManager:
        """Create StateManager instance."""
        return StateManager(str(tmp_path / "state.json"), "test")

    def test_save_creates_file(self, manager: StateManager) -> None:
        """save should create state file."""
        manager.save()
        assert manager.state_file.exists()

    def test_save_updates_last_save(self, manager: StateManager) -> None:
        """save should update last_save timestamp."""
        assert manager.state["last_save"] is None
        manager.save()
        assert manager.state["last_save"] is not None

    def test_save_persists_state(self, manager: StateManager, tmp_path: Path) -> None:
        """save should persist state that can be reloaded."""
        manager.state["custom"] = "value"
        manager.save()

        # Load in new manager
        new_manager = StateManager(str(tmp_path / "state.json"), "test")
        assert new_manager.state["custom"] == "value"


class TestStateManagerCheckpoints:
    """Tests for checkpoint functionality."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> StateManager:
        """Create StateManager instance."""
        return StateManager(str(tmp_path / "state.json"), "test")

    def test_set_checkpoint(self, manager: StateManager) -> None:
        """set_checkpoint should store checkpoint data."""
        manager.set_checkpoint("step1", {"status": "complete"})
        # Checkpoint is stored with value and timestamp
        assert manager.state["checkpoints"]["step1"]["value"] == {"status": "complete"}
        assert "timestamp" in manager.state["checkpoints"]["step1"]

    def test_get_checkpoint(self, manager: StateManager) -> None:
        """get_checkpoint should retrieve checkpoint value."""
        manager.set_checkpoint("step1", 42)
        result = manager.get_checkpoint("step1")
        assert result == 42

    def test_get_checkpoint_missing(self, manager: StateManager) -> None:
        """get_checkpoint should return default for missing checkpoint."""
        result = manager.get_checkpoint("nonexistent")
        assert result is None

        result = manager.get_checkpoint("nonexistent", "default_val")
        assert result == "default_val"


class TestStateManagerStats:
    """Tests for statistics functionality."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> StateManager:
        """Create StateManager instance."""
        return StateManager(str(tmp_path / "state.json"), "test")

    def test_update_stats(self, manager: StateManager) -> None:
        """update_stats should store statistics."""
        manager.update_stats(processed=100, failed=5)
        assert manager.state["stats"]["processed"] == 100
        assert manager.state["stats"]["failed"] == 5

    def test_update_stats_merges(self, manager: StateManager) -> None:
        """update_stats should merge with existing stats."""
        manager.update_stats(count=10)
        manager.update_stats(other=20)
        assert manager.state["stats"]["count"] == 10
        assert manager.state["stats"]["other"] == 20

    def test_increment_stat(self, manager: StateManager) -> None:
        """increment_stat should increment numeric stat."""
        manager.increment_stat("counter")
        assert manager.state["stats"]["counter"] == 1
        manager.increment_stat("counter")
        assert manager.state["stats"]["counter"] == 2
        manager.increment_stat("counter", 5)
        assert manager.state["stats"]["counter"] == 7


class TestStateManagerMetadata:
    """Tests for metadata functionality."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> StateManager:
        """Create StateManager instance."""
        return StateManager(str(tmp_path / "state.json"), "test")

    def test_set_metadata(self, manager: StateManager) -> None:
        """set_metadata should store metadata."""
        manager.set_metadata("version", "1.0")
        assert manager.state["metadata"]["version"] == "1.0"

    def test_get_metadata(self, manager: StateManager) -> None:
        """get_metadata should retrieve metadata."""
        manager.set_metadata("key", "value")
        assert manager.get_metadata("key") == "value"

    def test_get_metadata_default(self, manager: StateManager) -> None:
        """get_metadata should return default for missing key."""
        result = manager.get_metadata("missing", "default")
        assert result == "default"


class TestStateManagerPersistence:
    """Tests for state persistence across sessions."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """State should survive save and reload."""
        state_file = tmp_path / "state.json"

        # First session
        manager1 = StateManager(str(state_file), "persist_test")
        manager1.set_checkpoint("step1", {"done": True})
        manager1.update_stats(total=100)
        manager1.set_metadata("version", "2.0")
        manager1.save()

        # Second session
        manager2 = StateManager(str(state_file), "persist_test")
        assert manager2.get_checkpoint("step1") == {"done": True}
        assert manager2.state["stats"]["total"] == 100
        assert manager2.get_metadata("version") == "2.0"

    def test_atomic_save(self, tmp_path: Path) -> None:
        """save should not leave corrupted state."""
        state_file = tmp_path / "state.json"
        manager = StateManager(str(state_file), "test")
        manager.set_checkpoint("important", "data")
        manager.save()

        # Verify file is valid JSON
        with open(state_file) as f:
            data = json.load(f)
        assert data["checkpoints"]["important"]["value"] == "data"

    def test_clear(self, tmp_path: Path) -> None:
        """clear should reset state and remove file."""
        state_file = tmp_path / "state.json"
        manager = StateManager(str(state_file), "test")
        manager.set_checkpoint("step1", "done")
        manager.save()
        assert state_file.exists()

        manager.clear()
        assert not state_file.exists()
        assert manager.state["checkpoints"] == {}

    def test_get_progress_summary(self, tmp_path: Path) -> None:
        """get_progress_summary should return summary dict."""
        state_file = tmp_path / "state.json"
        manager = StateManager(str(state_file), "test")
        manager.set_checkpoint("step1", True)
        manager.set_checkpoint("step2", True)
        manager.update_stats(count=10)

        summary = manager.get_progress_summary()
        assert summary["process"] == "test"
        assert summary["checkpoints"] == 2
        assert summary["stats"]["count"] == 10


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> CheckpointManager:
        """Create CheckpointManager instance."""
        return CheckpointManager(str(tmp_path / "checkpoints.json"))

    def test_mark_processed(self, manager: CheckpointManager) -> None:
        """mark_processed should add item to processed set."""
        manager.mark_processed("item1")
        assert manager.is_processed("item1")
        assert not manager.is_processed("item2")

    def test_mark_many_processed(self, manager: CheckpointManager) -> None:
        """mark_many_processed should add multiple items."""
        manager.mark_many_processed(["a", "b", "c"])
        assert manager.is_processed("a")
        assert manager.is_processed("b")
        assert manager.is_processed("c")

    def test_get_unprocessed(self, manager: CheckpointManager) -> None:
        """get_unprocessed should filter processed items."""
        manager.mark_many_processed(["a", "b"])
        result = manager.get_unprocessed(["a", "b", "c", "d"])
        assert result == ["c", "d"]

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Checkpoints should survive save and reload."""
        file_path = str(tmp_path / "checkpoints.json")

        manager1 = CheckpointManager(file_path)
        manager1.mark_many_processed(["x", "y", "z"])
        manager1.save()

        manager2 = CheckpointManager(file_path)
        assert manager2.is_processed("x")
        assert manager2.is_processed("y")
        assert manager2.is_processed("z")

    def test_clear(self, manager: CheckpointManager) -> None:
        """clear should remove all processed items."""
        manager.mark_many_processed(["a", "b"])
        manager.save()

        manager.clear()
        assert not manager.is_processed("a")
        assert len(manager.processed) == 0

    def test_get_stats(self, manager: CheckpointManager) -> None:
        """get_stats should return count of processed items."""
        manager.mark_many_processed(["a", "b", "c"])
        stats = manager.get_stats()
        assert stats["total_processed"] == 3
