"""Unit tests for Workflow Base classes.

Tests for:
- WorkflowConfig dataclass
- WorkflowResult dataclass and computed properties
- WorkflowBase abstract class methods
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path


class TestWorkflowConfig:
    """Tests for WorkflowConfig dataclass."""

    def test_default_values(self):
        """Should create config with default values."""
        from core.workflows.workflow_base import WorkflowConfig

        config = WorkflowConfig(name="test_workflow")

        assert config.name == "test_workflow"
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.use_gpu is True
        assert config.checkpoint_enabled is True
        assert config.checkpoint_interval == 100
        assert config.timeout_seconds == 300

    def test_custom_values(self):
        """Should accept custom configuration values."""
        from core.workflows.workflow_base import WorkflowConfig

        config = WorkflowConfig(
            name="custom_workflow",
            batch_size=64,
            num_workers=8,
            use_gpu=False,
            checkpoint_enabled=False,
            checkpoint_interval=50,
            timeout_seconds=600,
        )

        assert config.name == "custom_workflow"
        assert config.batch_size == 64
        assert config.num_workers == 8
        assert config.use_gpu is False
        assert config.checkpoint_enabled is False
        assert config.checkpoint_interval == 50
        assert config.timeout_seconds == 600

    def test_staging_path_default(self):
        """Should create default staging path in temp directory."""
        from core.workflows.workflow_base import WorkflowConfig

        config = WorkflowConfig(name="test")

        assert "hades_workflow_staging" in str(config.staging_path)

    def test_custom_staging_path(self):
        """Should accept custom staging path."""
        from core.workflows.workflow_base import WorkflowConfig

        custom_path = Path("/custom/staging")
        config = WorkflowConfig(name="test", staging_path=custom_path)

        assert config.staging_path == custom_path


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_creates_result(self):
        """Should create workflow result with required fields."""
        from core.workflows.workflow_base import WorkflowResult

        start = datetime.now()
        end = start + timedelta(seconds=60)

        result = WorkflowResult(
            workflow_name="test",
            success=True,
            items_processed=100,
            items_failed=5,
            start_time=start,
            end_time=end,
        )

        assert result.workflow_name == "test"
        assert result.success is True
        assert result.items_processed == 100
        assert result.items_failed == 5

    def test_duration_seconds(self):
        """Should calculate duration in seconds."""
        from core.workflows.workflow_base import WorkflowResult

        start = datetime.now()
        end = start + timedelta(seconds=90)

        result = WorkflowResult(
            workflow_name="test",
            success=True,
            items_processed=100,
            items_failed=0,
            start_time=start,
            end_time=end,
        )

        assert result.duration_seconds == 90.0

    def test_success_rate_calculation(self):
        """Should calculate success rate as percentage."""
        from core.workflows.workflow_base import WorkflowResult

        start = datetime.now()
        end = start + timedelta(seconds=60)

        # 80 processed, 20 failed = 80% success rate
        result = WorkflowResult(
            workflow_name="test",
            success=True,
            items_processed=80,
            items_failed=20,
            start_time=start,
            end_time=end,
        )

        assert result.success_rate == 80.0

    def test_success_rate_zero_items(self):
        """Should return 0% when no items processed."""
        from core.workflows.workflow_base import WorkflowResult

        start = datetime.now()
        end = start + timedelta(seconds=60)

        result = WorkflowResult(
            workflow_name="test",
            success=False,
            items_processed=0,
            items_failed=0,
            start_time=start,
            end_time=end,
        )

        assert result.success_rate == 0.0

    def test_success_rate_all_success(self):
        """Should return 100% when all items processed."""
        from core.workflows.workflow_base import WorkflowResult

        start = datetime.now()
        end = start + timedelta(seconds=60)

        result = WorkflowResult(
            workflow_name="test",
            success=True,
            items_processed=100,
            items_failed=0,
            start_time=start,
            end_time=end,
        )

        assert result.success_rate == 100.0

    def test_metadata_default_empty(self):
        """Should default metadata to empty dict."""
        from core.workflows.workflow_base import WorkflowResult

        start = datetime.now()
        end = start + timedelta(seconds=60)

        result = WorkflowResult(
            workflow_name="test",
            success=True,
            items_processed=10,
            items_failed=0,
            start_time=start,
            end_time=end,
        )

        assert result.metadata == {}
        assert result.errors == []

    def test_custom_metadata_and_errors(self):
        """Should accept custom metadata and errors."""
        from core.workflows.workflow_base import WorkflowResult

        start = datetime.now()
        end = start + timedelta(seconds=60)

        result = WorkflowResult(
            workflow_name="test",
            success=False,
            items_processed=5,
            items_failed=3,
            start_time=start,
            end_time=end,
            metadata={"source": "arxiv", "batch_id": 1},
            errors=["Error 1", "Error 2"],
        )

        assert result.metadata == {"source": "arxiv", "batch_id": 1}
        assert result.errors == ["Error 1", "Error 2"]


class TestWorkflowBase:
    """Tests for WorkflowBase abstract class."""

    def _create_concrete_workflow(self):
        """Create a concrete workflow implementation for testing."""
        from core.workflows.workflow_base import WorkflowBase, WorkflowResult

        class ConcreteWorkflow(WorkflowBase):
            def validate_inputs(self, **kwargs) -> bool:
                return kwargs.get("valid", True)

            def execute(self, **kwargs) -> WorkflowResult:
                from datetime import datetime
                return WorkflowResult(
                    workflow_name=self.name,
                    success=True,
                    items_processed=kwargs.get("items", 0),
                    items_failed=0,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                )

        return ConcreteWorkflow

    def test_default_config(self):
        """Should use default config when none provided."""
        ConcreteWorkflow = self._create_concrete_workflow()

        workflow = ConcreteWorkflow()

        assert workflow.config.name == "unnamed_workflow"
        assert workflow.config.batch_size == 32

    def test_custom_config(self):
        """Should use provided config."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        config = WorkflowConfig(name="my_workflow", batch_size=64)
        workflow = ConcreteWorkflow(config=config)

        assert workflow.config.name == "my_workflow"
        assert workflow.config.batch_size == 64

    def test_creates_staging_directory(self):
        """Should create staging directory on init."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "test_staging"
            config = WorkflowConfig(name="test", staging_path=staging)

            assert not staging.exists()

            ConcreteWorkflow(config=config)  # Creates staging directory on init

            assert staging.exists()

    def test_name_property(self):
        """Should expose workflow name via property."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        config = WorkflowConfig(name="my_workflow")
        workflow = ConcreteWorkflow(config=config)

        assert workflow.name == "my_workflow"

    def test_supports_batch_default(self):
        """Should default to supporting batch processing."""
        ConcreteWorkflow = self._create_concrete_workflow()

        workflow = ConcreteWorkflow()

        assert workflow.supports_batch is True

    def test_supports_streaming_default(self):
        """Should default to not supporting streaming."""
        ConcreteWorkflow = self._create_concrete_workflow()

        workflow = ConcreteWorkflow()

        assert workflow.supports_streaming is False

    def test_get_workflow_info(self):
        """Should return workflow info dict."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        config = WorkflowConfig(
            name="info_test",
            batch_size=16,
            num_workers=2,
            use_gpu=False,
        )
        workflow = ConcreteWorkflow(config=config)

        info = workflow.get_workflow_info()

        assert info["name"] == "info_test"
        assert info["class"] == "ConcreteWorkflow"
        assert info["batch_size"] == 16
        assert info["num_workers"] == 2
        assert info["use_gpu"] is False
        assert info["checkpoint_enabled"] is True
        assert info["supports_batch"] is True
        assert info["supports_streaming"] is False


class TestWorkflowBaseCheckpointing:
    """Tests for WorkflowBase checkpoint methods."""

    def _create_concrete_workflow(self):
        """Create a concrete workflow implementation for testing."""
        from core.workflows.workflow_base import WorkflowBase, WorkflowResult

        class ConcreteWorkflow(WorkflowBase):
            def validate_inputs(self, **kwargs) -> bool:
                return True

            def execute(self, **kwargs) -> WorkflowResult:
                from datetime import datetime
                return WorkflowResult(
                    workflow_name=self.name,
                    success=True,
                    items_processed=0,
                    items_failed=0,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                )

        return ConcreteWorkflow

    def test_save_and_load_checkpoint(self):
        """Should save and load checkpoint data."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir)
            config = WorkflowConfig(name="checkpoint_test", staging_path=staging)
            workflow = ConcreteWorkflow(config=config)

            # Save checkpoint
            checkpoint_data = {"step": 5, "items": ["a", "b", "c"]}
            workflow.save_checkpoint(checkpoint_data)

            # Load checkpoint
            loaded = workflow.load_checkpoint()

            assert loaded == checkpoint_data

    def test_checkpoint_disabled(self):
        """Should not save/load when checkpointing disabled."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir)
            config = WorkflowConfig(
                name="no_checkpoint",
                staging_path=staging,
                checkpoint_enabled=False,
            )
            workflow = ConcreteWorkflow(config=config)

            # Save should do nothing
            workflow.save_checkpoint({"data": "test"})

            # Load should return None
            loaded = workflow.load_checkpoint()

            assert loaded is None

            # No checkpoint file should exist
            checkpoint_path = staging / "no_checkpoint_checkpoint.json"
            assert not checkpoint_path.exists()

    def test_load_checkpoint_missing_file(self):
        """Should return None when checkpoint file doesn't exist."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir)
            config = WorkflowConfig(name="missing_checkpoint", staging_path=staging)
            workflow = ConcreteWorkflow(config=config)

            loaded = workflow.load_checkpoint()

            assert loaded is None

    def test_clear_checkpoint(self):
        """Should clear checkpoint file."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir)
            config = WorkflowConfig(name="clear_test", staging_path=staging)
            workflow = ConcreteWorkflow(config=config)

            # Save checkpoint
            workflow.save_checkpoint({"data": "test"})
            checkpoint_path = staging / "clear_test_checkpoint.json"
            assert checkpoint_path.exists()

            # Clear checkpoint
            workflow.clear_checkpoint()

            assert not checkpoint_path.exists()

    def test_clear_checkpoint_missing_file(self):
        """Should handle clearing non-existent checkpoint gracefully."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir)
            config = WorkflowConfig(name="clear_missing", staging_path=staging)
            workflow = ConcreteWorkflow(config=config)

            # Should not raise
            workflow.clear_checkpoint()

    def test_checkpoint_atomic_save(self):
        """Should save checkpoint atomically (temp then rename)."""
        from core.workflows.workflow_base import WorkflowConfig

        ConcreteWorkflow = self._create_concrete_workflow()

        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir)
            config = WorkflowConfig(name="atomic_test", staging_path=staging)
            workflow = ConcreteWorkflow(config=config)

            # Save checkpoint
            workflow.save_checkpoint({"large": "data" * 1000})

            # Checkpoint should exist
            checkpoint_path = staging / "atomic_test_checkpoint.json"
            assert checkpoint_path.exists()

            # Temp file should not exist
            temp_path = checkpoint_path.with_suffix(".tmp")
            assert not temp_path.exists()
