"""Integration tests for workflow modules."""

from datetime import datetime
from pathlib import Path

import pytest

from core.workflows.workflow_base import (
    WorkflowBase,
    WorkflowConfig,
    WorkflowResult,
)


class TestWorkflowConfig:
    """Tests for WorkflowConfig dataclass."""

    def test_default_values(self) -> None:
        """WorkflowConfig should have sensible defaults."""
        config = WorkflowConfig(name="test_workflow")
        assert config.name == "test_workflow"
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.use_gpu is True
        assert config.checkpoint_enabled is True
        assert config.checkpoint_interval == 100
        assert config.timeout_seconds == 300

    def test_custom_values(self) -> None:
        """WorkflowConfig should accept custom values."""
        config = WorkflowConfig(
            name="custom",
            batch_size=16,
            num_workers=2,
            use_gpu=False,
            checkpoint_enabled=False,
        )
        assert config.name == "custom"
        assert config.batch_size == 16
        assert config.num_workers == 2
        assert config.use_gpu is False
        assert config.checkpoint_enabled is False

    def test_staging_path_is_path(self) -> None:
        """staging_path should be a Path object."""
        config = WorkflowConfig(name="test")
        assert isinstance(config.staging_path, Path)


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""

    def test_basic_creation(self) -> None:
        """WorkflowResult should store all fields."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 5, 0)
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

    def test_duration_seconds(self) -> None:
        """duration_seconds should calculate correct duration."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 5, 0)  # 5 minutes later
        result = WorkflowResult(
            workflow_name="test",
            success=True,
            items_processed=10,
            items_failed=0,
            start_time=start,
            end_time=end,
        )
        assert result.duration_seconds == 300.0  # 5 minutes = 300 seconds

    def test_success_rate_all_success(self) -> None:
        """success_rate should be 100% when no failures."""
        result = WorkflowResult(
            workflow_name="test",
            success=True,
            items_processed=100,
            items_failed=0,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert result.success_rate == 100.0

    def test_success_rate_partial(self) -> None:
        """success_rate should calculate correctly for partial success."""
        result = WorkflowResult(
            workflow_name="test",
            success=True,
            items_processed=80,
            items_failed=20,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert result.success_rate == 80.0

    def test_success_rate_zero_items(self) -> None:
        """success_rate should be 0 when no items processed."""
        result = WorkflowResult(
            workflow_name="test",
            success=False,
            items_processed=0,
            items_failed=0,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert result.success_rate == 0.0

    def test_errors_and_metadata(self) -> None:
        """WorkflowResult should store errors and metadata."""
        result = WorkflowResult(
            workflow_name="test",
            success=False,
            items_processed=5,
            items_failed=10,
            start_time=datetime.now(),
            end_time=datetime.now(),
            errors=["Error 1", "Error 2"],
            metadata={"source": "test"},
        )
        assert len(result.errors) == 2
        assert result.metadata["source"] == "test"


class ConcreteWorkflow(WorkflowBase):
    """Concrete implementation for testing WorkflowBase."""

    def validate_inputs(self, **kwargs) -> bool:
        """Validate inputs."""
        return "data" in kwargs

    def execute(self, **kwargs) -> WorkflowResult:
        """Execute workflow."""
        return WorkflowResult(
            workflow_name=self.config.name,
            success=True,
            items_processed=len(kwargs.get("data", [])),
            items_failed=0,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )


class TestWorkflowBase:
    """Tests for WorkflowBase abstract class."""

    @pytest.fixture
    def workflow(self, tmp_path: Path) -> ConcreteWorkflow:
        """Create concrete workflow instance."""
        config = WorkflowConfig(
            name="test_workflow",
            staging_path=tmp_path / "staging",
        )
        return ConcreteWorkflow(config)

    def test_default_config(self) -> None:
        """Workflow should have default config when none provided."""
        workflow = ConcreteWorkflow()
        assert workflow.config.name == "unnamed_workflow"

    def test_staging_directory_created(self, workflow: ConcreteWorkflow) -> None:
        """Workflow should create staging directory on init."""
        assert workflow.config.staging_path.exists()

    def test_validate_inputs(self, workflow: ConcreteWorkflow) -> None:
        """validate_inputs should return True for valid inputs."""
        assert workflow.validate_inputs(data=[1, 2, 3]) is True
        assert workflow.validate_inputs(other="value") is False

    def test_execute_returns_result(self, workflow: ConcreteWorkflow) -> None:
        """execute should return WorkflowResult."""
        result = workflow.execute(data=[1, 2, 3])
        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert result.items_processed == 3

    def test_checkpoint_data_initialized(self, workflow: ConcreteWorkflow) -> None:
        """Workflow should have empty checkpoint_data dict."""
        assert workflow.checkpoint_data == {}


class TestWorkflowImports:
    """Tests for workflow module imports."""

    def test_workflow_pdf_exports_document_processor(self) -> None:
        """workflow_pdf should re-export DocumentProcessor."""
        from core.workflows.workflow_pdf import DocumentProcessor

        assert DocumentProcessor is not None

    def test_workflow_pdf_exports_processing_config(self) -> None:
        """workflow_pdf should re-export ProcessingConfig."""
        from core.workflows.workflow_pdf import ProcessingConfig

        config = ProcessingConfig()
        assert config is not None

    def test_workflow_pdf_exports_processing_result(self) -> None:
        """workflow_pdf should re-export ProcessingResult."""
        from core.workflows.workflow_pdf import ProcessingResult

        assert ProcessingResult is not None
