#!/usr/bin/env python3
"""Base Workflow Class.

Defines the contract for all workflow implementations in HADES.
Workflows orchestrate the processing of documents through extraction,
embedding, and storage phases with checkpointing for fault tolerance.
"""

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for workflows."""
    name: str
    batch_size: int = 32
    num_workers: int = 4
    use_gpu: bool = True
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 100
    staging_path: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "hades_workflow_staging")
    timeout_seconds: int = 300


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_name: str
    success: bool
    items_processed: int
    items_failed: int
    start_time: datetime
    end_time: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Calculate workflow duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.items_processed + self.items_failed
        if total == 0:
            return 0.0
        return (self.items_processed / total) * 100


class WorkflowBase(ABC):
    """
    Abstract base class for all workflows.

    Provides common infrastructure for checkpointing, error handling,
    and progress tracking while enforcing a consistent interface.
    """

    def __init__(self, config: WorkflowConfig | None = None):
        """
        Initialize workflow with configuration.

        Args:
            config: Workflow configuration
        """
        self.config = config or WorkflowConfig(name="unnamed_workflow")
        self.checkpoint_data: dict[str, Any] = {}
        self._ensure_staging_directory()

    def _ensure_staging_directory(self):
        """Ensure staging directory exists."""
        if self.config.staging_path:
            self.config.staging_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate workflow inputs.

        Returns:
            True if inputs are valid, False otherwise
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the workflow.

        Returns:
            WorkflowResult with execution details
        """
        pass

    def save_checkpoint(self, checkpoint_data: dict[str, Any]):
        """
        Save workflow checkpoint with atomic write.

        Args:
            checkpoint_data: Data to checkpoint
        """
        if not self.config.checkpoint_enabled:
            return

        checkpoint_path = self.config.staging_path / f"{self.config.name}_checkpoint.json"

        try:
            import json
            # Write to temp file first, then atomically rename
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, default=str, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk

            # Atomic rename (on POSIX systems)
            temp_path.replace(checkpoint_path)
            logger.debug(f"Checkpoint saved atomically to {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def load_checkpoint(self) -> dict[str, Any] | None:
        """
        Load workflow checkpoint if exists.

        Returns:
            Checkpoint data or None
        """
        if not self.config.checkpoint_enabled:
            return None

        checkpoint_path = self.config.staging_path / f"{self.config.name}_checkpoint.json"

        if not checkpoint_path.exists():
            return None

        try:
            import json
            with open(checkpoint_path) as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self):
        """Clear workflow checkpoint."""
        checkpoint_path = self.config.staging_path / f"{self.config.name}_checkpoint.json"

        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.debug(f"Checkpoint cleared: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to clear checkpoint: {e}")

    @property
    def name(self) -> str:
        """Get workflow name."""
        return self.config.name

    @property
    def supports_batch(self) -> bool:
        """Whether this workflow supports batch processing."""
        return True

    @property
    def supports_streaming(self) -> bool:
        """Whether this workflow supports streaming processing."""
        return False

    def get_workflow_info(self) -> dict[str, Any]:
        """
        Get information about the workflow.

        Returns:
            Dictionary with workflow metadata
        """
        return {
            "name": self.config.name,
            "class": self.__class__.__name__,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "use_gpu": self.config.use_gpu,
            "checkpoint_enabled": self.config.checkpoint_enabled,
            "supports_batch": self.supports_batch,
            "supports_streaming": self.supports_streaming
        }
