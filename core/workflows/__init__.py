"""
Workflows Module

Provides orchestration and pipeline management for document processing.
Workflows coordinate the flow of data through extraction, embedding,
and storage phases while managing state and error recovery.
"""

from .workflow_base import WorkflowBase, WorkflowConfig, WorkflowResult

# Import PDF workflow if available
try:
    from .workflow_pdf import PDFWorkflow
except ImportError:
    PDFWorkflow = None  # type: ignore[misc]

# State management
from .state import CheckpointManager, StateManager

__all__ = [
    'WorkflowBase',
    'WorkflowConfig',
    'WorkflowResult',
    'StateManager',
    'CheckpointManager',
    'PDFWorkflow',
]
