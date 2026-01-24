"""
Workflows Module

Provides orchestration and pipeline management for document processing.
Workflows coordinate the flow of data through extraction, embedding,
and storage phases while managing state and error recovery.
"""

from .workflow_base import WorkflowBase, WorkflowConfig, WorkflowResult

# Import PDF workflow components if available
from typing import Optional, Type

DocumentProcessor: Optional[Type] = None
ProcessingConfig: Optional[Type] = None
ProcessingResult: Optional[Type] = None

try:
    from .workflow_pdf import DocumentProcessor as _DocumentProcessor
    from .workflow_pdf import ProcessingConfig as _ProcessingConfig
    from .workflow_pdf import ProcessingResult as _ProcessingResult
    DocumentProcessor = _DocumentProcessor
    ProcessingConfig = _ProcessingConfig
    ProcessingResult = _ProcessingResult
except ImportError:
    pass

# State management
from .state import CheckpointManager, StateManager

__all__ = [
    'WorkflowBase',
    'WorkflowConfig',
    'WorkflowResult',
    'StateManager',
    'CheckpointManager',
    'DocumentProcessor',
    'ProcessingConfig',
    'ProcessingResult',
]
