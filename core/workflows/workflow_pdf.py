#!/usr/bin/env python3
"""Backward-compatible DocumentProcessor wrapper.

The core implementation now lives in :mod:`core.processors.document_processor`.
This module simply re-exports the public API so existing imports under
``core.workflows.workflow_pdf`` continue to function.
"""

from core.processors.document_processor import (
    DocumentProcessor,
    ExtractionResult,
    ProcessingConfig,
    ProcessingResult,
)

__all__ = [
    "DocumentProcessor",
    "ProcessingConfig",
    "ProcessingResult",
    "ExtractionResult",
]

