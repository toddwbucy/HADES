"""
Extractors Module

Provides document extraction capabilities for various file formats.
"""

import os
from pathlib import Path
from typing import Union

from .extractors_base import ExtractionResult, ExtractorBase, ExtractorConfig

# Import extractors - use Optional types for conditional imports
from typing import Optional, Type

DoclingExtractor: Optional[Type] = None
LaTeXExtractor: Optional[Type] = None
CodeExtractor: Optional[Type] = None
TreeSitterExtractor: Optional[Type] = None
RobustExtractor: Optional[Type] = None

try:
    from .extractors_docling import DoclingExtractor as _DoclingExtractor
    DoclingExtractor = _DoclingExtractor
except ImportError:
    pass

try:
    from .extractors_latex import LaTeXExtractor as _LaTeXExtractor
    LaTeXExtractor = _LaTeXExtractor
except ImportError:
    pass

try:
    from .extractors_code import CodeExtractor as _CodeExtractor
    CodeExtractor = _CodeExtractor
except ImportError:
    pass

try:
    from .extractors_treesitter import TreeSitterExtractor as _TreeSitterExtractor
    TreeSitterExtractor = _TreeSitterExtractor
except ImportError:
    pass

try:
    from .extractors_robust import RobustExtractor as _RobustExtractor
    RobustExtractor = _RobustExtractor
except ImportError:
    pass


def get_extractor(file_path: str | Path | os.PathLike, **kwargs):
    """
    Get an appropriate extractor for a file.

    Args:
        file_path: Path to the file (str, Path, or os.PathLike)
        **kwargs: Additional configuration for the extractor

    Returns:
        Extractor instance
    """
    file_path_lower = str(file_path).lower()

    if file_path_lower.endswith('.pdf'):
        if DoclingExtractor is not None:
            return DoclingExtractor(**kwargs)
        if RobustExtractor is not None:
            return RobustExtractor(**kwargs)
        raise ImportError("No PDF extractor available (DoclingExtractor and RobustExtractor unavailable)")

    if file_path_lower.endswith('.tex'):
        if LaTeXExtractor is not None:
            return LaTeXExtractor(**kwargs)
        raise ImportError("LaTeXExtractor not available")

    if file_path_lower.endswith(('.py', '.js', '.ts', '.go', '.rs', '.java', '.c', '.cpp', '.h')):
        if CodeExtractor is not None:
            return CodeExtractor(**kwargs)
        # TreeSitterExtractor does not implement ExtractorBase interface
        raise ImportError("No code extractor available (CodeExtractor unavailable)")

    # Default to robust extractor for unknown types
    if RobustExtractor is not None:
        return RobustExtractor(**kwargs)
    if DoclingExtractor is not None:
        return DoclingExtractor(**kwargs)

    raise ImportError("No extractor available for this file type")


__all__ = [
    'ExtractorBase',
    'ExtractorConfig',
    'ExtractionResult',
    'DoclingExtractor',
    'LaTeXExtractor',
    'CodeExtractor',
    'TreeSitterExtractor',
    'RobustExtractor',
    'get_extractor',
]
