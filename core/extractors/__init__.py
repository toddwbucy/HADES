"""
Extractors Module

Provides document extraction capabilities for various file formats.

Usage:
    # Recommended: Use ExtractorFactory
    from core.extractors import ExtractorFactory
    extractor = ExtractorFactory.for_file("document.pdf")

    # Legacy: Direct function call (uses factory internally)
    from core.extractors import get_extractor
    extractor = get_extractor("document.pdf")
"""

import os
from pathlib import Path

from .extractor_factory import ExtractorFactory
from .extractors_base import ExtractionResult, ExtractorBase, ExtractorConfig

# Import extractors - use Optional types for conditional imports
DoclingExtractor: type | None = None
LaTeXExtractor: type | None = None
CodeExtractor: type | None = None
TreeSitterExtractor: type | None = None
RobustExtractor: type | None = None

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


def get_extractor(file_path: str | Path | os.PathLike, **kwargs) -> ExtractorBase:
    """
    Get an appropriate extractor for a file.

    This function uses ExtractorFactory.for_file() internally.
    Prefer using ExtractorFactory directly for more control.

    Args:
        file_path: Path to the file (str, Path, or os.PathLike)
        **kwargs: Additional configuration for the extractor

    Returns:
        Extractor instance

    Raises:
        ValueError: If no extractor available for the file type
    """
    return ExtractorFactory.for_file(file_path, **kwargs)


__all__ = [
    # Factory (recommended)
    'ExtractorFactory',
    # Base classes
    'ExtractorBase',
    'ExtractorConfig',
    'ExtractionResult',
    # Extractor classes (for direct use if needed)
    'DoclingExtractor',
    'LaTeXExtractor',
    'CodeExtractor',
    'TreeSitterExtractor',
    'RobustExtractor',
    # Legacy function
    'get_extractor',
]
