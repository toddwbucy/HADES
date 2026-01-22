"""
Extractors Module

Provides document extraction capabilities for various file formats.
"""

from .extractors_base import ExtractorBase, ExtractorConfig, ExtractionResult

# Import extractors
try:
    from .extractors_docling import DoclingExtractor
except ImportError:
    DoclingExtractor = None  # type: ignore[misc]

try:
    from .extractors_latex import LaTeXExtractor
except ImportError:
    LaTeXExtractor = None  # type: ignore[misc]

try:
    from .extractors_code import CodeExtractor
except ImportError:
    CodeExtractor = None  # type: ignore[misc]

try:
    from .extractors_treesitter import TreeSitterExtractor
except ImportError:
    TreeSitterExtractor = None  # type: ignore[misc]

try:
    from .extractors_robust import RobustExtractor
except ImportError:
    RobustExtractor = None  # type: ignore[misc]


def get_extractor(file_path: str, **kwargs):
    """
    Get an appropriate extractor for a file.

    Args:
        file_path: Path to the file
        **kwargs: Additional configuration for the extractor

    Returns:
        Extractor instance
    """
    file_path_lower = file_path.lower()

    if file_path_lower.endswith('.pdf'):
        if DoclingExtractor is not None:
            return DoclingExtractor(**kwargs)
        raise ImportError("DoclingExtractor not available")

    if file_path_lower.endswith('.tex'):
        if LaTeXExtractor is not None:
            return LaTeXExtractor(**kwargs)
        raise ImportError("LaTeXExtractor not available")

    if file_path_lower.endswith(('.py', '.js', '.ts', '.go', '.rs', '.java', '.c', '.cpp', '.h')):
        if CodeExtractor is not None:
            return CodeExtractor(**kwargs)
        if TreeSitterExtractor is not None:
            return TreeSitterExtractor(**kwargs)
        raise ImportError("No code extractor available")

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
