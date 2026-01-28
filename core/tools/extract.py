"""Standalone document extraction tool.

Wraps Docling to extract structured text from any supported document format:
PDF, DOCX, PPTX, XLSX, HTML, XML, Markdown, AsciiDoc, plain text, and images.

This is a thin, focused interface — no embedder loaded, no database involved.

Usage:
    from core.tools.extract import extract_document

    result = extract_document("paper.pdf")
    print(result["text"])
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_document(
    path: str | Path,
    *,
    extract_tables: bool = True,
    extract_equations: bool = True,
    extract_images: bool = True,
    ocr_enabled: bool = False,
) -> dict[str, Any]:
    """Extract structured text from a document file.

    Args:
        path: Path to the document file.
        extract_tables: Extract tables from the document.
        extract_equations: Extract equations from the document.
        extract_images: Extract images from the document.
        ocr_enabled: Enable OCR for scanned documents.

    Returns:
        JSON-serializable dict with keys:
            text, tables, equations, images, metadata,
            extraction_time, source_path

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If Docling is not available.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    from core.extractors import get_extractor
    from core.extractors.extractors_base import ExtractorConfig

    config = ExtractorConfig(
        extract_tables=extract_tables,
        extract_equations=extract_equations,
        extract_images=extract_images,
        ocr_enabled=ocr_enabled,
    )

    start = time.time()
    extractor = get_extractor(str(path), config=config)
    result = extractor.extract(str(path))
    elapsed = time.time() - start

    return {
        "text": result.text or "",
        "tables": result.tables or [],
        "equations": result.equations or [],
        "images": result.images or [],
        "metadata": result.metadata or {},
        "extraction_time": round(elapsed, 3),
        "source_path": str(path),
    }
