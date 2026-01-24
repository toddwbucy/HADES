#!/usr/bin/env python3
"""
Base Extractor Interface

Defines the contract for all document extraction implementations.
Extractors transform raw documents into structured information for
downstream processing and embedding generation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of document extraction."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    equations: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    code_blocks: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    processing_time: float = 0.0


@dataclass
class ExtractorConfig:
    """Configuration for extractors."""
    use_gpu: bool = True
    batch_size: int = 1
    timeout_seconds: int = 300
    extract_equations: bool = True
    extract_tables: bool = True
    extract_images: bool = True
    extract_code: bool = True
    extract_references: bool = True
    max_pages: int | None = None
    ocr_enabled: bool = False


class ExtractorBase(ABC):
    """
    Abstract base class for all extractors.

    Defines the interface that all extraction implementations must follow
    to ensure consistency across different document types and approaches.
    """

    def __init__(self, config: ExtractorConfig | None = None):
        """
        Initialize extractor with configuration.

        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractorConfig()

    @abstractmethod
    def extract(self,
               file_path: str | Path,
               **kwargs) -> ExtractionResult:
        """
        Extract content from a document.

        Args:
            file_path: Path to the document
            **kwargs: Additional extraction options

        Returns:
            ExtractionResult with extracted content
        """
        pass

    @abstractmethod
    def extract_batch(self,
                     file_paths: list[str | Path],
                     **kwargs) -> list[ExtractionResult]:
        """
        Extract content from multiple documents.

        Args:
            file_paths: List of document paths
            **kwargs: Additional extraction options

        Returns:
            List of ExtractionResult objects
        """
        pass

    def validate_file(self, file_path: str | Path) -> bool:
        """
        Validate that a file can be processed.

        Args:
            file_path: Path to validate

        Returns:
            True if file can be processed
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File does not exist: {path}")
            return False
        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            return False
        if path.stat().st_size == 0:
            logger.error(f"File is empty: {path}")
            return False
        return True

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        pass

    @property
    def supports_gpu(self) -> bool:
        """Whether this extractor can use GPU acceleration."""
        return False

    @property
    def supports_batch(self) -> bool:
        """Whether this extractor supports batch processing."""
        return True

    @property
    def supports_ocr(self) -> bool:
        """Whether this extractor supports OCR."""
        return False

    def get_extractor_info(self) -> dict[str, Any]:
        """
        Get information about the extractor.

        Returns:
            Dictionary with extractor metadata
        """
        return {
            "class": self.__class__.__name__,
            "supported_formats": self.supported_formats,
            "supports_gpu": self.supports_gpu,
            "supports_batch": self.supports_batch,
            "supports_ocr": self.supports_ocr,
            "config": {
                "use_gpu": self.config.use_gpu,
                "batch_size": self.config.batch_size,
                "timeout_seconds": self.config.timeout_seconds
            }
        }
