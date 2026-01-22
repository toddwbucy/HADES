"""
Robust PDF Extractor with Fallback
===================================

Wraps Docling with timeout protection and PyMuPDF fallback for problematic PDFs.
Provides resilience against segfaults and extraction failures.
"""

import logging
import signal
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing as mp

from .extractors_base import ExtractorBase, ExtractorConfig, ExtractionResult

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .extractors_docling import DoclingExtractor

logger = logging.getLogger(__name__)


def _extract_with_docling(pdf_path: str, use_ocr: bool, extract_tables: bool) -> Dict[str, Any]:
    """Extract using Docling (runs in subprocess)."""
    try:
        extractor = DoclingExtractor(
            use_ocr=use_ocr,
            extract_tables=extract_tables,
            use_fallback=False  # Don't use fallback here, we handle it at higher level
        )
        return extractor.extract(pdf_path)
    except Exception as e:
        return {'error': str(e)}


def _extract_with_pymupdf(pdf_path: str) -> Optional[Dict[str, Any]]:
    """Fallback extraction using PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        return None
        
    try:
        with fitz.open(pdf_path) as doc:
            text_parts = []
            num_pages = len(doc)
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            
            full_text = "\n\n".join(text_parts)
            
            return {
                'full_text': full_text,
                'text': full_text,
                'markdown': full_text,
                'num_pages': num_pages,
                'extractor': 'pymupdf_fallback'
            }
    except Exception as e:
        logger.error(f"PyMuPDF fallback failed: {e}")
        return None


class RobustExtractor(ExtractorBase):
    """
    Robust PDF extractor with timeout protection and fallback.

    Ensures reliable extraction even when primary tools fail by using
    subprocess isolation and fallback strategies.
    """

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        use_ocr: bool = False,
        extract_tables: bool = True,
        timeout: int = 30,
        use_fallback: bool = True,
        **kwargs
    ):
        """
        Initialize robust extractor.

        Args:
            config: ExtractorConfig object or None for defaults
            use_ocr: Whether to use OCR for scanned PDFs
            extract_tables: Whether to extract table structures
            timeout: Maximum seconds to wait for extraction
            use_fallback: Whether to use PyMuPDF fallback on failure
            **kwargs: Additional options
        """
        super().__init__(config)
        self.use_ocr = kwargs.get('use_ocr', use_ocr)
        self.extract_tables = kwargs.get('extract_tables', extract_tables)
        self.timeout = kwargs.get('timeout', timeout)
        self.use_fallback = kwargs.get('use_fallback', use_fallback)

        logger.info(f"Initialized RobustExtractor (timeout: {self.timeout}s, fallback: {self.use_fallback})")

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return ['.pdf']

    def _dict_to_result(self, data: Optional[Dict[str, Any]], pdf_path: str) -> ExtractionResult:
        """Convert internal dict format to ExtractionResult."""
        if data is None:
            return ExtractionResult(
                text='',
                metadata={'pdf_path': pdf_path},
                error='Extraction failed'
            )
        return ExtractionResult(
            text=data.get('full_text', data.get('text', '')),
            metadata=data.get('metadata', {'extractor': data.get('extractor', 'robust')}),
            tables=data.get('tables', []),
            equations=data.get('equations', []),
            images=data.get('images', []),
            processing_time=data.get('metadata', {}).get('processing_time', 0.0)
        )

    def _cleanup_executor_processes(self, executor: ProcessPoolExecutor) -> None:
        """Clean up only the executor's own worker processes."""
        try:
            if hasattr(executor, '_processes'):
                for process in executor._processes.values():
                    if process and process.is_alive():
                        logger.debug(f"Terminating executor worker: {process.pid}")
                        process.terminate()
                        process.join(timeout=2)
                        if process.is_alive():
                            logger.warning(f"Force killing executor worker: {process.pid}")
                            process.kill()
                            process.join(timeout=1)
        except (AttributeError, Exception) as e:
            logger.debug(f"Could not access executor processes: {e}")

    def extract(self, pdf_path: Union[str, Path], **kwargs) -> ExtractionResult:
        """
        Extract text and structures from PDF with timeout protection.

        Args:
            pdf_path: Path to PDF file
            **kwargs: Additional extraction options

        Returns:
            ExtractionResult object
        """
        pdf_path = str(pdf_path)
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return ExtractionResult(
                text='',
                metadata={'pdf_path': pdf_path},
                error=f"PDF not found: {pdf_path}"
            )
        
        # Try Docling with timeout
        logger.debug(f"Attempting Docling extraction: {pdf_file.name}")
        
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context('spawn')
        )
        
        try:
            future = executor.submit(
                _extract_with_docling,
                pdf_path,
                self.use_ocr,
                self.extract_tables
            )
            
            try:
                result = future.result(timeout=self.timeout)
                
                if result and 'error' not in result:
                    logger.debug(f"Docling extraction successful: {pdf_file.name}")
                    return self._dict_to_result(result, pdf_path)
                else:
                    logger.warning(f"Docling extraction failed: {pdf_file.name}")
                    
            except TimeoutError:
                logger.warning(f"Docling timeout after {self.timeout}s: {pdf_file.name}")
                # Cancel the future and shutdown executor
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

                # Clean up only the executor's own worker processes (not mp.active_children)
                self._cleanup_executor_processes(executor)

            except Exception as e:
                logger.error(f"Docling crashed: {pdf_file.name} - {e}")
                executor.shutdown(wait=False, cancel_futures=True)

                # Clean up only the executor's own worker processes
                self._cleanup_executor_processes(executor)
                    
        finally:
            # Ensure executor is shut down
            executor.shutdown(wait=False)
            # Clear executor reference
            executor = None
        
        # Try fallback if enabled
        if self.use_fallback and PYMUPDF_AVAILABLE:
            logger.info(f"Using PyMuPDF fallback: {pdf_file.name}")
            return self._dict_to_result(_extract_with_pymupdf(pdf_path), pdf_path)

        return ExtractionResult(
            text='',
            metadata={'pdf_path': pdf_path},
            error='Extraction failed and fallback disabled'
        )

    def extract_batch(self, pdf_paths: List[Union[str, Path]], **kwargs) -> List[ExtractionResult]:
        """
        Extract from multiple PDFs.

        Args:
            pdf_paths: List of PDF file paths
            **kwargs: Additional extraction options

        Returns:
            List of ExtractionResult objects
        """
        results = []
        for pdf_path in pdf_paths:
            try:
                result = self.extract(pdf_path, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {pdf_path}: {e}")
                results.append(ExtractionResult(
                    text='',
                    metadata={'pdf_path': str(pdf_path)},
                    error=str(e)
                ))
        return results