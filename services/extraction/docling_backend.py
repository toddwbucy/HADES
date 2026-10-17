"""Docling extraction backend — the ML boundary for document processing.

Wraps docling's DocumentConverter behind a minimal interface.
Everything else (batching, chunking, embedding, DB writes) is handled
by the Rust orchestrator.
"""

from __future__ import annotations

import gc
from importlib.metadata import version
from io import BytesIO
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Structured extraction output matching the gRPC ExtractResponse."""

    text: str = ""
    tables: list[dict[str, Any]] = field(default_factory=list)
    equations: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    processing_time: float = 0.0


class DoclingExtractor:
    """Extracts structured documents using the pinned Docling converter.

    Lazy-loads the docling converter on first use. Call cleanup() to
    release GPU memory when the model is no longer needed.
    """

    # Pinned Docling registry, plus the HADES Markdown spelling alias.
    # Some formats require upstream extras/system tools. Those errors are fatal.
    from docling.datamodel.base_models import FormatToExtensions
    SUPPORTED_EXTENSIONS = {
        "." + ext.lower() for extensions in FormatToExtensions.values() for ext in extensions
    } | {".markdown"}

    def __init__(self, *, use_ocr: bool = False, formula_enrichment: bool = False, device: str = "cpu") -> None:
        self._converter = None
        self._converter_lock = threading.Lock()
        self._use_ocr = use_ocr
        self._formula_enrichment = formula_enrichment
        self._device = device

    @staticmethod
    def is_mets_archive(path: str | Path) -> bool:
        # The pinned converter uses this format probe itself. Reuse it
        # to distinguish METS bundles from existing arXiv LaTeX archive inputs.
        from docling.datamodel.document import _DocumentConversionInput
        return _DocumentConversionInput._detect_mets_gbs(Path(path)) is not None

    @property
    def converter(self):
        """Lazy-load the Docling document converter (thread-safe)."""
        if self._converter is None:
            with self._converter_lock:
                if self._converter is not None:
                    return self._converter

                logger.info("Loading Docling document converter...")
                start = time.time()
                from docling.datamodel.accelerator_options import AcceleratorOptions
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    PdfPipelineOptions,
                    TableStructureOptions,
                )
                from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption

                pdf_options = PdfPipelineOptions(
                    do_table_structure=True,
                    accelerator_options=AcceleratorOptions(device=self._device),
                    do_ocr=self._use_ocr,
                    do_formula_enrichment=self._formula_enrichment,
                    table_structure_options=TableStructureOptions(do_cell_matching=True),
                )
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
                        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_options),
                    }
                )
                logger.info("Docling converter loaded in %.2fs", time.time() - start)
        return self._converter

    def extract(
        self,
        file_path: str | Path,
        *,
        extract_tables: bool = True,
        extract_equations: bool = True,
        extract_images: bool = True,
        use_ocr: bool | None = None,
    ) -> ExtractionResult:
        """Extract structured content from a document.

        This is the only method that touches ML models.
        """
        path = Path(file_path)
        start = time.time()
        effective_ocr = use_ocr if use_ocr is not None else self._use_ocr

        if not path.exists():
            return ExtractionResult(error=f"File not found: {path}")

        if not path.is_file():
            return ExtractionResult(error=f"Not a file: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS | {".gz", ".tgz"}:
            return ExtractionResult(error=f"Unsupported format: {path.suffix}")
        if effective_ocr != self._use_ocr:
            return ExtractionResult(error="Per-request OCR differs from service policy")
        try:
            return self._extract_with_docling(path, start, extract_tables, extract_equations, extract_images)
        except Exception as e:
            logger.error("Docling extraction failed for %s: %s", path, e)
            return ExtractionResult(
                error=f"Docling extraction failed: {e}",
                processing_time=time.time() - start,
            )

    def _extract_with_docling(
        self,
        path: Path,
        start: float,
        extract_tables: bool,
        extract_equations: bool,
        extract_images: bool,
    ) -> ExtractionResult:
        """Run the pinned converter and retain structured document exports."""
        source = str(path)
        if path.suffix.lower() == ".markdown":
            from docling.datamodel.base_models import DocumentStream
            source = DocumentStream(name=path.stem + ".md", stream=BytesIO(path.read_bytes()))
        result = self.converter.convert(source)
        status = getattr(result.status, "value", result.status)
        if status != "success":
            raise RuntimeError(f"Docling conversion did not complete: {status}; {getattr(result, 'errors', [])}")
        doc = result.document
        if doc is None:
            raise RuntimeError("Docling returned no document")

        # Extract markdown as the full text
        full_text = ""
        if doc is not None and hasattr(doc, "export_to_markdown"):
            full_text = doc.export_to_markdown()

        tables = []
        equations = []
        images = []

        if extract_tables and doc is not None:
            for i, table in enumerate(getattr(doc, "tables", [])):
                tables.append({
                    "content": table.export_to_markdown(doc=doc),
                    "caption": table.caption_text(doc),
                    "index": i,
                })

        if extract_equations and doc is not None:
            from docling_core.types.doc import DocItemLabel
            for item in doc.texts:
                if item.label == DocItemLabel.FORMULA:
                    equations.append({
                        "latex": item.text if self._formula_enrichment else "",
                        "text": item.text,
                        "index": len(equations),
                        "is_inline": False,
                    })

        if extract_images and doc is not None:
            for i, fig in enumerate(getattr(doc, "pictures", [])):
                images.append({
                    "path": getattr(fig, "uri", ""),
                    "caption": fig.caption_text(doc),
                    "index": i,
                })

        metadata = {
            "source": str(path),
            "format": path.suffix.lower().lstrip("."),
            "extractor": "docling",
            "docling_version": version("docling"),
            "docling_core_version": version("docling-core"),
            "formula_enrichment": str(self._formula_enrichment).lower(),
            "ocr": str(self._use_ocr).lower(),
            "conversion_status": "success",
            "num_pages": str(doc.num_pages()),
        }

        return ExtractionResult(
            text=full_text,
            tables=tables,
            equations=equations,
            images=images,
            metadata=metadata,
            processing_time=time.time() - start,
        )

    def cleanup(self) -> None:
        """Release GPU memory held by the Docling converter."""
        if self._converter is not None:
            logger.info("Unloading Docling converter...")
            del self._converter
            self._converter = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
            except ImportError:
                pass
            logger.info("Docling converter unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._converter is not None
