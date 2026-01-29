"""Generic document processing pipeline built around Docling and late chunking.

This module isolates the expensive extraction/embedding path so multiple
workflows (ArXiv, PDF batches, etc.) can reuse the same logic without
duplicating code. It ensures late chunking and contextual embeddings
while staying source-agnostic.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Suppress noisy third-party warnings produced during docling imports
warnings.filterwarnings(
    "ignore",
    message="Use `ConversionResult` instead.",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPy.* has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"\[Errno 13\] Permission denied.  joblib will operate in serial mode",
    category=UserWarning,
)

from core.embedders import EmbedderFactory, EmbeddingConfig
from core.embedders.embedders_jina import ChunkWithEmbedding
from core.extractors import DoclingExtractor, LaTeXExtractor
from core.extractors.extractors_base import ExtractorConfig as DoclingConfig
from core.processors.text.chunking_strategies import ChunkingStrategyFactory

logger = logging.getLogger(__name__)


def _get_default_staging_dir() -> str:
    """Return a portable default staging directory path.

    Uses /dev/shm on Linux if available (RAM-backed), otherwise falls back
    to the system temp directory.
    """
    if os.path.exists("/dev/shm") and os.path.isdir("/dev/shm"):
        return "/dev/shm/document_staging"
    return os.path.join(tempfile.gettempdir(), "document_staging")


@dataclass
class ProcessingConfig:
    """Configuration for generic document processing."""

    # Extraction settings
    use_gpu: bool = True
    extract_tables: bool = True
    extract_equations: bool = True
    extract_images: bool = True
    use_ocr: bool = False

    # Embedding settings
    embedding_model: str = "jina-v4"
    embedder_type: str = "jina"
    embedding_dim: int = 2048
    use_fp16: bool = True
    device: str | None = None

    # Chunking settings
    chunk_size_tokens: int = 1000
    chunk_overlap_tokens: int = 200
    chunking_strategy: str = "late"  # 'late', 'semantic', 'sliding', 'sliding_window', 'token', 'traditional'
    max_chunk_size: int = 8192

    # Processing settings
    batch_size: int = 1
    num_workers: int = 1
    timeout_seconds: int = 300

    # Performance settings
    cache_embeddings: bool = True
    use_ramfs_staging: bool = True
    staging_dir: str = field(default_factory=_get_default_staging_dir)


@dataclass
class ExtractionResult:
    """Raw extraction output prior to chunking or embedding."""

    full_text: str
    tables: list[dict[str, Any]] = field(default_factory=list)
    equations: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    latex_source: str | None = None
    has_latex: bool = False
    extraction_time: float = 0.0
    extractor_version: str = ""


@dataclass
class ProcessingResult:
    """Complete result from document processing."""

    extraction: ExtractionResult
    chunks: list[ChunkWithEmbedding]
    processing_metadata: dict[str, Any]
    total_processing_time: float
    extraction_time: float
    chunking_time: float
    embedding_time: float
    success: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the processing result."""

        return {
            "extraction": {
                "full_text": self.extraction.full_text,
                "tables": self.extraction.tables,
                "equations": self.extraction.equations,
                "images": self.extraction.images,
                "figures": self.extraction.figures,
                "metadata": self.extraction.metadata,
                "has_latex": self.extraction.has_latex,
                "extraction_time": self.extraction.extraction_time,
            },
            "chunks": [
                {
                    "text": chunk.text,
                    "embedding": np.asarray(chunk.embedding).tolist(),
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "context_window_used": chunk.context_window_used,
                }
                for chunk in self.chunks
            ]
            if self.chunks
            else [],
            "processing_metadata": self.processing_metadata,
            "performance": {
                "total_time": self.total_processing_time,
                "extraction_time": self.extraction_time,
                "chunking_time": self.chunking_time,
                "embedding_time": self.embedding_time,
            },
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DocumentProcessor:
    """Source-agnostic document processor leveraging Docling and late chunking."""

    def __init__(self, config: ProcessingConfig | None = None, embedder: Any | None = None):
        self.config = config or ProcessingConfig()

        # Store extractor config for lazy loading - don't load VLM models yet
        # This allows multiple DocumentProcessor instances without GPU memory overflow
        self._docling_config = DoclingConfig(
            ocr_enabled=self.config.use_ocr,
            extract_tables=self.config.extract_tables,
            extract_equations=self.config.extract_equations,
            extract_images=self.config.extract_images,
        )
        self._docling_extractor: Any | None = None
        self._latex_extractor: Any | None = None

        # Store embedder config for lazy loading - don't load the model yet
        # This allows us to unload extraction models before loading embedder
        embedder_type = (self.config.embedder_type or "jina").lower()
        model_name = self.config.embedding_model or "jinaai/jina-embeddings-v4"
        if model_name.lower() in {"jina-v4", "jinaai/jina-v4"}:
            model_name = "jinaai/jina-embeddings-v4"

        if embedder_type in {"sentence", "sentence-transformers"} and "sentence-transformers" not in model_name:
            logger.warning(
                "SentenceTransformers embedder requested without explicit model; "
                "defaulting to sentence-transformers/all-mpnet-base-v2.",
            )
            model_name = "sentence-transformers/all-mpnet-base-v2"

        self._embed_config = EmbeddingConfig(
            model_name=model_name,
            device=self.config.device or "cuda",
            batch_size=self.config.batch_size,
            use_fp16=self.config.use_fp16,
            chunk_size_tokens=self.config.chunk_size_tokens,
            chunk_overlap_tokens=self.config.chunk_overlap_tokens,
        )
        self._embedder_type = embedder_type
        self.embedding_model = model_name
        # Use injected embedder if provided, otherwise lazy-load when needed
        self._embedder = embedder

        # Initialize staging directory management
        self._staging_tempdir: tempfile.TemporaryDirectory | None = None
        self.staging_dir: Path | None = None

        if self.config.use_ramfs_staging:
            # Create base directory if needed (parent only)
            base_dir = Path(self.config.staging_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            # Create secure per-run subdirectory using TemporaryDirectory for auto-cleanup
            self._staging_tempdir = tempfile.TemporaryDirectory(
                dir=str(base_dir), prefix="staging-"
            )
            self.staging_dir = Path(self._staging_tempdir.name)
            # Set strict permissions and validate
            os.chmod(self.staging_dir, 0o700)
            stat_info = os.lstat(self.staging_dir)
            if os.path.islink(self.staging_dir):
                self._staging_tempdir.cleanup()
                raise RuntimeError(f"Staging directory is a symlink: {self.staging_dir}")
            if stat_info.st_uid != os.getuid():
                self._staging_tempdir.cleanup()
                raise RuntimeError(f"Staging directory not owned by current user: {self.staging_dir}")

        logger.info("Initialized DocumentProcessor with config: %s", self.config)

    @property
    def embedder(self):
        """Lazy-load the embedder only when first needed.

        This ensures we don't have both the extraction VLM models and
        embedding models loaded simultaneously, which would exceed GPU memory.
        """
        if self._embedder is None:
            logger.info("Lazy-loading embedder: %s", self.embedding_model)
            if self._embedder_type in {"sentence", "sentence-transformers"}:
                logger.warning(
                    "SentenceTransformersEmbedder is deprecated and will be removed after migration; "
                    "routing to JinaV4Embedder fallback.",
                )
            elif self._embedder_type not in {"jina", "transformer"}:
                logger.warning("Unknown embedder_type '%s'; defaulting to JinaV4Embedder", self._embedder_type)

            self._embedder = EmbedderFactory.create(
                model_name=self.embedding_model,
                config=self._embed_config,
            )
        return self._embedder

    @property
    def docling_extractor(self):
        """Lazy-load Docling extractor only when first needed.

        Docling's VLM models consume ~8GB VRAM. Lazy loading allows:
        1. Multiple processor instances without immediate memory pressure
        2. Explicit unloading before embedding via unload_extractor()
        3. Batch processing that shares extractor across documents
        """
        if self._docling_extractor is None:
            if DoclingExtractor is None:
                raise ImportError("DoclingExtractor not available - install docling package")
            logger.info("Lazy-loading Docling extractor (VLM models)")
            self._docling_extractor = DoclingExtractor(self._docling_config)
        return self._docling_extractor

    @property
    def latex_extractor(self):
        """Lazy-load LaTeX extractor only when first needed."""
        if self._latex_extractor is None and self.config.extract_equations:
            if LaTeXExtractor is None:
                raise ImportError("LaTeXExtractor not available")
            logger.info("Lazy-loading LaTeX extractor")
            self._latex_extractor = LaTeXExtractor()
        return self._latex_extractor

    def unload_extractor(self) -> None:
        """Explicitly unload extraction models to free GPU memory.

        Call this before embedding when processing batches, or when you need
        to reclaim VRAM. The extractor will be lazily reloaded on next use.
        """
        if self._docling_extractor is not None:
            logger.info("Unloading Docling extractor to free VLM memory")
            if hasattr(self._docling_extractor, "cleanup"):
                self._docling_extractor.cleanup()
            del self._docling_extractor
            self._docling_extractor = None

        if self._latex_extractor is not None:
            del self._latex_extractor
            self._latex_extractor = None

        # Clear CUDA cache
        try:
            import gc

            import torch
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as exc:
            logger.warning("Failed during CUDA cache cleanup: %s", exc, exc_info=True)

    def cleanup(self) -> None:
        """Clean up staging directory and other resources.

        Call this when done processing to release disk space. Also called
        automatically by __del__ if not explicitly invoked.
        """
        staging_tempdir = getattr(self, "_staging_tempdir", None)
        if staging_tempdir is not None:
            try:
                staging_tempdir.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to cleanup staging directory: %s", exc)
            finally:
                self._staging_tempdir = None
                self.staging_dir = None

    def __del__(self) -> None:
        """Ensure staging directory is cleaned up on garbage collection."""
        self.cleanup()

    def _clear_gpu_memory(self, unload_extractor: bool = True) -> None:
        """Clear CUDA cache to free GPU memory between processing phases.

        This is essential when processing large documents where the extraction
        VLM models (8GB+) need to release memory before embedding starts.

        Args:
            unload_extractor: If True, also unload the Docling extractor's models
                to fully free GPU memory. The extractor will be lazily reloaded
                if needed for batch processing.
        """
        try:
            import gc

            import torch

            # Unload extraction models if requested
            if unload_extractor:
                self.unload_extractor()

            if torch.cuda.is_available():
                # Run garbage collection to release Python references
                gc.collect()
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Run GC again after cache clear
                gc.collect()
                # Synchronize to ensure all operations complete
                torch.cuda.synchronize()

                # Log memory status
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU memory after clearing: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        except Exception as exc:
            logger.warning("Could not clear CUDA cache: %s", exc)

    def process_document(
        self,
        pdf_path: str | Path,
        latex_path: str | Path | None = None,
        document_id: str | None = None,
    ) -> ProcessingResult:
        start_time = time.time()
        errors: list[str] = []
        warnings: list[str] = []

        pdf_path = Path(pdf_path)
        if latex_path:
            latex_path = Path(latex_path)

        doc_id = document_id or pdf_path.stem
        logger.info("Processing document: %s", doc_id)

        # Configure CUDA memory allocation for better fragmentation handling
        try:
            import torch
            if torch.cuda.is_available():
                # This helps with memory fragmentation on smaller GPUs
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        except Exception:
            pass

        try:
            extraction_start = time.time()
            extraction_result = self._extract_content(pdf_path, latex_path)
            extraction_time = time.time() - extraction_start

            # Clear CUDA cache after extraction to free VLM memory before embedding
            self._clear_gpu_memory()

            strategy = self.config.chunking_strategy
            if strategy == "late":
                embedding_start = time.time()
                chunks = self._create_late_chunks(extraction_result.full_text)
                embedding_time = time.time() - embedding_start
                chunking_time = 0.0
            elif strategy in ("semantic", "sliding", "sliding_window", "token"):
                # Use ChunkingStrategyFactory with strategy-specific kwargs
                chunking_start = time.time()
                if strategy == "token":
                    chunker = ChunkingStrategyFactory.create_strategy(
                        strategy,
                        chunk_size=self.config.chunk_size_tokens,
                        chunk_overlap=self.config.chunk_overlap_tokens,
                    )
                elif strategy == "semantic":
                    chunker = ChunkingStrategyFactory.create_strategy(
                        strategy,
                        max_chunk_size=self.config.chunk_size_tokens,
                        min_chunk_size=self.config.chunk_overlap_tokens,
                        respect_sentences=True,
                        respect_paragraphs=True,
                    )
                else:  # sliding or sliding_window
                    chunker = ChunkingStrategyFactory.create_strategy(
                        strategy,
                        window_size=self.config.chunk_size_tokens,
                        step_size=self.config.chunk_size_tokens - self.config.chunk_overlap_tokens,
                    )
                text_chunks = chunker.create_chunks(extraction_result.full_text)
                # Convert TextChunk to dict format expected by _embed_chunks
                chunks = [
                    {
                        "text": tc.text,
                        "start_char": tc.start_char,
                        "end_char": tc.end_char,
                        "chunk_index": tc.chunk_index,
                        **tc.metadata,
                    }
                    for tc in text_chunks
                ]
                chunking_time = time.time() - chunking_start

                embedding_start = time.time()
                if chunks:
                    chunks = self._embed_chunks(chunks)
                embedding_time = time.time() - embedding_start
            elif strategy == "traditional":
                chunking_start = time.time()
                chunks = self._create_traditional_chunks(extraction_result.full_text)
                chunking_time = time.time() - chunking_start

                embedding_start = time.time()
                if chunks:
                    chunks = self._embed_chunks(chunks)
                embedding_time = time.time() - embedding_start
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")

            if not chunks:
                logger.warning("No chunks produced for document %s. Document may be empty.", doc_id)
                return ProcessingResult(
                    extraction=extraction_result,
                    chunks=[],
                    processing_metadata={"error": "No chunks produced"},
                    total_processing_time=time.time() - start_time,
                    extraction_time=extraction_time,
                    chunking_time=chunking_time,
                    embedding_time=embedding_time,
                    success=False,
                    errors=["Document produced no chunks"],
                    warnings=["Empty or unprocessable document"],
                )

            total_time = time.time() - start_time

            processing_metadata = {
                "processor_version": "1.0",
                "embedding_model": self.embedding_model,
                "chunking_strategy": self.config.chunking_strategy,
                "chunk_size_tokens": self.config.chunk_size_tokens,
                "chunk_overlap_tokens": self.config.chunk_overlap_tokens,
                "document_id": doc_id,
                "source_path": str(pdf_path),
                "timestamp": datetime.now(UTC).isoformat(),
                "chunk_count": len(chunks),
                "has_latex": extraction_result.has_latex,
                "has_tables": len(extraction_result.tables) > 0,
                "has_equations": len(extraction_result.equations) > 0,
            }

            return ProcessingResult(
                extraction=extraction_result,
                chunks=chunks,
                processing_metadata=processing_metadata,
                total_processing_time=total_time,
                extraction_time=extraction_time,
                chunking_time=chunking_time,
                embedding_time=embedding_time,
                success=True,
                errors=errors,
                warnings=warnings,
            )

        except Exception as exc:
            logger.error("Failed to process document %s: %s", doc_id, exc)
            errors.append(str(exc))
            return ProcessingResult(
                extraction=ExtractionResult(full_text=""),
                chunks=[],
                processing_metadata={},
                total_processing_time=time.time() - start_time,
                extraction_time=0.0,
                chunking_time=0.0,
                embedding_time=0.0,
                success=False,
                errors=errors,
                warnings=warnings,
            )

    def _extract_content(
        self,
        pdf_path: Path,
        latex_path: Path | None = None,
    ) -> ExtractionResult:
        start_time = time.time()

        docling_result = self.docling_extractor.extract(str(pdf_path))

        latex_source = None
        has_latex = False
        equations = docling_result.equations or []

        if latex_path and latex_path.exists() and self.latex_extractor:
            try:
                # Use the public extract API - pass the Path, get ExtractionResult
                latex_extraction = self.latex_extractor.extract(latex_path)
                has_latex = True
                latex_source = latex_path.read_text(encoding="utf-8")
                equations = latex_extraction.equations or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to process LaTeX source: %s", exc)

        # Access ExtractionResult fields as attributes
        full_text = docling_result.text or ""
        extractor_version = docling_result.metadata.get("version", "unknown") if docling_result.metadata else "unknown"

        return ExtractionResult(
            full_text=full_text,
            tables=docling_result.tables or [],
            equations=equations,
            images=docling_result.images or [],
            figures=[],  # Not in base ExtractionResult
            metadata=docling_result.metadata or {},
            latex_source=latex_source,
            has_latex=has_latex,
            extraction_time=time.time() - start_time,
            extractor_version=extractor_version,
        )

    def _create_late_chunks(self, text: str) -> list[ChunkWithEmbedding]:
        return self.embedder.embed_with_late_chunking(text)

    def _create_traditional_chunks(self, text: str) -> list[dict[str, Any]]:
        chunk_size = self.config.chunk_size_tokens
        overlap = self.config.chunk_overlap_tokens

        if chunk_size <= 0:
            raise ValueError(f"chunk_size_tokens must be positive, got {chunk_size}")

        if overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap_tokens (%s) must be less than chunk_size_tokens (%s)"
                % (overlap, chunk_size)
            )

        if overlap < 0:
            overlap = 0

        step = chunk_size - overlap
        if step <= 0:
            raise ValueError(
                "Invalid chunking parameters: step size would be %s. "
                "Ensure chunk_size_tokens > chunk_overlap_tokens." % step,
            )

        chunks: list[dict[str, Any]] = []
        tokens = text.split()

        if not tokens:
            return []

        # Build token_positions: starting character index for each token
        token_positions: list[int] = []
        search_offset = 0
        for token in tokens:
            pos = text.find(token, search_offset)
            if pos == -1:
                # Fallback: approximate position
                pos = search_offset
            token_positions.append(pos)
            search_offset = pos + len(token)

        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            end_token_idx = min(i + chunk_size, len(tokens)) - 1

            start_char = token_positions[i]
            # end_char is start of last token + length of last token
            end_char = token_positions[end_token_idx] + len(tokens[end_token_idx])

            chunks.append(
                {
                    "text": chunk_text,
                    "start_token": i,
                    "end_token": min(i + chunk_size, len(tokens)),
                    "chunk_index": len(chunks),
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

        return chunks

    def _embed_chunks(self, chunks: list[dict[str, Any]]) -> list[ChunkWithEmbedding]:
        if not chunks:
            return []

        # Use configured batch size to avoid OOM on large documents
        # Default batch_size=1 may be too conservative, but batch_size=len(texts) is dangerous
        texts = [chunk["text"] for chunk in chunks]
        batch_size = min(self.config.batch_size or 8, 32)  # Cap at 32 to avoid OOM
        embeddings = self.embedder.embed_texts(texts, batch_size=batch_size)

        # Build ChunkWithEmbedding instances from chunks and embeddings
        embedded_chunks: list[ChunkWithEmbedding] = []
        zero_vector = np.zeros(self.config.embedding_dim, dtype=np.float32)

        for i, chunk in enumerate(chunks):
            # Get embedding or substitute zero vector if missing
            raw_embedding = embeddings[i] if i < len(embeddings) else None

            if raw_embedding is not None:
                embedding_array = np.asarray(raw_embedding, dtype=np.float32)
                if embedding_array.ndim != 1:
                    embedding_array = embedding_array.reshape(-1)
            else:
                embedding_array = zero_vector.copy()

            embedded_chunks.append(
                ChunkWithEmbedding(
                    text=chunk["text"],
                    embedding=embedding_array,
                    start_char=chunk.get("start_char", 0),
                    end_char=chunk.get("end_char", len(chunk["text"])),
                    start_token=chunk.get("start_token", 0),
                    end_token=chunk.get("end_token", 0),
                    chunk_index=i,
                    total_chunks=len(chunks),
                    context_window_used=len(chunk["text"].split()),
                )
            )

        return embedded_chunks

    def process_batch(
        self,
        document_paths: list[tuple[Path, Path | None]],
        document_ids: list[str] | None = None,
        optimize_memory: bool = True,
    ) -> list[ProcessingResult]:
        """Process multiple documents with optimized model loading.

        Args:
            document_paths: List of (pdf_path, latex_path) tuples
            document_ids: Optional list of document IDs
            optimize_memory: If True (default), performs two-phase processing:
                1. Extract all documents (keeping extractor loaded)
                2. Unload extractor, load embedder
                3. Embed all chunks
                This minimizes GPU memory pressure by avoiding simultaneous
                model loading. Set to False for sequential per-document processing.

        Returns:
            List of ProcessingResult for each document
        """
        if not optimize_memory:
            # Fall back to simple sequential processing
            results: list[ProcessingResult] = []
            for i, (pdf_path, latex_path) in enumerate(document_paths):
                doc_id = document_ids[i] if document_ids and i < len(document_ids) else None
                result = self.process_document(pdf_path, latex_path, doc_id)
                results.append(result)
            return results

        # Optimized batch mode: separate extraction and embedding phases
        # Use indexed lists to preserve input order even when failures occur
        logger.info("Starting optimized batch processing for %d documents", len(document_paths))
        n_docs = len(document_paths)
        results: list[ProcessingResult | None] = [None] * n_docs
        extractions: list[tuple[ExtractionResult, str, Path, float] | None] = [None] * n_docs

        # Phase 1: Extract all documents (extractor stays loaded)
        logger.info("Phase 1: Extracting %d documents", n_docs)
        for i, (pdf_path_raw, latex_path_raw) in enumerate(document_paths):
            # Normalize paths to Path objects (input may be str or Path)
            pdf_path = Path(pdf_path_raw)
            latex_path = Path(latex_path_raw) if latex_path_raw else None
            doc_id = document_ids[i] if document_ids and i < len(document_ids) else pdf_path.stem
            start_time = time.time()
            try:
                extraction_result = self._extract_content(pdf_path, latex_path)
                extractions[i] = (extraction_result, doc_id, pdf_path, start_time)
                logger.info("Extracted document %d/%d: %s", i + 1, n_docs, doc_id)
            except Exception as exc:
                logger.error("Failed to extract document %s: %s", doc_id, exc)
                results[i] = ProcessingResult(
                    extraction=ExtractionResult(full_text=""),
                    chunks=[],
                    processing_metadata={},
                    total_processing_time=time.time() - start_time,
                    extraction_time=0.0,
                    chunking_time=0.0,
                    embedding_time=0.0,
                    success=False,
                    errors=[str(exc)],
                )

        # Unload extractor to free GPU memory before embedding
        logger.info("Unloading extractor before embedding phase")
        self.unload_extractor()
        self._clear_gpu_memory(unload_extractor=False)

        # Phase 2: Chunk and embed all extractions
        logger.info("Phase 2: Chunking and embedding extractions")
        for i, extraction_data in enumerate(extractions):
            if extraction_data is None:
                continue  # Already added error result at this index

            extraction_result, doc_id, pdf_path, start_time = extraction_data
            extraction_time = extraction_result.extraction_time

            try:
                strategy = self.config.chunking_strategy
                if strategy == "late":
                    embedding_start = time.time()
                    chunks = self._create_late_chunks(extraction_result.full_text)
                    embedding_time = time.time() - embedding_start
                    chunking_time = 0.0
                elif strategy in ("semantic", "sliding", "sliding_window", "token"):
                    chunking_start = time.time()
                    if strategy == "token":
                        chunker = ChunkingStrategyFactory.create_strategy(
                            strategy,
                            chunk_size=self.config.chunk_size_tokens,
                            chunk_overlap=self.config.chunk_overlap_tokens,
                        )
                    elif strategy == "semantic":
                        chunker = ChunkingStrategyFactory.create_strategy(
                            strategy,
                            max_chunk_size=self.config.chunk_size_tokens,
                            min_chunk_size=self.config.chunk_overlap_tokens,
                            respect_sentences=True,
                            respect_paragraphs=True,
                        )
                    else:  # sliding or sliding_window
                        chunker = ChunkingStrategyFactory.create_strategy(
                            strategy,
                            window_size=self.config.chunk_size_tokens,
                            step_size=self.config.chunk_size_tokens - self.config.chunk_overlap_tokens,
                        )
                    text_chunks = chunker.create_chunks(extraction_result.full_text)
                    chunks = [
                        {
                            "text": tc.text,
                            "start_char": tc.start_char,
                            "end_char": tc.end_char,
                            "chunk_index": tc.chunk_index,
                            **tc.metadata,
                        }
                        for tc in text_chunks
                    ]
                    chunking_time = time.time() - chunking_start

                    embedding_start = time.time()
                    if chunks:
                        chunks = self._embed_chunks(chunks)
                    embedding_time = time.time() - embedding_start
                elif strategy == "traditional":
                    chunking_start = time.time()
                    chunks = self._create_traditional_chunks(extraction_result.full_text)
                    chunking_time = time.time() - chunking_start

                    embedding_start = time.time()
                    if chunks:
                        chunks = self._embed_chunks(chunks)
                    embedding_time = time.time() - embedding_start
                else:
                    raise ValueError(f"Unknown chunking strategy: {strategy}")

                # Use sum of phase times for accurate per-doc metrics in batch mode
                # (wall-clock time would include time spent on other documents)
                total_time = extraction_time + chunking_time + embedding_time
                processing_metadata = {
                    "processor_version": "1.0",
                    "embedding_model": self.embedding_model,
                    "chunking_strategy": self.config.chunking_strategy,
                    "chunk_size_tokens": self.config.chunk_size_tokens,
                    "chunk_overlap_tokens": self.config.chunk_overlap_tokens,
                    "document_id": doc_id,
                    "source_path": str(pdf_path),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "chunk_count": len(chunks) if chunks else 0,
                    "has_latex": extraction_result.has_latex,
                    "has_tables": len(extraction_result.tables) > 0,
                    "has_equations": len(extraction_result.equations) > 0,
                    "batch_mode": True,
                }

                results[i] = ProcessingResult(
                    extraction=extraction_result,
                    chunks=chunks if chunks else [],
                    processing_metadata=processing_metadata,
                    total_processing_time=total_time,
                    extraction_time=extraction_time,
                    chunking_time=chunking_time,
                    embedding_time=embedding_time,
                    success=bool(chunks),
                    errors=[] if chunks else ["Document produced no chunks"],
                    warnings=[] if chunks else ["Empty or unprocessable document"],
                )
                logger.info("Embedded document %d/%d: %s (%d chunks)",
                           i + 1, n_docs, doc_id, len(chunks) if chunks else 0)

            except Exception as exc:
                logger.error("Failed to embed document %s: %s", doc_id, exc)
                results[i] = ProcessingResult(
                    extraction=extraction_result,
                    chunks=[],
                    processing_metadata={},
                    total_processing_time=time.time() - start_time,
                    extraction_time=extraction_time,
                    chunking_time=0.0,
                    embedding_time=0.0,
                    success=False,
                    errors=[str(exc)],
                )

        # Filter out None entries and return (all should be filled, but be safe)
        final_results = [r for r in results if r is not None]
        logger.info("Batch processing complete: %d/%d successful",
                   sum(1 for r in final_results if r.success), len(final_results))
        return final_results


__all__ = [
    "DocumentProcessor",
    "ExtractionResult",
    "ProcessingConfig",
    "ProcessingResult",
]
