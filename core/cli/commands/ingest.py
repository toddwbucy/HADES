"""Unified ingest command — auto-detects arxiv IDs vs file paths.

Usage:
    hades ingest 2501.12345                    # arxiv ID
    hades ingest document.pdf                  # local file
    hades ingest 2501.12345 2501.67890         # multiple arxiv IDs
    hades ingest paper1.pdf paper2.pdf         # multiple files
    hades ingest 2501.12345 paper.pdf          # mixed
    hades ingest /papers/*.pdf --batch         # batch mode with progress
    hades ingest --resume                      # resume after failure

The command auto-detects inputs:
- Patterns like "2501.12345" or "hep-th/9901001" → arxiv ID
- File paths (existing files) → local file ingest

Batch mode provides:
- JSON progress to stderr (stdout stays clean for final result)
- Per-document error isolation (one failure doesn't stop the batch)
- Resume via state file recording completed IDs
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.cli.config import get_arango_config, get_config
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    progress,
    success_response,
)
from core.database.collections import get_profile
from core.database.keys import chunk_key, embedding_key, normalize_document_key
from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

# Pattern for arxiv IDs: YYMM.NNNNN or category/NNNNNNN
_ARXIV_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$|^[a-z-]+/\d{7}(v\d+)?$", re.IGNORECASE)

# Extensions that should be routed through CodeProcessor + Jina Code LoRA
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp",
    ".h", ".hpp", ".cu", ".cuh", ".rb", ".swift", ".kt",
}


def _is_arxiv_id(s: str) -> bool:
    """Check if a string looks like an arxiv ID."""
    return bool(_ARXIV_PATTERN.match(s))


def _is_code_file(path: Path, task: str | None) -> bool:
    """Return True if this file should be routed through the code pipeline."""
    if task == "code":
        return True
    return task is None and path.suffix.lower() in _CODE_EXTENSIONS


def _classify_inputs(inputs: list[str]) -> tuple[list[str], list[str]]:
    """Classify inputs into arxiv IDs and file paths.

    Args:
        inputs: List of strings (arxiv IDs or file paths)

    Returns:
        Tuple of (arxiv_ids, file_paths)
    """
    arxiv_ids = []
    file_paths = []

    for inp in inputs:
        if _is_arxiv_id(inp):
            arxiv_ids.append(inp)
        elif Path(inp).exists():
            file_paths.append(inp)
        elif _is_arxiv_id(Path(inp).name):
            # Handle case where user passes full path that looks like arxiv ID
            arxiv_ids.append(Path(inp).name)
        else:
            # Assume it's a file path that doesn't exist yet (will error later)
            file_paths.append(inp)

    return arxiv_ids, file_paths


def ingest(
    inputs: list[str],
    document_id: str | None = None,
    force: bool = False,
    batch: bool = False,
    resume: bool = False,
    task: str | None = None,
    start_time: float = 0.0,
    extra_metadata: dict[str, Any] | None = None,
) -> CLIResponse:
    """Ingest documents into the knowledge base.

    Auto-detects arxiv IDs vs file paths and routes to appropriate handler.
    Code files (.rs, .cu, .py, etc.) are automatically routed through
    CodeProcessor with Jina V4 Code LoRA embeddings.

    Args:
        inputs: List of arxiv IDs or file paths to ingest.
        document_id: Custom document ID (only for single file ingest).
        force: Force reprocessing even if already exists.
        batch: Enable batch mode with progress reporting and error isolation.
        resume: Resume from previous batch state (implies batch=True).
        task: Embedding task type ('code' forces code pipeline). Auto-detected by extension.
        start_time: Command start timestamp.
        extra_metadata: Custom metadata fields to merge into the document metadata record.

    Returns:
        CLIResponse with ingestion results.
    """
    # Resume implies batch mode
    if resume:
        batch = True

    # Handle resume-only mode (no inputs needed)
    if resume and not inputs:
        return _ingest_batch([], None, force, resume, start_time)

    if not inputs:
        return error_response(
            command="ingest",
            code=ErrorCode.CONFIG_ERROR,
            message="No inputs provided. Specify arxiv IDs or file paths.",
            start_time=start_time,
        )

    # Classify inputs
    arxiv_ids, file_paths = _classify_inputs(inputs)

    # If custom document_id specified, only allow single file
    if document_id and (len(file_paths) != 1 or arxiv_ids):
        return error_response(
            command="ingest",
            code=ErrorCode.CONFIG_ERROR,
            message="--id option only valid for single file ingest",
            start_time=start_time,
        )

    # Batch mode not compatible with custom document_id
    if batch and document_id:
        return error_response(
            command="ingest",
            code=ErrorCode.CONFIG_ERROR,
            message="--id option not compatible with batch mode",
            start_time=start_time,
        )

    # Use batch processor for batch mode or large input sets
    if batch or len(arxiv_ids) + len(file_paths) > 5:
        all_items = arxiv_ids + file_paths
        return _ingest_batch(all_items, None, force, resume, start_time, task=task, extra_metadata=extra_metadata)

    # Standard (non-batch) mode
    progress(f"Ingesting {len(arxiv_ids)} arxiv papers and {len(file_paths)} files...")

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="ingest",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    results: list[dict[str, Any]] = []

    # Process arxiv IDs
    if arxiv_ids:
        for arxiv_id in arxiv_ids:
            result = _ingest_arxiv_paper(arxiv_id, config, force, extra_metadata=extra_metadata)
            results.append(result)

    # Process file paths
    if file_paths:
        for file_path in file_paths:
            result = _ingest_file(file_path, config, document_id, force, task=task, extra_metadata=extra_metadata)
            results.append(result)

    # Summarize
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    if failed and not successful:
        return error_response(
            command="ingest",
            code=ErrorCode.PROCESSING_FAILED,
            message=f"All {len(failed)} items failed to ingest",
            details={"results": results},
            start_time=start_time,
        )

    return success_response(
        command="ingest",
        data={
            "ingested": len(successful),
            "failed": len(failed),
            "results": results,
        },
        start_time=start_time,
    )


def _ingest_batch(
    items: list[str],
    document_id: str | None,
    force: bool,
    resume: bool,
    start_time: float,
    task: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> CLIResponse:
    """Ingest items using batch processor with progress and resume.

    Args:
        items: List of arxiv IDs or file paths.
        document_id: Not used in batch mode (passed for signature compatibility).
        force: Force reprocessing even if already exists.
        resume: Resume from previous batch state.
        start_time: Command start timestamp.
        task: Embedding task type ('code' forces code pipeline).
        extra_metadata: Custom metadata fields to merge into document records.

    Returns:
        CLIResponse with batch results.
    """
    from core.processors.batch import BatchProcessor

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="ingest",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    def process_single(item_id: str) -> dict[str, Any]:
        """Process a single item (arxiv ID or file path)."""
        if _is_arxiv_id(item_id):
            return _ingest_arxiv_paper(item_id, config, force, extra_metadata=extra_metadata)
        else:
            return _ingest_file(item_id, config, None, force, task=task, extra_metadata=extra_metadata)

    processor = BatchProcessor(
        state_file=".hades-batch-state.json",
        progress_interval=0.5,
    )

    result = processor.process_batch(items, process_single, resume=resume)

    if result.failed > 0 and result.completed == 0:
        return error_response(
            command="ingest",
            code=ErrorCode.PROCESSING_FAILED,
            message=f"All {result.failed} items failed to ingest",
            details={
                "results": result.results,
                "errors": result.errors,
            },
            start_time=start_time,
        )

    return success_response(
        command="ingest",
        data={
            "total": result.total,
            "completed": result.completed,
            "failed": result.failed,
            "skipped": result.skipped,
            "results": result.results,
            "errors": result.errors if result.errors else None,
        },
        start_time=start_time,
        metadata={
            "batch_mode": True,
            "duration_seconds": result.duration_seconds,
        },
    )


def _ingest_file(
    file_path: str,
    config: Any,
    document_id: str | None,
    force: bool,
    task: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest a local file (any supported format).

    Code files (.rs, .cu, .py, etc.) are automatically routed through
    CodeProcessor with Jina V4 Code LoRA embeddings. Use task='code' to
    force code pipeline for any extension.

    Args:
        file_path: Path to the file.
        config: CLI configuration.
        document_id: Custom document ID (optional).
        force: Overwrite existing data.
        task: Embedding task type. 'code' forces code pipeline; auto-detected by extension.
        extra_metadata: Custom metadata fields to merge into the document metadata record.

    Returns:
        Result dict with success status.
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "path": file_path,
            "success": False,
            "error": f"File not found: {file_path}",
        }

    doc_id = document_id or path.stem

    # Check if already exists (mirrors _ingest_arxiv_paper logic)
    if not force:
        exists = _check_paper_in_db(doc_id, config)
        if exists:
            progress(f"Document {doc_id} already in database, skipping (use --force to reprocess)")
            return {
                "path": file_path,
                "document_id": doc_id,
                "success": True,
                "skipped": True,
                "message": "Already in database",
            }

    # Route code files through the code pipeline
    if _is_code_file(path, task):
        progress(f"Ingesting {path.name} as {doc_id} (code pipeline, Code LoRA)...")
        try:
            result = _process_and_store_code(
                file_path=path,
                config=config,
                document_id=doc_id,
                force=force,
                extra_metadata=extra_metadata,
            )
            if not result["success"]:
                return {
                    "path": file_path,
                    "document_id": doc_id,
                    "success": False,
                    "error": result.get("error", "Code processing failed"),
                }
            return {
                "path": file_path,
                "document_id": doc_id,
                "success": True,
                "num_chunks": result.get("num_chunks", 0),
                "pipeline": "code",
            }
        except Exception as e:
            return {
                "path": file_path,
                "document_id": doc_id,
                "success": False,
                "error": str(e),
            }

    progress(f"Ingesting {path.name} as {doc_id}...")

    try:
        result = _process_and_store(
            arxiv_id=None,
            pdf_path=path,
            latex_path=None,
            metadata=None,
            config=config,
            document_id=doc_id,
            force=force,
            extra_metadata=extra_metadata,
        )

        if not result["success"]:
            return {
                "path": file_path,
                "document_id": doc_id,
                "success": False,
                "error": result.get("error", "Processing failed"),
            }

        return {
            "path": file_path,
            "document_id": doc_id,
            "success": True,
            "num_chunks": result.get("num_chunks", 0),
        }

    except Exception as e:
        return {
            "path": file_path,
            "document_id": doc_id,
            "success": False,
            "error": str(e),
        }


# =============================================================================
# Internal Helpers — Ingest (moved from arxiv.py)
# =============================================================================


def _ingest_arxiv_paper(
    arxiv_id: str,
    config: Any,
    force: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest a single arxiv paper.

    1. Check if already exists (unless force=True)
    2. Download PDF from arxiv
    3. Process with DocumentProcessor
    4. Store in ArangoDB
    """
    progress(f"Ingesting arxiv paper: {arxiv_id}")

    client = ArXivAPIClient(rate_limit_delay=1.0)

    try:
        # Validate ID
        if not client.validate_arxiv_id(arxiv_id):
            return {
                "arxiv_id": arxiv_id,
                "success": False,
                "error": f"Invalid arxiv ID format: {arxiv_id}",
            }

        # Check if already exists
        if not force:
            exists = _check_paper_in_db(arxiv_id, config)
            if exists:
                progress(f"Paper {arxiv_id} already in database, skipping (use --force to reprocess)")
                return {
                    "arxiv_id": arxiv_id,
                    "success": True,
                    "skipped": True,
                    "message": "Already in database",
                }

        # Download PDF
        progress(f"Downloading PDF for {arxiv_id}...")
        download_result = client.download_paper(
            arxiv_id,
            pdf_dir=config.pdf_base_path,
            latex_dir=config.latex_base_path,
            force=force,
        )

        if not download_result.success:
            return {
                "arxiv_id": arxiv_id,
                "success": False,
                "error": download_result.error_message or "Download failed",
            }

        progress(f"Downloaded {download_result.pdf_path} ({download_result.file_size_bytes:,} bytes)")

        # Process the PDF
        progress("Processing document (extracting text, generating embeddings)...")
        processing_result = _process_and_store(
            arxiv_id=arxiv_id,
            pdf_path=download_result.pdf_path,
            latex_path=download_result.latex_path,
            metadata=download_result.metadata,
            config=config,
            force=force,
            extra_metadata=extra_metadata,
        )

        if not processing_result["success"]:
            return {
                "arxiv_id": arxiv_id,
                "success": False,
                "error": processing_result.get("error", "Processing failed"),
            }

        progress(f"Stored {processing_result['num_chunks']} chunks for {arxiv_id}")

        return {
            "arxiv_id": arxiv_id,
            "success": True,
            "num_chunks": processing_result["num_chunks"],
            "title": download_result.metadata.title if download_result.metadata else None,
        }

    except Exception as e:
        return {
            "arxiv_id": arxiv_id,
            "success": False,
            "error": str(e),
        }
    finally:
        client.close()


def _check_paper_in_db(arxiv_id: str, config: Any) -> bool:
    """Check if a paper already exists in the database."""
    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

        arango_config = get_arango_config(config, read_only=True)
        client_config = ArangoHttp2Config(
            database=arango_config["database"],
            socket_path=arango_config.get("socket_path"),
            base_url=f"http://{arango_config['host']}:{arango_config['port']}",
            username=arango_config["username"],
            password=arango_config["password"],
        )

        client = ArangoHttp2Client(client_config)
        try:
            col = get_profile("arxiv")
            sanitized_id = normalize_document_key(arxiv_id)
            client.get_document(col.metadata, sanitized_id)
            return True
        except Exception:
            return False
        finally:
            client.close()
    except Exception:
        return False


def _process_and_store(
    arxiv_id: str | None,
    pdf_path: Path,
    latex_path: Path | None,
    metadata: Any,
    config: Any,
    document_id: str | None = None,
    force: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process a PDF and store results in the database."""
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config
    from core.processors.document_processor import DocumentProcessor, ProcessingConfig

    # Try to use the persistent embedding service instead of loading model in-process.
    # Use a longer timeout for ingest (model may need to load from idle + embed many chunks).
    # Fall back to in-process model if the service is unavailable.
    embedder = None
    try:
        from core.services.embedder_client import EmbedderClient

        embed_client = EmbedderClient(
            socket_path=config.embedding.service_socket,
            timeout=300.0,
            fallback_to_local=True,
        )
        if embed_client.is_service_available():
            embedder = embed_client
            progress("Using persistent embedding service")
        else:
            embed_client.close()
            progress("Embedding service unavailable, loading model in-process")
    except Exception:
        progress("Embedding service unavailable, loading model in-process")

    # Configure processor
    # Use semantic chunking to respect document structure (paragraphs/sentences)
    # Larger chunks (1000 tokens) preserve more context for better retrieval
    proc_config = ProcessingConfig(
        use_gpu=config.use_gpu,
        device=config.device,
        chunking_strategy="semantic",
        chunk_size_tokens=1000,
        chunk_overlap_tokens=200,
    )

    processor = DocumentProcessor(proc_config, embedder=embedder)

    try:
        # Process the document
        doc_id = document_id or arxiv_id or pdf_path.stem
        result = processor.process_document(
            pdf_path=pdf_path,
            latex_path=latex_path,
            document_id=doc_id,
        )

        if not result.success:
            return {
                "success": False,
                "error": "; ".join(result.errors) if result.errors else "Processing failed",
            }

        # Store in database
        arango_config = get_arango_config(config, read_only=False)
        client_config = ArangoHttp2Config(
            database=arango_config["database"],
            socket_path=arango_config.get("socket_path"),
            base_url=f"http://{arango_config['host']}:{arango_config['port']}",
            username=arango_config["username"],
            password=arango_config["password"],
        )

        client = ArangoHttp2Client(client_config)

        try:
            col = get_profile("arxiv")
            sanitized_id = normalize_document_key(doc_id)
            now_iso = datetime.now(UTC).isoformat()

            # Prepare metadata document
            meta_doc = {
                "_key": sanitized_id,
                "document_id": doc_id,
                "title": metadata.title if metadata else doc_id,
                "source": "arxiv" if arxiv_id else "local",
                "num_chunks": len(result.chunks),
                "processing_timestamp": now_iso,
                "status": "PROCESSED",
            }

            if arxiv_id and metadata:
                meta_doc.update(
                    {
                        "arxiv_id": arxiv_id,
                        "authors": metadata.authors,
                        "abstract": metadata.abstract,
                        "categories": metadata.categories,
                        "published": metadata.published.isoformat() if metadata.published else None,
                    }
                )

            # Merge user-provided custom metadata (last, so it can override)
            if extra_metadata:
                meta_doc.update(extra_metadata)
                # Reassert identity fields — these must stay consistent with chunks/embeddings
                meta_doc["_key"] = sanitized_id
                meta_doc["document_id"] = doc_id
                meta_doc["num_chunks"] = len(result.chunks)

            # Prepare chunk documents
            chunk_docs = []
            embedding_docs = []

            for chunk in result.chunks:
                ck = chunk_key(sanitized_id, chunk.chunk_index)

                chunk_docs.append(
                    {
                        "_key": ck,
                        "document_id": doc_id,
                        "paper_key": sanitized_id,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "text": chunk.text,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "created_at": now_iso,
                    }
                )

                embedding_docs.append(
                    {
                        "_key": embedding_key(ck),
                        "chunk_key": ck,
                        "document_id": doc_id,
                        "paper_key": sanitized_id,
                        "embedding": chunk.embedding.tolist(),
                        "embedding_dim": int(chunk.embedding.shape[0]),
                        "created_at": now_iso,
                    }
                )

            # Insert chunks and embeddings first so that metadata only
            # records success after the data is actually persisted.
            #
            # When force=True, use overwrite to replace existing documents,
            # then clean up orphaned chunks (safe insert-first approach).
            # This avoids the non-atomic purge+insert pattern where a failed
            # insert after purge would permanently lose data.
            progress(f"Storing {len(chunk_docs)} chunks in database...")

            if chunk_docs:
                client.insert_documents(col.chunks, chunk_docs, overwrite=force)
            if embedding_docs:
                client.insert_documents(col.embeddings, embedding_docs, overwrite=force)
            client.insert_documents(col.metadata, [meta_doc], overwrite=force)

            # After successful insert with force, clean up orphaned chunks/embeddings
            # from previous ingestion that had more chunks than the current one.
            if force:
                new_chunk_count = len(chunk_docs)
                # Remove chunks with index >= new count (orphans from prior ingestion)
                removed = client.query(
                    f"""
                    FOR c IN {col.chunks}
                        FILTER c.paper_key == @key AND c.chunk_index >= @max_idx
                        REMOVE c IN {col.chunks}
                        RETURN OLD._key
                    """,
                    {"key": sanitized_id, "max_idx": new_chunk_count},
                )
                # Remove corresponding orphaned embeddings
                if removed:
                    client.query(
                        f"""
                        FOR e IN {col.embeddings}
                            FILTER e.paper_key == @key AND e.chunk_key IN @orphan_keys
                            REMOVE e IN {col.embeddings}
                        """,
                        {"key": sanitized_id, "orphan_keys": removed},
                    )

            return {
                "success": True,
                "num_chunks": len(result.chunks),
            }

        finally:
            client.close()

    finally:
        processor.cleanup()
        if embedder is not None and hasattr(embedder, "close"):
            embedder.close()


def _process_and_store_code(
    file_path: Path,
    config: Any,
    document_id: str | None,
    force: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process a code file via CodeProcessor and store in arxiv profile collections.

    Uses Jina V4 Code LoRA embeddings (task="code") for semantically meaningful
    code representations. Chunks are AST-aligned (top-level functions/classes).

    Args:
        file_path: Path to the source code file.
        config: CLI configuration.
        document_id: Custom document ID (optional, defaults to file stem).
        force: Overwrite existing chunks/embeddings.
        extra_metadata: Custom metadata fields to merge into the document record.

    Returns:
        Result dict with success status and num_chunks.
    """
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config
    from core.processors.code_processor import CodeProcessor

    embedder = None
    try:
        from core.services.embedder_client import EmbedderClient

        embed_client = EmbedderClient(
            socket_path=config.embedding.service_socket,
            timeout=300.0,
            fallback_to_local=True,
        )
        if embed_client.is_service_available():
            embedder = embed_client
            progress("Using persistent embedding service (code task)")
        else:
            embed_client.close()
            progress("Embedding service unavailable, loading model in-process")
    except Exception:
        progress("Embedding service unavailable, loading model in-process")

    processor = CodeProcessor(embedder=embedder)

    try:
        # Use file's parent as repo_root so rel_path is just the filename.
        # process_file() extracts+chunks but does NOT embed (embedding is batched
        # in process_files). Call embed_chunks() explicitly to get vectors.
        result = processor.process_file(file_path, repo_root=file_path.parent)

        if not result.chunks:
            return {"success": False, "error": "CodeProcessor produced no chunks"}

        if embedder and result.chunks:
            texts = [c.text for c in result.chunks]
            # EmbedderClient exposes embed_texts; CodeProcessor.embed_chunks uses
            # embed_batch which is not on the client — call embed_texts directly.
            if hasattr(embedder, "embed_texts"):
                vecs = embedder.embed_texts(texts, task="code")
                result.embedding_vectors = [v.tolist() if hasattr(v, "tolist") else v for v in vecs]
            else:
                result.embedding_vectors = processor.embed_chunks(result.chunks)

        # Store in arxiv profile collections (same as document ingest)
        arango_config = get_arango_config(config, read_only=False)
        client_config = ArangoHttp2Config(
            database=arango_config["database"],
            socket_path=arango_config.get("socket_path"),
            base_url=f"http://{arango_config['host']}:{arango_config['port']}",
            username=arango_config["username"],
            password=arango_config["password"],
        )
        client = ArangoHttp2Client(client_config)

        try:
            col = get_profile("arxiv")
            # Ensure the required collections exist (new databases start empty)
            for collection_name in (col.metadata, col.chunks, col.embeddings):
                try:
                    client.request("POST", "/_api/collection", json={"name": collection_name})
                except Exception:
                    pass  # Already exists or creation failed — insert will surface real errors

            doc_id = document_id or file_path.stem
            sanitized_id = normalize_document_key(doc_id)
            now_iso = datetime.now(UTC).isoformat()

            meta_doc: dict[str, Any] = {
                "_key": sanitized_id,
                "document_id": doc_id,
                "title": doc_id,
                "source": "code",
                "language": file_path.suffix.lstrip("."),
                "file_path": str(file_path),
                "num_chunks": len(result.chunks),
                "processing_timestamp": now_iso,
                "status": "PROCESSED",
            }
            if extra_metadata:
                meta_doc.update(extra_metadata)
                meta_doc["_key"] = sanitized_id
                meta_doc["document_id"] = doc_id

            chunk_docs: list[dict[str, Any]] = []
            embedding_docs: list[dict[str, Any]] = []

            for i, chunk in enumerate(result.chunks):
                ck = chunk_key(sanitized_id, i)
                chunk_docs.append(
                    {
                        "_key": ck,
                        "document_id": doc_id,
                        "paper_key": sanitized_id,
                        "chunk_index": i,
                        "total_chunks": len(result.chunks),
                        "text": chunk.text,
                        "chunk_type": chunk.chunk_type,
                        # start_line/end_line stored as start_char/end_char for schema compat
                        "start_char": chunk.start_line,
                        "end_char": chunk.end_line,
                        "created_at": now_iso,
                    }
                )

            # Build embedding docs from parallel embedding_vectors list
            for i, (_chunk, vec) in enumerate(
                zip(result.chunks, result.embedding_vectors, strict=False)
            ):
                ck = chunk_key(sanitized_id, i)
                embedding_docs.append(
                    {
                        "_key": embedding_key(ck),
                        "chunk_key": ck,
                        "document_id": doc_id,
                        "paper_key": sanitized_id,
                        "embedding": vec,
                        "embedding_dim": len(vec),
                        "created_at": now_iso,
                    }
                )

            progress(f"Storing {len(chunk_docs)} code chunks in database...")

            # IMPORTANT: `insert_documents(..., overwrite=True)` uses ArangoDB's
            # /_api/import?overwrite=true which TRUNCATES the entire collection.
            # For force re-ingestion of a single document, we must delete only
            # this document's existing data via AQL, then insert with overwrite=False.
            if force:
                client.query(
                    f"FOR c IN {col.chunks} FILTER c.paper_key == @key REMOVE c IN {col.chunks}",
                    {"key": sanitized_id},
                )
                client.query(
                    f"FOR e IN {col.embeddings} FILTER e.paper_key == @key REMOVE e IN {col.embeddings}",
                    {"key": sanitized_id},
                )
                client.query(
                    f"FOR m IN {col.metadata} FILTER m._key == @key REMOVE m IN {col.metadata}",
                    {"key": sanitized_id},
                )

            if chunk_docs:
                client.insert_documents(col.chunks, chunk_docs, overwrite=False)
            if embedding_docs:
                client.insert_documents(col.embeddings, embedding_docs, overwrite=False)
            client.insert_documents(col.metadata, [meta_doc], overwrite=False)

            return {"success": True, "num_chunks": len(chunk_docs)}

        finally:
            client.close()

    finally:
        if embedder is not None and hasattr(embedder, "close"):
            embedder.close()
