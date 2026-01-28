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
from pathlib import Path
from typing import Any

from core.cli.config import get_config
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    progress,
    success_response,
)

# Pattern for arxiv IDs: YYMM.NNNNN or category/NNNNNNN
_ARXIV_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$|^[a-z-]+/\d{7}(v\d+)?$", re.IGNORECASE)


def _is_arxiv_id(s: str) -> bool:
    """Check if a string looks like an arxiv ID."""
    return bool(_ARXIV_PATTERN.match(s))


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
        elif _is_arxiv_id(inp.split("/")[-1]):
            # Handle case where user passes full path that looks like arxiv
            arxiv_ids.append(inp)
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
    start_time: float = 0.0,
) -> CLIResponse:
    """Ingest documents into the knowledge base.

    Auto-detects arxiv IDs vs file paths and routes to appropriate handler.

    Args:
        inputs: List of arxiv IDs or file paths to ingest.
        document_id: Custom document ID (only for single file ingest).
        force: Force reprocessing even if already exists.
        batch: Enable batch mode with progress reporting and error isolation.
        resume: Resume from previous batch state (implies batch=True).
        start_time: Command start timestamp.

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
        return _ingest_batch(all_items, None, force, resume, start_time)

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
        from core.cli.commands.arxiv import _ingest_arxiv_paper

        for arxiv_id in arxiv_ids:
            result = _ingest_arxiv_paper(arxiv_id, config, force)
            results.append(result)

    # Process file paths
    if file_paths:
        for file_path in file_paths:
            result = _ingest_file(file_path, config, document_id, force)
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
) -> CLIResponse:
    """Ingest items using batch processor with progress and resume.

    Args:
        items: List of arxiv IDs or file paths.
        document_id: Not used in batch mode (passed for signature compatibility).
        force: Force reprocessing even if already exists.
        resume: Resume from previous batch state.
        start_time: Command start timestamp.

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
            from core.cli.commands.arxiv import _ingest_arxiv_paper

            return _ingest_arxiv_paper(item_id, config, force)
        else:
            return _ingest_file(item_id, config, None, force)

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
) -> dict[str, Any]:
    """Ingest a local file (any supported format).

    Args:
        file_path: Path to the file.
        config: CLI configuration.
        document_id: Custom document ID (optional).
        force: Overwrite existing data.

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
    progress(f"Ingesting {path.name} as {doc_id}...")

    try:
        # Use _process_and_store from arxiv.py for now
        # This will be refactored to use the new tools in future PRs
        from core.cli.commands.arxiv import _process_and_store

        result = _process_and_store(
            arxiv_id=None,
            pdf_path=path,
            latex_path=None,
            metadata=None,
            config=config,
            document_id=doc_id,
            force=force,
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
