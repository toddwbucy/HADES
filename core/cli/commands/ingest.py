"""Paper ingestion commands for HADES CLI."""

from __future__ import annotations

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


def ingest_papers(
    arxiv_ids: list[str] | None = None,
    pdf_paths: list[str] | None = None,
    force: bool = False,
    start_time: float = 0.0,
) -> CLIResponse:
    """Ingest papers into the knowledge base.

    Args:
        arxiv_ids: List of arxiv paper IDs to download and ingest
        pdf_paths: List of local PDF file paths to ingest
        force: Force reprocessing even if paper already exists
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with ingestion results
    """
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

    if arxiv_ids:
        for arxiv_id in arxiv_ids:
            result = _ingest_arxiv_paper(arxiv_id, config, force)
            results.append(result)

    if pdf_paths:
        for pdf_path in pdf_paths:
            result = _ingest_local_pdf(pdf_path, config, force)
            results.append(result)

    # Summarize results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    if failed and not successful:
        return error_response(
            command="ingest",
            code=ErrorCode.PROCESSING_FAILED,
            message=f"All {len(failed)} papers failed to ingest",
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


def _ingest_arxiv_paper(arxiv_id: str, config: Any, force: bool) -> dict[str, Any]:
    """Ingest a single arxiv paper.

    1. Check if already exists (unless force=True)
    2. Download PDF from arxiv
    3. Process with DocumentProcessor
    4. Store in ArangoDB
    """
    from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

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


def _ingest_local_pdf(pdf_path: str, config: Any, force: bool) -> dict[str, Any]:
    """Ingest a local PDF file."""
    progress(f"Ingesting local PDF: {pdf_path}")

    path = Path(pdf_path)
    if not path.exists():
        return {
            "path": pdf_path,
            "success": False,
            "error": f"File not found: {pdf_path}",
        }

    if not path.suffix.lower() == ".pdf":
        return {
            "path": pdf_path,
            "success": False,
            "error": f"Not a PDF file: {pdf_path}",
        }

    # Use filename as document ID
    doc_id = path.stem

    try:
        processing_result = _process_and_store(
            arxiv_id=None,
            pdf_path=path,
            latex_path=None,
            metadata=None,
            config=config,
            document_id=doc_id,
        )

        if not processing_result["success"]:
            return {
                "path": pdf_path,
                "success": False,
                "error": processing_result.get("error", "Processing failed"),
            }

        progress(f"Stored {processing_result['num_chunks']} chunks for {doc_id}")

        return {
            "path": pdf_path,
            "document_id": doc_id,
            "success": True,
            "num_chunks": processing_result["num_chunks"],
        }

    except Exception as e:
        return {
            "path": pdf_path,
            "success": False,
            "error": str(e),
        }


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
            # Check arxiv_metadata collection
            sanitized_id = arxiv_id.replace(".", "_").replace("/", "_")
            client.get_document("arxiv_metadata", sanitized_id)
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
) -> dict[str, Any]:
    """Process a PDF and store results in the database."""
    from datetime import UTC, datetime

    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config
    from core.processors.document_processor import DocumentProcessor, ProcessingConfig

    # Configure processor
    # Use traditional chunking for now - late chunking has a dimension bug
    proc_config = ProcessingConfig(
        use_gpu=config.use_gpu,
        device=config.device,
        chunking_strategy="traditional",
        chunk_size_tokens=500,
        chunk_overlap_tokens=100,
    )

    processor = DocumentProcessor(proc_config)

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
            sanitized_id = doc_id.replace(".", "_").replace("/", "_")
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
                meta_doc.update({
                    "arxiv_id": arxiv_id,
                    "authors": metadata.authors,
                    "abstract": metadata.abstract,
                    "categories": metadata.categories,
                    "published": metadata.published.isoformat() if metadata.published else None,
                })

            # Prepare chunk documents
            chunk_docs = []
            embedding_docs = []

            for chunk in result.chunks:
                chunk_key = f"{sanitized_id}_chunk_{chunk.chunk_index}"

                chunk_docs.append({
                    "_key": chunk_key,
                    "document_id": doc_id,
                    "paper_key": sanitized_id,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "created_at": now_iso,
                })

                embedding_docs.append({
                    "_key": f"{chunk_key}_emb",
                    "chunk_key": chunk_key,
                    "document_id": doc_id,
                    "paper_key": sanitized_id,
                    "embedding": chunk.embedding.tolist(),
                    "embedding_dim": int(chunk.embedding.shape[0]),
                    "created_at": now_iso,
                })

            # Insert documents
            # Note: Using simple insert; for production, use transactions
            progress(f"Storing {len(chunk_docs)} chunks in database...")

            client.insert_documents("arxiv_metadata", [meta_doc])
            if chunk_docs:
                client.insert_documents("arxiv_abstract_chunks", chunk_docs)
            if embedding_docs:
                client.insert_documents("arxiv_abstract_embeddings", embedding_docs)

            return {
                "success": True,
                "num_chunks": len(result.chunks),
            }

        finally:
            client.close()

    finally:
        processor.cleanup()
