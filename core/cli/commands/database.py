"""Database management commands for HADES CLI.

Consolidates: list, stats, check, query, chunks, create, delete, ingest commands.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from core.cli.config import get_arango_config, get_config
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    progress,
    success_response,
)
from core.database.collections import get_default_profile_name, get_profile
from core.database.keys import normalize_document_key

# =============================================================================
# Paper Management Commands (list, stats, check)
# =============================================================================


def list_stored_papers(
    limit: int,
    category: str | None,
    start_time: float,
    collection: str | None = None,
) -> CLIResponse:
    """List documents stored in the database.

    Args:
        limit: Maximum number of documents to return
        category: Optional category filter
        start_time: Start time for duration calculation
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        CLIResponse with document list
    """
    profile_name = collection or get_default_profile_name()

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.list",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

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
            col = get_profile(profile_name)
            # Build query with optional category filter
            if category:
                aql = f"""
                    FOR doc IN {col.metadata}
                        FILTER @category IN doc.categories
                        SORT doc.processing_timestamp DESC
                        LIMIT @limit
                        RETURN {{
                            arxiv_id: doc.arxiv_id,
                            document_id: doc.document_id,
                            title: doc.title,
                            authors: doc.authors,
                            categories: doc.categories,
                            num_chunks: doc.num_chunks,
                            source: doc.source,
                            processing_timestamp: doc.processing_timestamp
                        }}
                """
                bind_vars = {"category": category, "limit": limit}
            else:
                aql = f"""
                    FOR doc IN {col.metadata}
                        SORT doc.processing_timestamp DESC
                        LIMIT @limit
                        RETURN {{
                            arxiv_id: doc.arxiv_id,
                            document_id: doc.document_id,
                            title: doc.title,
                            authors: doc.authors,
                            categories: doc.categories,
                            num_chunks: doc.num_chunks,
                            source: doc.source,
                            processing_timestamp: doc.processing_timestamp
                        }}
                """
                bind_vars = {"limit": limit}

            results = client.query(aql, bind_vars=bind_vars)

            return success_response(
                command="database.list",
                data={"papers": results},
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.list",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def get_stats(start_time: float, collection: str | None = None) -> CLIResponse:
    """Get database statistics.

    Args:
        start_time: Start time for duration calculation
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        CLIResponse with database statistics
    """
    profile_name = collection or get_default_profile_name()

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.stats",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

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
            col = get_profile(profile_name)
            stats = {}

            # Count papers
            paper_count = client.query(f"RETURN LENGTH({col.metadata})")
            stats["total_papers"] = paper_count[0] if paper_count else 0

            # Count chunks
            chunk_count = client.query(f"RETURN LENGTH({col.chunks})")
            stats["total_chunks"] = chunk_count[0] if chunk_count else 0

            # Count embeddings
            embedding_count = client.query(f"RETURN LENGTH({col.embeddings})")
            stats["total_embeddings"] = embedding_count[0] if embedding_count else 0

            # Get category distribution
            category_aql = f"""
                FOR doc IN {col.metadata}
                    FOR cat IN (doc.categories || [])
                        COLLECT category = cat WITH COUNT INTO count
                        SORT count DESC
                        LIMIT 10
                        RETURN {{category: category, count: count}}
            """
            categories = client.query(category_aql)
            stats["top_categories"] = categories

            # Get source distribution
            source_aql = f"""
                FOR doc IN {col.metadata}
                    COLLECT source = (doc.source || "arxiv") WITH COUNT INTO count
                    RETURN {{source: source, count: count}}
            """
            sources = client.query(source_aql)
            stats["sources"] = sources

            # Get recent papers
            recent_aql = f"""
                FOR doc IN {col.metadata}
                    SORT doc.processing_timestamp DESC
                    LIMIT 5
                    RETURN {{
                        arxiv_id: doc.arxiv_id,
                        title: doc.title,
                        processing_timestamp: doc.processing_timestamp
                    }}
            """
            recent = client.query(recent_aql)
            stats["recent_papers"] = recent

            return success_response(
                command="database.stats",
                data=stats,
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.stats",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def check_paper_exists(
    document_id: str, start_time: float, collection: str | None = None
) -> CLIResponse:
    """Check if a document exists in the database.

    Args:
        document_id: Document ID to check
        start_time: Start time for duration calculation
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        CLIResponse with existence check result
    """
    profile_name = collection or get_default_profile_name()

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.check",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

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
            col = get_profile(profile_name)
            sanitized_id = normalize_document_key(document_id)

            # Try to get the document
            from core.database.arango.optimized_client import ArangoHttpError

            try:
                doc = client.get_document(col.metadata, sanitized_id)
                exists = True
                doc_info = {
                    "document_id": doc.get("document_id") or doc.get("arxiv_id"),
                    "title": doc.get("title"),
                    "num_chunks": doc.get("num_chunks"),
                    "processing_timestamp": doc.get("processing_timestamp"),
                    "status": doc.get("status"),
                }
            except ArangoHttpError as e:
                if e.status_code == 404:
                    exists = False
                    doc_info = None
                else:
                    raise

            return success_response(
                command="database.check",
                data={
                    "document_id": document_id,
                    "collection": profile_name,
                    "exists": exists,
                    "document": doc_info,
                },
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.check",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def get_recent_papers(
    limit: int, start_time: float, collection: str | None = None
) -> CLIResponse:
    """Get recently ingested papers sorted by ingestion time.

    Args:
        limit: Maximum number of papers to return
        start_time: Start time for duration calculation
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        CLIResponse with recent papers list
    """
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

    profile_name = collection or get_default_profile_name()

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.recent",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
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
            col = get_profile(profile_name)
            results = client.query(
                f"""
                FOR doc IN {col.metadata}
                    SORT doc.processing_timestamp DESC
                    LIMIT @limit
                    RETURN {{
                        document_id: doc.document_id OR doc._key,
                        arxiv_id: doc.arxiv_id,
                        title: doc.title,
                        authors: doc.authors,
                        num_chunks: doc.num_chunks,
                        source: doc.source,
                        ingested: doc.processing_timestamp
                    }}
                """,
                {"limit": limit},
            )

            return success_response(
                command="database.recent",
                data={"papers": results, "collection": profile_name},
                start_time=start_time,
                count=len(results),
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.recent",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def check_health(start_time: float, collection: str | None = None) -> CLIResponse:
    """Check database health and data integrity.

    Detects:
    - Orphaned chunks (chunks without metadata)
    - Orphaned embeddings (embeddings without chunks)
    - Missing embeddings (chunks without embeddings)
    - Papers with mismatched chunk counts

    Args:
        start_time: Start time for duration calculation
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        CLIResponse with health report
    """
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

    profile_name = collection or get_default_profile_name()

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.health",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
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
            col = get_profile(profile_name)
            issues = []

            # Count totals
            metadata_count = client.query(f"RETURN LENGTH({col.metadata})")[0]
            chunks_count = client.query(f"RETURN LENGTH({col.chunks})")[0]
            embeddings_count = client.query(f"RETURN LENGTH({col.embeddings})")[0]

            # Check for orphaned chunks (chunks without metadata)
            orphaned_chunks = client.query(
                f"""
                FOR c IN {col.chunks}
                    LET meta = DOCUMENT({col.metadata}, c.paper_key)
                    FILTER meta == null
                    COLLECT WITH COUNT INTO cnt
                    RETURN cnt
                """
            )
            orphaned_chunks_count = orphaned_chunks[0] if orphaned_chunks else 0
            if orphaned_chunks_count > 0:
                issues.append({
                    "type": "orphaned_chunks",
                    "count": orphaned_chunks_count,
                    "description": "Chunks without corresponding metadata document",
                })

            # Check for orphaned embeddings (embeddings without chunks)
            orphaned_embeddings = client.query(
                f"""
                FOR e IN {col.embeddings}
                    LET chunk = DOCUMENT({col.chunks}, e.chunk_key)
                    FILTER chunk == null
                    COLLECT WITH COUNT INTO cnt
                    RETURN cnt
                """
            )
            orphaned_embeddings_count = orphaned_embeddings[0] if orphaned_embeddings else 0
            if orphaned_embeddings_count > 0:
                issues.append({
                    "type": "orphaned_embeddings",
                    "count": orphaned_embeddings_count,
                    "description": "Embeddings without corresponding chunk document",
                })

            # Check for missing embeddings by comparing counts per paper
            # This is more efficient than checking each chunk individually
            if chunks_count != embeddings_count:
                diff = abs(chunks_count - embeddings_count)
                if chunks_count > embeddings_count:
                    issues.append({
                        "type": "missing_embeddings",
                        "count": diff,
                        "description": f"Fewer embeddings ({embeddings_count}) than chunks ({chunks_count})",
                    })
                else:
                    issues.append({
                        "type": "extra_embeddings",
                        "count": diff,
                        "description": f"More embeddings ({embeddings_count}) than chunks ({chunks_count})",
                    })

            # Check for chunk count mismatches
            mismatched_papers = client.query(
                f"""
                FOR m IN {col.metadata}
                    LET actual_chunks = LENGTH(
                        FOR c IN {col.chunks}
                            FILTER c.paper_key == m._key
                            RETURN 1
                    )
                    FILTER m.num_chunks != null AND m.num_chunks != actual_chunks
                    RETURN {{
                        document_id: m.document_id OR m._key,
                        expected: m.num_chunks,
                        actual: actual_chunks
                    }}
                """
            )
            if mismatched_papers:
                issues.append({
                    "type": "chunk_count_mismatch",
                    "count": len(mismatched_papers),
                    "description": "Papers where stored chunk count differs from metadata",
                    "papers": mismatched_papers[:10],  # Limit to first 10
                })

            health_status = "healthy" if not issues else "issues_found"

            return success_response(
                command="database.health",
                data={
                    "status": health_status,
                    "collection": profile_name,
                    "totals": {
                        "metadata": metadata_count,
                        "chunks": chunks_count,
                        "embeddings": embeddings_count,
                    },
                    "issues": issues,
                },
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.health",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def purge_paper(
    document_id: str, start_time: float, collection: str | None = None
) -> CLIResponse:
    """Remove all data for a document from all collections.

    Deletes metadata, chunks, and embeddings for the given document ID.

    Args:
        document_id: Document ID to purge
        start_time: Start time for duration calculation
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        CLIResponse with deletion counts
    """
    profile_name = collection or get_default_profile_name()

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.purge",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

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
            col = get_profile(profile_name)
            doc_key = normalize_document_key(document_id)

            aql = f"""
                LET meta = (FOR d IN {col.metadata} FILTER d._key == @key REMOVE d IN {col.metadata} RETURN 1)
                LET chunks = (FOR d IN {col.chunks} FILTER d.paper_key == @key REMOVE d IN {col.chunks} RETURN 1)
                LET embs = (FOR d IN {col.embeddings} FILTER d.paper_key == @key REMOVE d IN {col.embeddings} RETURN 1)
                RETURN {{metadata: LENGTH(meta), chunks: LENGTH(chunks), embeddings: LENGTH(embs)}}
            """

            results = client.query(aql, bind_vars={"key": doc_key})
            counts = results[0] if results else {"metadata": 0, "chunks": 0, "embeddings": 0}

            total = counts["metadata"] + counts["chunks"] + counts["embeddings"]
            if total == 0:
                return error_response(
                    command="database.purge",
                    code=ErrorCode.PAPER_NOT_FOUND,
                    message=f"No data found for document: {document_id}",
                    start_time=start_time,
                )

            return success_response(
                command="database.purge",
                data={
                    "document_id": document_id,
                    "collection": profile_name,
                    "document_key": doc_key,
                    "deleted": counts,
                    "total_deleted": total,
                },
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.purge",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def ingest_file(
    file_path: str,
    document_id: str | None,
    force: bool,
    start_time: float,
) -> CLIResponse:
    """Ingest a local file into the knowledge base.

    Supports any format handled by the Docling extractor (PDF, DOCX, PPTX,
    HTML, Markdown, plain text, images).

    Args:
        file_path: Path to the file to ingest
        document_id: Optional document ID (defaults to filename stem)
        force: Overwrite existing data if present
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with ingestion result
    """
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return error_response(
            command="database.ingest",
            code=ErrorCode.PAPER_NOT_FOUND,
            message=f"File not found: {path}",
            start_time=start_time,
        )

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.ingest",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    doc_id = document_id or path.stem

    try:
        from core.cli.commands.ingest import _process_and_store

        result = _process_and_store(
            arxiv_id=None,
            pdf_path=path,
            latex_path=None,
            metadata=None,
            config=config,
            document_id=doc_id,
            force=force,
        )

        if not result.get("success"):
            return error_response(
                command="database.ingest",
                code=ErrorCode.PROCESSING_FAILED,
                message=result.get("error", "Processing failed"),
                start_time=start_time,
            )

        return success_response(
            command="database.ingest",
            data={
                "file": str(path),
                "document_id": doc_id,
                "num_chunks": result.get("num_chunks", 0),
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="database.ingest",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )


# =============================================================================
# Query Commands (query, chunks)
# =============================================================================


def semantic_query(
    search_text: str,
    limit: int,
    start_time: float,
    paper_filter: str | None = None,
    context: int = 0,
    cite_only: bool = False,
    hybrid: bool = False,
    decompose: bool = False,
    rerank: bool = False,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    collection: str | None = None,
) -> CLIResponse:
    """Perform semantic search over stored chunks.

    Args:
        search_text: Query text to search for
        limit: Maximum number of results
        start_time: Start time for duration calculation
        paper_filter: Optional document ID to limit search to specific document
        context: Number of adjacent chunks to include (0 = none)
        cite_only: If True, return minimal citation format
        hybrid: If True, combine semantic similarity with keyword matching
        decompose: If True, split compound queries and merge results
        rerank: If True, re-rank top results with cross-encoder for better precision
        rerank_model: Cross-encoder model to use for re-ranking
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        CLIResponse with search results
    """
    profile_name = collection or get_default_profile_name()
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.query",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        # Decompose query if requested
        if decompose:
            sub_queries = _decompose_query(search_text)
            if len(sub_queries) > 1:
                progress(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
                results = _search_with_decomposition(
                    sub_queries, limit, config, paper_filter, hybrid, search_text,
                    collection=profile_name
                )
            else:
                # Single query, proceed normally
                decompose = False

        if not decompose:
            progress("Generating query embedding...")
            # Generate query embedding
            query_embedding = _get_query_embedding(search_text, config)

            if paper_filter:
                progress(f"Searching document {paper_filter} for {limit} most similar chunks...")
            else:
                progress(f"Searching {limit} most similar chunks...")

            # Search database - fetch extra results for reranking stages
            # Fetch more if we'll be doing any reranking
            if rerank:
                fetch_limit = max(limit * 5, 50)  # Get more candidates for cross-encoder
            elif hybrid:
                fetch_limit = limit * 3
            else:
                fetch_limit = limit
            results = _search_embeddings(
                query_embedding, fetch_limit, config, paper_filter=paper_filter,
                collection=profile_name
            )

            # Apply hybrid reranking if requested (fast, keyword-based)
            if hybrid and results:
                progress("Re-ranking with keyword matching...")
                results = _hybrid_rerank_results(results, search_text)
                if not rerank:
                    results = results[:limit]  # Trim unless cross-encoder will further refine

        # Apply cross-encoder reranking if requested (slower, more accurate)
        if rerank and results:
            progress(f"Re-ranking with cross-encoder ({rerank_model})...")
            results = _crossencoder_rerank(results, search_text, limit, rerank_model)

        # Add context chunks if requested
        if context > 0 and results:
            results = _add_context_chunks(results, context, config, collection=profile_name)

        # Format for citation if requested
        if cite_only:
            results = _format_citations(results)

        response_data = {
            "query": search_text,
            "results": results,
        }
        if paper_filter:
            response_data["paper_filter"] = paper_filter

        return success_response(
            command="database.query",
            data=response_data,
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="database.query",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )


def get_paper_chunks(
    paper_id: str,
    limit: int,
    start_time: float,
    collection: str | None = None,
) -> CLIResponse:
    """Get all chunks for a specific document.

    Args:
        paper_id: Document ID
        limit: Maximum number of chunks to return
        start_time: Start time for duration calculation
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        CLIResponse with document chunks
    """
    profile_name = collection or get_default_profile_name()

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.chunks",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    progress(f"Retrieving chunks for document: {paper_id}")

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
            col = get_profile(profile_name)
            sanitized_id = normalize_document_key(paper_id)

            # Query chunks for this paper
            aql = f"""
                FOR chunk IN {col.chunks}
                    FILTER chunk.paper_key == @paper_key OR chunk.document_id == @paper_id
                    SORT chunk.chunk_index
                    LIMIT @limit
                    RETURN {{
                        chunk_index: chunk.chunk_index,
                        total_chunks: chunk.total_chunks,
                        text: chunk.text,
                        start_char: chunk.start_char,
                        end_char: chunk.end_char
                    }}
            """

            results = client.query(
                aql,
                bind_vars={
                    "paper_key": sanitized_id,
                    "paper_id": paper_id,
                    "limit": limit,
                },
            )

            if not results:
                return error_response(
                    command="database.chunks",
                    code=ErrorCode.PAPER_NOT_FOUND,
                    message=f"No chunks found for paper: {paper_id}",
                    start_time=start_time,
                )

            return success_response(
                command="database.chunks",
                data={
                    "paper_id": paper_id,
                    "chunks": results,
                },
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.chunks",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


# =============================================================================
# Collection Management Commands (create, delete)
# =============================================================================


def create_collection(name: str, start_time: float) -> CLIResponse:
    """Create a new ArangoDB collection.

    Args:
        name: Collection name
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with creation result
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.create",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

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
            client.request(
                "POST",
                f"/_db/{arango_config['database']}/_api/collection",
                json={"name": name},
            )

            return success_response(
                command="database.create",
                data={"collection": name, "created": True},
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.create",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def delete_document(collection: str, key: str, start_time: float) -> CLIResponse:
    """Delete a document from an ArangoDB collection.

    Args:
        collection: Collection name
        key: Document key
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with deletion result
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.delete",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

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
            client.request(
                "DELETE",
                f"/_db/{arango_config['database']}/_api/document/{collection}/{key}",
            )

            return success_response(
                command="database.delete",
                data={"collection": collection, "key": key, "deleted": True},
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.delete",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


# =============================================================================
# CRUD Commands (collections, count, insert, get, update)
# =============================================================================


def _make_client(read_only: bool = True) -> tuple[Any, Any, str]:
    """Create an ArangoDB client with config.

    Returns:
        Tuple of (client, config, database_name)
    """
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

    config = get_config()
    arango_config = get_arango_config(config, read_only=read_only)

    client_config = ArangoHttp2Config(
        database=arango_config["database"],
        socket_path=arango_config.get("socket_path"),
        base_url=f"http://{arango_config['host']}:{arango_config['port']}",
        username=arango_config["username"],
        password=arango_config["password"],
    )

    client = ArangoHttp2Client(client_config)
    return client, arango_config, arango_config["database"]


# ArangoDB collection names: alphanumeric, hyphens, underscores (1-256 chars)
_VALID_COLLECTION_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,256}$")


def _validate_collection_name(name: str) -> str | None:
    """Validate a user-provided collection name. Returns error message or None."""
    if not _VALID_COLLECTION_NAME.match(name):
        return f"Invalid collection name '{name}': must be alphanumeric, hyphens, or underscores (1-256 chars)"
    return None


def list_collections(
    start_time: float,
    prefix: str | None = None,
    exclude_system: bool = True,
) -> CLIResponse:
    """List all collections in the database.

    Args:
        start_time: Start time for duration calculation
        prefix: Optional prefix filter (e.g., 'arxiv_' to show only arxiv collections)
        exclude_system: If True, exclude system collections (starting with '_')

    Returns:
        CLIResponse with collection list
    """
    try:
        client, _, db_name = _make_client(read_only=True)
    except ValueError as e:
        return error_response(
            command="database.collections",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        results = client.query(
            """
            FOR c IN COLLECTIONS()
                FILTER !@exclude_system OR !STARTS_WITH(c.name, '_')
                FILTER !@has_prefix OR STARTS_WITH(c.name, @prefix)
                SORT c.name
                RETURN c.name
            """,
            {
                "exclude_system": exclude_system,
                "has_prefix": prefix is not None,
                "prefix": prefix or "",
            },
        )

        return success_response(
            command="database.collections",
            data={"collections": results, "database": db_name},
            start_time=start_time,
            count=len(results),
        )

    except Exception as e:
        return error_response(
            command="database.collections",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def count_collection(
    collection_name: str,
    start_time: float,
) -> CLIResponse:
    """Count documents in a collection.

    Args:
        collection_name: Name of the collection
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with document count
    """
    validation_err = _validate_collection_name(collection_name)
    if validation_err:
        return error_response(
            command="database.count",
            code=ErrorCode.VALIDATION_ERROR,
            message=validation_err,
            start_time=start_time,
        )

    try:
        client, _, _ = _make_client(read_only=True)
    except ValueError as e:
        return error_response(
            command="database.count",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        results = client.query(
            "RETURN LENGTH(@@col)",
            bind_vars={"@col": collection_name},
            batch_size=1,
        )
        count = results[0] if results else 0

        return success_response(
            command="database.count",
            data={"collection": collection_name, "count": count},
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="database.count",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def insert_documents(
    collection_name: str,
    data: str | None,
    file_path: str | None,
    start_time: float,
) -> CLIResponse:
    """Insert one or more documents into a collection.

    Accepts either inline JSON (single object or array) or a JSONL file.
    Creates the collection if it doesn't exist.

    Args:
        collection_name: Target collection name
        data: JSON string (object or array of objects)
        file_path: Path to JSONL file (one JSON object per line)
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with insertion result
    """
    import json

    validation_err = _validate_collection_name(collection_name)
    if validation_err:
        return error_response(
            command="database.insert",
            code=ErrorCode.VALIDATION_ERROR,
            message=validation_err,
            start_time=start_time,
        )

    if not data and not file_path:
        return error_response(
            command="database.insert",
            code=ErrorCode.VALIDATION_ERROR,
            message="Provide either --data or --file",
            start_time=start_time,
        )

    # Parse documents from input
    try:
        if file_path:
            docs = []
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line))
        else:
            parsed = json.loads(data)
            docs = parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, OSError) as e:
        return error_response(
            command="database.insert",
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid input: {e}",
            start_time=start_time,
        )

    if not docs:
        return error_response(
            command="database.insert",
            code=ErrorCode.VALIDATION_ERROR,
            message="No documents to insert",
            start_time=start_time,
        )

    try:
        client, arango_config, db_name = _make_client(read_only=False)
    except ValueError as e:
        return error_response(
            command="database.insert",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        # Auto-create collection if it doesn't exist
        try:
            client.request(
                "POST",
                f"/_db/{db_name}/_api/collection",
                json={"name": collection_name},
            )
        except Exception as e:
            # Ignore "duplicate name" errors (collection already exists)
            error_str = str(e).lower()
            if "duplicate" not in error_str and "1207" not in error_str:
                raise

        # Insert documents
        client.insert_documents(collection_name, docs, overwrite=False)

        # Return keys of inserted documents
        keys = [d.get("_key") for d in docs if d.get("_key")]

        return success_response(
            command="database.insert",
            data={
                "collection": collection_name,
                "inserted": len(docs),
                "keys": keys if keys else None,
            },
            start_time=start_time,
            count=len(docs),
        )

    except Exception as e:
        return error_response(
            command="database.insert",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def get_document(
    collection_name: str,
    key: str,
    start_time: float,
) -> CLIResponse:
    """Get a single document by collection and key.

    Args:
        collection_name: Collection name
        key: Document key (_key)
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with document data
    """
    validation_err = _validate_collection_name(collection_name)
    if validation_err:
        return error_response(
            command="database.get",
            code=ErrorCode.VALIDATION_ERROR,
            message=validation_err,
            start_time=start_time,
        )

    try:
        client, _, _ = _make_client(read_only=True)
    except ValueError as e:
        return error_response(
            command="database.get",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        doc = client.get_document(collection_name, key)

        return success_response(
            command="database.get",
            data={"collection": collection_name, "key": key, "document": doc},
            start_time=start_time,
        )

    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return error_response(
                command="database.get",
                code=ErrorCode.PAPER_NOT_FOUND,
                message=f"Document '{key}' not found in '{collection_name}'",
                start_time=start_time,
            )
        return error_response(
            command="database.get",
            code=ErrorCode.DATABASE_ERROR,
            message=error_msg,
            start_time=start_time,
        )
    finally:
        client.close()


def update_document(
    collection_name: str,
    key: str,
    data: str,
    start_time: float,
    replace: bool = False,
) -> CLIResponse:
    """Update a document by collection and key.

    By default, merges the provided fields into the existing document.
    Use replace=True to fully replace the document.

    Args:
        collection_name: Collection name
        key: Document key (_key)
        data: JSON string with fields to update
        start_time: Start time for duration calculation
        replace: If True, replace entire document instead of merging

    Returns:
        CLIResponse with update result
    """
    import json

    validation_err = _validate_collection_name(collection_name)
    if validation_err:
        return error_response(
            command="database.update",
            code=ErrorCode.VALIDATION_ERROR,
            message=validation_err,
            start_time=start_time,
        )

    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as e:
        return error_response(
            command="database.update",
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid JSON: {e}",
            start_time=start_time,
        )

    if not isinstance(parsed, dict):
        return error_response(
            command="database.update",
            code=ErrorCode.VALIDATION_ERROR,
            message="Data must be a JSON object",
            start_time=start_time,
        )

    try:
        client, arango_config, db_name = _make_client(read_only=False)
    except ValueError as e:
        return error_response(
            command="database.update",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        method = "PUT" if replace else "PATCH"
        client.request(
            method,
            f"/_db/{db_name}/_api/document/{collection_name}/{key}",
            json=parsed,
        )

        return success_response(
            command="database.update",
            data={
                "collection": collection_name,
                "key": key,
                "updated": True,
                "mode": "replace" if replace else "merge",
            },
            start_time=start_time,
        )

    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return error_response(
                command="database.update",
                code=ErrorCode.PAPER_NOT_FOUND,
                message=f"Document '{key}' not found in '{collection_name}'",
                start_time=start_time,
            )
        return error_response(
            command="database.update",
            code=ErrorCode.DATABASE_ERROR,
            message=error_msg,
            start_time=start_time,
        )
    finally:
        client.close()


def export_collection(
    collection_name: str,
    output_path: str | None,
    start_time: float,
    limit: int | None = None,
) -> CLIResponse:
    """Export collection documents as JSONL.

    Args:
        collection_name: Collection to export
        output_path: Output file path (None = stdout)
        start_time: Start time for duration calculation
        limit: Maximum documents to export (None = all)

    Returns:
        CLIResponse with export result
    """
    import json

    validation_err = _validate_collection_name(collection_name)
    if validation_err:
        return error_response(
            command="database.export",
            code=ErrorCode.VALIDATION_ERROR,
            message=validation_err,
            start_time=start_time,
        )

    try:
        client, _, _ = _make_client(read_only=True)
    except ValueError as e:
        return error_response(
            command="database.export",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        aql = "FOR doc IN @@col"
        bind_vars: dict[str, Any] = {"@col": collection_name}

        if limit:
            aql += " LIMIT @limit"
            bind_vars["limit"] = limit

        aql += " RETURN doc"

        docs = client.query(aql, bind_vars)

        if output_path:
            with open(output_path, "w") as f:
                for doc in docs:
                    f.write(json.dumps(doc, default=str) + "\n")
        else:
            # Write to stdout directly (not through CLIResponse)
            import sys

            for doc in docs:
                sys.stdout.write(json.dumps(doc, default=str) + "\n")

        return success_response(
            command="database.export",
            data={
                "collection": collection_name,
                "exported": len(docs),
                "output": output_path or "stdout",
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="database.export",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def get_all_stats(start_time: float) -> CLIResponse:
    """Get database-wide statistics across all collections.

    Returns:
        CLIResponse with per-collection document counts
    """
    try:
        client, _, db_name = _make_client(read_only=True)
    except ValueError as e:
        return error_response(
            command="database.stats",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        collections = client.query(
            """
            FOR c IN COLLECTIONS()
                FILTER !STARTS_WITH(c.name, '_')
                SORT c.name
                RETURN c.name
            """
        )

        stats = []
        total_docs = 0
        for col_name in collections:
            count_result = client.query(
                "RETURN LENGTH(@@col)",
                {"@col": col_name},
                batch_size=1,
            )
            count = count_result[0] if count_result else 0
            total_docs += count
            stats.append({"collection": col_name, "count": count})

        return success_response(
            command="database.stats",
            data={
                "database": db_name,
                "collections": stats,
                "total_documents": total_docs,
                "total_collections": len(stats),
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="database.stats",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


# =============================================================================
# Internal Helpers — Query
# =============================================================================


def _get_query_embedding(text: str, config: Any) -> np.ndarray:
    """Generate embedding for query text.

    Uses the embedding service for consistent similarity search.
    """
    from core.cli.config import get_embedder_client

    with get_embedder_client(config) as client:
        return client.embed_query(text)


def _search_embeddings(
    query_embedding: np.ndarray,
    limit: int,
    config: Any,
    paper_filter: str | None = None,
    collection: str | None = None,
) -> list[dict[str, Any]]:
    """Search for similar embeddings in the database.

    Uses cosine similarity for matching.

    Args:
        query_embedding: Query vector
        limit: Maximum results to return
        config: CLI configuration
        paper_filter: Optional document ID to limit search to specific document
        collection: Collection profile name (default: from env or 'arxiv')
    """
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

    profile_name = collection or get_default_profile_name()

    # Use read-write socket because cursor pagination requires PUT requests,
    # which the read-only socket doesn't support
    arango_config = get_arango_config(config, read_only=False)
    client_config = ArangoHttp2Config(
        database=arango_config["database"],
        socket_path=arango_config.get("socket_path"),
        base_url=f"http://{arango_config['host']}:{arango_config['port']}",
        username=arango_config["username"],
        password=arango_config["password"],
    )

    client = ArangoHttp2Client(client_config)

    # Normalize paper filter if provided
    paper_key_filter = None
    if paper_filter:
        paper_key_filter = normalize_document_key(paper_filter)

    try:
        col = get_profile(profile_name)
        all_embeddings = []

        # Search chunks in the specified collection
        try:
            if paper_key_filter:
                aql_chunks = f"""
                    FOR emb IN {col.embeddings}
                        FILTER emb.chunk_key != null
                        FILTER emb.paper_key == @paper_key
                        LET chunk = DOCUMENT(CONCAT("{col.chunks}/", emb.chunk_key))
                        LET meta = DOCUMENT(CONCAT("{col.metadata}/", emb.paper_key))
                        RETURN {{
                            paper_key: emb.paper_key,
                            embedding: emb.embedding,
                            text: chunk.text,
                            chunk_index: chunk.chunk_index,
                            total_chunks: chunk.total_chunks,
                            title: meta.title,
                            arxiv_id: meta.arxiv_id,
                            source: @collection
                        }}
                """
                # Use large batch to avoid cursor pagination issues with proxy
                chunk_results = client.query(
                    aql_chunks,
                    bind_vars={"paper_key": paper_key_filter, "collection": profile_name},
                    batch_size=50000
                )
            else:
                aql_chunks = f"""
                    FOR emb IN {col.embeddings}
                        FILTER emb.chunk_key != null
                        LET chunk = DOCUMENT(CONCAT("{col.chunks}/", emb.chunk_key))
                        LET meta = DOCUMENT(CONCAT("{col.metadata}/", emb.paper_key))
                        RETURN {{
                            paper_key: emb.paper_key,
                            embedding: emb.embedding,
                            text: chunk.text,
                            chunk_index: chunk.chunk_index,
                            total_chunks: chunk.total_chunks,
                            title: meta.title,
                            arxiv_id: meta.arxiv_id,
                            source: @collection
                        }}
                """
                # Use large batch to avoid cursor pagination issues with proxy
                chunk_results = client.query(
                    aql_chunks,
                    bind_vars={"collection": profile_name},
                    batch_size=50000
                )
            all_embeddings.extend(chunk_results or [])
        except Exception as e:
            # Collection may not exist
            progress(f"Note: Could not query {profile_name} chunks: {e}")

        if not all_embeddings:
            return []

        # Compute cosine similarities
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        scored_results = []
        for item in all_embeddings:
            emb = np.array(item["embedding"], dtype=np.float32)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            similarity = float(np.dot(query_norm, emb_norm))

            scored_results.append(
                {
                    "similarity": round(similarity, 4),
                    "arxiv_id": item.get("arxiv_id"),
                    "title": item.get("title"),
                    "source": item.get("source", "full_paper"),
                    "chunk_index": item.get("chunk_index"),
                    "total_chunks": item.get("total_chunks"),
                    "text": item.get("text"),
                }
            )

        # Sort by similarity and take top results
        scored_results.sort(key=lambda x: x["similarity"], reverse=True)

        return scored_results[:limit]

    finally:
        client.close()


def _hybrid_rerank_results(
    results: list[dict[str, Any]],
    query: str,
    semantic_weight: float = 0.7,
) -> list[dict[str, Any]]:
    """Re-rank results by combining semantic similarity with keyword matching.

    Uses a simple term frequency score combined with semantic similarity.
    This improves retrieval for queries with specific technical terms that
    should appear in the matching text.

    Args:
        results: List of search results with 'similarity' and 'text' fields
        query: Original query text for keyword matching
        semantic_weight: Weight for semantic score (1 - this = keyword weight)

    Returns:
        Re-ranked results with additional score fields
    """
    # Tokenize query into terms
    query_terms = set(query.lower().split())

    reranked = []
    for result in results:
        semantic_score = result.get("similarity", 0)

        # Compute keyword score from text and title
        text = (result.get("text") or "").lower()
        title = (result.get("title") or "").lower()
        combined_text = title + " " + text

        # Simple term frequency score: fraction of query terms found
        text_terms = set(combined_text.split())
        if query_terms:
            matches = sum(1 for term in query_terms if term in text_terms)
            keyword_score = matches / len(query_terms)
        else:
            keyword_score = 0

        # Combine scores
        combined_score = (semantic_weight * semantic_score) + ((1 - semantic_weight) * keyword_score)

        # Create result with additional scores
        reranked_result = {
            **result,
            "keyword_score": round(keyword_score, 4),
            "combined_score": round(combined_score, 4),
        }
        reranked.append(reranked_result)

    # Sort by combined score descending
    reranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return reranked


def _decompose_query(query: str) -> list[str]:
    """Decompose a compound query into sub-queries.

    Uses rule-based splitting on conjunctions and punctuation.
    Keeps multi-word technical terms together.

    Args:
        query: Original query text

    Returns:
        List of sub-queries (may be single-element if not compound)
    """
    import re

    # Normalize whitespace
    query = " ".join(query.split())

    # Split patterns: conjunctions and punctuation
    # Order matters - try more specific patterns first
    split_patterns = [
        r"\s+and\s+",
        r"\s+or\s+",
        r"\s+with\s+",
        r"\s+vs\.?\s+",
        r"\s+versus\s+",
        r",\s*",
        r";\s*",
    ]

    # Try each pattern
    sub_queries = [query]
    for pattern in split_patterns:
        new_subs = []
        for sq in sub_queries:
            parts = re.split(pattern, sq, flags=re.IGNORECASE)
            new_subs.extend(parts)
        sub_queries = new_subs

    # Clean up: strip whitespace, filter empty/short, deduplicate
    cleaned = []
    seen = set()
    for sq in sub_queries:
        sq = sq.strip()
        # Skip if too short (single word with < 4 chars) or empty
        if not sq or (len(sq.split()) == 1 and len(sq) < 4):
            continue
        # Deduplicate (case-insensitive)
        sq_lower = sq.lower()
        if sq_lower not in seen:
            seen.add(sq_lower)
            cleaned.append(sq)

    # If we ended up with nothing useful, return original
    if not cleaned:
        return [query]

    return cleaned


def _search_with_decomposition(
    sub_queries: list[str],
    limit: int,
    config: Any,
    paper_filter: str | None,
    hybrid: bool,
    original_query: str,
    collection: str | None = None,
) -> list[dict[str, Any]]:
    """Search with multiple sub-queries and merge results.

    Runs each sub-query, aggregates scores, and deduplicates.

    Args:
        sub_queries: List of decomposed queries
        limit: Final number of results to return
        config: CLI configuration
        paper_filter: Optional paper filter
        hybrid: Whether to use hybrid reranking
        original_query: Original query for hybrid reranking
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        Merged and deduplicated results
    """
    from collections import defaultdict

    profile_name = collection or get_default_profile_name()

    # Track results by unique key (arxiv_id + chunk_index)
    result_scores: dict[str, dict[str, Any]] = {}
    result_counts: defaultdict[str, int] = defaultdict(int)

    # Fetch more results per sub-query to get good coverage
    per_query_limit = max(limit, 10)

    for i, sq in enumerate(sub_queries):
        progress(f"Sub-query {i + 1}/{len(sub_queries)}: {sq}")

        # Generate embedding for sub-query
        query_embedding = _get_query_embedding(sq, config)

        # Search
        fetch_limit = per_query_limit * 2 if hybrid else per_query_limit
        results = _search_embeddings(
            query_embedding, fetch_limit, config, paper_filter=paper_filter, collection=profile_name
        )

        # Apply hybrid reranking per sub-query
        if hybrid and results:
            results = _hybrid_rerank_results(results, sq)
            results = results[:per_query_limit]

        # Aggregate results
        for result in results:
            # Create unique key
            arxiv_id = result.get("arxiv_id", "")
            chunk_idx = result.get("chunk_index", 0)
            key = f"{arxiv_id}:{chunk_idx}"

            score = result.get("combined_score") if hybrid else result.get("similarity", 0)

            if key in result_scores:
                # Update: keep max score, increment count
                existing_score = (
                    result_scores[key].get("combined_score")
                    if hybrid
                    else result_scores[key].get("similarity", 0)
                )
                if score > existing_score:
                    result_scores[key] = result
            else:
                result_scores[key] = result

            result_counts[key] += 1

    # Boost results that appear in multiple sub-queries
    progress("Merging results from sub-queries...")
    merged = []
    for key, result in result_scores.items():
        count = result_counts[key]
        base_score = result.get("combined_score") if hybrid else result.get("similarity", 0)

        # Boost: results appearing in multiple queries get higher scores
        # Formula: base_score * (1 + 0.1 * (count - 1))
        # This gives 10% boost per additional query match
        boost_factor = 1 + 0.1 * (count - 1)
        boosted_score = base_score * boost_factor

        merged_result = {
            **result,
            "sub_query_matches": count,
            "aggregated_score": round(boosted_score, 4),
        }
        merged.append(merged_result)

    # Sort by aggregated score
    merged.sort(key=lambda x: x["aggregated_score"], reverse=True)

    # Apply final hybrid reranking with original query for coherence
    if hybrid:
        merged = _hybrid_rerank_results(merged, original_query)

    return merged[:limit]


def _crossencoder_rerank(
    results: list[dict[str, Any]],
    query: str,
    limit: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict[str, Any]]:
    """Re-rank results using a cross-encoder for better precision.

    Cross-encoders score (query, document) pairs together, allowing
    them to model token interactions that bi-encoders miss.

    Args:
        results: Initial search results with 'text' field
        query: Original query text
        limit: Number of results to return after reranking
        model_name: HuggingFace cross-encoder model identifier

    Returns:
        Re-ranked results with cross-encoder scores
    """
    from sentence_transformers import CrossEncoder

    from core.cli.output import progress_bar

    if not results:
        return results

    # Load cross-encoder model with progress indicator (may take a few seconds on first load)
    with progress_bar("Loading cross-encoder", total=None) as (prog, task):
        model = CrossEncoder(model_name, max_length=512)
        prog.update(task, description="Cross-encoder ready")

    # Prepare (query, passage) pairs
    pairs = []
    for result in results:
        text = result.get("text", "")
        title = result.get("title", "")
        # Combine title and text for better context
        passage = f"{title}. {text}" if title else text
        pairs.append([query, passage])

    # Score all pairs in batch
    scores = model.predict(pairs, show_progress_bar=False)

    # Add scores to results
    reranked = []
    for result, score in zip(results, scores, strict=True):
        reranked_result = {
            **result,
            "cross_encoder_score": round(float(score), 4),
        }
        reranked.append(reranked_result)

    # Sort by cross-encoder score (descending)
    reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    return reranked[:limit]


def _add_context_chunks(
    results: list[dict[str, Any]],
    context: int,
    config: Any,
    collection: str | None = None,
) -> list[dict[str, Any]]:
    """Add neighboring chunks to each result for context.

    Args:
        results: Search results with chunk_index
        context: Number of chunks before/after to include
        collection: Collection profile name (default: from env or 'arxiv')

    Returns:
        Results with context_before and context_after lists
    """
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

    profile_name = collection or get_default_profile_name()

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
        col = get_profile(profile_name)
        for result in results:
            arxiv_id = result.get("arxiv_id")
            chunk_index = result.get("chunk_index")
            source = result.get("source")

            # Skip abstracts (they're single chunks)
            if source == "abstract" or chunk_index is None or arxiv_id is None:
                result["context_before"] = []
                result["context_after"] = []
                continue

            # Normalize arxiv_id to paper_key
            paper_key = normalize_document_key(arxiv_id)

            # Fetch context chunks
            context_before = []
            context_after = []

            try:
                # Get chunks before
                if chunk_index > 0:
                    aql_before = f"""
                        FOR chunk IN {col.chunks}
                            FILTER chunk.paper_key == @paper_key
                            FILTER chunk.chunk_index >= @start_idx
                            FILTER chunk.chunk_index < @current_idx
                            SORT chunk.chunk_index
                            RETURN {{
                                chunk_index: chunk.chunk_index,
                                text: chunk.text
                            }}
                    """
                    start_idx = max(0, chunk_index - context)
                    before_results = client.query(
                        aql_before,
                        bind_vars={
                            "paper_key": paper_key,
                            "start_idx": start_idx,
                            "current_idx": chunk_index,
                        },
                    )
                    context_before = before_results or []

                # Get chunks after
                aql_after = f"""
                    FOR chunk IN {col.chunks}
                        FILTER chunk.paper_key == @paper_key
                        FILTER chunk.chunk_index > @current_idx
                        FILTER chunk.chunk_index <= @end_idx
                        SORT chunk.chunk_index
                        RETURN {{
                            chunk_index: chunk.chunk_index,
                            text: chunk.text
                        }}
                """
                total_chunks = result.get("total_chunks", 999)
                end_idx = min(total_chunks - 1, chunk_index + context)
                after_results = client.query(
                    aql_after,
                    bind_vars={
                        "paper_key": paper_key,
                        "current_idx": chunk_index,
                        "end_idx": end_idx,
                    },
                )
                context_after = after_results or []

            except Exception:
                # If context fetch fails, log and continue without context
                progress(f"Note: Could not fetch context for chunk {chunk_index}: skipped")

            result["context_before"] = context_before
            result["context_after"] = context_after

        return results

    finally:
        client.close()


# =============================================================================
# AQL Execution
# =============================================================================


def execute_aql(
    aql: str,
    bind_vars_json: str | None,
    start_time: float,
) -> CLIResponse:
    """Execute an arbitrary AQL query.

    Args:
        aql: AQL query string
        bind_vars_json: Optional JSON string of bind variables
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with query results
    """
    # Parse bind vars
    bind_vars: dict[str, Any] | None = None
    if bind_vars_json:
        try:
            import json

            bind_vars = json.loads(bind_vars_json)
        except (json.JSONDecodeError, TypeError) as e:
            return error_response(
                command="database.aql",
                code=ErrorCode.CONFIG_ERROR,
                message=f"Invalid bind vars JSON: {e}",
                start_time=start_time,
            )

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.aql",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

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
            results = client.query(aql, bind_vars=bind_vars)

            return success_response(
                command="database.aql",
                data={"results": results, "count": len(results)},
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.aql",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )


# =============================================================================
# Graph Commands
# =============================================================================


def graph_create(
    name: str,
    edge_definitions_json: str,
    start_time: float,
) -> CLIResponse:
    """Create a named graph in ArangoDB.

    Args:
        name: Graph name
        edge_definitions_json: JSON string of edge definitions
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with creation result
    """
    try:
        import json

        edge_definitions = json.loads(edge_definitions_json)
    except (json.JSONDecodeError, TypeError) as e:
        return error_response(
            command="database.graph.create",
            code=ErrorCode.CONFIG_ERROR,
            message=f"Invalid edge definitions JSON: {e}",
            start_time=start_time,
        )

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.graph.create",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

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
            client.request(
                "POST",
                f"/_db/{arango_config['database']}/_api/gharial",
                json={"name": name, "edgeDefinitions": edge_definitions},
            )

            return success_response(
                command="database.graph.create",
                data={"graph": name, "created": True, "edge_definitions": edge_definitions},
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.graph.create",
            code=ErrorCode.GRAPH_ERROR,
            message=str(e),
            start_time=start_time,
        )


def graph_list(start_time: float) -> CLIResponse:
    """List all named graphs.

    Args:
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with graph list
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.graph.list",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

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
            result = client.request(
                "GET",
                f"/_db/{arango_config['database']}/_api/gharial",
            )

            graphs = []
            for g in result.get("graphs", []):
                graphs.append(
                    {
                        "name": g.get("_key") or g.get("name"),
                        "edge_definitions": g.get("edgeDefinitions", []),
                    }
                )

            return success_response(
                command="database.graph.list",
                data={"graphs": graphs},
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.graph.list",
            code=ErrorCode.GRAPH_ERROR,
            message=str(e),
            start_time=start_time,
        )


def graph_drop(
    name: str,
    drop_collections: bool,
    start_time: float,
) -> CLIResponse:
    """Drop a named graph.

    Args:
        name: Graph name
        drop_collections: Whether to also drop the collections used by the graph
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with drop result
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.graph.drop",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

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
            drop_param = "true" if drop_collections else "false"
            client.request(
                "DELETE",
                f"/_db/{arango_config['database']}/_api/gharial/{name}?dropCollections={drop_param}",
            )

            return success_response(
                command="database.graph.drop",
                data={"graph": name, "dropped": True, "collections_dropped": drop_collections},
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.graph.drop",
            code=ErrorCode.GRAPH_ERROR,
            message=str(e),
            start_time=start_time,
        )


def graph_traverse(
    start_vertex: str,
    graph: str,
    direction: str,
    min_depth: int,
    max_depth: int,
    limit: int,
    start_time: float,
) -> CLIResponse:
    """Traverse a named graph from a start vertex.

    Args:
        start_vertex: Starting document ID (e.g., "collection/key")
        graph: Graph name
        direction: Traversal direction (outbound, inbound, any)
        min_depth: Minimum traversal depth
        max_depth: Maximum traversal depth
        limit: Maximum number of results
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with traversal results
    """
    direction_upper = direction.upper()
    if direction_upper not in ("OUTBOUND", "INBOUND", "ANY"):
        return error_response(
            command="database.graph.traverse",
            code=ErrorCode.CONFIG_ERROR,
            message=f"Invalid direction: {direction}. Must be outbound, inbound, or any.",
            start_time=start_time,
        )

    if min_depth < 0:
        return error_response(
            command="database.graph.traverse",
            code=ErrorCode.CONFIG_ERROR,
            message="min_depth must be >= 0",
            start_time=start_time,
        )

    if max_depth < min_depth:
        return error_response(
            command="database.graph.traverse",
            code=ErrorCode.CONFIG_ERROR,
            message="max_depth must be >= min_depth",
            start_time=start_time,
        )

    if limit <= 0:
        return error_response(
            command="database.graph.traverse",
            code=ErrorCode.CONFIG_ERROR,
            message="limit must be > 0",
            start_time=start_time,
        )

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.graph.traverse",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

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
            aql = f"""
                FOR v, e, p IN @min_depth..@max_depth {direction_upper} @start GRAPH @graph
                    LIMIT @limit
                    RETURN {{vertex: v, edge: e}}
            """
            results = client.query(
                aql,
                bind_vars={
                    "min_depth": min_depth,
                    "max_depth": max_depth,
                    "start": start_vertex,
                    "graph": graph,
                    "limit": limit,
                },
            )

            return success_response(
                command="database.graph.traverse",
                data={"results": results, "start": start_vertex, "graph": graph, "direction": direction},
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.graph.traverse",
            code=ErrorCode.GRAPH_ERROR,
            message=str(e),
            start_time=start_time,
        )


def graph_shortest_path(
    from_vertex: str,
    to_vertex: str,
    graph: str,
    direction: str,
    start_time: float,
) -> CLIResponse:
    """Find the shortest path between two vertices in a named graph.

    Args:
        from_vertex: Source document ID
        to_vertex: Target document ID
        graph: Graph name
        direction: Edge direction (outbound, inbound, any)
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with shortest path
    """
    direction_upper = direction.upper()
    if direction_upper not in ("OUTBOUND", "INBOUND", "ANY"):
        return error_response(
            command="database.graph.shortest-path",
            code=ErrorCode.CONFIG_ERROR,
            message=f"Invalid direction: {direction}. Must be outbound, inbound, or any.",
            start_time=start_time,
        )

    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="database.graph.shortest-path",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

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
            aql = f"""
                FOR v, e IN {direction_upper} SHORTEST_PATH @from_v TO @to_v GRAPH @graph
                    RETURN {{vertex: v, edge: e}}
            """
            results = client.query(
                aql,
                bind_vars={
                    "from_v": from_vertex,
                    "to_v": to_vertex,
                    "graph": graph,
                },
            )

            return success_response(
                command="database.graph.shortest-path",
                data={
                    "results": results,
                    "from": from_vertex,
                    "to": to_vertex,
                    "graph": graph,
                    "direction": direction,
                },
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="database.graph.shortest-path",
            code=ErrorCode.GRAPH_ERROR,
            message=str(e),
            start_time=start_time,
        )


def graph_neighbors(
    start_vertex: str,
    graph: str,
    direction: str,
    limit: int,
    start_time: float,
) -> CLIResponse:
    """Get immediate neighbors of a vertex (depth 1 traversal).

    Args:
        start_vertex: Starting document ID
        graph: Graph name
        direction: Edge direction (outbound, inbound, any)
        limit: Maximum number of results
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with neighbor vertices
    """
    return graph_traverse(
        start_vertex=start_vertex,
        graph=graph,
        direction=direction,
        min_depth=1,
        max_depth=1,
        limit=limit,
        start_time=start_time,
    )


def _format_citations(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format results as minimal citations.

    Args:
        results: Full search results

    Returns:
        Minimal citation format with arxiv_id, title, and quote
    """
    citations = []
    for result in results:
        text = result.get("text", "")
        # Truncate text to ~200 chars for quote
        if len(text) > 200:
            quote = text[:200].rsplit(" ", 1)[0] + "..."
        else:
            quote = text

        citations.append(
            {
                "arxiv_id": result.get("arxiv_id"),
                "title": result.get("title"),
                "similarity": result.get("similarity"),
                "chunk_index": result.get("chunk_index"),
                "quote": quote,
            }
        )

    return citations
