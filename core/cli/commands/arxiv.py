"""ArXiv commands for HADES CLI.

Consolidates: search, info, abstract search, bulk search, similar, refine,
sync, sync-status, and ingest commands.
"""

from __future__ import annotations

import heapq
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from defusedxml import ElementTree as ET

from core.cli.config import get_arango_config, get_config
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    progress,
    success_response,
)
from core.database.collections import get_profile
from core.database.keys import chunk_key, embedding_key, normalize_document_key, strip_version
from core.tools.arxiv.arxiv_api_client import ArXivAPIClient, ArXivMetadata

# Sync metadata collection and document key
SYNC_METADATA_COLLECTION = "sync_metadata"
SYNC_WATERMARK_KEY = "abstracts"


# =============================================================================
# ArXiv API Commands (search, info)
# =============================================================================


def search_arxiv(
    query: str,
    max_results: int,
    categories: str | None,
    start_time: float,
) -> CLIResponse:
    """Search arxiv for papers matching a query.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        categories: Comma-separated arxiv categories (e.g., "cs.AI,cs.CL")
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with search results
    """
    progress(f"Searching arxiv for: {query}")

    client = ArXivAPIClient(rate_limit_delay=0.5)

    try:
        # Build search query
        search_query = query

        # Add category filter if specified
        if categories:
            cat_list = [c.strip() for c in categories.split(",")]
            cat_query = " OR ".join(f"cat:{cat}" for cat in cat_list)
            search_query = f"({query}) AND ({cat_query})"

        # Use arxiv API search endpoint
        params = {
            "search_query": f"all:{search_query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = client._make_request(client.api_base_url, params)
        root = ET.fromstring(response.content)

        # Parse results
        results = []
        entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for entry in entries:
            try:
                metadata = client._parse_entry(entry)
                results.append(_metadata_to_dict(metadata))
            except Exception:
                # Skip entries that fail to parse
                continue

        progress(f"Found {len(results)} papers")

        return success_response(
            command="arxiv.search",
            data={"query": query, "results": results},
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="arxiv.search",
            code=ErrorCode.SEARCH_FAILED,
            message=f"Search failed: {str(e)}",
            start_time=start_time,
        )
    finally:
        client.close()


def get_paper_info(arxiv_id: str, start_time: float) -> CLIResponse:
    """Get detailed metadata for a specific arxiv paper.

    Args:
        arxiv_id: ArXiv paper ID (e.g., "2401.12345")
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with paper metadata
    """
    progress(f"Fetching metadata for: {arxiv_id}")

    client = ArXivAPIClient(rate_limit_delay=0.5)

    try:
        # Validate ID format
        if not client.validate_arxiv_id(arxiv_id):
            return error_response(
                command="arxiv.info",
                code=ErrorCode.INVALID_ARXIV_ID,
                message=f"Invalid arxiv ID format: {arxiv_id}",
                start_time=start_time,
            )

        # Fetch metadata
        metadata = client.get_paper_metadata(arxiv_id)

        if metadata is None:
            return error_response(
                command="arxiv.info",
                code=ErrorCode.PAPER_NOT_FOUND,
                message=f"Paper not found: {arxiv_id}",
                start_time=start_time,
            )

        return success_response(
            command="arxiv.info",
            data=_metadata_to_dict(metadata),
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="arxiv.info",
            code=ErrorCode.NETWORK_ERROR,
            message=f"Failed to fetch paper info: {str(e)}",
            start_time=start_time,
        )
    finally:
        client.close()


def _metadata_to_dict(metadata: ArXivMetadata) -> dict[str, Any]:
    """Convert ArXivMetadata to a JSON-serializable dictionary."""
    return {
        "arxiv_id": metadata.arxiv_id,
        "title": metadata.title,
        "abstract": metadata.abstract,
        "authors": metadata.authors,
        "categories": metadata.categories,
        "primary_category": metadata.primary_category,
        "published": metadata.published.isoformat() if metadata.published else None,
        "updated": metadata.updated.isoformat() if metadata.updated else None,
        "doi": metadata.doi,
        "journal_ref": metadata.journal_ref,
        "pdf_url": metadata.pdf_url,
        "has_latex": metadata.has_latex,
    }


# =============================================================================
# Abstract Search Commands (search, bulk-search, similar, refine)
# =============================================================================


def search_abstracts(
    query: str,
    limit: int,
    start_time: float,
    category: str | None = None,
    hybrid: bool = False,
) -> CLIResponse:
    """Search the synced abstract embeddings (2.8M vectors).

    Args:
        query: Search query text
        limit: Maximum number of results
        start_time: Start time for duration calculation
        category: Optional arxiv category filter (e.g., "cs.AI")
        hybrid: If True, combine BM25 keyword search with semantic search

    Returns:
        CLIResponse with search results including local availability status
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="abstract.search",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    if limit <= 0:
        return error_response(
            command="abstract.search",
            code=ErrorCode.CONFIG_ERROR,
            message="limit must be >= 1",
            start_time=start_time,
        )

    progress("Generating query embedding...")

    try:
        # Generate query embedding
        query_embedding = _get_query_embedding(query, config)

        search_mode = "hybrid (BM25 + semantic)" if hybrid else "semantic"
        progress(f"Searching {limit} most similar abstracts ({search_mode})...")

        # Search the abstract embeddings
        results, total_processed = _search_abstract_embeddings(
            query_embedding,
            limit,
            config,
            category_filter=category,
            hybrid_query=query if hybrid else None,
        )

        return success_response(
            command="abstract.search",
            data={
                "query": query,
                "mode": "hybrid" if hybrid else "semantic",
                "results": results,
                "total_searched": total_processed,
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="abstract.search",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )


def search_abstracts_bulk(
    queries: list[str],
    limit: int,
    start_time: float,
    category: str | None = None,
) -> CLIResponse:
    """Search abstracts with multiple queries in a single optimized pass.

    This is more efficient than calling search_abstracts() multiple times because:
    - Embedder model is loaded once (not N times)
    - Single pass over 2.8M embeddings (not N passes)
    - Uses matrix multiplication for batch similarity computation

    Args:
        queries: List of search query texts
        limit: Maximum number of results per query
        start_time: Start time for duration calculation
        category: Optional arxiv category filter (e.g., "cs.AI")

    Returns:
        CLIResponse with results grouped by query
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="abstract.search-bulk",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    if not queries:
        return error_response(
            command="abstract.search-bulk",
            code=ErrorCode.CONFIG_ERROR,
            message="queries list must not be empty",
            start_time=start_time,
        )

    if limit <= 0:
        return error_response(
            command="abstract.search-bulk",
            code=ErrorCode.CONFIG_ERROR,
            message="limit must be >= 1",
            start_time=start_time,
        )

    progress(f"Generating embeddings for {len(queries)} queries...")

    try:
        # Generate all query embeddings in one batch
        query_embeddings = _get_bulk_query_embeddings(queries, config)

        progress(f"Searching {limit} results per query across {len(queries)} queries...")

        # Single-pass search for all queries
        results_by_query, total_processed = _search_abstract_embeddings_bulk(
            query_embeddings,
            queries,
            limit,
            config,
            category_filter=category,
        )

        return success_response(
            command="abstract.search-bulk",
            data={
                "queries": queries,
                "results_by_query": results_by_query,
                "total_searched": total_processed,
                "query_count": len(queries),
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="abstract.search-bulk",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )


def find_similar(
    arxiv_id: str,
    limit: int,
    start_time: float,
    category: str | None = None,
) -> CLIResponse:
    """Find papers similar to a given paper using its embedding.

    This is a "query by example" - uses the paper's existing embedding
    as the query vector, skipping the embedding generation step.

    Args:
        arxiv_id: ArXiv paper ID to find similar papers for
        limit: Maximum number of results
        start_time: Start time for duration calculation
        category: Optional arxiv category filter

    Returns:
        CLIResponse with similar papers
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="abstract.similar",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    if limit <= 0:
        return error_response(
            command="abstract.similar",
            code=ErrorCode.CONFIG_ERROR,
            message="limit must be >= 1",
            start_time=start_time,
        )

    progress(f"Finding papers similar to {arxiv_id}...")

    try:
        # Fetch the paper's existing embedding
        paper_embedding = _get_paper_embedding(arxiv_id, config)

        if paper_embedding is None:
            return error_response(
                command="abstract.similar",
                code=ErrorCode.PAPER_NOT_FOUND,
                message=f"Paper {arxiv_id} not found in synced abstracts",
                start_time=start_time,
            )

        progress(f"Searching {limit} most similar papers...")

        # Search using the paper's embedding (add 1 to limit since we'll exclude the source paper)
        results, total_processed = _search_abstract_embeddings(
            paper_embedding,
            limit + 1,
            config,
            category_filter=category,
            exclude_arxiv_id=arxiv_id,
        )

        # Filter out the source paper if it appears in results
        results = [r for r in results if r.get("arxiv_id") != arxiv_id][:limit]

        # Fetch source paper info for context
        source_info = _get_paper_info(arxiv_id, config)

        return success_response(
            command="abstract.similar",
            data={
                "source_paper": {
                    "arxiv_id": arxiv_id,
                    "title": source_info.get("title") if source_info else None,
                },
                "results": results,
                "total_searched": total_processed,
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="abstract.similar",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )


def refine_search(
    query: str,
    positive_ids: list[str],
    limit: int,
    start_time: float,
    negative_ids: list[str] | None = None,
    category: str | None = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> CLIResponse:
    """Refine search using relevance feedback (Rocchio algorithm).

    Takes a query and list of relevant papers (positive exemplars) to refine
    the search. Computes a modified query vector as a weighted combination of
    the original query and the positive exemplar embeddings.

    Rocchio formula: q' = alpha*q + beta*mean(positive) - gamma*mean(negative)

    Args:
        query: Original search query text
        positive_ids: ArXiv IDs of relevant papers (positive exemplars)
        limit: Maximum number of results
        start_time: Start time for duration calculation
        negative_ids: Optional ArXiv IDs of irrelevant papers (negative exemplars)
        category: Optional arxiv category filter (e.g., "cs.AI")
        alpha: Weight for original query (default 1.0)
        beta: Weight for positive exemplars (default 0.75)
        gamma: Weight for negative exemplars (default 0.15)

    Returns:
        CLIResponse with refined search results
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="abstract.refine",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    if limit <= 0:
        return error_response(
            command="abstract.refine",
            code=ErrorCode.CONFIG_ERROR,
            message="limit must be >= 1",
            start_time=start_time,
        )

    if not positive_ids:
        return error_response(
            command="abstract.refine",
            code=ErrorCode.CONFIG_ERROR,
            message="At least one positive exemplar is required",
            start_time=start_time,
        )

    progress("Generating query embedding...")

    try:
        # Generate original query embedding
        query_embedding = _get_query_embedding(query, config)

        progress(f"Fetching embeddings for {len(positive_ids)} positive exemplars...")

        # Fetch positive exemplar embeddings
        positive_embeddings = _get_multiple_paper_embeddings(positive_ids, config)
        if not positive_embeddings:
            return error_response(
                command="abstract.refine",
                code=ErrorCode.PAPER_NOT_FOUND,
                message="None of the positive exemplar papers were found",
                start_time=start_time,
            )

        # Fetch negative exemplar embeddings if provided
        negative_embeddings: list[np.ndarray] = []
        if negative_ids:
            progress(f"Fetching embeddings for {len(negative_ids)} negative exemplars...")
            negative_embeddings = _get_multiple_paper_embeddings(negative_ids, config)

        progress("Computing refined query using Rocchio algorithm...")

        # Compute Rocchio centroid
        refined_embedding = _compute_rocchio_centroid(
            query_embedding,
            positive_embeddings,
            negative_embeddings,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        progress(f"Searching {limit} most similar abstracts with refined query...")

        # Build exclude set: exclude all exemplar papers from results
        exclude_ids = set(positive_ids)
        if negative_ids:
            exclude_ids.update(negative_ids)

        # Search using refined embedding
        results, total_processed = _search_abstract_embeddings(
            refined_embedding,
            limit + len(exclude_ids),  # Fetch extra to account for exclusions
            config,
            category_filter=category,
        )

        # Filter out exemplar papers from results
        filtered_results = []
        for r in results:
            result_arxiv_id = r.get("arxiv_id", "")
            # Normalize for comparison
            base_id = strip_version(result_arxiv_id)
            if not any(strip_version(eid) == base_id for eid in exclude_ids):
                filtered_results.append(r)
                if len(filtered_results) >= limit:
                    break

        # Get info about the positive exemplars used
        positive_info = []
        for pid in positive_ids:
            info = _get_paper_info(pid, config)
            if info:
                positive_info.append({"arxiv_id": pid, "title": info.get("title")})

        return success_response(
            command="abstract.refine",
            data={
                "query": query,
                "mode": "relevance_feedback",
                "positive_exemplars": positive_info,
                "negative_exemplars": negative_ids or [],
                "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
                "results": filtered_results,
                "total_searched": total_processed,
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="abstract.refine",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )


# =============================================================================
# Sync Commands
# =============================================================================


def get_sync_status(start_time: float) -> CLIResponse:
    """Get the current sync status including last sync time and history.

    Args:
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with sync status
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="sync.status",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        metadata = _get_sync_metadata(config)

        if metadata is None:
            return success_response(
                command="sync.status",
                data={
                    "last_sync": None,
                    "total_synced": 0,
                    "sync_history": [],
                    "message": "No sync history found. Run 'hades arxiv sync' to begin.",
                },
                start_time=start_time,
            )

        return success_response(
            command="sync.status",
            data={
                "last_sync": metadata.get("last_sync"),
                "total_synced": metadata.get("total_synced", 0),
                "sync_history": metadata.get("sync_history", [])[-10:],  # Last 10 syncs
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="sync.status",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def sync_abstracts(
    from_date: str | None,
    categories: str | None,
    max_results: int,
    batch_size: int,
    start_time: float,
    incremental: bool = False,
) -> CLIResponse:
    """Sync abstracts from arxiv to the database.

    Fetches metadata and abstracts, generates embeddings, and stores them
    for semantic search - without downloading full PDFs.

    Args:
        from_date: Start date in YYYY-MM-DD format (default: 7 days ago)
        categories: Comma-separated arxiv categories (e.g., "cs.AI,cs.CL")
        max_results: Maximum number of papers to fetch
        batch_size: Batch size for embedding generation
        start_time: Start time for duration calculation
        incremental: If True, sync only papers newer than last sync watermark

    Returns:
        CLIResponse with sync results
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="sync",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    # Determine start date
    if incremental:
        # Use watermark from last sync
        last_sync = _get_last_sync_date(config)
        if last_sync is None:
            progress("No previous sync found, using default (7 days ago)...")
            start_date = datetime.now() - timedelta(days=7)
        else:
            # Convert to naive local datetime for comparison
            # Must use astimezone() first to convert UTC to local time
            if last_sync.tzinfo is not None:
                start_date = last_sync.astimezone().replace(tzinfo=None)
            else:
                start_date = last_sync
            progress(f"Incremental sync from {start_date.strftime('%Y-%m-%d %H:%M')}...")
    elif from_date:
        try:
            start_date = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            return error_response(
                command="sync",
                code=ErrorCode.CONFIG_ERROR,
                message=f"Invalid date format: {from_date}. Use YYYY-MM-DD",
                start_time=start_time,
            )
    else:
        start_date = datetime.now() - timedelta(days=7)

    # Validate batch_size
    if batch_size <= 0:
        return error_response(
            command="sync",
            code=ErrorCode.CONFIG_ERROR,
            message="batch_size must be >= 1",
            start_time=start_time,
        )

    progress(f"Syncing abstracts from {start_date.strftime('%Y-%m-%d')}...")

    try:
        # Fetch papers from arxiv
        papers = _fetch_recent_papers(start_date, categories, max_results)

        if not papers:
            return success_response(
                command="sync",
                data={"synced": 0, "skipped": 0, "message": "No new papers found"},
                start_time=start_time,
            )

        progress(f"Found {len(papers)} papers, checking for duplicates...")

        # Filter out already-synced papers
        new_papers = _filter_existing(papers, config)

        if not new_papers:
            return success_response(
                command="sync",
                data={
                    "synced": 0,
                    "skipped": len(papers),
                    "message": "All papers already in database",
                },
                start_time=start_time,
            )

        progress(f"Embedding {len(new_papers)} new abstracts...")

        # Embed abstracts and store
        synced = _embed_and_store_abstracts(new_papers, config, batch_size)

        # Update sync metadata watermark with current time (when sync ran)
        try:
            _update_sync_metadata(config, added=synced, updated=0, sync_date=datetime.now(UTC))
        except Exception as meta_err:
            progress(f"Warning: Failed to update sync metadata: {meta_err}")

        return success_response(
            command="sync",
            data={
                "synced": synced,
                "skipped": len(papers) - len(new_papers),
                "mode": "incremental" if incremental else "manual",
                "from_date": start_date.strftime("%Y-%m-%d"),
                "papers": [
                    {
                        "arxiv_id": p["arxiv_id"],
                        "title": p["title"],
                        "categories": p.get("categories", []),
                    }
                    for p in new_papers[:10]  # Show first 10
                ],
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="sync",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )


# =============================================================================
# Ingest Commands
# =============================================================================


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
            result = _ingest_local_pdf(pdf_path, config)
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


def ingest_from_abstract(
    arxiv_ids: list[str],
    force: bool,
    start_time: float,
) -> CLIResponse:
    """Ingest papers identified from abstract search.

    This is a convenience wrapper around ingest_papers,
    designed for the abstract search → ingest workflow.

    Args:
        arxiv_ids: ArXiv paper IDs to ingest
        force: Force reprocessing even if already exists
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with ingestion results
    """
    return ingest_papers(
        arxiv_ids=arxiv_ids,
        pdf_paths=None,
        force=force,
        start_time=start_time,
    )


# =============================================================================
# Internal Helpers — Abstract Search
# =============================================================================


def _get_query_embedding(text: str, config: Any) -> np.ndarray:
    """Generate embedding for query text."""
    from core.cli.config import get_embedder_client

    with get_embedder_client(config) as client:
        return client.embed_query(text)


def _get_bulk_query_embeddings(texts: list[str], config: Any) -> np.ndarray:
    """Generate embeddings for multiple query texts in one batch.

    Args:
        texts: List of query texts
        config: CLI configuration

    Returns:
        Array of shape (num_queries, embedding_dim)
    """
    from core.cli.config import get_embedder_client

    with get_embedder_client(config) as client:
        return client.embed_texts(texts, task="retrieval.query")


def _get_multiple_paper_embeddings(arxiv_ids: list[str], config: Any) -> list[np.ndarray]:
    """Fetch embeddings for multiple papers.

    Args:
        arxiv_ids: List of ArXiv paper IDs
        config: CLI configuration

    Returns:
        List of embedding vectors for papers that were found
    """
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
        sync_col = get_profile("sync")
        # Normalize arxiv_ids to keys
        keys = [normalize_document_key(aid) for aid in arxiv_ids]

        # Fetch all embeddings in one query
        aql = f"""
            FOR key IN @keys
                LET doc = DOCUMENT(CONCAT("{sync_col.embeddings}/", key))
                FILTER doc != null AND doc.combined_embedding != null
                RETURN doc.combined_embedding
        """
        results = client.query(aql, bind_vars={"keys": keys})

        embeddings = []
        for emb in results:
            if emb:
                embeddings.append(np.array(emb, dtype=np.float32))

        return embeddings

    finally:
        client.close()


def _compute_rocchio_centroid(
    query_embedding: np.ndarray,
    positive_embeddings: list[np.ndarray],
    negative_embeddings: list[np.ndarray] | None = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
) -> np.ndarray:
    """Compute Rocchio relevance feedback centroid.

    Rocchio formula: q' = alpha*q + beta*mean(positive) - gamma*mean(negative)

    Args:
        query_embedding: Original query vector
        positive_embeddings: List of positive exemplar vectors
        negative_embeddings: Optional list of negative exemplar vectors
        alpha: Weight for original query
        beta: Weight for positive exemplars
        gamma: Weight for negative exemplars

    Returns:
        Refined query vector
    """
    # Start with weighted original query
    refined = alpha * query_embedding

    # Add positive centroid
    if positive_embeddings:
        positive_centroid = np.mean(positive_embeddings, axis=0)
        refined = refined + (beta * positive_centroid)

    # Subtract negative centroid
    if negative_embeddings:
        negative_centroid = np.mean(negative_embeddings, axis=0)
        refined = refined - (gamma * negative_centroid)

    return refined


def _get_paper_embedding(arxiv_id: str, config: Any) -> np.ndarray | None:
    """Fetch the existing embedding for a paper.

    Args:
        arxiv_id: ArXiv paper ID
        config: CLI configuration

    Returns:
        The paper's embedding vector, or None if not found

    Raises:
        ArangoHttpError: For non-404 database errors (5xx, connection issues, etc.)
    """
    from core.database.arango.optimized_client import (
        ArangoHttp2Client,
        ArangoHttp2Config,
        ArangoHttpError,
    )

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
        sync_col = get_profile("sync")
        key = normalize_document_key(arxiv_id)

        # Fetch the embedding - only catch 404 (document not found)
        try:
            doc = client.get_document(sync_col.embeddings, key)
            if doc and doc.get("combined_embedding"):
                return np.array(doc["combined_embedding"], dtype=np.float32)
            return None
        except ArangoHttpError as e:
            if e.status_code == 404:
                return None
            raise  # Re-raise non-404 errors (5xx, auth issues, etc.)

    finally:
        client.close()


def _get_paper_info(arxiv_id: str, config: Any) -> dict[str, Any] | None:
    """Fetch basic info for a paper.

    Args:
        arxiv_id: ArXiv paper ID
        config: CLI configuration

    Returns:
        Dict with paper info, or None if not found

    Raises:
        ArangoHttpError: For non-404 database errors (5xx, connection issues, etc.)
    """
    from core.database.arango.optimized_client import (
        ArangoHttp2Client,
        ArangoHttp2Config,
        ArangoHttpError,
    )

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
        sync_col = get_profile("sync")
        key = normalize_document_key(arxiv_id)

        # Fetch the paper info - only catch 404 (document not found)
        try:
            doc = client.get_document(sync_col.chunks, key)
            if doc:
                return {
                    "arxiv_id": arxiv_id,
                    "title": doc.get("title"),
                    "abstract": doc.get("abstract"),
                }
            return None
        except ArangoHttpError as e:
            if e.status_code == 404:
                return None
            raise  # Re-raise non-404 errors (5xx, auth issues, etc.)

    finally:
        client.close()


def _search_abstract_embeddings(
    query_embedding: np.ndarray,
    limit: int,
    config: Any,
    category_filter: str | None = None,
    hybrid_query: str | None = None,
    exclude_arxiv_id: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Search abstract embeddings with batched processing.

    Processes 2.8M embeddings in batches to avoid memory issues.
    Uses a min-heap to efficiently track top-k results across batches.

    Args:
        query_embedding: Query vector (2048-dim)
        limit: Number of results to return
        config: CLI configuration
        category_filter: Optional category to filter by
        hybrid_query: If provided, combine semantic with BM25 keyword search
        exclude_arxiv_id: ArXiv ID to exclude from results (for similar search)

    Returns:
        Tuple of (results list, total_processed count)
    """
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
        sync_col = get_profile("sync")
        # Normalize query embedding once
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Use a min-heap to track top-k results (heap stores negative similarity for max-heap behavior)
        # Format: (negative_similarity, arxiv_id, data_dict)
        top_results: list[tuple[float, str, dict]] = []

        # Batch size for processing embeddings
        batch_size = 10000
        offset = 0
        total_processed = 0

        # Build category filter if specified (use bind variable to prevent injection)
        category_clause = ""
        category_bind: dict[str, str] = {}
        if category_filter:
            # Join with arxiv_papers to filter by category
            category_clause = f"""
                LET paper = DOCUMENT(CONCAT("{sync_col.metadata}/", emb._key))
                FILTER paper != null AND @category_filter IN paper.categories
            """
            category_bind = {"category_filter": category_filter}

        progress(f"Processing embeddings in batches of {batch_size}...")

        while True:
            # Fetch batch of embeddings
            aql = f"""
                FOR emb IN {sync_col.embeddings}
                    {category_clause}
                    LIMIT @offset, @batch_size
                    RETURN {{
                        arxiv_id: emb.arxiv_id,
                        embedding: emb.combined_embedding
                    }}
            """

            try:
                batch = client.query(
                    aql,
                    bind_vars={"offset": offset, "batch_size": batch_size, **category_bind},
                )
            except Exception as e:
                # If we get a chunked encoding error, try smaller batches
                if "chunked" in str(e).lower() or "501" in str(e):
                    progress(f"Reducing batch size due to: {e}")
                    batch_size = batch_size // 2
                    if batch_size < 1000:
                        raise RuntimeError("Batch size too small, cannot process") from e
                    continue
                raise

            if not batch:
                break

            # Process batch
            for item in batch:
                if item.get("embedding") is None:
                    continue

                item_arxiv_id = item.get("arxiv_id", "")

                # Skip excluded paper (for similar search)
                if exclude_arxiv_id:
                    if strip_version(exclude_arxiv_id) == strip_version(item_arxiv_id):
                        continue

                emb = np.array(item["embedding"], dtype=np.float32)
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                similarity = float(np.dot(query_norm, emb_norm))

                # Use heap to track top-k
                if len(top_results) < limit:
                    heapq.heappush(top_results, (similarity, item_arxiv_id, {"similarity": similarity}))
                elif similarity > top_results[0][0]:
                    heapq.heapreplace(top_results, (similarity, item_arxiv_id, {"similarity": similarity}))

            total_processed += len(batch)
            offset += batch_size

            # Progress update every 100k
            if total_processed % 100000 < batch_size:
                progress(f"Processed {total_processed:,} embeddings...")

            # Early termination if we've processed fewer than batch_size (end of data)
            if len(batch) < batch_size:
                break

        progress(f"Processed {total_processed:,} total embeddings")

        # Extract results from heap and sort by similarity (descending)
        heap_results = sorted(top_results, key=lambda x: x[0], reverse=True)

        # Fetch metadata and check local status for top results
        results = []
        result_arxiv_ids = [r[1] for r in heap_results]

        if result_arxiv_ids:
            # Fetch abstracts and titles
            # Normalize both "." and "/" for legacy arxiv IDs like hep-th/9901001
            arxiv_col = get_profile("arxiv")
            abstracts_aql = f"""
                FOR id IN @ids
                    LET key = SUBSTITUTE(SUBSTITUTE(id, ".", "_"), "/", "_")
                    LET abstract = DOCUMENT(CONCAT("{sync_col.chunks}/", key))
                    LET paper = DOCUMENT(CONCAT("{sync_col.metadata}/", key))
                    LET local = DOCUMENT(CONCAT("{arxiv_col.metadata}/", key))
                    RETURN {{
                        arxiv_id: id,
                        title: abstract.title,
                        abstract: abstract.abstract,
                        categories: paper.categories,
                        local: local != null,
                        local_chunks: local.num_chunks
                    }}
            """
            metadata = client.query(abstracts_aql, bind_vars={"ids": result_arxiv_ids})
            metadata_map = {m["arxiv_id"]: m for m in metadata if m}

            # If hybrid search, compute keyword scores and re-rank
            if hybrid_query:
                progress("Re-ranking with keyword matching...")
                heap_results = _hybrid_rerank(heap_results, metadata_map, hybrid_query)

            for sim, result_id, extra in heap_results:
                meta = metadata_map.get(result_id, {})

                # Truncate abstract for display
                abstract_text = meta.get("abstract", "") or ""
                if len(abstract_text) > 300:
                    abstract_snippet = abstract_text[:300].rsplit(" ", 1)[0] + "..."
                else:
                    abstract_snippet = abstract_text

                # After hybrid rerank, sim is combined score; use semantic score from extra
                semantic_score = extra.get("similarity", sim)

                result_entry = {
                    "arxiv_id": result_id,
                    "title": meta.get("title"),
                    "similarity": round(semantic_score, 4),
                    "abstract": abstract_snippet,
                    "categories": meta.get("categories", []),
                    "local": meta.get("local", False),
                    "local_chunks": meta.get("local_chunks"),
                }

                # Add hybrid scores if present
                if extra.get("keyword_score") is not None:
                    result_entry["keyword_score"] = round(extra["keyword_score"], 4)
                    result_entry["combined_score"] = round(extra["combined_score"], 4)

                results.append(result_entry)

        return results, total_processed

    finally:
        client.close()


def _hybrid_rerank(
    heap_results: list[tuple[float, str, dict]],
    metadata_map: dict[str, dict],
    query: str,
    semantic_weight: float = 0.7,
) -> list[tuple[float, str, dict]]:
    """Re-rank results by combining semantic similarity with keyword matching.

    Uses a simple BM25-inspired keyword score combined with semantic similarity.

    Args:
        heap_results: List of (similarity, arxiv_id, extra_dict) tuples
        metadata_map: Map of arxiv_id to metadata (title, abstract)
        query: Original query text for keyword matching
        semantic_weight: Weight for semantic score (1 - this = keyword weight)

    Returns:
        Re-ranked list of (combined_score, arxiv_id, extra_dict) tuples
    """
    # Tokenize query
    query_terms = set(query.lower().split())

    reranked = []
    for semantic_score, result_id, _extra in heap_results:
        meta = metadata_map.get(result_id, {})

        # Compute keyword score from title and abstract
        title = (meta.get("title") or "").lower()
        abstract = (meta.get("abstract") or "").lower()
        text = title + " " + abstract

        # Simple term frequency score
        text_terms = text.split()
        if text_terms:
            matches = sum(1 for term in query_terms if term in text)
            keyword_score = matches / len(query_terms) if query_terms else 0
        else:
            keyword_score = 0

        # Combine scores
        combined = (semantic_weight * semantic_score) + ((1 - semantic_weight) * keyword_score)

        # Store scores in extra dict
        new_extra = {
            "similarity": semantic_score,
            "keyword_score": keyword_score,
            "combined_score": combined,
        }

        reranked.append((combined, result_id, new_extra))

    # Sort by combined score descending
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked


def _search_abstract_embeddings_bulk(
    query_embeddings: np.ndarray,
    queries: list[str],
    limit: int,
    config: Any,
    category_filter: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Search abstract embeddings with multiple queries in a single pass.

    Uses matrix multiplication for efficient batch similarity computation.
    Maintains separate heaps for each query's top-k results.

    Args:
        query_embeddings: Array of shape (num_queries, embedding_dim)
        queries: Original query texts (for result attribution)
        limit: Number of results per query
        config: CLI configuration
        category_filter: Optional category to filter by

    Returns:
        Tuple of (results dict keyed by query, total_processed count)
    """
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
        sync_col = get_profile("sync")
        num_queries = len(queries)

        # Normalize all query embeddings at once
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8
        query_norms = query_embeddings / norms

        # Separate heap for each query: heap[i] = [(similarity, arxiv_id, {}), ...]
        top_results: list[list[tuple[float, str, dict]]] = [[] for _ in range(num_queries)]

        batch_size = 10000
        offset = 0
        total_processed = 0

        # Build category filter if specified (use bind variable to prevent injection)
        category_clause = ""
        category_bind: dict[str, str] = {}
        if category_filter:
            category_clause = f"""
                LET paper = DOCUMENT(CONCAT("{sync_col.metadata}/", emb._key))
                FILTER paper != null AND @category_filter IN paper.categories
            """
            category_bind = {"category_filter": category_filter}

        progress(f"Processing embeddings in batches of {batch_size}...")

        while True:
            aql = f"""
                FOR emb IN {sync_col.embeddings}
                    {category_clause}
                    LIMIT @offset, @batch_size
                    RETURN {{
                        arxiv_id: emb.arxiv_id,
                        embedding: emb.combined_embedding
                    }}
            """

            try:
                batch = client.query(
                    aql,
                    bind_vars={"offset": offset, "batch_size": batch_size, **category_bind},
                )
            except Exception as e:
                if "chunked" in str(e).lower() or "501" in str(e):
                    progress(f"Reducing batch size due to: {e}")
                    batch_size = batch_size // 2
                    if batch_size < 1000:
                        raise RuntimeError("Batch size too small, cannot process") from e
                    continue
                raise

            if not batch:
                break

            # Build embedding matrix for this batch
            batch_embeddings = []
            batch_arxiv_ids = []
            for item in batch:
                if item.get("embedding") is not None:
                    batch_embeddings.append(item["embedding"])
                    batch_arxiv_ids.append(item.get("arxiv_id", ""))

            if batch_embeddings:
                # Convert to numpy and normalize
                emb_matrix = np.array(batch_embeddings, dtype=np.float32)
                emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
                emb_matrix_norm = emb_matrix / emb_norms

                # Compute all similarities at once: (num_queries, batch_size)
                similarities = query_norms @ emb_matrix_norm.T

                # Update heaps for each query
                for q_idx in range(num_queries):
                    for b_idx, batch_arxiv_id in enumerate(batch_arxiv_ids):
                        sim = float(similarities[q_idx, b_idx])

                        if len(top_results[q_idx]) < limit:
                            heapq.heappush(top_results[q_idx], (sim, batch_arxiv_id, {"similarity": sim}))
                        elif sim > top_results[q_idx][0][0]:
                            heapq.heapreplace(top_results[q_idx], (sim, batch_arxiv_id, {"similarity": sim}))

            total_processed += len(batch)
            offset += batch_size

            if total_processed % 100000 < batch_size:
                progress(f"Processed {total_processed:,} embeddings...")

            if len(batch) < batch_size:
                break

        progress(f"Processed {total_processed:,} total embeddings")

        # Collect all unique arxiv_ids for metadata fetch
        all_arxiv_ids = set()
        for heap in top_results:
            for _, heap_arxiv_id, _ in heap:
                all_arxiv_ids.add(heap_arxiv_id)

        # Fetch metadata for all results
        metadata_map: dict[str, dict] = {}
        if all_arxiv_ids:
            arxiv_col = get_profile("arxiv")
            abstracts_aql = f"""
                FOR id IN @ids
                    LET key = SUBSTITUTE(SUBSTITUTE(id, ".", "_"), "/", "_")
                    LET abstract = DOCUMENT(CONCAT("{sync_col.chunks}/", key))
                    LET paper = DOCUMENT(CONCAT("{sync_col.metadata}/", key))
                    LET local = DOCUMENT(CONCAT("{arxiv_col.metadata}/", key))
                    RETURN {{
                        arxiv_id: id,
                        title: abstract.title,
                        abstract: abstract.abstract,
                        categories: paper.categories,
                        local: local != null,
                        local_chunks: local.num_chunks
                    }}
            """
            metadata_results = client.query(abstracts_aql, bind_vars={"ids": list(all_arxiv_ids)})
            metadata_map = {m["arxiv_id"]: m for m in metadata_results if m}

        # Build results dict keyed by query
        results_by_query: dict[str, list[dict[str, Any]]] = {}

        for q_idx, query in enumerate(queries):
            heap_results = sorted(top_results[q_idx], key=lambda x: x[0], reverse=True)
            query_results = []

            for sim, result_id, extra in heap_results:
                meta = metadata_map.get(result_id, {})

                abstract_text = meta.get("abstract", "") or ""
                if len(abstract_text) > 300:
                    abstract_snippet = abstract_text[:300].rsplit(" ", 1)[0] + "..."
                else:
                    abstract_snippet = abstract_text

                query_results.append(
                    {
                        "arxiv_id": result_id,
                        "title": meta.get("title"),
                        "similarity": round(extra.get("similarity", sim), 4),
                        "abstract": abstract_snippet,
                        "categories": meta.get("categories", []),
                        "local": meta.get("local", False),
                        "local_chunks": meta.get("local_chunks"),
                    }
                )

            results_by_query[query] = query_results

        return results_by_query, total_processed

    finally:
        client.close()


# =============================================================================
# Internal Helpers — Sync
# =============================================================================


def _get_sync_metadata(config: Any) -> dict[str, Any] | None:
    """Fetch sync metadata from the database.

    Returns None if no metadata exists (first sync).
    """
    from core.database.arango.optimized_client import (
        ArangoHttp2Client,
        ArangoHttp2Config,
        ArangoHttpError,
    )

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
        try:
            doc = client.get_document(SYNC_METADATA_COLLECTION, SYNC_WATERMARK_KEY)
            return doc
        except ArangoHttpError as e:
            if e.status_code == 404:
                return None
            raise
    finally:
        client.close()


def _update_sync_metadata(
    config: Any,
    added: int,
    updated: int,
    sync_date: datetime,
) -> None:
    """Update sync metadata after a successful sync.

    Creates the sync_metadata collection if it doesn't exist.
    """
    from core.database.arango.optimized_client import (
        ArangoHttp2Client,
        ArangoHttp2Config,
        ArangoHttpError,
    )

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
        # Ensure collection exists
        try:
            client.request(
                "POST",
                f"/_db/{arango_config['database']}/_api/collection",
                json={"name": SYNC_METADATA_COLLECTION},
            )
        except ArangoHttpError as e:
            # 409 = collection already exists, which is fine
            if e.status_code != 409:
                raise

        # Get existing metadata or create new
        existing = None
        try:
            existing = client.get_document(SYNC_METADATA_COLLECTION, SYNC_WATERMARK_KEY)
        except ArangoHttpError as e:
            if e.status_code != 404:
                raise

        now_iso = datetime.now(UTC).isoformat()
        sync_entry = {
            "date": sync_date.strftime("%Y-%m-%d"),
            "added": added,
            "updated": updated,
            "timestamp": now_iso,
        }

        if existing:
            # Update existing document
            total_synced = existing.get("total_synced", 0) + added
            history = existing.get("sync_history", [])
            history.append(sync_entry)
            # Keep last 100 entries
            history = history[-100:]

            # Use REPLACE to update the document
            client.request(
                "PUT",
                f"/_db/{arango_config['database']}/_api/document/{SYNC_METADATA_COLLECTION}/{SYNC_WATERMARK_KEY}",
                json={
                    "_key": SYNC_WATERMARK_KEY,
                    "last_sync": now_iso,
                    "total_synced": total_synced,
                    "sync_history": history,
                },
            )
        else:
            # Create new document
            client.insert_documents(
                SYNC_METADATA_COLLECTION,
                [
                    {
                        "_key": SYNC_WATERMARK_KEY,
                        "last_sync": now_iso,
                        "total_synced": added,
                        "sync_history": [sync_entry],
                    }
                ],
            )

    finally:
        client.close()


def _get_last_sync_date(config: Any) -> datetime | None:
    """Get the date of the last successful sync.

    Returns None if no previous sync exists.
    """
    metadata = _get_sync_metadata(config)
    if metadata and metadata.get("last_sync"):
        # Parse ISO format timestamp
        last_sync_str = metadata["last_sync"]
        # Handle both with and without timezone
        try:
            return datetime.fromisoformat(last_sync_str.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _fetch_recent_papers(
    start_date: datetime,
    categories: str | None,
    max_results: int,
) -> list[dict[str, Any]]:
    """Fetch recent papers from arxiv API.

    ArXiv API has pagination limits (~10000 results per query), so we
    query month by month to avoid hitting those limits.
    """

    client = ArXivAPIClient(rate_limit_delay=0.5)

    try:
        papers = []
        current_date = start_date
        today = datetime.now()

        # Query month by month to avoid pagination limits
        while current_date <= today and len(papers) < max_results:
            # Calculate month end
            if current_date.month == 12:
                month_end = datetime(current_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)

            if month_end > today:
                month_end = today

            progress(f"Fetching papers from {current_date.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}...")

            # Build date query for this month
            date_str = current_date.strftime("%Y%m%d")
            end_str = month_end.strftime("%Y%m%d")
            date_query = f"submittedDate:[{date_str}0000 TO {end_str}2359]"

            if categories:
                cat_list = [c.strip() for c in categories.split(",")]
                cat_query = " OR ".join(f"cat:{cat}" for cat in cat_list)
                search_query = f"({cat_query}) AND {date_query}"
            else:
                search_query = date_query

            # Fetch papers for this month (limit to 5000 per month to stay safe)
            month_papers = _fetch_month_papers(client, search_query, min(5000, max_results - len(papers)))
            papers.extend(month_papers)

            progress(f"Found {len(month_papers)} papers for {current_date.strftime('%Y-%m')}")

            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

        return papers[:max_results]

    finally:
        client.close()


def _fetch_month_papers(
    client: Any,
    search_query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Fetch papers for a single month query."""
    papers = []
    start = 0
    batch_size = 100  # arxiv max per request

    while len(papers) < max_results:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = client._make_request(client.api_base_url, params)
            root = ET.fromstring(response.content)

            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if not entries:
                break

            for entry in entries:
                try:
                    metadata = client._parse_entry(entry)
                    papers.append(
                        {
                            "arxiv_id": metadata.arxiv_id,
                            "title": metadata.title,
                            "abstract": metadata.abstract,
                            "authors": metadata.authors,
                            "categories": metadata.categories,
                            "primary_category": metadata.primary_category,
                            "published": metadata.published.isoformat() if metadata.published else None,
                            "updated": metadata.updated.isoformat() if metadata.updated else None,
                        }
                    )
                except Exception:
                    continue

            start += batch_size
            if len(entries) < batch_size:
                break

            # Rate limit
            time.sleep(0.5)

        except Exception as e:
            # If we hit pagination limits or server errors, stop for this month
            progress(f"Warning: Stopped at {start} papers due to: {e}")
            break

    return papers


def _filter_existing(
    papers: list[dict[str, Any]],
    config: Any,
) -> list[dict[str, Any]]:
    """Filter out papers that already exist in the database."""
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

    if not papers:
        return []

    # Use read-write socket - cursor operations require write access
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
        sync_col = get_profile("sync")
        # Build list of base IDs to check (strip version suffixes)
        check_ids = [strip_version(p["arxiv_id"]) for p in papers]

        # Query only for the specific IDs we're checking (not all 2.8M!)
        # This avoids chunked transfer encoding issues with large result sets
        existing_ids = set()
        try:
            results = client.query(
                f"FOR doc IN {sync_col.metadata} FILTER doc.arxiv_id IN @ids RETURN doc.arxiv_id",
                bind_vars={"ids": check_ids},
            )
            existing_ids.update(r for r in results if r)
        except Exception as e:
            # Collection may not exist yet - treat as empty
            progress(f"Note: Could not query existing papers: {e}")

        # Filter out existing
        new_papers = []
        for p in papers:
            base_id = strip_version(p["arxiv_id"])
            if base_id not in existing_ids:
                new_papers.append(p)

        return new_papers

    finally:
        client.close()


def _embed_and_store_abstracts(
    papers: list[dict[str, Any]],
    config: Any,
    batch_size: int,
) -> int:
    """Embed abstracts and store in database.

    Uses the existing database schema:
    - arxiv_papers: metadata (arxiv_id, title, authors, categories, etc.)
    - arxiv_abstracts: abstract text (arxiv_id, title, abstract)
    - arxiv_embeddings: embeddings (arxiv_id, combined_embedding)
    """
    from core.cli.config import get_embedder_client
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

    # Initialize embedder client (uses service, no local fallback)
    embedder_client = get_embedder_client(config)

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
        sync_col = get_profile("sync")
        synced = 0
        now_iso = datetime.now(UTC).isoformat()

        # Process in batches
        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            progress(f"Processing batch {i // batch_size + 1} ({len(batch)} papers)...")

            # Extract abstracts for embedding
            abstracts = [p["abstract"] for p in batch]

            # Generate embeddings using embedding service
            embeddings = embedder_client.embed_texts(abstracts, task="retrieval.passage")

            # Prepare documents matching existing schema
            paper_docs = []
            abstract_docs = []
            embedding_docs = []

            for j, paper in enumerate(batch):
                paper_arxiv_id = paper["arxiv_id"]
                base_id = strip_version(paper_arxiv_id)
                sanitized_key = normalize_document_key(paper_arxiv_id)

                # Parse year_month from arxiv_id (YYMM.NNNNN format)
                if "." in base_id:
                    yymm = base_id.split(".")[0]
                    year = 2000 + int(yymm[:2])
                    month = int(yymm[2:4])
                else:
                    year = 2024
                    month = 1
                    yymm = "2401"

                # arxiv_papers document (metadata)
                paper_docs.append(
                    {
                        "_key": sanitized_key,
                        "arxiv_id": base_id,
                        "title": paper["title"],
                        "authors": paper["authors"],
                        "categories": paper["categories"],
                        "primary_category": paper["primary_category"],
                        "year": year,
                        "month": month,
                        "year_month": f"{year}{month:02d}",
                        "created_at": now_iso,
                    }
                )

                # arxiv_abstracts document
                abstract_docs.append(
                    {
                        "_key": sanitized_key,
                        "arxiv_id": base_id,
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                    }
                )

                # arxiv_embeddings document (matching existing schema)
                embedding_docs.append(
                    {
                        "_key": sanitized_key,
                        "arxiv_id": base_id,
                        "combined_embedding": embeddings[j].tolist(),
                        "abstract_embedding": [],  # Empty to match existing schema
                        "title_embedding": [],  # Empty to match existing schema
                    }
                )

            # Insert documents (skip duplicates - they raise unique constraint errors)
            try:
                duplicates = 0
                for doc in paper_docs:
                    try:
                        client.insert_documents(sync_col.metadata, [doc])
                    except Exception:
                        duplicates += 1  # Likely duplicate key

                for doc in abstract_docs:
                    try:
                        client.insert_documents(sync_col.chunks, [doc])
                    except Exception:
                        pass  # Skip if exists

                for doc in embedding_docs:
                    try:
                        client.insert_documents(sync_col.embeddings, [doc])
                    except Exception:
                        pass  # Skip if exists

                synced += len(batch) - duplicates
            except Exception as e:
                progress(f"Warning: Failed to store batch: {e}")

        return synced

    finally:
        embedder_client.close()
        client.close()


# =============================================================================
# Internal Helpers — Ingest
# =============================================================================


def _ingest_arxiv_paper(arxiv_id: str, config: Any, force: bool) -> dict[str, Any]:
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


def _ingest_local_pdf(pdf_path: str, config: Any) -> dict[str, Any]:
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
    # Use traditional chunking for now - late chunking has a dimension bug
    proc_config = ProcessingConfig(
        use_gpu=config.use_gpu,
        device=config.device,
        chunking_strategy="traditional",
        chunk_size_tokens=500,
        chunk_overlap_tokens=100,
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
            progress(f"Storing {len(chunk_docs)} chunks in database...")

            if chunk_docs:
                client.insert_documents(col.chunks, chunk_docs, overwrite=force)
            if embedding_docs:
                client.insert_documents(col.embeddings, embedding_docs, overwrite=force)
            client.insert_documents(col.metadata, [meta_doc], overwrite=force)

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
