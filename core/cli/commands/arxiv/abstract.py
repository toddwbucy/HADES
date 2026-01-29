"""Abstract search commands for HADES CLI.

Provides semantic search over the 2.8M synced abstract embeddings,
including bulk search, query-by-example (similar), and relevance
feedback (Rocchio algorithm).
"""

from __future__ import annotations

import numpy as np

from core.cli.config import get_config
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    progress,
    success_response,
)
from core.database.keys import strip_version

from .helpers import (
    _compute_rocchio_centroid,
    _get_bulk_query_embeddings,
    _get_multiple_paper_embeddings,
    _get_paper_embedding,
    _get_paper_info,
    _get_query_embedding,
    _search_abstract_embeddings,
    _search_abstract_embeddings_bulk,
)


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
