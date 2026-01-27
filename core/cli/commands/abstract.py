"""Abstract search commands for HADES CLI.

Search the 2.8M synced abstract embeddings and manage paper ingestion.
"""

from __future__ import annotations

import heapq
import re
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


def ingest_from_abstract(
    arxiv_ids: list[str],
    force: bool,
    start_time: float,
) -> CLIResponse:
    """Ingest papers identified from abstract search.

    This is a convenience wrapper around the standard ingest command,
    designed for the abstract search → ingest workflow.

    Args:
        arxiv_ids: ArXiv paper IDs to ingest
        force: Force reprocessing even if already exists
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with ingestion results
    """
    # Delegate to the existing ingest logic
    from core.cli.commands.ingest import ingest_papers

    return ingest_papers(
        arxiv_ids=arxiv_ids,
        pdf_paths=None,
        force=force,
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
            arxiv_id = r.get("arxiv_id", "")
            # Normalize for comparison
            base_id = re.sub(r"v\d+$", "", arxiv_id)
            if not any(re.sub(r"v\d+$", "", eid) == base_id for eid in exclude_ids):
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
        # Normalize arxiv_ids to keys
        keys = []
        for arxiv_id in arxiv_ids:
            base_id = re.sub(r"v\d+$", "", arxiv_id)
            key = base_id.replace(".", "_").replace("/", "_")
            keys.append(key)

        # Fetch all embeddings in one query
        aql = """
            FOR key IN @keys
                LET doc = DOCUMENT(CONCAT("arxiv_embeddings/", key))
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
        # Normalize arxiv_id to key format
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        key = base_id.replace(".", "_").replace("/", "_")

        # Fetch the embedding - only catch 404 (document not found)
        try:
            doc = client.get_document("arxiv_embeddings", key)
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
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        key = base_id.replace(".", "_").replace("/", "_")

        # Fetch the paper info - only catch 404 (document not found)
        try:
            doc = client.get_document("arxiv_abstracts", key)
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


def _get_query_embedding(text: str, config: Any) -> np.ndarray:
    """Generate embedding for query text."""
    from core.embedders.embedders_jina import JinaV4Embedder

    embedder = JinaV4Embedder(
        config={
            "device": config.device,
            "use_fp16": True,
        }
    )

    embedding = embedder.embed_texts([text], task="retrieval")[0]
    return embedding


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
            category_clause = """
                LET paper = DOCUMENT(CONCAT("arxiv_papers/", emb._key))
                FILTER paper != null AND @category_filter IN paper.categories
            """
            category_bind = {"category_filter": category_filter}

        progress(f"Processing embeddings in batches of {batch_size}...")

        while True:
            # Fetch batch of embeddings
            aql = f"""
                FOR emb IN arxiv_embeddings
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

                arxiv_id = item.get("arxiv_id", "")

                # Skip excluded paper (for similar search)
                if exclude_arxiv_id:
                    base_exclude = re.sub(r"v\d+$", "", exclude_arxiv_id)
                    base_current = re.sub(r"v\d+$", "", arxiv_id)
                    if base_exclude == base_current:
                        continue

                emb = np.array(item["embedding"], dtype=np.float32)
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                similarity = float(np.dot(query_norm, emb_norm))

                # Use heap to track top-k
                if len(top_results) < limit:
                    heapq.heappush(top_results, (similarity, arxiv_id, {"similarity": similarity}))
                elif similarity > top_results[0][0]:
                    heapq.heapreplace(top_results, (similarity, arxiv_id, {"similarity": similarity}))

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
        arxiv_ids = [r[1] for r in heap_results]

        if arxiv_ids:
            # Fetch abstracts and titles
            # Normalize both "." and "/" for legacy arxiv IDs like hep-th/9901001
            abstracts_aql = """
                FOR id IN @ids
                    LET key = SUBSTITUTE(SUBSTITUTE(id, ".", "_"), "/", "_")
                    LET abstract = DOCUMENT(CONCAT("arxiv_abstracts/", key))
                    LET paper = DOCUMENT(CONCAT("arxiv_papers/", key))
                    LET local = DOCUMENT(CONCAT("arxiv_metadata/", key))
                    RETURN {
                        arxiv_id: id,
                        title: abstract.title,
                        abstract: abstract.abstract,
                        categories: paper.categories,
                        local: local != null,
                        local_chunks: local.num_chunks
                    }
            """
            metadata = client.query(abstracts_aql, bind_vars={"ids": arxiv_ids})
            metadata_map = {m["arxiv_id"]: m for m in metadata if m}

            # If hybrid search, compute keyword scores and re-rank
            if hybrid_query:
                progress("Re-ranking with keyword matching...")
                heap_results = _hybrid_rerank(heap_results, metadata_map, hybrid_query)

            for sim, arxiv_id, extra in heap_results:
                meta = metadata_map.get(arxiv_id, {})

                # Truncate abstract for display
                abstract_text = meta.get("abstract", "") or ""
                if len(abstract_text) > 300:
                    abstract_snippet = abstract_text[:300].rsplit(" ", 1)[0] + "..."
                else:
                    abstract_snippet = abstract_text

                # After hybrid rerank, sim is combined score; use semantic score from extra
                semantic_score = extra.get("similarity", sim)

                result_entry = {
                    "arxiv_id": arxiv_id,
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
    for semantic_score, arxiv_id, _extra in heap_results:
        meta = metadata_map.get(arxiv_id, {})

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

        reranked.append((combined, arxiv_id, new_extra))

    # Sort by combined score descending
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked


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


def _get_bulk_query_embeddings(texts: list[str], config: Any) -> np.ndarray:
    """Generate embeddings for multiple query texts in one batch.

    Args:
        texts: List of query texts
        config: CLI configuration

    Returns:
        Array of shape (num_queries, embedding_dim)
    """
    from core.embedders.embedders_jina import JinaV4Embedder

    embedder = JinaV4Embedder(
        config={
            "device": config.device,
            "use_fp16": True,
        }
    )

    embeddings = embedder.embed_texts(texts, task="retrieval")
    return np.array(embeddings, dtype=np.float32)


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
            category_clause = """
                LET paper = DOCUMENT(CONCAT("arxiv_papers/", emb._key))
                FILTER paper != null AND @category_filter IN paper.categories
            """
            category_bind = {"category_filter": category_filter}

        progress(f"Processing embeddings in batches of {batch_size}...")

        while True:
            aql = f"""
                FOR emb IN arxiv_embeddings
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
                    for b_idx, arxiv_id in enumerate(batch_arxiv_ids):
                        sim = float(similarities[q_idx, b_idx])

                        if len(top_results[q_idx]) < limit:
                            heapq.heappush(top_results[q_idx], (sim, arxiv_id, {"similarity": sim}))
                        elif sim > top_results[q_idx][0][0]:
                            heapq.heapreplace(top_results[q_idx], (sim, arxiv_id, {"similarity": sim}))

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
            for _, arxiv_id, _ in heap:
                all_arxiv_ids.add(arxiv_id)

        # Fetch metadata for all results
        metadata_map: dict[str, dict] = {}
        if all_arxiv_ids:
            abstracts_aql = """
                FOR id IN @ids
                    LET key = SUBSTITUTE(SUBSTITUTE(id, ".", "_"), "/", "_")
                    LET abstract = DOCUMENT(CONCAT("arxiv_abstracts/", key))
                    LET paper = DOCUMENT(CONCAT("arxiv_papers/", key))
                    LET local = DOCUMENT(CONCAT("arxiv_metadata/", key))
                    RETURN {
                        arxiv_id: id,
                        title: abstract.title,
                        abstract: abstract.abstract,
                        categories: paper.categories,
                        local: local != null,
                        local_chunks: local.num_chunks
                    }
            """
            metadata = client.query(abstracts_aql, bind_vars={"ids": list(all_arxiv_ids)})
            metadata_map = {m["arxiv_id"]: m for m in metadata if m}

        # Build results dict keyed by query
        results_by_query: dict[str, list[dict[str, Any]]] = {}

        for q_idx, query in enumerate(queries):
            heap_results = sorted(top_results[q_idx], key=lambda x: x[0], reverse=True)
            results = []

            for sim, arxiv_id, extra in heap_results:
                meta = metadata_map.get(arxiv_id, {})

                abstract_text = meta.get("abstract", "") or ""
                if len(abstract_text) > 300:
                    abstract_snippet = abstract_text[:300].rsplit(" ", 1)[0] + "..."
                else:
                    abstract_snippet = abstract_text

                results.append({
                    "arxiv_id": arxiv_id,
                    "title": meta.get("title"),
                    "similarity": round(extra.get("similarity", sim), 4),
                    "abstract": abstract_snippet,
                    "categories": meta.get("categories", []),
                    "local": meta.get("local", False),
                    "local_chunks": meta.get("local_chunks"),
                })

            results_by_query[query] = results

        return results_by_query, total_processed

    finally:
        client.close()
