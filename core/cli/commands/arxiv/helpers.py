"""Shared utilities for arxiv abstract search operations.

Contains embedding and search helper functions used by abstract.py
for semantic search over the 2.8M synced abstract embeddings.
"""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np

from core.cli.config import get_arango_config
from core.cli.output import progress
from core.database.collections import get_profile
from core.database.keys import normalize_document_key, strip_version


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
