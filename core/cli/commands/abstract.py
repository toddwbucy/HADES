"""Abstract search commands for HADES CLI.

Search the 2.8M synced abstract embeddings and manage paper ingestion.
"""

from __future__ import annotations

import heapq
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
) -> CLIResponse:
    """Search the synced abstract embeddings (2.8M vectors).

    Args:
        query: Search query text
        limit: Maximum number of results
        start_time: Start time for duration calculation
        category: Optional arxiv category filter (e.g., "cs.AI")

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

    progress("Generating query embedding...")

    try:
        # Generate query embedding
        query_embedding = _get_query_embedding(query, config)

        progress(f"Searching {limit} most similar abstracts...")

        # Search the abstract embeddings
        results = _search_abstract_embeddings(
            query_embedding,
            limit,
            config,
            category_filter=category,
        )

        return success_response(
            command="abstract.search",
            data={
                "query": query,
                "results": results,
                "total_searched": results[0]["total_searched"] if results else 0,
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
) -> list[dict[str, Any]]:
    """Search abstract embeddings with batched processing.

    Processes 2.8M embeddings in batches to avoid memory issues.
    Uses a min-heap to efficiently track top-k results across batches.

    Args:
        query_embedding: Query vector (2048-dim)
        limit: Number of results to return
        config: CLI configuration
        category_filter: Optional category to filter by

    Returns:
        List of results with similarity scores and local status
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

        # Build category filter if specified
        category_clause = ""
        if category_filter:
            # Join with arxiv_papers to filter by category
            category_clause = f"""
                LET paper = DOCUMENT(CONCAT("arxiv_papers/", emb._key))
                FILTER paper != null AND "{category_filter}" IN paper.categories
            """

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
                    bind_vars={"offset": offset, "batch_size": batch_size},
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

                emb = np.array(item["embedding"], dtype=np.float32)
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                similarity = float(np.dot(query_norm, emb_norm))

                arxiv_id = item.get("arxiv_id", "")

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
            abstracts_aql = """
                FOR id IN @ids
                    LET abstract = DOCUMENT(CONCAT("arxiv_abstracts/", SUBSTITUTE(id, ".", "_")))
                    LET paper = DOCUMENT(CONCAT("arxiv_papers/", SUBSTITUTE(id, ".", "_")))
                    LET local = DOCUMENT(CONCAT("arxiv_metadata/", SUBSTITUTE(id, ".", "_")))
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

            for sim, arxiv_id, _ in heap_results:
                meta = metadata_map.get(arxiv_id, {})

                # Truncate abstract for display
                abstract_text = meta.get("abstract", "") or ""
                if len(abstract_text) > 300:
                    abstract_snippet = abstract_text[:300].rsplit(" ", 1)[0] + "..."
                else:
                    abstract_snippet = abstract_text

                results.append({
                    "arxiv_id": arxiv_id,
                    "title": meta.get("title"),
                    "similarity": round(sim, 4),
                    "abstract": abstract_snippet,
                    "categories": meta.get("categories", []),
                    "local": meta.get("local", False),
                    "local_chunks": meta.get("local_chunks"),
                    "total_searched": total_processed,
                })

        return results

    finally:
        client.close()
