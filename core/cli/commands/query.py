"""Query commands for HADES CLI - semantic search over the knowledge base."""

from __future__ import annotations

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


def semantic_query(
    search_text: str,
    limit: int,
    start_time: float,
    paper_filter: str | None = None,
    context: int = 0,
    cite_only: bool = False,
) -> CLIResponse:
    """Perform semantic search over stored chunks.

    Args:
        search_text: Query text to search for
        limit: Maximum number of results
        start_time: Start time for duration calculation
        paper_filter: Optional arxiv ID to limit search to specific paper
        context: Number of adjacent chunks to include (0 = none)
        cite_only: If True, return minimal citation format

    Returns:
        CLIResponse with search results
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="query",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    progress("Generating query embedding...")

    try:
        # Generate query embedding
        query_embedding = _get_query_embedding(search_text, config)

        if paper_filter:
            progress(f"Searching paper {paper_filter} for {limit} most similar chunks...")
        else:
            progress(f"Searching {limit} most similar chunks...")

        # Search database
        results = _search_embeddings(query_embedding, limit, config, paper_filter=paper_filter)

        # Add context chunks if requested
        if context > 0 and results:
            results = _add_context_chunks(results, context, config)

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
            command="query",
            data=response_data,
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="query",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )


def get_paper_chunks(
    paper_id: str,
    limit: int,
    start_time: float,
) -> CLIResponse:
    """Get all chunks for a specific paper.

    Args:
        paper_id: ArXiv ID or document ID
        limit: Maximum number of chunks to return
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with paper chunks
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="query",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    progress(f"Retrieving chunks for paper: {paper_id}")

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
            # Normalize paper ID for lookup (strip version suffix like "v1", "v2")
            import re
            base_id = re.sub(r"v\d+$", "", paper_id)
            sanitized_id = base_id.replace(".", "_").replace("/", "_")

            # Query chunks for this paper
            aql = """
                FOR chunk IN arxiv_abstract_chunks
                    FILTER chunk.paper_key == @paper_key OR chunk.document_id == @paper_id
                    SORT chunk.chunk_index
                    LIMIT @limit
                    RETURN {
                        chunk_index: chunk.chunk_index,
                        total_chunks: chunk.total_chunks,
                        text: chunk.text,
                        start_char: chunk.start_char,
                        end_char: chunk.end_char
                    }
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
                    command="query",
                    code=ErrorCode.PAPER_NOT_FOUND,
                    message=f"No chunks found for paper: {paper_id}",
                    start_time=start_time,
                )

            return success_response(
                command="query",
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
            command="query",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def _get_query_embedding(text: str, config: Any) -> np.ndarray:
    """Generate embedding for query text.

    Uses the same embedder as document processing for consistent similarity search.
    """
    from core.embedders.embedders_jina import JinaV4Embedder

    embedder = JinaV4Embedder(
        config={
            "device": config.device,
            "use_fp16": True,
        }
    )

    # Use retrieval task for query embeddings
    embedding = embedder.embed_texts([text], task="retrieval")[0]

    return embedding


def _search_embeddings(
    query_embedding: np.ndarray,
    limit: int,
    config: Any,
    paper_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search for similar embeddings in the database.

    Uses cosine similarity for matching.

    Args:
        query_embedding: Query vector
        limit: Maximum results to return
        config: CLI configuration
        paper_filter: Optional arxiv ID to limit search to specific paper
    """
    import re

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

    # Normalize paper filter if provided
    paper_key_filter = None
    if paper_filter:
        base_id = re.sub(r"v\d+$", "", paper_filter)
        paper_key_filter = base_id.replace(".", "_").replace("/", "_")

    try:
        all_embeddings = []

        # Search full paper chunks (from ingested PDFs)
        try:
            if paper_key_filter:
                aql_chunks = """
                    FOR emb IN arxiv_abstract_embeddings
                        FILTER emb.chunk_key != null
                        FILTER emb.paper_key == @paper_key
                        LET chunk = DOCUMENT(CONCAT("arxiv_abstract_chunks/", emb.chunk_key))
                        LET meta = DOCUMENT(CONCAT("arxiv_metadata/", emb.paper_key))
                        RETURN {
                            paper_key: emb.paper_key,
                            embedding: emb.embedding,
                            text: chunk.text,
                            chunk_index: chunk.chunk_index,
                            total_chunks: chunk.total_chunks,
                            title: meta.title,
                            arxiv_id: meta.arxiv_id,
                            source: "full_paper"
                        }
                """
                chunk_results = client.query(aql_chunks, bind_vars={"paper_key": paper_key_filter})
            else:
                aql_chunks = """
                    FOR emb IN arxiv_abstract_embeddings
                        FILTER emb.chunk_key != null
                        LET chunk = DOCUMENT(CONCAT("arxiv_abstract_chunks/", emb.chunk_key))
                        LET meta = DOCUMENT(CONCAT("arxiv_metadata/", emb.paper_key))
                        RETURN {
                            paper_key: emb.paper_key,
                            embedding: emb.embedding,
                            text: chunk.text,
                            chunk_index: chunk.chunk_index,
                            total_chunks: chunk.total_chunks,
                            title: meta.title,
                            arxiv_id: meta.arxiv_id,
                            source: "full_paper"
                        }
                """
                chunk_results = client.query(aql_chunks)
            all_embeddings.extend(chunk_results or [])
        except Exception as e:
            # Collection may not exist - continue with other sources
            progress(f"Note: Could not query full paper chunks: {e}")

        # Search synced abstracts (from sync command) - skip if filtering by paper
        # (abstracts are single chunks, full paper search is more useful when filtering)
        if not paper_key_filter:
            try:
                aql_abstracts = """
                    FOR emb IN arxiv_abstract_embeddings
                        FILTER emb.text_type == "abstract"
                        LET abstract = DOCUMENT(CONCAT("arxiv_abstracts/", emb.paper_key))
                        RETURN {
                            paper_key: emb.paper_key,
                            embedding: emb.embedding,
                            text: abstract.abstract,
                            chunk_index: 0,
                            total_chunks: 1,
                            title: abstract.title,
                            arxiv_id: abstract.arxiv_id,
                            source: "abstract"
                        }
                """
                abstract_results = client.query(aql_abstracts)
                all_embeddings.extend(abstract_results or [])
            except Exception as e:
                # Collection may not exist - continue with other sources
                progress(f"Note: Could not query synced abstracts: {e}")

        if not all_embeddings:
            return []

        # Compute cosine similarities
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        scored_results = []
        for item in all_embeddings:
            emb = np.array(item["embedding"], dtype=np.float32)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            similarity = float(np.dot(query_norm, emb_norm))

            scored_results.append({
                "similarity": round(similarity, 4),
                "arxiv_id": item.get("arxiv_id"),
                "title": item.get("title"),
                "source": item.get("source", "full_paper"),
                "chunk_index": item.get("chunk_index"),
                "total_chunks": item.get("total_chunks"),
                "text": item.get("text"),
            })

        # Sort by similarity and take top results
        scored_results.sort(key=lambda x: x["similarity"], reverse=True)

        return scored_results[:limit]

    finally:
        client.close()


def _add_context_chunks(
    results: list[dict[str, Any]],
    context: int,
    config: Any,
) -> list[dict[str, Any]]:
    """Add neighboring chunks to each result for context.

    Args:
        results: Search results with chunk_index
        context: Number of chunks before/after to include

    Returns:
        Results with context_before and context_after lists
    """
    import re

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
            base_id = re.sub(r"v\d+$", "", arxiv_id)
            paper_key = base_id.replace(".", "_").replace("/", "_")

            # Fetch context chunks
            context_before = []
            context_after = []

            try:
                # Get chunks before
                if chunk_index > 0:
                    aql_before = """
                        FOR chunk IN arxiv_abstract_chunks
                            FILTER chunk.paper_key == @paper_key
                            FILTER chunk.chunk_index >= @start_idx
                            FILTER chunk.chunk_index < @current_idx
                            SORT chunk.chunk_index
                            RETURN {
                                chunk_index: chunk.chunk_index,
                                text: chunk.text
                            }
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
                aql_after = """
                    FOR chunk IN arxiv_abstract_chunks
                        FILTER chunk.paper_key == @paper_key
                        FILTER chunk.chunk_index > @current_idx
                        FILTER chunk.chunk_index <= @end_idx
                        SORT chunk.chunk_index
                        RETURN {
                            chunk_index: chunk.chunk_index,
                            text: chunk.text
                        }
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
                # If context fetch fails, just continue without context
                pass

            result["context_before"] = context_before
            result["context_after"] = context_after

        return results

    finally:
        client.close()


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

        citations.append({
            "arxiv_id": result.get("arxiv_id"),
            "title": result.get("title"),
            "similarity": result.get("similarity"),
            "chunk_index": result.get("chunk_index"),
            "quote": quote,
        })

    return citations
