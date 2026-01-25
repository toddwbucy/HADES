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
) -> CLIResponse:
    """Perform semantic search over stored chunks.

    Args:
        search_text: Query text to search for
        limit: Maximum number of results
        start_time: Start time for duration calculation

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

        progress(f"Searching {limit} most similar chunks...")

        # Search database
        results = _search_embeddings(query_embedding, limit, config)

        return success_response(
            command="query",
            data={
                "query": search_text,
                "results": results,
            },
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
            # Normalize paper ID for lookup
            sanitized_id = paper_id.replace(".", "_").replace("/", "_")

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
) -> list[dict[str, Any]]:
    """Search for similar embeddings in the database.

    Uses cosine similarity for matching.
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
        all_embeddings = []

        # Search full paper chunks (from ingested PDFs)
        try:
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
        except Exception:
            pass

        # Search synced abstracts (from sync command)
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
        except Exception:
            pass

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
