"""Database management commands for HADES CLI.

Consolidates: list, stats, check, query, chunks, create, delete commands.
"""

from __future__ import annotations

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

# =============================================================================
# Paper Management Commands (list, stats, check)
# =============================================================================


def list_stored_papers(
    limit: int,
    category: str | None,
    start_time: float,
) -> CLIResponse:
    """List papers stored in the database.

    Args:
        limit: Maximum number of papers to return
        category: Optional arxiv category filter
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with paper list
    """
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
            # Build query with optional category filter
            if category:
                aql = """
                    FOR doc IN arxiv_metadata
                        FILTER @category IN doc.categories
                        SORT doc.processing_timestamp DESC
                        LIMIT @limit
                        RETURN {
                            arxiv_id: doc.arxiv_id,
                            document_id: doc.document_id,
                            title: doc.title,
                            authors: doc.authors,
                            categories: doc.categories,
                            num_chunks: doc.num_chunks,
                            source: doc.source,
                            processing_timestamp: doc.processing_timestamp
                        }
                """
                bind_vars = {"category": category, "limit": limit}
            else:
                aql = """
                    FOR doc IN arxiv_metadata
                        SORT doc.processing_timestamp DESC
                        LIMIT @limit
                        RETURN {
                            arxiv_id: doc.arxiv_id,
                            document_id: doc.document_id,
                            title: doc.title,
                            authors: doc.authors,
                            categories: doc.categories,
                            num_chunks: doc.num_chunks,
                            source: doc.source,
                            processing_timestamp: doc.processing_timestamp
                        }
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


def get_stats(start_time: float) -> CLIResponse:
    """Get database statistics.

    Args:
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with database statistics
    """
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
            stats = {}

            # Count papers
            paper_count = client.query("RETURN LENGTH(arxiv_metadata)")
            stats["total_papers"] = paper_count[0] if paper_count else 0

            # Count chunks
            chunk_count = client.query("RETURN LENGTH(arxiv_abstract_chunks)")
            stats["total_chunks"] = chunk_count[0] if chunk_count else 0

            # Count embeddings
            embedding_count = client.query("RETURN LENGTH(arxiv_abstract_embeddings)")
            stats["total_embeddings"] = embedding_count[0] if embedding_count else 0

            # Get category distribution
            category_aql = """
                FOR doc IN arxiv_metadata
                    FOR cat IN (doc.categories || [])
                        COLLECT category = cat WITH COUNT INTO count
                        SORT count DESC
                        LIMIT 10
                        RETURN {category: category, count: count}
            """
            categories = client.query(category_aql)
            stats["top_categories"] = categories

            # Get source distribution
            source_aql = """
                FOR doc IN arxiv_metadata
                    COLLECT source = (doc.source || "arxiv") WITH COUNT INTO count
                    RETURN {source: source, count: count}
            """
            sources = client.query(source_aql)
            stats["sources"] = sources

            # Get recent papers
            recent_aql = """
                FOR doc IN arxiv_metadata
                    SORT doc.processing_timestamp DESC
                    LIMIT 5
                    RETURN {
                        arxiv_id: doc.arxiv_id,
                        title: doc.title,
                        processing_timestamp: doc.processing_timestamp
                    }
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


def check_paper_exists(arxiv_id: str, start_time: float) -> CLIResponse:
    """Check if a paper exists in the database.

    Args:
        arxiv_id: ArXiv paper ID to check
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with existence check result
    """
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
            # Normalize paper ID (strip version suffix like "v1", "v2")
            base_id = re.sub(r"v\d+$", "", arxiv_id)
            sanitized_id = base_id.replace(".", "_").replace("/", "_")

            # Try to get the document
            from core.database.arango.optimized_client import ArangoHttpError

            try:
                doc = client.get_document("arxiv_metadata", sanitized_id)
                exists = True
                paper_info = {
                    "arxiv_id": doc.get("arxiv_id"),
                    "title": doc.get("title"),
                    "num_chunks": doc.get("num_chunks"),
                    "processing_timestamp": doc.get("processing_timestamp"),
                    "status": doc.get("status"),
                }
            except ArangoHttpError as e:
                if e.status_code == 404:
                    exists = False
                    paper_info = None
                else:
                    raise

            return success_response(
                command="database.check",
                data={
                    "arxiv_id": arxiv_id,
                    "exists": exists,
                    "paper": paper_info,
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
            command="database.query",
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
            command="database.chunks",
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
) -> list[dict[str, Any]]:
    """Search for similar embeddings in the database.

    Uses cosine similarity for matching.

    Args:
        query_embedding: Query vector
        limit: Maximum results to return
        config: CLI configuration
        paper_filter: Optional arxiv ID to limit search to specific paper
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
