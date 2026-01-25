"""Database management commands for HADES CLI."""

from __future__ import annotations

from core.cli.config import get_arango_config, get_config
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    success_response,
)


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
            command="list",
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
                command="list",
                data={"papers": results},
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="list",
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
            command="stats",
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
            paper_count = client.query(
                "RETURN LENGTH(arxiv_metadata)"
            )
            stats["total_papers"] = paper_count[0] if paper_count else 0

            # Count chunks
            chunk_count = client.query(
                "RETURN LENGTH(arxiv_abstract_chunks)"
            )
            stats["total_chunks"] = chunk_count[0] if chunk_count else 0

            # Count embeddings
            embedding_count = client.query(
                "RETURN LENGTH(arxiv_abstract_embeddings)"
            )
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
                command="stats",
                data=stats,
                start_time=start_time,
            )

        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="stats",
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
            command="check",
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
            # Normalize paper ID
            sanitized_id = arxiv_id.replace(".", "_").replace("/", "_")

            # Try to get the document
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
            except Exception:
                exists = False
                paper_info = None

            return success_response(
                command="check",
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
            command="check",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
