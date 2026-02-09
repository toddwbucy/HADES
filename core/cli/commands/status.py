"""HADES Status Command - Unified system overview for fresh workspaces.

Provides a single-command audit of the entire HADES system:
- Embedding service status
- Database connection and collection stats
- Recently ingested papers
- Sync status

Designed for AI agents to quickly understand what's available.
"""

from __future__ import annotations

import time
from typing import Any

from core.cli.config import get_arango_config, get_config
from core.cli.output import CLIResponse
from core.database.collections import get_profile, list_profiles


def get_status(start_time: float) -> CLIResponse:
    """Get comprehensive system status.

    Aggregates:
    - Version info
    - Embedding service status
    - Database connection and stats per collection
    - Recently ingested papers
    - Last sync timestamp

    Returns:
        CLIResponse with complete system status
    """
    from importlib.metadata import version as get_version

    status_data: dict[str, Any] = {}

    # Version
    try:
        status_data["version"] = get_version("hades")
    except Exception:
        status_data["version"] = "unknown"

    # Embedding service status
    status_data["embedding_service"] = _get_embedding_service_status()

    # Database status
    status_data["database"] = _get_database_status()

    # Recent papers (from arxiv collection)
    status_data["recent_papers"] = _get_recent_papers(limit=5)

    # Last sync timestamp
    status_data["last_sync"] = _get_last_sync()

    return CLIResponse(
        success=True,
        command="status",
        data=status_data,
        metadata={
            "duration_ms": int((time.time() - start_time) * 1000),
        },
    )


def _get_embedding_service_status() -> dict[str, Any]:
    """Check embedding service health."""
    try:
        from core.services.embedder_client import EmbedderClient

        config = get_config()
        client = EmbedderClient(
            socket_path=config.embedding.service_socket,
            timeout=5.0,
            fallback_to_local=False,
        )

        try:
            if client.is_service_available():
                health = client.get_health()
                return {
                    "status": "running",
                    "model_loaded": health.get("model_loaded", False),
                    "device": health.get("device", "unknown"),
                    "idle_seconds": health.get("idle_seconds"),
                }
            else:
                return {"status": "unavailable", "error": "Service not responding"}
        finally:
            client.close()

    except Exception as e:
        return {"status": "error", "error": str(e)}


def _get_database_status() -> dict[str, Any]:
    """Check database connection and get stats per collection."""
    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

        config = get_config()
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
            collections_status = {}

            for profile_name in list_profiles():
                try:
                    profile = get_profile(profile_name)
                    stats = _get_collection_stats(client, profile)
                    if stats:  # Only include if collection exists
                        collections_status[profile_name] = stats
                except Exception as e:
                    # Collection might not exist — log other errors for debugging
                    import logging

                    logging.debug("Could not get stats for %s: %s", profile_name, e)

            return {
                "connected": True,
                "collections": collections_status,
            }

        finally:
            client.close()

    except Exception as e:
        return {"connected": False, "error": str(e)}


def _get_collection_stats(client: Any, profile: Any) -> dict[str, int] | None:
    """Get stats for a specific collection profile."""
    try:
        # Check if metadata collection exists and has documents
        result = client.query(
            f"RETURN LENGTH({profile.metadata})",
            batch_size=1,
        )
        papers_count = result[0] if result else 0

        if papers_count == 0:
            return None  # Collection is empty or doesn't exist

        # Get chunks count
        chunks_result = client.query(
            f"RETURN LENGTH({profile.chunks})",
            batch_size=1,
        )
        chunks_count = chunks_result[0] if chunks_result else 0

        # Get embeddings count
        embeddings_result = client.query(
            f"RETURN LENGTH({profile.embeddings})",
            batch_size=1,
        )
        embeddings_count = embeddings_result[0] if embeddings_result else 0

        return {
            "papers": papers_count,
            "chunks": chunks_count,
            "embeddings": embeddings_count,
        }

    except Exception:
        return None


def _get_recent_papers(limit: int = 5) -> list[dict[str, Any]]:
    """Get recently ingested papers from arxiv collection."""
    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

        config = get_config()
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
            profile = get_profile("arxiv")
            results = client.query(
                f"""
                FOR doc IN {profile.metadata}
                    SORT doc.processing_timestamp DESC
                    LIMIT @limit
                    RETURN {{
                        arxiv_id: doc.arxiv_id,
                        document_id: doc.document_id,
                        title: doc.title,
                        num_chunks: doc.num_chunks,
                        ingested: doc.processing_timestamp
                    }}
                """,
                {"limit": limit},
            )
            return results

        finally:
            client.close()

    except Exception:
        return []


def _get_last_sync() -> str | None:
    """Get last arxiv sync timestamp."""
    try:
        from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

        config = get_config()
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
            # Try to get sync metadata
            result = client.query(
                """
                FOR doc IN sync_metadata
                    FILTER doc._key == "sync_state"
                    RETURN doc.last_sync_date
                """,
                batch_size=1,
            )
            return result[0] if result else None

        finally:
            client.close()

    except Exception:
        return None
