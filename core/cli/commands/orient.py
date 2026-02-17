"""Orient command — metadata-first context orientation.

Returns a compact map of what's available in a database so an LLM agent
can plan its query strategy from metadata alone, without loading content.

Inspired by the Recursive Language Models paper (Zhang, Kraska, Khattab):
the root model receives only constant-size metadata about what's available
before deciding what to examine.
"""

from __future__ import annotations

from typing import Any

from core.cli.commands.database import _make_client
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    success_response,
)
from core.database.collections import PROFILES, CollectionProfile


def _safe_count(client: Any, collection: str) -> int | None:
    """Count documents in a collection, returning None if it doesn't exist."""
    try:
        result = client.query(
            "RETURN LENGTH(@@col)",
            bind_vars={"@col": collection},
        )
        return result[0] if result else 0
    except Exception:
        return None


def _profile_stats(client: Any, profile: CollectionProfile) -> dict[str, int | None]:
    """Get document counts for a collection profile."""
    return {
        "metadata": _safe_count(client, profile.metadata),
        "chunks": _safe_count(client, profile.chunks),
        "embeddings": _safe_count(client, profile.embeddings),
    }


def _recent_papers(client: Any, profile: CollectionProfile, limit: int = 10) -> list[dict[str, Any]]:
    """Get recent papers from a profile's metadata collection."""
    try:
        return client.query(
            """
            FOR doc IN @@col
                SORT doc.processing_timestamp DESC
                LIMIT @limit
                RETURN {
                    id: doc.arxiv_id || doc.document_id || doc._key,
                    title: doc.title,
                    categories: doc.categories
                }
            """,
            bind_vars={"@col": profile.metadata, "limit": limit},
        )
    except Exception:
        return []


def orient(start_time: float, *, papers_limit: int = 10) -> CLIResponse:
    """Build a compact orientation map for the current database.

    Returns:
        CLIResponse with database name, collection profiles with counts,
        all user collections, and recent papers from the default profile.
    """
    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="orient",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        # 1. All user collections with counts
        collection_names = client.query(
            "FOR c IN COLLECTIONS() FILTER !STARTS_WITH(c.name, '_') SORT c.name RETURN c.name"
        )

        collection_counts: list[dict[str, Any]] = []
        for name in collection_names:
            count = _safe_count(client, name)
            if count is not None:
                collection_counts.append({"name": name, "count": count})

        # 2. Profile stats (which known profiles have data?)
        profiles: dict[str, Any] = {}
        for profile_name, profile in PROFILES.items():
            stats = _profile_stats(client, profile)
            # Only include profiles that have at least one existing collection
            if any(v is not None for v in stats.values()):
                profiles[profile_name] = {
                    "collections": {
                        "metadata": profile.metadata,
                        "chunks": profile.chunks,
                        "embeddings": profile.embeddings,
                    },
                    "counts": stats,
                }

        # 3. Recent papers from profiles that have metadata
        recent: dict[str, list[dict[str, Any]]] = {}
        for profile_name, profile in PROFILES.items():
            if profiles.get(profile_name, {}).get("counts", {}).get("metadata"):
                papers = _recent_papers(client, profile, limit=papers_limit)
                if papers:
                    recent[profile_name] = papers

        # 4. Persephone stats (if present)
        persephone: dict[str, int | None] | None = None
        persephone_collections = ["persephone_tasks", "persephone_sessions", "persephone_handoffs", "persephone_logs"]
        if any(name in collection_names for name in persephone_collections):
            persephone = {}
            for name in persephone_collections:
                count = _safe_count(client, name)
                if count is not None:
                    persephone[name.replace("persephone_", "")] = count

        # 5. Codebase stats (if present)
        codebase: dict[str, int | None] | None = None
        codebase_collections = ["codebase_files", "codebase_chunks", "codebase_edges"]
        if any(name in collection_names for name in codebase_collections):
            codebase = {}
            for name in codebase_collections:
                count = _safe_count(client, name)
                if count is not None:
                    codebase[name.replace("codebase_", "")] = count

        result: dict[str, Any] = {
            "database": db_name,
            "profiles": profiles,
            "recent_papers": recent,
            "all_collections": collection_counts,
            "total_collections": len(collection_counts),
        }
        if persephone:
            result["persephone"] = persephone
        if codebase:
            result["codebase"] = codebase

        return success_response(
            command="orient",
            data=result,
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="orient",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()
