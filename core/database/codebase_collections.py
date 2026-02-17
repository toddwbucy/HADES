"""Codebase knowledge graph collection management.

Defines the ArangoDB collections used by the codebase knowledge graph
and provides auto-creation on first use. Mirrors the pattern used by
Persephone (core/persephone/collections.py).

Collections store code files as graph nodes, AST-aligned chunks,
embedding vectors, and import-relationship edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodebaseCollections:
    """ArangoDB collection names for the codebase knowledge graph."""

    files: str = "codebase_files"
    chunks: str = "codebase_chunks"
    embeddings: str = "codebase_embeddings"
    edges: str = "codebase_edges"


CODEBASE_COLLECTIONS = CodebaseCollections()


def ensure_codebase_collections(
    client: Any,
    db_name: str,
    collections: CodebaseCollections | None = None,
) -> None:
    """Create missing codebase collections.

    Idempotent — skips collections that already exist.
    The edges collection is created as type=3 (edge collection).

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        collections: Collection names (defaults to CODEBASE_COLLECTIONS)
    """
    cols = collections or CODEBASE_COLLECTIONS

    resp = client.request("GET", f"/_db/{db_name}/_api/collection")
    if resp.get("error"):
        raise RuntimeError(f"Failed to list collections: {resp.get('errorMessage', 'unknown error')}")
    existing = {c["name"] for c in resp.get("result", [])}

    # Document collections (type=2)
    for name in (cols.files, cols.chunks, cols.embeddings):
        if name not in existing:
            create_resp = client.request(
                "POST",
                f"/_db/{db_name}/_api/collection",
                json={"name": name, "type": 2},
            )
            if create_resp.get("error"):
                raise RuntimeError(
                    f"Failed to create collection '{name}': {create_resp.get('errorMessage', 'unknown error')}"
                )
            logger.info("Created collection %s", name)

    # Edge collection (type=3)
    if cols.edges not in existing:
        create_resp = client.request(
            "POST",
            f"/_db/{db_name}/_api/collection",
            json={"name": cols.edges, "type": 3},
        )
        if create_resp.get("error"):
            raise RuntimeError(
                f"Failed to create edge collection '{cols.edges}': "
                f"{create_resp.get('errorMessage', 'unknown error')}"
            )
        logger.info("Created edge collection %s", cols.edges)
