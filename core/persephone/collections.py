"""Persephone collection management.

Defines the ArangoDB collections used by Persephone and provides
auto-creation on first use. Collections are kept separate from the
production CollectionProfile used by the document pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersephoneCollections:
    """ArangoDB collection names for Persephone task management."""

    tasks: str = "persephone_tasks"
    sessions: str = "persephone_sessions"
    handoffs: str = "persephone_handoffs"
    logs: str = "persephone_logs"
    edges: str = "persephone_edges"


PERSEPHONE_COLLECTIONS = PersephoneCollections()


def ensure_collections(
    client: Any,
    db_name: str,
    collections: PersephoneCollections | None = None,
) -> None:
    """Create missing Persephone collections.

    Idempotent — skips collections that already exist.
    The edges collection is created as type=3 (edge collection).

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        collections: Collection names (defaults to PERSEPHONE_COLLECTIONS)
    """
    cols = collections or PERSEPHONE_COLLECTIONS

    # Get existing collection names
    resp = client.request("GET", f"/_db/{db_name}/_api/collection")
    existing = {c["name"] for c in resp.get("result", [])}

    # Document collections (type=2)
    for name in (cols.tasks, cols.sessions, cols.handoffs, cols.logs):
        if name not in existing:
            client.request(
                "POST",
                f"/_db/{db_name}/_api/collection",
                json={"name": name, "type": 2},
            )
            logger.info("Created collection %s", name)

    # Edge collection (type=3)
    if cols.edges not in existing:
        client.request(
            "POST",
            f"/_db/{db_name}/_api/collection",
            json={"name": cols.edges, "type": 3},
        )
        logger.info("Created edge collection %s", cols.edges)
