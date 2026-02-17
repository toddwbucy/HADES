"""Persephone activity logging.

Writes structured log entries to persephone_logs for every significant
Persephone action (task create/update/transition, handoff create,
session start/end). Provides the audit trail for graph-node project
management.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from core.persephone.collections import PERSEPHONE_COLLECTIONS, PersephoneCollections
from core.persephone.models import LogCreate

logger = logging.getLogger(__name__)


def _generate_log_key() -> str:
    """Generate a log key: log_XXXXXX (6-char random hex)."""
    return f"log_{secrets.token_hex(3)}"


def _utcnow() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def create_log(
    client: Any,
    db_name: str,
    *,
    action: str,
    task_key: str | None = None,
    session_key: str | None = None,
    details: dict | None = None,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Write a structured activity log entry to persephone_logs.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        action: Action type (e.g., "task.created", "task.transitioned")
        task_key: Related task key (optional)
        session_key: Related session key (optional)
        details: Additional structured details (optional)
        collections: Collection names override

    Returns:
        The created log document dict

    Raises:
        ValueError: If action is empty
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    now = _utcnow()
    key = _generate_log_key()

    try:
        validated = LogCreate(
            action=action,
            task_key=task_key,
            session_key=session_key,
            details=details,
            created_at=now,
        )
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    doc: dict[str, Any] = {"_key": key, **validated.model_dump()}

    resp = client.request(
        "POST",
        f"/_db/{db_name}/_api/document/{cols.logs}",
        json=doc,
    )

    doc["_id"] = resp.get("_id", f"{cols.logs}/{key}")
    doc["_rev"] = resp.get("_rev")
    return doc


def list_logs(
    client: Any,
    db_name: str,
    *,
    task_key: str | None = None,
    session_key: str | None = None,
    action: str | None = None,
    limit: int = 50,
    collections: PersephoneCollections | None = None,
) -> list[dict[str, Any]]:
    """Query activity logs with optional filters.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        task_key: Filter by task key
        session_key: Filter by session key
        action: Filter by action type (e.g., "task.transitioned")
        limit: Maximum results (default 50)
        collections: Collection names override

    Returns:
        List of log document dicts, newest first
    """
    _ = db_name  # Used by client config; kept for API consistency
    cols = collections or PERSEPHONE_COLLECTIONS
    if limit <= 0:
        raise ValueError("limit must be positive")
    filters = []
    bind_vars: dict[str, Any] = {"@col": cols.logs, "limit": limit}

    if task_key:
        filters.append("FILTER doc.task_key == @task_key")
        bind_vars["task_key"] = task_key
    if session_key:
        filters.append("FILTER doc.session_key == @session_key")
        bind_vars["session_key"] = session_key
    if action:
        filters.append("FILTER doc.action == @action")
        bind_vars["action"] = action

    filter_clause = "\n                    ".join(filters)
    aql = f"""
        FOR doc IN @@col
            {filter_clause}
            SORT doc.created_at DESC
            LIMIT @limit
            RETURN doc
    """

    return client.query(aql, bind_vars=bind_vars)
