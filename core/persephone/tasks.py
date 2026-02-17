"""Persephone task CRUD operations.

Pure business logic — receives an ArangoDB client and database name,
returns dicts or None. No CLI concerns here.
"""

from __future__ import annotations

import logging
import secrets
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from core.database.arango.optimized_client import ArangoHttpError
from core.persephone.collections import PERSEPHONE_COLLECTIONS, PersephoneCollections
from core.persephone.models import TaskCreate, TaskUpdate

# ArangoDB error number for document not found
_ARANGO_DOC_NOT_FOUND = 1202


def _generate_key() -> str:
    """Generate a task key: task_XXXXXX (6-char random hex)."""
    return f"task_{secrets.token_hex(3)}"


def _utcnow() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _is_not_found(exc: ArangoHttpError) -> bool:
    """Check if an ArangoHttpError represents a document-not-found."""
    return exc.status_code == 404 or exc.details.get("errorNum") == _ARANGO_DOC_NOT_FOUND


def create_task(
    client: Any,
    db_name: str,
    title: str,
    *,
    description: str | None = None,
    priority: str = "medium",
    type_: str = "task",
    labels: list[str] | None = None,
    parent_key: str | None = None,
    acceptance: str | None = None,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Create a new task document.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        title: Task title (required)
        description: Optional detailed description
        priority: One of critical|high|medium|low
        type_: One of task|bug|epic
        labels: Optional list of label strings
        parent_key: Optional parent task key (for epics)
        acceptance: Optional acceptance criteria text
        collections: Collection names override

    Returns:
        The created task document dict

    Raises:
        ValueError: If priority or type_ is invalid
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    now = _utcnow()
    key = _generate_key()

    try:
        validated = TaskCreate(
            title=title,
            description=description,
            priority=priority,
            type=type_,
            labels=labels or [],
            parent_key=parent_key,
            acceptance=acceptance,
            created_at=now,
            updated_at=now,
        )
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    doc: dict[str, Any] = {"_key": key, **validated.model_dump()}

    resp = client.request(
        "POST",
        f"/_db/{db_name}/_api/document/{cols.tasks}",
        json=doc,
    )

    # Return the full doc with ArangoDB metadata
    doc["_id"] = resp.get("_id", f"{cols.tasks}/{key}")
    doc["_rev"] = resp.get("_rev")

    # Best-effort activity log
    try:
        from core.persephone.logging import create_log

        create_log(
            client, db_name,
            action="task.created",
            task_key=key,
            details={"title": title, "priority": priority, "type": type_},
            collections=cols,
        )
    except Exception:
        logging.getLogger(__name__).warning("Failed to log task.created for %s", key)

    return doc


def get_task(
    client: Any,
    db_name: str,
    key: str,
    *,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any] | None:
    """Get a task by key.

    Returns:
        Task document dict, or None if not found

    Raises:
        ArangoHttpError: On network/auth/unexpected errors
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    try:
        return client.request(
            "GET",
            f"/_db/{db_name}/_api/document/{cols.tasks}/{key}",
        )
    except ArangoHttpError as e:
        if _is_not_found(e):
            return None
        raise


def list_tasks(
    client: Any,
    db_name: str,
    *,
    status: str | None = None,
    priority: str | None = None,
    type_: str | None = None,
    parent_key: str | None = None,
    limit: int = 50,
    collections: PersephoneCollections | None = None,
) -> list[dict[str, Any]]:
    """List tasks with optional filters.

    Note: db_name is accepted for API consistency but the AQL query runs
    against the database configured on the client (set by _make_client).

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name (unused — client is pre-configured)
        status: Filter by status
        priority: Filter by priority
        type_: Filter by type
        parent_key: Filter by parent task
        limit: Maximum results (default 50)
        collections: Collection names override

    Returns:
        List of task document dicts
    """
    _ = db_name  # Used by client config; kept for API consistency
    cols = collections or PERSEPHONE_COLLECTIONS
    filters = []
    bind_vars: dict[str, Any] = {"@col": cols.tasks, "limit": limit}

    if status:
        filters.append("FILTER doc.status == @status")
        bind_vars["status"] = status
    if priority:
        filters.append("FILTER doc.priority == @priority")
        bind_vars["priority"] = priority
    if type_:
        filters.append("FILTER doc.type == @type")
        bind_vars["type"] = type_
    if parent_key:
        filters.append("FILTER doc.parent_key == @parent_key")
        bind_vars["parent_key"] = parent_key

    filter_clause = "\n                    ".join(filters)
    aql = f"""
        FOR doc IN @@col
            {filter_clause}
            SORT doc.created_at DESC
            LIMIT @limit
            RETURN doc
    """

    return client.query(aql, bind_vars=bind_vars)


def update_task(
    client: Any,
    db_name: str,
    key: str,
    *,
    collections: PersephoneCollections | None = None,
    **fields: Any,
) -> dict[str, Any] | None:
    """Update a task by merging fields.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        key: Task document key
        **fields: Fields to update (status, priority, title, etc.)

    Returns:
        Updated task document, or None if not found

    Raises:
        ValueError: If an enum field value is invalid
        ArangoHttpError: On network/auth/unexpected errors
    """
    try:
        validated = TaskUpdate(**fields)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    cols = collections or PERSEPHONE_COLLECTIONS
    clean = validated.model_dump(exclude_unset=True)
    clean["updated_at"] = _utcnow()

    try:
        resp = client.request(
            "PATCH",
            f"/_db/{db_name}/_api/document/{cols.tasks}/{key}",
            json=clean,
            params={"returnNew": "true"},
        )
        result = resp.get("new", resp)
    except ArangoHttpError as e:
        if _is_not_found(e):
            return None
        raise

    # Best-effort activity log
    try:
        from core.persephone.logging import create_log

        create_log(
            client, db_name,
            action="task.updated",
            task_key=key,
            details={"fields": list(clean.keys())},
            collections=cols,
        )
    except Exception:
        logging.getLogger(__name__).warning("Failed to log task.updated for %s", key)

    return result


def delete_task(
    client: Any,
    db_name: str,
    key: str,
    *,
    collections: PersephoneCollections | None = None,
) -> bool:
    """Delete a task by key.

    Returns:
        True if deleted, False if not found

    Raises:
        ArangoHttpError: On network/auth/unexpected errors
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    try:
        client.request(
            "DELETE",
            f"/_db/{db_name}/_api/document/{cols.tasks}/{key}",
        )
        return True
    except ArangoHttpError as e:
        if _is_not_found(e):
            return False
        raise
