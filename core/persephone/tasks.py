"""Persephone task CRUD operations.

Pure business logic — receives an ArangoDB client and database name,
returns dicts or None. No CLI concerns here.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

from core.persephone.collections import PERSEPHONE_COLLECTIONS, PersephoneCollections

# Valid enum values
VALID_STATUSES = {"open", "in_progress", "in_review", "closed", "blocked"}
VALID_PRIORITIES = {"critical", "high", "medium", "low"}
VALID_TYPES = {"task", "bug", "epic"}


def _generate_key() -> str:
    """Generate a task key: task_XXXXXX (6-char random hex)."""
    return f"task_{secrets.token_hex(3)}"


def _utcnow() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


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
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority '{priority}': must be one of {VALID_PRIORITIES}")
    if type_ not in VALID_TYPES:
        raise ValueError(f"Invalid type '{type_}': must be one of {VALID_TYPES}")

    cols = collections or PERSEPHONE_COLLECTIONS
    now = _utcnow()
    key = _generate_key()

    doc: dict[str, Any] = {
        "_key": key,
        "title": title,
        "description": description,
        "status": "open",
        "priority": priority,
        "type": type_,
        "labels": labels or [],
        "parent_key": parent_key,
        "acceptance": acceptance,
        "minor": False,
        "created_at": now,
        "updated_at": now,
    }

    resp = client.request(
        "POST",
        f"/_db/{db_name}/_api/document/{cols.tasks}",
        json=doc,
    )

    # Return the full doc with ArangoDB metadata
    doc["_id"] = resp.get("_id", f"{cols.tasks}/{key}")
    doc["_rev"] = resp.get("_rev")
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
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    try:
        resp = client.request(
            "GET",
            f"/_db/{db_name}/_api/document/{cols.tasks}/{key}",
        )
        return resp
    except Exception:
        return None


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

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        status: Filter by status
        priority: Filter by priority
        type_: Filter by type
        parent_key: Filter by parent task
        limit: Maximum results (default 50)
        collections: Collection names override

    Returns:
        List of task document dicts
    """
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
    """
    # Validate enum fields if provided
    if "status" in fields and fields["status"] not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{fields['status']}': must be one of {VALID_STATUSES}")
    if "priority" in fields and fields["priority"] not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority '{fields['priority']}': must be one of {VALID_PRIORITIES}")
    if "type" in fields and fields["type"] not in VALID_TYPES:
        raise ValueError(f"Invalid type '{fields['type']}': must be one of {VALID_TYPES}")

    cols = collections or PERSEPHONE_COLLECTIONS
    fields["updated_at"] = _utcnow()

    try:
        resp = client.request(
            "PATCH",
            f"/_db/{db_name}/_api/document/{cols.tasks}/{key}",
            json=fields,
            params={"returnNew": "true"},
        )
        return resp.get("new", resp)
    except Exception:
        return None


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
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    try:
        client.request(
            "DELETE",
            f"/_db/{db_name}/_api/document/{cols.tasks}/{key}",
        )
        return True
    except Exception:
        return False
