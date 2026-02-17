"""Persephone workflow state machine.

Governs task lifecycle transitions with guards that enforce rules like
"reviewer != implementer" and dependency blocking. All state changes
create audit edges in persephone_edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.persephone.collections import PERSEPHONE_COLLECTIONS, PersephoneCollections
from core.persephone.sessions import (
    create_session_task_edge,
    get_or_create_session,
)
from core.persephone.tasks import get_task, update_task

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Transition table
# ------------------------------------------------------------------

# Valid (from_status, to_status) pairs
VALID_TRANSITIONS: set[tuple[str, str]] = {
    ("open", "in_progress"),
    ("in_progress", "in_review"),
    ("in_progress", "blocked"),
    ("in_progress", "open"),
    ("blocked", "in_progress"),
    ("blocked", "open"),
    ("in_review", "closed"),
    ("in_review", "in_progress"),
    ("closed", "open"),
}


# ------------------------------------------------------------------
# Guard errors
# ------------------------------------------------------------------


class TransitionError(Exception):
    """Raised when a state transition is invalid or blocked by a guard."""

    def __init__(self, message: str, code: str = "TRANSITION_ERROR") -> None:
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------
# Guards
# ------------------------------------------------------------------


@dataclass(frozen=True)
class GuardContext:
    """Context passed to guard functions."""

    task: dict[str, Any]
    from_status: str
    to_status: str
    session: dict[str, Any]
    client: Any
    db_name: str
    collections: PersephoneCollections
    human_override: bool = False
    block_reason: str | None = None


def _guard_different_reviewer(ctx: GuardContext) -> None:
    """in_review → closed requires a different session than the implementer.

    Bypassed if task is marked minor or human_override is True.
    """
    if ctx.from_status != "in_review" or ctx.to_status != "closed":
        return

    if ctx.human_override:
        return

    task = ctx.task
    if task.get("minor"):
        return

    # Find sessions that implemented or submitted this task for review
    involved_sessions = ctx.client.query(
        """
        FOR e IN @@edges
            FILTER e._to == @task_id
            FILTER e.type IN ["implements", "submitted_review"]
            RETURN e._from
        """,
        bind_vars={
            "@edges": ctx.collections.edges,
            "task_id": f"{ctx.collections.tasks}/{task['_key']}",
        },
    )

    current_session_id = f"{ctx.collections.sessions}/{ctx.session['_key']}"
    if current_session_id in involved_sessions:
        raise TransitionError(
            f"Cannot approve own work: session {ctx.session['_key']} implemented or reviewed this task. "
            "A different session must approve, or use --human to override.",
            code="SAME_REVIEWER",
        )


def _guard_dependency(ctx: GuardContext) -> None:
    """Cannot move to in_progress if blocked_by dependencies are unresolved."""
    if ctx.to_status != "in_progress":
        return

    blockers = check_blocked(ctx.client, ctx.db_name, ctx.task["_key"], collections=ctx.collections)
    if blockers:
        titles = ", ".join(f"{b['_key']} ({b.get('title', '?')})" for b in blockers)
        raise TransitionError(
            f"Task is blocked by: {titles}",
            code="BLOCKED",
        )


def _guard_session_required(ctx: GuardContext) -> None:
    """Ownership transitions (in_progress, in_review) require an active session."""
    if ctx.to_status in ("in_progress", "in_review"):
        if not ctx.session or not ctx.session.get("_key"):
            raise TransitionError(
                "An active session is required for this transition.",
                code="NO_SESSION",
            )


def _guard_block_reason(ctx: GuardContext) -> None:
    """Moving to blocked requires a reason."""
    if ctx.to_status == "blocked" and not ctx.block_reason:
        raise TransitionError(
            "A --reason is required when blocking a task.",
            code="NO_BLOCK_REASON",
        )


# Guards applied in order for each transition
_ALL_GUARDS = [
    _guard_session_required,
    _guard_dependency,
    _guard_block_reason,
    _guard_different_reviewer,
]


# ------------------------------------------------------------------
# State machine
# ------------------------------------------------------------------


def transition(
    client: Any,
    db_name: str,
    task_key: str,
    new_status: str,
    *,
    session: dict[str, Any] | None = None,
    human_override: bool = False,
    block_reason: str | None = None,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Transition a task to a new status with guard enforcement.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        task_key: Task document key
        new_status: Target status
        session: Active session (auto-detected if None)
        human_override: Bypass DifferentReviewerGuard
        block_reason: Required when transitioning to 'blocked'
        collections: Collection names override

    Returns:
        Updated task document

    Raises:
        TransitionError: If transition is invalid or guard fails
    """
    cols = collections or PERSEPHONE_COLLECTIONS

    # Fetch task
    task = get_task(client, db_name, task_key, collections=cols)
    if task is None:
        raise TransitionError(f"Task '{task_key}' not found", code="NOT_FOUND")

    from_status = task["status"]

    # Check transition is valid
    if (from_status, new_status) not in VALID_TRANSITIONS:
        raise TransitionError(
            f"Invalid transition: {from_status} → {new_status}",
            code="INVALID_TRANSITION",
        )

    # Get or auto-detect session
    if session is None:
        session = get_or_create_session(client, db_name, collections=cols)

    # Build guard context
    ctx = GuardContext(
        task=task,
        from_status=from_status,
        to_status=new_status,
        session=session,
        client=client,
        db_name=db_name,
        collections=cols,
        human_override=human_override,
        block_reason=block_reason,
    )

    # Run all guards
    for guard in _ALL_GUARDS:
        guard(ctx)

    # Apply transition
    update_fields: dict[str, Any] = {"status": new_status}
    if block_reason and new_status == "blocked":
        update_fields["block_reason"] = block_reason
    if new_status != "blocked":
        # Clear block_reason when leaving blocked state
        update_fields["block_reason"] = None

    prev_block_reason = task.get("block_reason")
    updated = update_task(client, db_name, task_key, collections=cols, **update_fields)

    # Create audit edge — rollback status on failure to keep state + audit in sync
    edge_type = _edge_type_for_transition(from_status, new_status)
    try:
        create_session_task_edge(client, db_name, session["_key"], task_key, edge_type, collections=cols)
    except Exception:
        update_task(
            client, db_name, task_key, collections=cols,
            status=from_status, block_reason=prev_block_reason,
        )
        raise

    logger.info(
        "Task %s: %s → %s (session=%s, edge=%s)",
        task_key, from_status, new_status, session["_key"], edge_type,
    )

    return updated


def _edge_type_for_transition(from_status: str, to_status: str) -> str:
    """Map a transition to an edge type for audit trail."""
    mapping = {
        ("open", "in_progress"): "implements",
        ("in_progress", "in_review"): "submitted_review",
        ("in_progress", "blocked"): "blocked",
        ("in_progress", "open"): "abandoned",
        ("blocked", "in_progress"): "unblocked",
        ("blocked", "open"): "abandoned",
        ("in_review", "closed"): "approved",
        ("in_review", "in_progress"): "rejected",
        ("closed", "open"): "reopened",
    }
    return mapping.get((from_status, to_status), "transitioned")


# ------------------------------------------------------------------
# Dependency management
# ------------------------------------------------------------------


def add_dependency(
    client: Any,
    db_name: str,
    task_key: str,
    depends_on_key: str,
    *,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Add a blocked_by dependency edge.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        task_key: Task that is blocked
        depends_on_key: Task that blocks it

    Returns:
        Created edge document
    """
    cols = collections or PERSEPHONE_COLLECTIONS

    # Verify both tasks exist
    task = get_task(client, db_name, task_key, collections=cols)
    if task is None:
        raise TransitionError(f"Task '{task_key}' not found", code="NOT_FOUND")

    blocker = get_task(client, db_name, depends_on_key, collections=cols)
    if blocker is None:
        raise TransitionError(f"Blocker task '{depends_on_key}' not found", code="NOT_FOUND")

    # Prevent self-dependency
    if task_key == depends_on_key:
        raise TransitionError("A task cannot depend on itself", code="SELF_DEPENDENCY")

    key = f"{task_key}__blocked_by__{depends_on_key}"
    doc = {
        "_key": key,
        "_from": f"{cols.tasks}/{task_key}",
        "_to": f"{cols.tasks}/{depends_on_key}",
        "type": "blocked_by",
        "created_at": _utcnow(),
    }

    resp = client.request(
        "POST",
        f"/_db/{db_name}/_api/document/{cols.edges}",
        json=doc,
        params={"overwriteMode": "replace"},
    )

    doc["_id"] = resp.get("_id", f"{cols.edges}/{key}")
    doc["_rev"] = resp.get("_rev")
    logger.info("Dependency: %s blocked by %s", task_key, depends_on_key)
    return doc


def remove_dependency(
    client: Any,
    db_name: str,
    task_key: str,
    depends_on_key: str,
    *,
    collections: PersephoneCollections | None = None,
) -> bool:
    """Remove a blocked_by dependency edge.

    Returns:
        True if removed, False if edge didn't exist
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    key = f"{task_key}__blocked_by__{depends_on_key}"

    from core.database.arango.optimized_client import ArangoHttpError

    try:
        client.request(
            "DELETE",
            f"/_db/{db_name}/_api/document/{cols.edges}/{key}",
        )
        return True
    except ArangoHttpError as e:
        if e.status_code == 404:
            return False
        raise


def check_blocked(
    client: Any,
    db_name: str,
    task_key: str,
    *,
    collections: PersephoneCollections | None = None,
) -> list[dict[str, Any]]:
    """Return list of unresolved blocking tasks.

    Only tasks that are NOT closed count as blockers.
    """
    cols = collections or PERSEPHONE_COLLECTIONS

    return client.query(
        """
        FOR e IN @@edges
            FILTER e._from == @task_id
            FILTER e.type == "blocked_by"
            LET blocker = DOCUMENT(e._to)
            FILTER blocker.status != "closed"
            RETURN blocker
        """,
        bind_vars={
            "@edges": cols.edges,
            "task_id": f"{cols.tasks}/{task_key}",
        },
    )


# Re-export for convenience
from core.persephone.sessions import _utcnow  # noqa: E402
