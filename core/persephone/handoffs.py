"""Persephone handoff management.

Structured handoff documents capture context between agent sessions:
what was done, what remains, key decisions, and uncertainties. Stored
as first-class ArangoDB graph nodes linked to sessions and tasks via
edges in persephone_edges.
"""

from __future__ import annotations

import logging
import os
import secrets
import subprocess
from typing import Any

from core.persephone.collections import PERSEPHONE_COLLECTIONS, PersephoneCollections
from core.persephone.sessions import _create_edge, _utcnow, get_or_create_session

logger = logging.getLogger(__name__)


def _generate_handoff_key() -> str:
    """Generate a handoff key: hnd_XXXXXX (6-char random hex)."""
    return f"hnd_{secrets.token_hex(3)}"


def _capture_git_state() -> dict[str, Any]:
    """Capture current git state (best-effort).

    Returns dict with git_branch, git_sha, git_dirty_files.
    All values are None if git is unavailable.
    """
    cwd = os.environ.get("HADES_PROJECT_ROOT", ".")
    result: dict[str, Any] = {
        "git_branch": None,
        "git_sha": None,
        "git_dirty_files": None,
    }

    try:
        sha_out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        )
        if sha_out.returncode == 0:
            result["git_sha"] = sha_out.stdout.strip()

        branch_out = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        )
        if branch_out.returncode == 0:
            result["git_branch"] = branch_out.stdout.strip() or None

        status_out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        )
        if status_out.returncode == 0:
            lines = [ln for ln in status_out.stdout.strip().split("\n") if ln]
            result["git_dirty_files"] = len(lines)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return result


def create_handoff(
    client: Any,
    db_name: str,
    task_key: str,
    *,
    done: list[str] | None = None,
    remaining: list[str] | None = None,
    decisions: list[str] | None = None,
    uncertain: list[str] | None = None,
    note: str | None = None,
    session: dict[str, Any] | None = None,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Create a handoff document linked to a task and session.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        task_key: Task this handoff is for
        done: What was completed
        remaining: What still needs doing
        decisions: Key decisions made (and rationale)
        uncertain: Open questions / unknowns
        note: Free-form note
        session: Session dict (auto-detected if None)
        collections: Collection names override

    Returns:
        Created handoff document dict

    Raises:
        ValueError: If no content fields are provided
    """
    if not any([done, remaining, decisions, uncertain, note]):
        raise ValueError("At least one content field required (done, remaining, decisions, uncertain, or note)")

    cols = collections or PERSEPHONE_COLLECTIONS

    # Auto-detect session
    if session is None:
        session = get_or_create_session(client, db_name, collections=cols)
    session_key = session["_key"]

    # Capture git state
    git = _capture_git_state()

    key = _generate_handoff_key()
    now = _utcnow()

    doc: dict[str, Any] = {
        "_key": key,
        "task_key": task_key,
        "session_key": session_key,
        "done": done or [],
        "remaining": remaining or [],
        "decisions": decisions or [],
        "uncertain": uncertain or [],
        "note": note,
        "git_branch": git["git_branch"],
        "git_sha": git["git_sha"],
        "git_dirty_files": git["git_dirty_files"],
        "created_at": now,
    }

    resp = client.request(
        "POST",
        f"/_db/{db_name}/_api/document/{cols.handoffs}",
        json=doc,
    )
    doc["_id"] = resp.get("_id", f"{cols.handoffs}/{key}")
    doc["_rev"] = resp.get("_rev")

    # Create edges: session → handoff, handoff → task
    _create_edge(
        client, db_name,
        _from=f"{cols.sessions}/{session_key}",
        _to=f"{cols.handoffs}/{key}",
        edge_type="authored_handoff",
        collections=cols,
    )
    _create_edge(
        client, db_name,
        _from=f"{cols.handoffs}/{key}",
        _to=f"{cols.tasks}/{task_key}",
        edge_type="handoff_for",
        collections=cols,
    )

    logger.info("Created handoff %s for task %s (session %s)", key, task_key, session_key)
    return doc


def get_latest_handoff(
    client: Any,
    db_name: str,
    task_key: str,
    *,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any] | None:
    """Get the most recent handoff for a task.

    Uses edge traversal: find handoff_for edges pointing to the task,
    sort by created_at descending, return the newest.
    """
    cols = collections or PERSEPHONE_COLLECTIONS

    results = client.query(
        """
        FOR e IN @@edges
            FILTER e._to == @task_id
            FILTER e.type == "handoff_for"
            FOR h IN @@handoffs
                FILTER h._id == e._from
                SORT h.created_at DESC
                LIMIT 1
                RETURN h
        """,
        bind_vars={
            "@edges": cols.edges,
            "@handoffs": cols.handoffs,
            "task_id": f"{cols.tasks}/{task_key}",
        },
    )

    return results[0] if results else None


def list_handoffs(
    client: Any,
    db_name: str,
    task_key: str,
    *,
    limit: int = 10,
    collections: PersephoneCollections | None = None,
) -> list[dict[str, Any]]:
    """List handoffs for a task, newest first."""
    cols = collections or PERSEPHONE_COLLECTIONS

    return client.query(
        """
        FOR e IN @@edges
            FILTER e._to == @task_id
            FILTER e.type == "handoff_for"
            FOR h IN @@handoffs
                FILTER h._id == e._from
                SORT h.created_at DESC
                LIMIT @limit
                RETURN h
        """,
        bind_vars={
            "@edges": cols.edges,
            "@handoffs": cols.handoffs,
            "task_id": f"{cols.tasks}/{task_key}",
            "limit": limit,
        },
    )
