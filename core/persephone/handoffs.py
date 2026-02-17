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

from pydantic import ValidationError

from core.persephone.collections import PERSEPHONE_COLLECTIONS, PersephoneCollections
from core.persephone.models import HandoffCreate
from core.persephone.sessions import _create_edge, _utcnow, get_or_create_session

logger = logging.getLogger(__name__)


def _generate_handoff_key() -> str:
    """Generate a handoff key: hnd_XXXXXX (6-char random hex)."""
    return f"hnd_{secrets.token_hex(3)}"


def _capture_git_state() -> dict[str, Any]:
    """Capture current git state (best-effort).

    Returns dict with git_branch, git_sha, git_dirty_files,
    and git_changed_files (list of modified/added .py files).
    All values are None/empty if git is unavailable.
    """
    cwd = os.environ.get("HADES_PROJECT_ROOT", ".")
    result: dict[str, Any] = {
        "git_branch": None,
        "git_sha": None,
        "git_dirty_files": None,
        "git_changed_files": [],
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

        # Capture changed .py files (staged + unstaged)
        changed_files: set[str] = set()
        for diff_cmd in (
            ["git", "diff", "--name-only", "HEAD"],
            ["git", "diff", "--name-only", "--cached"],
        ):
            diff_out = subprocess.run(
                diff_cmd,
                capture_output=True, text=True, timeout=5, cwd=cwd,
            )
            if diff_out.returncode == 0:
                for f in diff_out.stdout.strip().splitlines():
                    f = f.strip()
                    if f and f.endswith(".py"):
                        changed_files.add(f)
        result["git_changed_files"] = sorted(changed_files)

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
    cols = collections or PERSEPHONE_COLLECTIONS

    # Auto-detect session
    if session is None:
        session = get_or_create_session(client, db_name, collections=cols)
    session_key = session["_key"]

    # Capture git state
    git = _capture_git_state()

    key = _generate_handoff_key()
    now = _utcnow()

    try:
        validated = HandoffCreate(
            task_key=task_key,
            session_key=session_key,
            done=done or [],
            remaining=remaining or [],
            decisions=decisions or [],
            uncertain=uncertain or [],
            note=note,
            git_branch=git["git_branch"],
            git_sha=git["git_sha"],
            git_dirty_files=git["git_dirty_files"],
            git_changed_files=git.get("git_changed_files", []),
            created_at=now,
        )
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    doc: dict[str, Any] = {"_key": key, **validated.model_dump()}

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

    # Create task→file edges for changed .py files (links to codebase graph)
    _link_task_to_changed_files(client, db_name, task_key, git.get("git_changed_files", []), cols)

    logger.info("Created handoff %s for task %s (session %s)", key, task_key, session_key)

    # Best-effort activity log
    try:
        from core.persephone.logging import create_log

        create_log(
            client, db_name,
            action="handoff.created",
            task_key=task_key,
            session_key=session_key,
            details={"handoff_key": key},
            collections=cols,
        )
    except Exception:
        logger.warning("Failed to log handoff.created for %s", key)

    return doc


def _link_task_to_changed_files(
    client: Any,
    db_name: str,
    task_key: str,
    changed_files: list[str],
    collections: PersephoneCollections,
) -> None:
    """Create task→file edges for changed .py files.

    Best-effort: checks if the file exists in codebase_files before
    creating the edge. Silently skips files not in the codebase graph.
    """
    if not changed_files:
        return

    from core.database.codebase_collections import CODEBASE_COLLECTIONS
    from core.database.keys import file_key

    for rel_path in changed_files:
        fk = file_key(rel_path)
        # Check if codebase file doc exists
        resp = client.request(
            "GET",
            f"/_db/{db_name}/_api/document/{CODEBASE_COLLECTIONS.files}/{fk}",
        )
        if resp.get("error"):
            continue

        # Create task → file edge in persephone_edges
        _create_edge(
            client,
            db_name,
            _from=f"{collections.tasks}/{task_key}",
            _to=f"{CODEBASE_COLLECTIONS.files}/{fk}",
            edge_type="modifies",
            collections=collections,
        )

    logger.debug("Linked task %s to %d changed files", task_key, len(changed_files))


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
