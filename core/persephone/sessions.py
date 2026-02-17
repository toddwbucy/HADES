"""Persephone session management and agent fingerprinting.

Detects the calling agent by walking the process tree, manages session
lifecycle in ArangoDB, and creates session-task edges when tasks are
claimed.
"""

from __future__ import annotations

import logging
import os
import secrets
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from core.database.arango.optimized_client import ArangoHttpError
from core.persephone.collections import PERSEPHONE_COLLECTIONS, PersephoneCollections
from core.persephone.models import SessionCreate

logger = logging.getLogger(__name__)

# Known agent process names → canonical agent_type
_AGENT_PROCESS_MAP: dict[str, str] = {
    "claude": "claude_code",
    "claude-code": "claude_code",
    "cursor": "cursor",
    "codex": "codex",
    "windsurf": "windsurf",
    "zed": "zed",
    "aider": "aider",
    "copilot": "copilot",
    "gemini": "gemini",
    "kiro": "kiro",
    "amp": "amp",
    "opencode": "opencode",
    "pi-agent": "pi_agent",
    "warp": "warp",
}

# Environment variables that reveal the calling agent
_AGENT_ENV_VARS: dict[str, str] = {
    "CLAUDE_CODE": "claude_code",
    "CLAUDE_CODE_AGENT": "claude_code",
    "CURSOR_AGENT": "cursor",
    "CODEX_AGENT": "codex",
}

# ArangoDB error number for document not found
_ARANGO_DOC_NOT_FOUND = 1202


def _utcnow() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _generate_session_key() -> str:
    """Generate a session key: ses_XXXXXX (6-char random hex)."""
    return f"ses_{secrets.token_hex(3)}"


# ------------------------------------------------------------------
# Agent fingerprinting
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AgentFingerprint:
    """Detected agent identity."""

    agent_type: str
    agent_pid: int
    context_id: str


def _get_parent_processes(max_depth: int = 15) -> list[tuple[int, str]]:
    """Walk the process tree upward, returning (pid, name) pairs.

    Uses /proc on Linux for speed, falls back to ps.
    """
    results: list[tuple[int, str]] = []
    pid = os.getpid()

    for _ in range(max_depth):
        try:
            # Try /proc first (Linux)
            stat_path = f"/proc/{pid}/stat"
            if os.path.exists(stat_path):
                with open(stat_path) as f:
                    stat = f.read()
                # Format: pid (comm) state ppid ...
                comm_start = stat.index("(") + 1
                comm_end = stat.rindex(")")
                name = stat[comm_start:comm_end]
                rest = stat[comm_end + 2 :].split()
                ppid = int(rest[1])  # state is rest[0], ppid is rest[1]
            else:
                # Fallback to ps
                out = subprocess.run(
                    ["ps", "-o", "ppid=,comm=", "-p", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if out.returncode != 0:
                    break
                parts = out.stdout.strip().split(None, 1)
                if len(parts) < 2:
                    break
                ppid = int(parts[0])
                name = parts[1].strip()

            results.append((pid, name))
            if ppid <= 1:
                break
            pid = ppid
        except (OSError, ValueError, subprocess.TimeoutExpired):
            break

    return results


@lru_cache(maxsize=1)
def detect_agent() -> AgentFingerprint:
    """Detect the calling agent.

    Detection order:
    1. PERSEPHONE_SESSION_ID env var (explicit override — returns as-is)
    2. Agent-specific env vars (CLAUDE_CODE, CURSOR_AGENT, etc.)
    3. Walk parent process tree matching known agent names
    4. Terminal session ID (TERM_SESSION_ID, TMUX_PANE)
    5. Fallback: 'unknown'

    Cached for process lifetime.
    """
    # 1. Explicit override
    explicit_id = os.environ.get("PERSEPHONE_SESSION_ID", "").strip()
    if explicit_id:
        return AgentFingerprint(
            agent_type="explicit",
            agent_pid=os.getpid(),
            context_id=explicit_id,
        )

    # 2. Agent-specific env vars
    for env_var, agent_type in _AGENT_ENV_VARS.items():
        if os.environ.get(env_var):
            return AgentFingerprint(
                agent_type=agent_type,
                agent_pid=os.getppid(),
                context_id=os.environ.get(env_var, ""),
            )

    # 3. Walk process tree
    for pid, name in _get_parent_processes():
        name_lower = name.lower().replace("_", "-")
        for pattern, agent_type in _AGENT_PROCESS_MAP.items():
            if pattern in name_lower:
                return AgentFingerprint(
                    agent_type=agent_type,
                    agent_pid=pid,
                    context_id=str(pid),
                )

    # 4. Terminal session fallback
    term_id = os.environ.get("TERM_SESSION_ID") or os.environ.get("TMUX_PANE") or ""
    if term_id:
        return AgentFingerprint(
            agent_type="terminal",
            agent_pid=os.getppid(),
            context_id=term_id,
        )

    # 5. Unknown
    return AgentFingerprint(
        agent_type="unknown",
        agent_pid=os.getppid(),
        context_id=str(os.getppid()),
    )


def _get_current_branch() -> str:
    """Get current git branch name."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.environ.get("HADES_PROJECT_ROOT", "."),
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


# ------------------------------------------------------------------
# Session CRUD
# ------------------------------------------------------------------


def get_or_create_session(
    client: Any,
    db_name: str,
    *,
    fingerprint: AgentFingerprint | None = None,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Find existing session for (branch, agent_type, context_id) or create new.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        fingerprint: Agent fingerprint (auto-detected if None)
        collections: Collection names override

    Returns:
        Session document dict
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    fp = fingerprint or detect_agent()
    branch = _get_current_branch()

    # Look for existing active session
    results = client.query(
        """
        FOR s IN @@col
            FILTER s.agent_type == @agent_type
            FILTER s.branch == @branch
            FILTER s.context_id == @context_id
            FILTER s.ended_at == null
            SORT s.started_at DESC
            LIMIT 1
            RETURN s
        """,
        bind_vars={
            "@col": cols.sessions,
            "agent_type": fp.agent_type,
            "branch": branch,
            "context_id": fp.context_id,
        },
    )

    if results:
        # Heartbeat the existing session
        session = results[0]
        heartbeat(client, db_name, session["_key"], collections=cols)
        return session

    # Create new session
    return _create_session(client, db_name, fp, branch, collections=cols)


def force_new_session(
    client: Any,
    db_name: str,
    *,
    fingerprint: AgentFingerprint | None = None,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Always create a new session, linking to previous via edge.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        fingerprint: Agent fingerprint (auto-detected if None)
        collections: Collection names override

    Returns:
        New session document dict
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    fp = fingerprint or detect_agent()
    branch = _get_current_branch()

    # Find previous session to link (scoped to same context to avoid ending other sessions)
    previous = client.query(
        """
        FOR s IN @@col
            FILTER s.agent_type == @agent_type
            FILTER s.branch == @branch
            FILTER s.context_id == @context_id
            FILTER s.ended_at == null
            SORT s.started_at DESC
            LIMIT 1
            RETURN s
        """,
        bind_vars={
            "@col": cols.sessions,
            "agent_type": fp.agent_type,
            "branch": branch,
            "context_id": fp.context_id,
        },
    )

    # End previous session
    previous_key = None
    if previous:
        previous_key = previous[0]["_key"]
        end_session(client, db_name, previous_key, collections=cols)

    # Create new session
    session = _create_session(
        client, db_name, fp, branch, previous_session_key=previous_key, collections=cols
    )

    # Link to previous via edge
    if previous_key:
        _create_edge(
            client,
            db_name,
            _from=f"{cols.sessions}/{session['_key']}",
            _to=f"{cols.sessions}/{previous_key}",
            edge_type="continues",
            collections=cols,
        )

    return session


def _create_session(
    client: Any,
    db_name: str,
    fp: AgentFingerprint,
    branch: str,
    *,
    previous_session_key: str | None = None,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Create a session document."""
    cols = collections or PERSEPHONE_COLLECTIONS
    now = _utcnow()
    key = _generate_session_key()

    try:
        validated = SessionCreate(
            agent_type=fp.agent_type,
            agent_pid=fp.agent_pid,
            context_id=fp.context_id,
            branch=branch,
            previous_session_key=previous_session_key,
            started_at=now,
            last_activity=now,
        )
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    doc: dict[str, Any] = {"_key": key, **validated.model_dump()}

    resp = client.request(
        "POST",
        f"/_db/{db_name}/_api/document/{cols.sessions}",
        json=doc,
    )

    doc["_id"] = resp.get("_id", f"{cols.sessions}/{key}")
    doc["_rev"] = resp.get("_rev")
    logger.info("Created session %s (agent=%s, branch=%s)", key, fp.agent_type, branch)
    return doc


def heartbeat(
    client: Any,
    db_name: str,
    session_key: str,
    *,
    collections: PersephoneCollections | None = None,
) -> None:
    """Update last_activity timestamp on a session."""
    cols = collections or PERSEPHONE_COLLECTIONS
    try:
        client.request(
            "PATCH",
            f"/_db/{db_name}/_api/document/{cols.sessions}/{session_key}",
            json={"last_activity": _utcnow()},
        )
    except ArangoHttpError as e:
        if e.status_code != 404:
            raise
        logger.warning("Session %s not found for heartbeat", session_key)


def end_session(
    client: Any,
    db_name: str,
    session_key: str,
    *,
    collections: PersephoneCollections | None = None,
) -> None:
    """End a session by setting ended_at."""
    cols = collections or PERSEPHONE_COLLECTIONS
    try:
        client.request(
            "PATCH",
            f"/_db/{db_name}/_api/document/{cols.sessions}/{session_key}",
            json={"ended_at": _utcnow(), "last_activity": _utcnow()},
        )
    except ArangoHttpError as e:
        if e.status_code != 404:
            raise
        logger.warning("Session %s not found for end", session_key)


def get_session(
    client: Any,
    db_name: str,
    session_key: str,
    *,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any] | None:
    """Get a session by key."""
    cols = collections or PERSEPHONE_COLLECTIONS
    try:
        return client.request(
            "GET",
            f"/_db/{db_name}/_api/document/{cols.sessions}/{session_key}",
        )
    except ArangoHttpError as e:
        if e.status_code == 404 or e.details.get("errorNum") == _ARANGO_DOC_NOT_FOUND:
            return None
        raise


# ------------------------------------------------------------------
# Session-Task edges
# ------------------------------------------------------------------


def create_session_task_edge(
    client: Any,
    db_name: str,
    session_key: str,
    task_key: str,
    edge_type: str = "implements",
    *,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Create an edge linking a session to a task.

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        session_key: Session document key
        task_key: Task document key
        edge_type: Relationship type (implements, reviews, created)
        collections: Collection names override

    Returns:
        Created edge document
    """
    cols = collections or PERSEPHONE_COLLECTIONS
    return _create_edge(
        client,
        db_name,
        _from=f"{cols.sessions}/{session_key}",
        _to=f"{cols.tasks}/{task_key}",
        edge_type=edge_type,
        collections=cols,
    )


def _create_edge(
    client: Any,
    db_name: str,
    *,
    _from: str,
    _to: str,
    edge_type: str,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Create an edge in persephone_edges."""
    cols = collections or PERSEPHONE_COLLECTIONS
    now = _utcnow()

    # Deterministic key to prevent duplicates
    from_part = _from.split("/")[-1]
    to_part = _to.split("/")[-1]
    key = f"{from_part}__{to_part}__{edge_type}"

    doc = {
        "_key": key,
        "_from": _from,
        "_to": _to,
        "type": edge_type,
        "created_at": now,
    }

    resp = client.request(
        "POST",
        f"/_db/{db_name}/_api/document/{cols.edges}",
        json=doc,
        params={"overwriteMode": "replace"},
    )

    doc["_id"] = resp.get("_id", f"{cols.edges}/{key}")
    doc["_rev"] = resp.get("_rev")
    return doc


# ------------------------------------------------------------------
# Usage briefing
# ------------------------------------------------------------------


def build_usage_briefing(
    client: Any,
    db_name: str,
    session: dict[str, Any],
    *,
    collections: PersephoneCollections | None = None,
) -> dict[str, Any]:
    """Build a briefing for the current session.

    Returns a dict with:
    - session: current session info
    - in_progress: tasks this session is working on
    - reviewable: tasks in_review by other sessions
    - ready: open tasks sorted by priority
    """
    cols = collections or PERSEPHONE_COLLECTIONS

    # Tasks this session is implementing (via edges)
    in_progress = client.query(
        """
        FOR e IN @@edges
            FILTER e._from == @session_id
            FILTER e.type == "implements"
            FOR t IN @@tasks
                FILTER t._id == e._to
                FILTER t.status == "in_progress"
                RETURN t
        """,
        bind_vars={
            "@edges": cols.edges,
            "@tasks": cols.tasks,
            "session_id": f"{cols.sessions}/{session['_key']}",
        },
    )

    # Tasks in_review by OTHER sessions (available for this session to review)
    reviewable = client.query(
        """
        FOR t IN @@tasks
            FILTER t.status == "in_review"
            LET implementers = (
                FOR e IN @@edges
                    FILTER e._to == t._id
                    FILTER e.type == "implements"
                    RETURN DOCUMENT(e._from).agent_type
            )
            FILTER @agent_type NOT IN implementers
            RETURN t
        """,
        bind_vars={
            "@tasks": cols.tasks,
            "@edges": cols.edges,
            "agent_type": session.get("agent_type", "unknown"),
        },
    )

    # Open tasks ready to start (not blocked, not in_progress)
    ready = client.query(
        """
        FOR t IN @@tasks
            FILTER t.status == "open"
            LET prio_order = t.priority == "critical" ? 0
                : t.priority == "high" ? 1
                : t.priority == "medium" ? 2
                : 3
            SORT prio_order, t.created_at
            LIMIT 20
            RETURN t
        """,
        bind_vars={"@tasks": cols.tasks},
    )

    # For each in-progress task, fetch latest handoff for context
    handoffs: dict[str, Any] = {}
    if in_progress:
        from core.persephone.handoffs import get_latest_handoff

        for task in in_progress:
            handoff = get_latest_handoff(client, db_name, task["_key"], collections=cols)
            if handoff:
                handoffs[task["_key"]] = handoff

    return {
        "session": {
            "key": session["_key"],
            "agent_type": session.get("agent_type"),
            "branch": session.get("branch"),
            "started_at": session.get("started_at"),
        },
        "in_progress": in_progress,
        "handoffs": handoffs,
        "reviewable": reviewable,
        "ready": ready,
    }
