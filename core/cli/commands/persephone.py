"""Persephone task management commands for HADES CLI.

CLI wrappers around core.persephone.tasks, sessions, and workflow —
handles client creation, collection auto-setup, and CLIResponse formatting.
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
from core.persephone.collections import ensure_collections
from core.persephone.handoffs import (
    create_handoff,
    list_handoffs,
)
from core.persephone.logging import list_logs
from core.persephone.sessions import (
    build_usage_briefing,
    create_session_task_edge,
    force_new_session,
    get_or_create_session,
)
from core.persephone.tasks import (
    create_task,
    get_task,
    list_tasks,
    update_task,
)
from core.persephone.workflow import (
    TransitionError,
    add_dependency,
    check_blocked,
    remove_dependency,
    transition,
)


def task_create(
    title: str,
    start_time: float,
    *,
    description: str | None = None,
    priority: str = "medium",
    type_: str = "task",
    labels: list[str] | None = None,
    parent_key: str | None = None,
    acceptance: str | None = None,
) -> CLIResponse:
    """Create a new task."""
    try:
        client, _cfg, db_name = _make_client(read_only=False)
    except Exception as e:
        return error_response(
            command="task.create",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        ensure_collections(client, db_name)
        doc = create_task(
            client,
            db_name,
            title,
            description=description,
            priority=priority,
            type_=type_,
            labels=labels,
            parent_key=parent_key,
            acceptance=acceptance,
        )
        return success_response(
            command="task.create",
            data={"task": doc},
            start_time=start_time,
        )
    except ValueError as e:
        return error_response(
            command="task.create",
            code=ErrorCode.VALIDATION_ERROR,
            message=str(e),
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.create",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_list(
    start_time: float,
    *,
    status: str | None = None,
    priority: str | None = None,
    type_: str | None = None,
    parent_key: str | None = None,
    limit: int = 50,
) -> CLIResponse:
    """List tasks with optional filters."""
    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="task.list",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        tasks = list_tasks(
            client,
            db_name,
            status=status,
            priority=priority,
            type_=type_,
            parent_key=parent_key,
            limit=limit,
        )
        return success_response(
            command="task.list",
            data={"tasks": tasks},
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.list",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_show(
    key: str,
    start_time: float,
) -> CLIResponse:
    """Show a single task by key."""
    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="task.show",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        doc = get_task(client, db_name, key)
        if doc is None:
            return error_response(
                command="task.show",
                code=ErrorCode.TASK_ERROR,
                message=f"Task '{key}' not found",
                start_time=start_time,
            )
        return success_response(
            command="task.show",
            data={"task": doc},
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.show",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_update(
    key: str,
    start_time: float,
    **fields: Any,
) -> CLIResponse:
    """Update task fields."""
    try:
        client, _cfg, db_name = _make_client(read_only=False)
    except Exception as e:
        return error_response(
            command="task.update",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        ensure_collections(client, db_name)
        doc = update_task(client, db_name, key, **fields)
        if doc is None:
            return error_response(
                command="task.update",
                code=ErrorCode.TASK_ERROR,
                message=f"Task '{key}' not found",
                start_time=start_time,
            )

        # Auto-create session-task edge when claiming a task
        if fields.get("status") == "in_progress":
            try:
                session = get_or_create_session(client, db_name)
                create_session_task_edge(client, db_name, session["_key"], key, "implements")
            except Exception:
                pass  # Edge creation is best-effort

        return success_response(
            command="task.update",
            data={"task": doc},
            start_time=start_time,
        )
    except ValueError as e:
        return error_response(
            command="task.update",
            code=ErrorCode.VALIDATION_ERROR,
            message=str(e),
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.update",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_close(
    key: str,
    start_time: float,
) -> CLIResponse:
    """Close a task (set status=closed)."""
    return task_update(key, start_time, status="closed")


def task_usage(
    start_time: float,
    *,
    new_session: bool = False,
) -> CLIResponse:
    """Get session briefing: current session, in-progress tasks, ready tasks."""
    try:
        client, _cfg, db_name = _make_client(read_only=False)
    except Exception as e:
        return error_response(
            command="task.usage",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        ensure_collections(client, db_name)

        if new_session:
            session = force_new_session(client, db_name)
        else:
            session = get_or_create_session(client, db_name)

        briefing = build_usage_briefing(client, db_name, session)
        return success_response(
            command="task.usage",
            data=briefing,
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.usage",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


# ------------------------------------------------------------------
# Handoff commands (Phase 4)
# ------------------------------------------------------------------


def task_handoff(
    key: str,
    start_time: float,
    *,
    done: list[str] | None = None,
    remaining: list[str] | None = None,
    decisions: list[str] | None = None,
    uncertain: list[str] | None = None,
    note: str | None = None,
) -> CLIResponse:
    """Create a handoff document for a task."""
    try:
        client, _cfg, db_name = _make_client(read_only=False)
    except Exception as e:
        return error_response(
            command="task.handoff",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        ensure_collections(client, db_name)
        session = get_or_create_session(client, db_name)
        doc = create_handoff(
            client,
            db_name,
            key,
            done=done,
            remaining=remaining,
            decisions=decisions,
            uncertain=uncertain,
            note=note,
            session=session,
        )
        return success_response(
            command="task.handoff",
            data={"handoff": doc},
            start_time=start_time,
        )
    except ValueError as e:
        return error_response(
            command="task.handoff",
            code=ErrorCode.VALIDATION_ERROR,
            message=str(e),
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.handoff",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_handoff_show(
    key: str,
    start_time: float,
    *,
    limit: int = 10,
) -> CLIResponse:
    """Show handoffs for a task."""
    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="task.handoff-show",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        handoffs = list_handoffs(client, db_name, key, limit=limit)
        latest = handoffs[0] if handoffs else None
        return success_response(
            command="task.handoff-show",
            data={
                "task_key": key,
                "handoffs": handoffs,
                "latest": latest,
                "count": len(handoffs),
            },
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.handoff-show",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


# ------------------------------------------------------------------
# Workflow commands (Phase 3)
# ------------------------------------------------------------------


def task_transition(
    key: str,
    new_status: str,
    start_time: float,
    *,
    human_override: bool = False,
    block_reason: str | None = None,
) -> CLIResponse:
    """Transition a task with guard enforcement."""
    try:
        client, _cfg, db_name = _make_client(read_only=False)
    except Exception as e:
        return error_response(
            command="task.transition",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        ensure_collections(client, db_name)
        doc = transition(
            client, db_name, key, new_status,
            human_override=human_override,
            block_reason=block_reason,
        )
        return success_response(
            command="task.transition",
            data={"task": doc, "transition": f"→ {new_status}"},
            start_time=start_time,
        )
    except TransitionError as e:
        return error_response(
            command="task.transition",
            code=ErrorCode.VALIDATION_ERROR,
            message=str(e),
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.transition",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_dep_add(
    key: str,
    blocked_by: str,
    start_time: float,
) -> CLIResponse:
    """Add a dependency (blocked_by edge)."""
    try:
        client, _cfg, db_name = _make_client(read_only=False)
    except Exception as e:
        return error_response(
            command="task.dep",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        ensure_collections(client, db_name)
        edge = add_dependency(client, db_name, key, blocked_by)
        return success_response(
            command="task.dep",
            data={"edge": edge, "message": f"{key} is now blocked by {blocked_by}"},
            start_time=start_time,
        )
    except TransitionError as e:
        return error_response(
            command="task.dep",
            code=ErrorCode.VALIDATION_ERROR,
            message=str(e),
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.dep",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_dep_remove(
    key: str,
    blocked_by: str,
    start_time: float,
) -> CLIResponse:
    """Remove a dependency."""
    try:
        client, _cfg, db_name = _make_client(read_only=False)
    except Exception as e:
        return error_response(
            command="task.dep",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        ensure_collections(client, db_name)
        removed = remove_dependency(client, db_name, key, blocked_by)
        if not removed:
            return error_response(
                command="task.dep",
                code=ErrorCode.TASK_ERROR,
                message=f"No dependency from {key} to {blocked_by}",
                start_time=start_time,
            )
        return success_response(
            command="task.dep",
            data={"message": f"Removed: {key} no longer blocked by {blocked_by}"},
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.dep",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_context(
    key: str,
    start_time: float,
    *,
    include_imports: bool = True,
) -> CLIResponse:
    """Assemble full context for a task."""
    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="task.context",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        from core.persephone.context import assemble_task_context

        context = assemble_task_context(
            client,
            db_name,
            key,
            include_imports=include_imports,
        )
        if "error" in context:
            return error_response(
                command="task.context",
                code=ErrorCode.TASK_ERROR,
                message=context["error"],
                start_time=start_time,
            )
        return success_response(
            command="task.context",
            data=context,
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.context",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_blocked(
    key: str,
    start_time: float,
) -> CLIResponse:
    """Check what blocks a task."""
    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="task.blocked",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        blockers = check_blocked(client, db_name, key)
        return success_response(
            command="task.blocked",
            data={
                "task_key": key,
                "blocked": len(blockers) > 0,
                "blockers": blockers,
            },
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.blocked",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


# ------------------------------------------------------------------
# Log and session commands (Phase 7)
# ------------------------------------------------------------------


def task_log(
    key: str,
    start_time: float,
    *,
    limit: int = 50,
) -> CLIResponse:
    """List activity logs for a task."""
    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="task.log",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        logs = list_logs(client, db_name, task_key=key, limit=limit)
        return success_response(
            command="task.log",
            data={
                "task_key": key,
                "logs": logs,
                "count": len(logs),
            },
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.log",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()


def task_sessions(
    key: str,
    start_time: float,
    *,
    limit: int = 10,
) -> CLIResponse:
    """List sessions that worked on a task (via 'implements' edges)."""
    try:
        client, _cfg, db_name = _make_client(read_only=True)
    except Exception as e:
        return error_response(
            command="task.sessions",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        from core.persephone.collections import PERSEPHONE_COLLECTIONS

        cols = PERSEPHONE_COLLECTIONS
        sessions = client.query(
            """
            FOR e IN @@edges
                FILTER e._to == @task_id
                FILTER e.type IN ["implements", "submitted_review", "approved"]
                FOR s IN @@sessions
                    FILTER s._id == e._from
                    SORT s.started_at DESC
                    LIMIT @limit
                    RETURN MERGE(s, {edge_type: e.type})
            """,
            bind_vars={
                "@edges": cols.edges,
                "@sessions": cols.sessions,
                "task_id": f"{cols.tasks}/{key}",
                "limit": limit,
            },
        )
        return success_response(
            command="task.sessions",
            data={
                "task_key": key,
                "sessions": sessions,
                "count": len(sessions),
            },
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="task.sessions",
            code=ErrorCode.TASK_ERROR,
            message=str(e),
            start_time=start_time,
        )
    finally:
        client.close()
