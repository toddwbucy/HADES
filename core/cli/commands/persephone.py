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
