"""Context assembly for Persephone tasks.

Traverses both the Persephone task graph and the codebase knowledge
graph to assemble a complete context for a task: what was done, what
remains, which code files are involved, and their import dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

from core.database.codebase_collections import CODEBASE_COLLECTIONS, CodebaseCollections
from core.persephone.collections import PERSEPHONE_COLLECTIONS, PersephoneCollections

logger = logging.getLogger(__name__)


def assemble_task_context(
    client: Any,
    db_name: str,
    task_key: str,
    *,
    include_imports: bool = True,
    import_depth: int = 1,
    persephone_cols: PersephoneCollections | None = None,
    codebase_cols: CodebaseCollections | None = None,
) -> dict[str, Any]:
    """Assemble full context for a task.

    Runs two AQL queries:
    1. Persephone graph: task → sessions, handoffs, dependencies, modified files
    2. Codebase graph: modified files → import dependencies

    Args:
        client: ArangoHttp2Client instance
        db_name: Target database name
        task_key: Task key to assemble context for
        include_imports: Whether to include import graph traversal
        import_depth: Depth of import traversal (1 = direct imports only)
        persephone_cols: Persephone collection names override
        codebase_cols: Codebase collection names override

    Returns:
        Dict with task, latest_handoff, sessions, dependencies, modified_files.
    """
    p_cols = persephone_cols or PERSEPHONE_COLLECTIONS
    c_cols = codebase_cols or CODEBASE_COLLECTIONS

    # Query 1: Persephone graph
    persephone_result = _query_persephone_graph(client, db_name, task_key, p_cols)

    if persephone_result is None:
        return {"error": f"Task '{task_key}' not found"}

    # Extract file IDs from "modifies" edges
    file_ids: list[str] = persephone_result.get("files", [])

    # Query 2: Codebase graph (import dependencies)
    modified_files: list[dict[str, Any]] = []
    if file_ids:
        modified_files = _query_codebase_graph(
            client, db_name, file_ids, c_cols,
            include_imports=include_imports,
            import_depth=import_depth,
        )

    # Build final context
    context: dict[str, Any] = {
        "task": persephone_result.get("task"),
        "latest_handoff": persephone_result.get("latest_handoff"),
        "sessions": persephone_result.get("sessions", []),
        "dependencies": persephone_result.get("deps", []),
        "modified_files": modified_files,
    }

    return context


def _query_persephone_graph(
    client: Any,
    db_name: str,
    task_key: str,
    cols: PersephoneCollections,
) -> dict[str, Any] | None:
    """Query the Persephone graph for task context."""
    aql = """
    LET task = DOCUMENT(CONCAT(@tasks_col, "/", @key))

    LET sessions = (
        FOR e IN @@edges
            FILTER e._to == task._id AND e.type == "implements"
            LET s = DOCUMENT(e._from)
            FILTER s != null
            RETURN KEEP(s, "_key", "agent_type", "started_at", "last_activity")
    )

    LET handoffs = (
        FOR e IN @@edges
            FILTER e._to == task._id AND e.type == "handoff_for"
            LET h = DOCUMENT(e._from)
            FILTER h != null
            SORT h.created_at DESC
            RETURN h
    )

    LET deps = (
        FOR e IN @@edges
            FILTER e._from == task._id AND e.type IN ["blocked_by", "blocked"]
            LET d = DOCUMENT(e._to)
            FILTER d != null
            RETURN KEEP(d, "_key", "title", "status", "priority")
    )

    LET files = (
        FOR e IN @@edges
            FILTER e._from == task._id AND e.type == "modifies"
            RETURN e._to
    )

    RETURN {
        task: task != null ? KEEP(task, "_key", "title", "status", "priority", "description", "labels", "type") : null,
        sessions: sessions,
        latest_handoff: LENGTH(handoffs) > 0 ? handoffs[0] : null,
        handoff_count: LENGTH(handoffs),
        deps: deps,
        files: files
    }
    """
    try:
        results = client.query(
            aql,
            bind_vars={
                "tasks_col": cols.tasks,
                "@edges": cols.edges,
                "key": task_key,
            },
        )
    except Exception:
        logger.exception("Persephone graph query failed for task %s", task_key)
        return None

    if not results:
        return None

    result = results[0]
    if result.get("task") is None:
        return None

    return result


def _query_codebase_graph(
    client: Any,
    db_name: str,
    file_ids: list[str],
    cols: CodebaseCollections,
    *,
    include_imports: bool = True,
    import_depth: int = 1,
) -> list[dict[str, Any]]:
    """Query the codebase graph for file details + imports."""
    if not include_imports:
        # Simple: just get file details
        aql = """
        FOR file_id IN @file_ids
            LET f = DOCUMENT(file_id)
            FILTER f != null
            RETURN {
                path: f.rel_path,
                file_key: f._key,
                symbols: f.symbols
            }
        """
        try:
            return client.query(aql, bind_vars={"file_ids": file_ids})
        except Exception:
            logger.exception("Codebase file query failed")
            return []

    # With imports: traverse import edges up to import_depth
    aql = """
    FOR file_id IN @file_ids
        LET f = DOCUMENT(file_id)
        FILTER f != null
        LET imports = (
            FOR v, e IN 1..@depth OUTBOUND file_id @@code_edges
                FILTER e.type == "imports"
                RETURN DISTINCT v.rel_path
        )
        RETURN {
            path: f.rel_path,
            file_key: f._key,
            symbols: f.symbols,
            imports: imports
        }
    """
    try:
        return client.query(
            aql,
            bind_vars={
                "file_ids": file_ids,
                "@code_edges": cols.edges,
                "depth": import_depth,
            },
        )
    except Exception:
        logger.exception("Codebase graph query failed")
        return []
