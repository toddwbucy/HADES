"""HADES MCP Server.

Wraps the HADES CLI as MCP tools. Each tool calls the hades binary as a
subprocess and returns its structured JSON output as the tool result.

This approach:
- Avoids duplicating CLI argument parsing and global --database state
- Guarantees tool output is always consistent with CLI output
- Works regardless of which venv the caller is in

Transports (controlled by CLI args to hades-mcp):
  stdio               Claude Code native (default)
  Unix socket HTTP    Hermes integration (--socket PATH)
  Network HTTP        Fallback (--port PORT)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Locate the hades binary (same venv as this server)
# ---------------------------------------------------------------------------

_VENV_BIN = Path(sys.executable).parent
_HADES_BIN = str(_VENV_BIN / "hades")


# ---------------------------------------------------------------------------
# MCP app
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "hades",
    instructions="""HADES knowledge base — semantic graph RAG over papers and code.

Always use these tools instead of running hades via Bash. They are faster,
always in context, and survive context compression.

Common tools:
  hades_query              — semantic search (most common operation)
  hades_ingest             — add file or arxiv paper to knowledge base
  hades_orient             — metadata-first session orientation (start here)
  hades_status             — system health check
  hades_db_aql             — raw AQL query against ArangoDB
  hades_link               — create compliance edge (code → smell)

Database / CRUD:
  hades_db_collections     — list collections
  hades_db_databases       — list all accessible databases
  hades_db_create_database — create a new database
  hades_db_recent          — recently ingested documents
  hades_db_health          — chunk/embedding consistency check
  hades_db_get             — fetch a single document
  hades_db_insert          — insert document(s) into any collection
  hades_db_update          — merge-update a document
  hades_db_delete          — delete a document (requires HADES_DESTRUCTIVE_OPS)
  hades_db_export          — export a collection as JSONL
  hades_db_purge           — remove a paper and all its chunks

Graph:
  hades_db_graph_list      — list named graphs
  hades_db_graph_create    — create a named graph
  hades_db_graph_traverse  — depth-first traversal
  hades_db_graph_shortest_path — shortest path between nodes
  hades_db_graph_neighbors — immediate neighbors of a node
  hades_db_graph_drop      — delete a named graph

Vector index:
  hades_db_index_status    — check ANN index / search mode
  hades_db_create_index    — build FAISS-backed vector index

ArXiv:
  hades_arxiv_search       — search ArXiv live
  hades_arxiv_abstract     — search local 2.8M abstract database
  hades_arxiv_info         — metadata for a specific paper
  hades_arxiv_sync         — sync new abstracts from ArXiv
  hades_arxiv_sync_status  — check sync progress

Embedding service:
  hades_embed              — embed text via Jina V4
  hades_embed_service_status / start / stop
  hades_embed_gpu_status / list

Task management (Persephone):
  hades_task_usage         — session briefing (start here)
  hades_task_list          — open tasks
  hades_task_get           — full task detail
  hades_task_create        — create a task
  hades_task_update        — update task fields
  hades_task_start / review / approve / close / block / unblock
  hades_task_dep           — manage task dependencies
  hades_task_handoff       — structured context transfer
  hades_task_context       — full context assembly
  hades_task_log           — activity log
  hades_task_sessions      — session history

Codebase graph:
  hades_codebase_ingest    — index Python files (AST + import edges)
  hades_codebase_update    — incremental update
  hades_codebase_stats     — collection counts

The `database` parameter selects the ArangoDB database (e.g. "NL", "bident").
Omit it to use the default configured database.
""",
)


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run(
    *args: str,
    database: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run a hades CLI command and return parsed JSON output."""
    cmd = [_HADES_BIN]
    if database:
        cmd += ["--database", database]
    cmd += list(args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": {"code": "TIMEOUT", "message": f"Command timed out after {timeout}s"}}
    except FileNotFoundError:
        return {"success": False, "error": {"code": "NOT_FOUND", "message": f"hades binary not found at {_HADES_BIN}"}}

    stdout = result.stdout.strip()
    if not stdout:
        return {
            "success": False,
            "error": {"code": "NO_OUTPUT", "message": result.stderr.strip() or "No output from hades"},
        }

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"success": False, "error": {"code": "PARSE_ERROR", "message": stdout[:500]}}


def _result(data: dict[str, Any]) -> str:
    """Serialise a hades JSON response back to a compact string for MCP."""
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Search & retrieval
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_query(
    text: str,
    database: str | None = None,
    collection: str | None = None,
    limit: int = 10,
    context: int = 0,
    paper: str | None = None,
) -> str:
    """Semantic search over the HADES knowledge base.

    Args:
        text: Natural language search query.
        database: ArangoDB database name (default: configured default).
        collection: Collection profile — "arxiv", "sync", or "default".
        limit: Maximum number of results to return.
        context: Number of adjacent chunks to include with each result.
        paper: Restrict search to chunks from this paper/document ID.
    """
    args = ["db", "query", text, "--limit", str(limit), "--context", str(context)]
    if collection:
        args += ["--collection", collection]
    if paper:
        args += ["--paper", paper]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_db_aql(
    query: str,
    database: str | None = None,
) -> str:
    """Execute a raw AQL query against ArangoDB.

    Args:
        query: AQL query string.
        database: ArangoDB database name (default: configured default).
    """
    return _result(_run("db", "aql", query, database=database))


@mcp.tool()
def hades_db_check(
    document_id: str,
    database: str | None = None,
    collection: str | None = None,
) -> str:
    """Check if a document exists in the knowledge base and show its metadata.

    Args:
        document_id: Paper key or document ID (e.g. "2409.04701" or "conductor-rs").
        database: ArangoDB database name.
        collection: Collection profile name.
    """
    args = ["db", "check", document_id]
    if collection:
        args += ["--collection", collection]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_db_stats(
    database: str | None = None,
    collection: str | None = None,
    all_collections: bool = False,
) -> str:
    """Show database statistics: document counts, sources, recent activity.

    Args:
        database: ArangoDB database name.
        collection: Collection profile name.
        all_collections: Show stats for all collections (database-wide).
    """
    args = ["db", "stats"]
    if collection:
        args += ["--collection", collection]
    if all_collections:
        args.append("--all")
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_db_list(
    database: str | None = None,
    limit: int = 20,
    collection: str | None = None,
) -> str:
    """List documents stored in the knowledge base.

    Args:
        database: ArangoDB database name.
        limit: Maximum number of documents to return.
        collection: Collection profile name.
    """
    args = ["db", "list", "--limit", str(limit)]
    if collection:
        args += ["--collection", collection]
    return _result(_run(*args, database=database))


# ---------------------------------------------------------------------------
# Database management & CRUD
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_db_databases(
    database: str | None = None,
) -> str:
    """List all ArangoDB databases accessible to the configured user.

    Args:
        database: ArangoDB database name (used for connection credentials).
    """
    return _result(_run("db", "databases", database=database))


@mcp.tool()
def hades_db_create_database(
    name: str,
    database: str | None = None,
) -> str:
    """Create a new ArangoDB database.

    Args:
        name: Name of the database to create.
        database: ArangoDB database name (used for connection credentials).
    """
    return _result(_run("db", "create-database", name, database=database))


@mcp.tool()
def hades_db_collections(
    database: str | None = None,
    prefix: str | None = None,
) -> str:
    """List all collections in the database.

    Args:
        database: ArangoDB database name.
        prefix: Filter collections by name prefix (e.g. "arxiv_").
    """
    args = ["db", "collections"]
    if prefix:
        args += ["--prefix", prefix]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_db_count(
    collection: str,
    database: str | None = None,
) -> str:
    """Count documents in a collection.

    Args:
        collection: Collection name.
        database: ArangoDB database name.
    """
    return _result(_run("db", "count", collection, database=database))


@mcp.tool()
def hades_db_recent(
    database: str | None = None,
    limit: int = 10,
    collection: str | None = None,
) -> str:
    """Show recently ingested documents.

    Args:
        database: ArangoDB database name.
        limit: Number of recent documents to return.
        collection: Collection profile name.
    """
    args = ["db", "recent", "--limit", str(limit)]
    if collection:
        args += ["--collection", collection]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_db_health(
    database: str | None = None,
) -> str:
    """Check chunk/embedding consistency across the knowledge base.

    Args:
        database: ArangoDB database name.
    """
    return _result(_run("db", "health", database=database))


@mcp.tool()
def hades_db_get(
    collection: str,
    key: str,
    database: str | None = None,
) -> str:
    """Fetch a single document from any collection.

    Args:
        collection: Collection name.
        key: Document key (_key field).
        database: ArangoDB database name.
    """
    return _result(_run("db", "get", collection, key, database=database))


@mcp.tool()
def hades_db_create(
    collection: str,
    database: str | None = None,
) -> str:
    """Create a new collection in the database.

    Args:
        collection: Collection name to create.
        database: ArangoDB database name.
    """
    return _result(_run("db", "create", collection, database=database))


@mcp.tool()
def hades_db_insert(
    collection: str,
    data: str | None = None,
    file: str | None = None,
    database: str | None = None,
) -> str:
    """Insert one or more documents into a collection.

    Provide either `data` (inline JSON) or `file` (path to a JSONL file).

    Args:
        collection: Target collection name.
        data: JSON string for a single document, e.g. '{"key": "value"}'.
        file: Path to a JSONL file for bulk insert.
        database: ArangoDB database name.
    """
    args = ["db", "insert", collection]
    if data:
        args += ["--data", data]
    if file:
        args += ["--file", file]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_db_update(
    collection: str,
    key: str,
    data: str,
    replace: bool = False,
    database: str | None = None,
) -> str:
    """Update a document in a collection.

    Args:
        collection: Collection name.
        key: Document key (_key field).
        data: JSON string with fields to merge (or full replacement if replace=True).
        replace: If True, fully replace the document instead of merging.
        database: ArangoDB database name.
    """
    args = ["db", "update", collection, key, "--data", data]
    if replace:
        args.append("--replace")
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_db_delete(
    collection: str,
    key: str,
    database: str | None = None,
) -> str:
    """Delete a document from a collection.

    Requires HADES_DESTRUCTIVE_OPS=enabled environment variable.

    Args:
        collection: Collection name.
        key: Document key (_key field).
        database: ArangoDB database name.
    """
    return _result(_run("db", "delete", collection, key, database=database))


@mcp.tool()
def hades_db_export(
    collection: str,
    database: str | None = None,
) -> str:
    """Export all documents from a collection as JSONL.

    Args:
        collection: Collection name to export.
        database: ArangoDB database name.
    """
    return _result(_run("db", "export", collection, database=database))


@mcp.tool()
def hades_db_purge(
    document_id: str,
    database: str | None = None,
) -> str:
    """Remove a paper and all its associated chunks and embeddings.

    Requires HADES_DESTRUCTIVE_OPS=enabled environment variable.

    Args:
        document_id: Paper key or document ID (e.g. "2409.04701").
        database: ArangoDB database name.
    """
    return _result(_run("db", "purge", document_id, database=database))


# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_db_index_status(
    database: str | None = None,
    collection: str | None = None,
) -> str:
    """Check whether a vector index exists and the current search mode.

    Args:
        database: ArangoDB database name.
        collection: Collection profile name.
    """
    args = ["db", "index-status"]
    if collection:
        args += ["--collection", collection]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_db_create_index(
    database: str | None = None,
    collection: str | None = None,
    n_lists: int | None = None,
    n_probe: int | None = None,
    metric: str | None = None,
) -> str:
    """Build a FAISS-backed vector index for fast ANN search.

    Args:
        database: ArangoDB database name.
        collection: Collection profile name.
        n_lists: Number of IVF lists (default: 100).
        n_probe: Number of lists to probe at query time (default: 10).
        metric: Distance metric — "cosine" or "l2" (default: "cosine").
    """
    args = ["db", "create-index"]
    if collection:
        args += ["--collection", collection]
    if n_lists is not None:
        args += ["--n-lists", str(n_lists)]
    if n_probe is not None:
        args += ["--n-probe", str(n_probe)]
    if metric:
        args += ["--metric", metric]
    return _result(_run(*args, database=database))


# ---------------------------------------------------------------------------
# Graph operations
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_db_graph_list(
    database: str | None = None,
) -> str:
    """List all named graphs in the database.

    Args:
        database: ArangoDB database name.
    """
    return _result(_run("db", "graph", "list", database=database))


@mcp.tool()
def hades_db_graph_create(
    name: str,
    edge_defs: str,
    database: str | None = None,
) -> str:
    """Create a named graph.

    Args:
        name: Graph name.
        edge_defs: JSON array of edge definitions, e.g.
                   '[{"collection":"edges","from":["A"],"to":["B"]}]'.
        database: ArangoDB database name.
    """
    return _result(_run("db", "graph", "create", "--name", name, "--edge-defs", edge_defs, database=database))


@mcp.tool()
def hades_db_graph_traverse(
    start: str,
    graph: str,
    direction: str = "outbound",
    database: str | None = None,
) -> str:
    """Traverse a named graph from a starting node.

    Args:
        start: Start vertex ID in collection/key format (e.g. "nodes/abc123").
        graph: Named graph to traverse.
        direction: "outbound", "inbound", or "any".
        database: ArangoDB database name.
    """
    return _result(_run(
        "db", "graph", "traverse",
        "--start", start, "--graph", graph, "--direction", direction,
        database=database,
    ))


@mcp.tool()
def hades_db_graph_shortest_path(
    from_id: str,
    to_id: str,
    graph: str,
    database: str | None = None,
) -> str:
    """Find the shortest path between two nodes in a named graph.

    Args:
        from_id: Source vertex ID (e.g. "nodes/abc123").
        to_id: Target vertex ID (e.g. "nodes/def456").
        graph: Named graph to search.
        database: ArangoDB database name.
    """
    return _result(_run(
        "db", "graph", "shortest-path",
        "--from", from_id, "--to", to_id, "--graph", graph,
        database=database,
    ))


@mcp.tool()
def hades_db_graph_neighbors(
    start: str,
    graph: str,
    database: str | None = None,
) -> str:
    """Return immediate neighbors of a node in a named graph.

    Args:
        start: Start vertex ID (e.g. "nodes/abc123").
        graph: Named graph to query.
        database: ArangoDB database name.
    """
    return _result(_run(
        "db", "graph", "neighbors",
        "--start", start, "--graph", graph,
        database=database,
    ))


@mcp.tool()
def hades_db_graph_drop(
    name: str,
    database: str | None = None,
) -> str:
    """Delete a named graph (graph definition only; collections are preserved).

    Requires HADES_DESTRUCTIVE_OPS=enabled environment variable.

    Args:
        name: Graph name to delete.
        database: ArangoDB database name.
    """
    return _result(_run("db", "graph", "drop", "--name", name, database=database))


# ---------------------------------------------------------------------------
# Ingest & extract
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_ingest(
    target: str,
    database: str | None = None,
    id: str | None = None,
    force: bool = False,
    task: str | None = None,
    claims: str | None = None,
) -> str:
    """Ingest a file or ArXiv paper into the knowledge base.

    Accepts a local file path or an ArXiv ID (e.g. "2409.04701").
    Automatically extracts, embeds, and stores with late chunking.
    Code files (.rs, .cu, .py, etc.) are auto-detected and routed through
    the Jina V4 Code LoRA.

    Args:
        target: File path or ArXiv ID to ingest.
        database: ArangoDB database name.
        id: Custom document ID (overrides auto-generated).
        force: Re-process even if document already exists.
        task: Embedding task type — "code" activates Jina V4 Code LoRA.
        claims: Compliance claims in "CS-32:behavioral,CS-33:architectural" format.
    """
    args = ["ingest", target]
    if id:
        args += ["--id", id]
    if force:
        args.append("--force")
    if task:
        args += ["--task", task]
    if claims:
        args += ["--claims", claims]
    return _result(_run(*args, database=database, timeout=600))


@mcp.tool()
def hades_extract(
    file_path: str,
    format: str = "json",
) -> str:
    """Extract structured text from a document without storing it.

    Args:
        file_path: Path to the document (PDF, DOCX, HTML, code file, etc.).
        format: Output format — "json" (full structure) or "text" (plain text).
    """
    return _result(_run("extract", file_path, "--format", format))


# ---------------------------------------------------------------------------
# Compliance linking
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_link(
    source_id: str,
    smell: str,
    enforcement: str = "behavioral",
    database: str | None = None,
    summary: str | None = None,
    methods: list[str] | None = None,
    smell_collection: str = "nl_code_smells",
) -> str:
    """Create a compliance edge from an ingested code file to a smell node.

    Args:
        source_id: Document ID in arxiv_metadata (e.g. "conductor-rs").
        smell: Smell identifier — "CS-32" or full key "smell-032-observe-then-advance".
        enforcement: One of: static, behavioral, architectural, review, documentation.
        database: ArangoDB database name.
        summary: Human-readable compliance summary.
        methods: List of specific methods/functions that demonstrate compliance.
        smell_collection: ArangoDB collection containing smell nodes.
    """
    args = ["link", source_id, "--smell", smell, "--enforcement", enforcement,
            "--smell-collection", smell_collection]
    if summary:
        args += ["--summary", summary]
    if methods:
        for method in methods:
            args += ["--method", method]
    return _result(_run(*args, database=database))


# ---------------------------------------------------------------------------
# ArXiv
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_arxiv_search(
    query: str,
    max_results: int = 20,
) -> str:
    """Search ArXiv live for papers matching a query.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.
    """
    return _result(_run("arxiv", "search", query, "--max", str(max_results)))


@mcp.tool()
def hades_arxiv_abstract(
    query: str,
    limit: int = 20,
    database: str | None = None,
) -> str:
    """Search the local synced ArXiv abstract database (2.8M papers, fast).

    Args:
        query: Natural language search query.
        limit: Maximum number of results to return.
        database: ArangoDB database name.
    """
    return _result(_run("arxiv", "abstract", query, "--limit", str(limit), database=database))


@mcp.tool()
def hades_arxiv_info(
    arxiv_id: str,
) -> str:
    """Get metadata for a specific ArXiv paper.

    Args:
        arxiv_id: ArXiv paper ID (e.g. "2409.04701").
    """
    return _result(_run("arxiv", "info", arxiv_id))


@mcp.tool()
def hades_arxiv_sync(
    from_date: str | None = None,
    categories: str | None = None,
    max_results: int | None = None,
    batch: int | None = None,
) -> str:
    """Sync new abstracts from ArXiv into the local database.

    Args:
        from_date: Start date in YYYY-MM-DD format (e.g. "2025-01-01").
        categories: Comma-separated ArXiv categories (e.g. "cs.AI,cs.CL").
        max_results: Maximum number of papers to sync.
        batch: Embedding batch size (higher = faster, more VRAM).
    """
    args = ["arxiv", "sync"]
    if from_date:
        args += ["--from", from_date]
    if categories:
        args += ["--categories", categories]
    if max_results is not None:
        args += ["--max", str(max_results)]
    if batch is not None:
        args += ["--batch", str(batch)]
    return _result(_run(*args, timeout=3600))


@mcp.tool()
def hades_arxiv_sync_status() -> str:
    """Check the status of the ArXiv abstract sync (last run, counts, etc.)."""
    return _result(_run("arxiv", "sync-status"))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_embed(
    text: str,
    task: str = "retrieval.query",
) -> str:
    """Generate an embedding vector for text using Jina V4.

    Args:
        text: Text to embed.
        task: Embedding task — "retrieval.query", "retrieval.passage", or "code".
               Use "code" to activate the Jina V4 Code LoRA adapter.
    """
    return _result(_run("embed", "text", text, "--task", task))


@mcp.tool()
def hades_embed_service_status() -> str:
    """Check the health of the persistent embedding service."""
    return _result(_run("embed", "service", "status"))


@mcp.tool()
def hades_embed_service_start() -> str:
    """Start the persistent embedding service daemon."""
    return _result(_run("embed", "service", "start", timeout=60))


@mcp.tool()
def hades_embed_service_stop() -> str:
    """Stop the persistent embedding service daemon."""
    return _result(_run("embed", "service", "stop"))


@mcp.tool()
def hades_embed_gpu_status() -> str:
    """Show GPU memory usage and utilization for all visible GPUs."""
    return _result(_run("embed", "gpu", "status"))


@mcp.tool()
def hades_embed_gpu_list() -> str:
    """List all available GPUs with their indices and names."""
    return _result(_run("embed", "gpu", "list"))


# ---------------------------------------------------------------------------
# Task management (Persephone)
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_task_usage(
    database: str | None = None,
    new_session: bool = False,
) -> str:
    """Session briefing: auto-detects agent, resumes or creates a session.

    Run at the start of every session to get context on open tasks.

    Args:
        database: ArangoDB database containing persephone_* collections.
                  Use "bident" for the main project task database.
        new_session: Force creation of a new session even in the same context.
    """
    args = ["task", "usage"]
    if new_session:
        args.append("--new-session")
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_task_list(
    database: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> str:
    """List Persephone tasks from the knowledge base.

    Args:
        database: ArangoDB database containing persephone_tasks collection.
                  Use "bident" for the main project task database.
        status: Filter by status — "open", "in_progress", "in_review", "closed", "blocked".
        priority: Filter by priority — "high", "medium", "low".
    """
    args = ["task", "list"]
    if status:
        args += ["--status", status]
    if priority:
        args += ["--priority", priority]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_task_get(
    key: str,
    database: str | None = None,
) -> str:
    """Get full details for a single Persephone task.

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
    """
    return _result(_run("task", "show", key, database=database))


@mcp.tool()
def hades_task_create(
    title: str,
    database: str | None = None,
    priority: str | None = None,
    type: str | None = None,
) -> str:
    """Create a new Persephone task.

    Args:
        title: Task title.
        database: ArangoDB database name.
        priority: "high", "medium", or "low".
        type: Task type — "task", "bug", "feature", etc.
    """
    args = ["task", "create", title]
    if priority:
        args += ["--priority", priority]
    if type:
        args += ["--type", type]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_task_update(
    key: str,
    database: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> str:
    """Update fields on a Persephone task.

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
        status: New status value.
        priority: New priority value.
    """
    args = ["task", "update", key]
    if status:
        args += ["--status", status]
    if priority:
        args += ["--priority", priority]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_task_start(
    key: str,
    database: str | None = None,
) -> str:
    """Transition a task from open → in_progress (guarded state change).

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
    """
    return _result(_run("task", "start", key, database=database))


@mcp.tool()
def hades_task_review(
    key: str,
    database: str | None = None,
) -> str:
    """Transition a task from in_progress → in_review (guarded state change).

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
    """
    return _result(_run("task", "review", key, database=database))


@mcp.tool()
def hades_task_approve(
    key: str,
    database: str | None = None,
) -> str:
    """Transition a task from in_review → closed (guarded state change).

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
    """
    return _result(_run("task", "approve", key, database=database))


@mcp.tool()
def hades_task_close(
    key: str,
    database: str | None = None,
) -> str:
    """Close a task directly (shortcut for completed work).

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
    """
    return _result(_run("task", "close", key, database=database))


@mcp.tool()
def hades_task_block(
    key: str,
    reason: str,
    database: str | None = None,
) -> str:
    """Mark a task as blocked with a reason.

    Args:
        key: Task key (e.g. "task_88074f").
        reason: Human-readable reason for the block.
        database: ArangoDB database name.
    """
    return _result(_run("task", "block", key, "--reason", reason, database=database))


@mcp.tool()
def hades_task_unblock(
    key: str,
    database: str | None = None,
) -> str:
    """Unblock a task, returning it to in_progress.

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
    """
    return _result(_run("task", "unblock", key, database=database))


@mcp.tool()
def hades_task_dep(
    key: str,
    database: str | None = None,
    blocked_by: str | None = None,
    remove: str | None = None,
) -> str:
    """Manage task dependencies.

    Call with no options to show current blockers for a task.

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
        blocked_by: Key of a task that must complete before this one.
        remove: Key of a dependency to remove.
    """
    args = ["task", "dep", key]
    if blocked_by:
        args += ["--blocked-by", blocked_by]
    if remove:
        args += ["--remove", remove]
    return _result(_run(*args, database=database))


@mcp.tool()
def hades_task_handoff(
    key: str,
    done: str,
    remaining: str,
    database: str | None = None,
) -> str:
    """Record a structured context handoff for a task (between sessions/agents).

    Args:
        key: Task key (e.g. "task_88074f").
        done: Summary of what was completed in this session.
        remaining: What still needs to be done.
        database: ArangoDB database name.
    """
    return _result(_run("task", "handoff", key, "--done", done, "--remaining", remaining, database=database))


@mcp.tool()
def hades_task_context(
    key: str,
    database: str | None = None,
) -> str:
    """Assemble full context for a task (traverses task + codebase graphs).

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
    """
    return _result(_run("task", "context", key, database=database))


@mcp.tool()
def hades_task_log(
    key: str,
    database: str | None = None,
) -> str:
    """Show the activity log for a task.

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
    """
    return _result(_run("task", "log", key, database=database))


@mcp.tool()
def hades_task_sessions(
    key: str,
    database: str | None = None,
) -> str:
    """Show session history for a task.

    Args:
        key: Task key (e.g. "task_88074f").
        database: ArangoDB database name.
    """
    return _result(_run("task", "sessions", key, database=database))


# ---------------------------------------------------------------------------
# Codebase knowledge graph
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_codebase_ingest(
    path: str = ".",
    database: str | None = None,
    force: bool = False,
) -> str:
    """Index Python files in a directory (AST chunks + import edges).

    Args:
        path: Directory to index (default: current directory).
        database: ArangoDB database name.
        force: Re-process all files even if unchanged.
    """
    args = ["codebase", "ingest", path]
    if force:
        args.append("--force")
    return _result(_run(*args, database=database, timeout=600))


@mcp.tool()
def hades_codebase_update(
    path: str = ".",
    database: str | None = None,
) -> str:
    """Incrementally update the codebase index (only changed files).

    Args:
        path: Directory to update (default: current directory).
        database: ArangoDB database name.
    """
    return _result(_run("codebase", "update", path, database=database, timeout=600))


@mcp.tool()
def hades_codebase_stats(
    database: str | None = None,
) -> str:
    """Show codebase knowledge graph collection counts.

    Args:
        database: ArangoDB database name.
    """
    return _result(_run("codebase", "stats", database=database))


# ---------------------------------------------------------------------------
# System status & orientation
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_status() -> str:
    """Check HADES system health: GPU, embedding service, ArangoDB connection."""
    return _result(_run("status"))


@mcp.tool()
def hades_orient(
    papers: int = 5,
    database: str | None = None,
) -> str:
    """Metadata-first orientation: recent papers, open tasks, system state.

    Run at session start to orient a new agent session without loading full content.

    Args:
        papers: Number of recent papers to include in orientation.
        database: ArangoDB database name.
    """
    args = ["orient", "--papers", str(papers)]
    return _result(_run(*args, database=database))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """hades-mcp entry point.

    Usage:
        hades-mcp                              stdio (Claude Code default)
        hades-mcp --socket /run/hades/mcp.sock Unix socket HTTP (Hermes)
        hades-mcp --port 8765                  Network HTTP (fallback)
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="hades-mcp",
        description="HADES MCP server",
    )
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--socket",
        metavar="PATH",
        help="Run as Unix socket HTTP server (e.g. /run/hades/mcp.sock)",
    )
    transport_group.add_argument(
        "--port",
        type=int,
        metavar="PORT",
        help="Run as network HTTP server on PORT",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for network HTTP mode (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    if args.socket:
        _run_unix_socket(args.socket)
    elif args.port:
        _run_network(args.host, args.port)
    else:
        # Default: stdio for Claude Code
        mcp.run(transport="stdio")


def _run_unix_socket(socket_path: str) -> None:
    """Run MCP server on a Unix domain socket."""
    import asyncio

    import uvicorn

    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale socket
    if path.exists():
        path.unlink()

    starlette_app = mcp.sse_app()
    config = uvicorn.Config(
        starlette_app,
        uds=str(path),
        log_level="warning",
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


def _run_network(host: str, port: int) -> None:
    """Run MCP server on a network socket."""
    import asyncio

    import uvicorn

    starlette_app = mcp.sse_app()
    config = uvicorn.Config(
        starlette_app,
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
