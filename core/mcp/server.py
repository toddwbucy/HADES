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

Key tools:
  hades_query        — semantic search (most common operation)
  hades_ingest       — add file or arxiv paper to knowledge base
  hades_task_list    — open Persephone tasks
  hades_db_aql       — raw AQL query against ArangoDB
  hades_link         — create compliance edge (code → smell)
  hades_orient       — metadata-first session orientation
  hades_arxiv_search — search ArXiv live
  hades_status       — system health check

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
) -> str:
    """Show database statistics: document counts, sources, recent activity.

    Args:
        database: ArangoDB database name.
        collection: Collection profile name.
    """
    args = ["db", "stats"]
    if collection:
        args += ["--collection", collection]
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


# ---------------------------------------------------------------------------
# Task management (Persephone)
# ---------------------------------------------------------------------------

@mcp.tool()
def hades_task_list(
    database: str | None = None,
) -> str:
    """List open Persephone tasks from the knowledge base.

    Args:
        database: ArangoDB database containing persephone_tasks collection.
                  Use "bident" for the main project task database.
    """
    return _result(_run("task", "list", database=database))


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
