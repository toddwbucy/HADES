"""HADES CLI - Main entry point.

AI-focused CLI for interacting with the HADES knowledge base.
All commands output JSON for predictable parsing by AI models.

Core Tools (standalone):
    hades extract <file>             # Extract text from any document
    hades embed text "..."           # Generate embedding for text
    hades ingest <file_or_arxiv>     # Extract → embed → store (unified)

Database Operations:
    hades db query "text"            # Semantic search over stored chunks
    hades db list                    # List papers in database
    hades db stats                   # Database statistics
    hades db check <id>              # Check if document exists
    hades db purge <id>              # Remove document and its chunks

ArXiv Source Adapter:
    hades arxiv search "query"       # Search arxiv API for papers
    hades arxiv info <arxiv_id>      # Get paper metadata from arxiv
    hades arxiv abstract "query"     # Search 2.8M synced abstracts
    hades arxiv sync                 # Sync abstracts from arxiv
"""

from __future__ import annotations

import os
import sys
import time

import typer

from core.cli.decorators import cli_command
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    print_response,
)
from core.database.collections import get_default_profile_name, list_profiles


def _set_gpu(gpu: int | None) -> None:
    """Set CUDA_VISIBLE_DEVICES if gpu is specified.

    This must be called before importing torch or loading config,
    as it controls which GPU PyTorch will use.
    """
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _global_gpu_callback(
    ctx: typer.Context,
    gpu: int | None,
) -> None:
    """Global callback to set GPU before any command runs."""
    if gpu is not None:
        _set_gpu(gpu)
        # Store in context for commands that also accept --gpu
        ctx.ensure_object(dict)
        ctx.obj["gpu"] = gpu


# Create the main Typer app
# Note: rich_markup_mode=None disables rich help formatting to avoid
# typer/click compatibility issues with Parameter.make_metavar()
app = typer.Typer(
    name="hades",
    help="HADES Knowledge Base CLI - AI model interface for semantic search over academic papers.",
    no_args_is_help=True,
    add_completion=True,  # Enable shell completion (#43)
    rich_markup_mode=None,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit", is_eager=True),
    agent: str = typer.Option(None, "--agent", help="Install agent integration (claude or agent)", is_eager=True),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index for embedding commands (e.g., 0, 1, 2)"),
    database: str = typer.Option(
        None, "--database", "--db", help="Target ArangoDB database name (overrides config/env)"
    ),
) -> None:
    """HADES Knowledge Base CLI - AI model interface for semantic search over academic papers."""
    if database:
        os.environ["HADES_DATABASE"] = database

    if version:
        from importlib.metadata import version as get_version

        try:
            print(f"hades {get_version('hades')}", file=sys.stderr)
        except Exception:
            print("hades 0.1.0", file=sys.stderr)
        raise typer.Exit()

    if agent:
        from core.cli.agent_templates import install_agent

        install_agent(agent)
        raise typer.Exit()

    _global_gpu_callback(ctx, gpu)

    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        print(ctx.get_help(), file=sys.stderr)
        raise typer.Exit()


# =============================================================================
# Subcommand Groups
# =============================================================================

arxiv_app = typer.Typer(
    name="arxiv",
    help="ArXiv abstract sync and status. Search/API commands moved to arxiv-manager.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
app.add_typer(arxiv_app, name="arxiv")

db_app = typer.Typer(
    name="db",
    help="Database queries, paper management, and collection operations.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
app.add_typer(db_app, name="db")

graph_app = typer.Typer(
    name="graph",
    help="Named graph management and traversal operations.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
db_app.add_typer(graph_app, name="graph")

# Embed commands (text/image → vector)
embed_app = typer.Typer(
    name="embed",
    help="Generate embeddings for text or images.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
app.add_typer(embed_app, name="embed")

embed_service_app = typer.Typer(
    name="service",
    help="Manage the embedding service daemon.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
embed_app.add_typer(embed_service_app, name="service")

embed_gpu_app = typer.Typer(
    name="gpu",
    help="GPU status and management.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
embed_app.add_typer(embed_gpu_app, name="gpu")

# Persephone task management
task_app = typer.Typer(
    name="task",
    help="Persephone task management.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
app.add_typer(task_app, name="task")


# =============================================================================
# Top-Level Commands (Standalone Tools)
# =============================================================================


@app.command("status")
def status_cmd() -> None:
    """Show comprehensive system status for workspace discovery.

    Provides a single-command audit of the entire HADES system:
    - Version and embedding service status
    - Database connection and collection stats
    - Recently ingested papers
    - Last sync timestamp

    Designed for fresh AI sessions to quickly understand what's available.

    Examples:
        hades status
    """
    start_time = time.time()

    try:
        from core.cli.commands.status import get_status

        response = get_status(start_time)
        print_response(response)

    except Exception as e:
        response = error_response(
            command="status",
            code=ErrorCode.INTERNAL_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@app.command("extract")
def extract_cmd(
    file: str = typer.Argument(..., help="Path to document file", metavar="FILE"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or text"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
) -> None:
    """Extract structured text from a document.

    Supports PDF, DOCX, PPTX, HTML, Markdown, and plain text files.

    Examples:
        hades extract paper.pdf
        hades extract paper.pdf --format text
        hades extract paper.pdf --output extracted.json
    """
    start_time = time.time()

    try:
        from core.cli.commands.extract import extract_file

        response = extract_file(file, format, output, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="extract",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@app.command("ingest")
def ingest_cmd(
    inputs: list[str] = typer.Argument(None, help="ArXiv IDs or file paths to ingest"),
    id: str = typer.Option(None, "--id", help="Custom document ID (single file only)"),
    force: bool = typer.Option(False, "--force", help="Force reprocessing"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Enable batch mode with progress and error isolation"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from previous batch state"),
    metadata: str = typer.Option(None, "--metadata", "-m", help="Custom metadata JSON to merge into document record"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index"),
) -> None:
    """Ingest documents into the knowledge base.

    Auto-detects arxiv IDs vs file paths:
    - "2501.12345" → downloads and processes arxiv paper
    - "paper.pdf" → processes local file

    Batch mode (--batch) provides:
    - JSON progress to stderr (stdout stays clean for final result)
    - Per-document error isolation (one failure doesn't stop the batch)
    - Resume capability via state file

    Examples:
        hades ingest 2501.12345
        hades ingest paper.pdf --id my-doc
        hades ingest 2501.12345 2501.67890 --force
        hades ingest paper1.pdf paper2.pdf
        hades ingest /papers/*.pdf --batch
        hades ingest --resume
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.ingest import ingest

        # Parse custom metadata if provided
        extra_metadata = None
        if metadata:
            import json

            try:
                extra_metadata = json.loads(metadata)
                if not isinstance(extra_metadata, dict):
                    response = error_response(
                        command="ingest",
                        code=ErrorCode.PROCESSING_FAILED,
                        message="--metadata must be a JSON object",
                        start_time=start_time,
                    )
                    print_response(response)
                    raise typer.Exit(1) from None
            except json.JSONDecodeError as e:
                response = error_response(
                    command="ingest",
                    code=ErrorCode.PROCESSING_FAILED,
                    message=f"Invalid --metadata JSON: {e}",
                    start_time=start_time,
                )
                print_response(response)
                raise typer.Exit(1) from None

        # Handle resume-only mode (no inputs required)
        actual_inputs = inputs or []

        response = ingest(
            actual_inputs,
            document_id=id,
            force=force,
            batch=batch,
            resume=resume,
            start_time=start_time,
            extra_metadata=extra_metadata,
        )
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="ingest",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# ArXiv Commands
# =============================================================================


@arxiv_app.command("sync")
def arxiv_sync(
    from_date: str = typer.Option(None, "--from", "-f", help="Start date (YYYY-MM-DD, default: 7 days ago)"),
    categories: str = typer.Option(
        None, "--categories", "-c", help="Comma-separated arxiv categories (e.g., cs.AI,cs.CL)"
    ),
    max_results: int = typer.Option(1000, "--max", "-m", help="Maximum papers to sync"),
    batch_size: int = typer.Option(8, "--batch", "-b", help="Batch size for embedding (default 8 for 16GB GPU)"),
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Sync only papers newer than last sync"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Sync recent abstracts from arxiv for semantic search.

    Fetches metadata and abstracts, embeds them with Jina, and stores
    for fast semantic search - WITHOUT downloading full PDFs.

    Use this to keep your abstract database current, then use 'hades arxiv ingest'
    to download full papers you're interested in.

    Examples:
        hades arxiv sync --gpu 2 --batch 8
        hades arxiv sync --incremental                    # Sync since last sync
        hades arxiv sync --from 2025-01-01 --categories cs.AI,cs.CL --gpu 0
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.arxiv import sync_abstracts

        response = sync_abstracts(from_date, categories, max_results, batch_size, start_time, incremental=incremental)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="arxiv.sync",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@arxiv_app.command("sync-status")
def arxiv_sync_status() -> None:
    """Show sync status including last sync time and history.

    Examples:
        hades arxiv sync-status
    """
    start_time = time.time()

    try:
        from core.cli.commands.arxiv import get_sync_status

        response = get_sync_status(start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="arxiv.sync-status",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Database Commands
# =============================================================================


@db_app.command("query")
def database_query(
    search_text: str = typer.Argument(None, help="Search query text", metavar="SEARCH_TEXT"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    paper: str = typer.Option(None, "--paper", "-p", help="Filter results to a specific paper (document ID)"),
    context: int = typer.Option(0, "--context", "-c", help="Include N adjacent chunks for context"),
    cite: bool = typer.Option(False, "--cite", help="Output minimal citation format (arxiv_id, title, quote)"),
    chunks_only: bool = typer.Option(False, "--chunks", help="Get all chunks for --paper (no semantic search)"),
    hybrid: bool = typer.Option(False, "--hybrid", "-H", help="Combine semantic search with keyword matching"),
    decompose: bool = typer.Option(False, "--decompose", "-D", help="Split compound queries and merge results"),
    rerank: bool = typer.Option(False, "--rerank", "-R", help="Re-rank with cross-encoder for better precision"),
    rerank_model: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "--rerank-model",
        help="Cross-encoder model for re-ranking",
    ),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
    collection: str = typer.Option(
        None,
        "--collection",
        "-C",
        help=f"Collection profile to query (default: $HADES_DEFAULT_COLLECTION or 'arxiv'). Available: {', '.join(list_profiles())}",
    ),
) -> None:
    """Semantic search over the knowledge base.

    Returns relevant text chunks with similarity scores.

    Examples:
        hades db query "attention mechanism"                    # Search default collection
        hades db query "Newton-Schulz" --paper 2505.23735       # Search within paper
        hades db query "attention" --context 1                  # Include ±1 adjacent chunks
        hades db query "attention" --cite --limit 3             # Citation format, top 3
        hades db query --paper 2505.23735 --chunks              # Get all chunks (no search)
        hades db query "flash attention" --hybrid               # Semantic + keyword matching
        hades db query "attention" --collection sync            # Query synced abstracts
        hades db query "attention" --rerank                     # Cross-encoder precision
    """
    _set_gpu(gpu)
    start_time = time.time()
    # Use provided collection or fall back to default
    profile = collection or get_default_profile_name()

    try:
        from core.cli.commands.database import get_paper_chunks, semantic_query

        # Mode 1: Get all chunks for a paper (no semantic search)
        if chunks_only and paper:
            response = get_paper_chunks(paper, limit, start_time, collection=profile)
        # Mode 2: Semantic search (optionally filtered by paper)
        elif search_text:
            response = semantic_query(
                search_text,
                limit,
                start_time,
                paper_filter=paper,
                context=context,
                cite_only=cite,
                hybrid=hybrid,
                decompose=decompose,
                rerank=rerank,
                rerank_model=rerank_model,
                collection=profile,
            )
        # Mode 3: Must provide search text or --chunks
        else:
            response = error_response(
                command="database.query",
                code=ErrorCode.CONFIG_ERROR,
                message="Provide search text, or use --paper with --chunks",
                start_time=start_time,
            )
            print_response(response)
            raise typer.Exit(1) from None

        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.query",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@db_app.command("aql")
def database_aql(
    aql: str = typer.Argument(..., help="AQL query string", metavar="AQL"),
    bind: str = typer.Option(None, "--bind", "-b", help="Bind variables as JSON (e.g., '{\"x\":1}')"),
) -> None:
    """Execute an arbitrary AQL query.

    Uses the read-write socket since AQL can mutate data.

    Examples:
        hades database aql "FOR d IN arxiv_metadata LIMIT 5 RETURN d.title"
        hades database aql "FOR d IN col FILTER d.x == @x RETURN d" --bind '{"x":1}'
    """
    start_time = time.time()

    try:
        from core.cli.commands.database import execute_aql

        response = execute_aql(aql, bind, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.aql",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@db_app.command("list")
def database_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of documents to list"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by category (if supported by collection)"),
    collection: str = typer.Option(
        None,
        "--collection",
        "-C",
        help=f"Collection profile (default: $HADES_DEFAULT_COLLECTION or 'arxiv'). Available: {', '.join(list_profiles())}",
    ),
) -> None:
    """List documents stored in the database."""
    start_time = time.time()
    profile = collection or get_default_profile_name()

    try:
        from core.cli.commands.database import list_stored_papers

        response = list_stored_papers(limit, category, start_time, collection=profile)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.list",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@db_app.command("stats")
@cli_command("database.stats", ErrorCode.DATABASE_ERROR)
def database_stats(
    collection: str = typer.Option(
        None,
        "--collection",
        "-C",
        help=f"Collection profile (default: $HADES_DEFAULT_COLLECTION or 'arxiv'). Available: {', '.join(list_profiles())}",
    ),
    all: bool = typer.Option(False, "--all", "-a", help="Show database-wide stats for all collections"),
    start_time: float = typer.Option(0.0, hidden=True),  # Injected by decorator
) -> CLIResponse:
    """Show database statistics."""
    if all:
        from core.cli.commands.database import get_all_stats

        return get_all_stats(start_time)

    from core.cli.commands.database import get_stats

    profile = collection or get_default_profile_name()
    return get_stats(start_time, collection=profile)


@db_app.command("recent")
@cli_command("database.recent", ErrorCode.DATABASE_ERROR)
def database_recent(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recent papers to show"),
    collection: str = typer.Option(
        None,
        "--collection",
        "-C",
        help=f"Collection profile (default: $HADES_DEFAULT_COLLECTION or 'arxiv'). Available: {', '.join(list_profiles())}",
    ),
    start_time: float = typer.Option(0.0, hidden=True),  # Injected by decorator
) -> CLIResponse:
    """Show recently ingested papers.

    Lists papers in reverse chronological order by ingestion time.
    Useful for understanding what's new in the knowledge base.

    Examples:
        hades db recent
        hades db recent --limit 20
        hades db recent --collection sync
    """
    from core.cli.commands.database import get_recent_papers

    profile = collection or get_default_profile_name()
    return get_recent_papers(limit, start_time, collection=profile)


@db_app.command("health")
@cli_command("database.health", ErrorCode.DATABASE_ERROR)
def database_health(
    collection: str = typer.Option(
        None,
        "--collection",
        "-C",
        help=f"Collection profile (default: $HADES_DEFAULT_COLLECTION or 'arxiv'). Available: {', '.join(list_profiles())}",
    ),
    start_time: float = typer.Option(0.0, hidden=True),  # Injected by decorator
) -> CLIResponse:
    """Check database health and data integrity.

    Detects common issues:
    - Orphaned chunks (chunks without metadata)
    - Orphaned embeddings (embeddings without chunks)
    - Missing embeddings (chunks without embeddings)
    - Papers with mismatched chunk counts

    Examples:
        hades db health
        hades db health --collection arxiv
    """
    from core.cli.commands.database import check_health

    profile = collection or get_default_profile_name()
    return check_health(start_time, collection=profile)


@db_app.command("check")
@cli_command("database.check", ErrorCode.DATABASE_ERROR)
def database_check(
    document_id: str = typer.Argument(..., help="Document ID to check", metavar="DOCUMENT_ID"),
    collection: str = typer.Option(
        None,
        "--collection",
        "-C",
        help=f"Collection profile (default: $HADES_DEFAULT_COLLECTION or 'arxiv'). Available: {', '.join(list_profiles())}",
    ),
    start_time: float = typer.Option(0.0, hidden=True),  # Injected by decorator
) -> CLIResponse:
    """Check if a document exists in the database."""
    from core.cli.commands.database import check_paper_exists

    profile = collection or get_default_profile_name()
    return check_paper_exists(document_id, start_time, collection=profile)


@db_app.command("purge")
def database_purge(
    document_id: str = typer.Argument(..., help="Document ID to purge", metavar="DOCUMENT_ID"),
    force: bool = typer.Option(
        False,
        "--force",
        "-y",
        help="Skip interactive confirmation (still requires HADES_DESTRUCTIVE_OPS=enabled)",
    ),
    collection: str = typer.Option(
        None,
        "--collection",
        "-C",
        help=f"Collection profile (default: $HADES_DEFAULT_COLLECTION or 'arxiv'). Available: {', '.join(list_profiles())}",
    ),
) -> None:
    """Remove all data for a document from all collections (metadata, chunks, embeddings).

    Requires HADES_DESTRUCTIVE_OPS=enabled environment variable.
    Interactive confirmation is designed to require human involvement when using Claude Code.
    """
    start_time = time.time()
    profile = collection or get_default_profile_name()

    try:
        from core.cli.destructive import check_destructive_allowed

        # Check if destructive operation is allowed
        blocked = check_destructive_allowed(
            command="database.purge",
            operation_desc=f"purge document {document_id}",
            confirm_text=f"PURGE {document_id}",
            start_time=start_time,
            force=force,
        )
        if blocked:
            print_response(blocked)
            raise typer.Exit(1) from None

        from core.cli.commands.database import purge_paper

        response = purge_paper(document_id, start_time, collection=profile)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.purge",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@db_app.command("create")
def database_create(
    name: str = typer.Argument(..., help="Collection name to create", metavar="NAME"),
) -> None:
    """Create a new ArangoDB collection."""
    start_time = time.time()

    try:
        from core.cli.commands.database import create_collection

        response = create_collection(name, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.create",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@db_app.command("delete")
def database_delete(
    collection: str = typer.Argument(..., help="Collection name", metavar="COLLECTION"),
    key: str = typer.Argument(..., help="Document key to delete", metavar="KEY"),
    force: bool = typer.Option(
        False,
        "--force",
        "-y",
        help="Skip interactive confirmation (still requires HADES_DESTRUCTIVE_OPS=enabled)",
    ),
) -> None:
    """Delete a document from an ArangoDB collection.

    Requires HADES_DESTRUCTIVE_OPS=enabled environment variable.
    Interactive confirmation is designed to require human involvement when using Claude Code.
    """
    start_time = time.time()

    try:
        from core.cli.destructive import check_destructive_allowed

        # Check if destructive operation is allowed
        blocked = check_destructive_allowed(
            command="database.delete",
            operation_desc=f"delete {collection}/{key}",
            confirm_text=f"DELETE {collection}/{key}",
            start_time=start_time,
            force=force,
        )
        if blocked:
            print_response(blocked)
            raise typer.Exit(1) from None

        from core.cli.commands.database import delete_document

        response = delete_document(collection, key, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.delete",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@db_app.command("collections")
@cli_command("database.collections", ErrorCode.DATABASE_ERROR)
def database_collections(
    prefix: str = typer.Option(None, "--prefix", "-p", help="Filter by collection name prefix"),
    show_system: bool = typer.Option(False, "--system", "-s", help="Include system collections (starting with '_')"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """List all collections in the database.

    Shows all user collections by default. Use --system to include system collections,
    or --prefix to filter by name prefix.

    Examples:
        hades db collections
        hades db collections --prefix arxiv_
        hades db collections --system
    """
    from core.cli.commands.database import list_collections

    return list_collections(start_time, prefix=prefix, exclude_system=not show_system)


@db_app.command("databases")
@cli_command("database.databases", ErrorCode.DATABASE_ERROR)
def database_databases(
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """List all accessible ArangoDB databases.

    Shows all databases the configured user can access, and marks the
    currently active database.

    Examples:
        hades db databases
        hades --database NL db databases
    """
    from core.cli.commands.database import list_databases

    return list_databases(start_time)


@db_app.command("count")
@cli_command("database.count", ErrorCode.DATABASE_ERROR)
def database_count(
    collection: str = typer.Argument(..., help="Collection name", metavar="COLLECTION"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Count documents in a collection.

    Examples:
        hades db count arxiv_metadata
        hades db count my_nodes
    """
    from core.cli.commands.database import count_collection

    return count_collection(collection, start_time)


@db_app.command("get")
@cli_command("database.get", ErrorCode.DATABASE_ERROR)
def database_get(
    collection: str = typer.Argument(..., help="Collection name", metavar="COLLECTION"),
    key: str = typer.Argument(..., help="Document key", metavar="KEY"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Get a single document by collection and key.

    Examples:
        hades db get arxiv_metadata 2409_04701
        hades db get my_nodes node_001
    """
    from core.cli.commands.database import get_document

    return get_document(collection, key, start_time)


@db_app.command("insert")
def database_insert(
    collection: str = typer.Argument(..., help="Target collection name", metavar="COLLECTION"),
    data: str = typer.Option(None, "--data", "-d", help='JSON document or array (e.g., \'{"name": "test"}\')'),
    file: str = typer.Option(None, "--file", "-f", help="Path to JSONL file (one JSON object per line)"),
) -> None:
    """Insert documents into a collection.

    Creates the collection automatically if it doesn't exist.
    Accepts inline JSON (single object or array) or a JSONL file for bulk insert.

    Examples:
        hades db insert my_nodes --data '{"_key": "n1", "label": "test"}'
        hades db insert my_nodes --data '[{"a": 1}, {"a": 2}]'
        hades db insert my_nodes --file nodes.jsonl
    """
    start_time = time.time()

    try:
        from core.cli.commands.database import insert_documents

        response = insert_documents(collection, data, file, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.insert",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@db_app.command("update")
def database_update(
    collection: str = typer.Argument(..., help="Collection name", metavar="COLLECTION"),
    key: str = typer.Argument(..., help="Document key", metavar="KEY"),
    data: str = typer.Option(..., "--data", "-d", help='JSON fields to merge (e.g., \'{"status": "reviewed"}\')'),
    replace: bool = typer.Option(False, "--replace", help="Replace entire document instead of merging fields"),
) -> None:
    """Update a document by merging fields (or full replace with --replace).

    By default, merges the provided fields into the existing document (PATCH).
    Use --replace to overwrite the entire document (PUT).

    Examples:
        hades db update my_nodes n1 --data '{"confidence": 0.95}'
        hades db update my_nodes n1 --data '{"label": "new"}' --replace
    """
    start_time = time.time()

    try:
        from core.cli.commands.database import update_document

        response = update_document(collection, key, data, start_time, replace=replace)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.update",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@db_app.command("export")
def database_export(
    collection: str = typer.Argument(..., help="Collection name to export", metavar="COLLECTION"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path (default: stdout)"),
    limit: int = typer.Option(None, "--limit", "-n", help="Maximum documents to export"),
) -> None:
    """Export collection documents as JSONL (one JSON object per line).

    Outputs to stdout by default (pipe-friendly). Use --output to write to a file.

    Examples:
        hades db export my_collection > backup.jsonl
        hades db export arxiv_metadata --output papers.jsonl
        hades db export my_nodes --limit 100 -o sample.jsonl
    """
    start_time = time.time()

    try:
        from core.cli.commands.database import export_collection

        response = export_collection(collection, output, start_time, limit=limit)
        # Only print the response JSON if writing to a file (not stdout)
        if output:
            print_response(response)
        elif not response.success:
            # Keep stdout clean for JSONL; surface errors on stderr
            import sys

            print(response.to_json(), file=sys.stderr)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.export",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Vector Index Commands
# =============================================================================


@db_app.command("create-index")
@cli_command("database.create-index", ErrorCode.DATABASE_ERROR)
def database_create_index(
    collection: str = typer.Option(None, "--collection", "-c", help="Collection profile (arxiv, sync, default)"),
    n_lists: int = typer.Option(None, "--n-lists", help="Number of IVF cells (auto if omitted)"),
    n_probe: int = typer.Option(10, "--n-probe", help="Cells to probe per query (recall vs speed)"),
    metric: str = typer.Option("cosine", "--metric", help="Distance metric: cosine, l2, innerProduct"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Create a vector index on the embeddings collection for ANN search.

    Enables server-side approximate nearest neighbor search via ArangoDB's
    FAISS-backed vector index. Falls back to brute-force if not created.

    Examples:
        hades db create-index
        hades db create-index --collection sync
        hades db create-index --n-lists 200 --n-probe 20
        hades db create-index --metric l2
    """
    from core.cli.commands.database import create_vector_index

    return create_vector_index(start_time, collection=collection, n_lists=n_lists, n_probe=n_probe, metric=metric)


@db_app.command("index-status")
@cli_command("database.index-status", ErrorCode.DATABASE_ERROR)
def database_index_status(
    collection: str = typer.Option(None, "--collection", "-c", help="Collection profile (arxiv, sync, default)"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Show vector index status for the embeddings collection.

    Reports whether a vector index exists, its configuration, and the
    current search mode (ann or brute_force).

    Examples:
        hades db index-status
        hades db index-status --collection sync
    """
    from core.cli.commands.database import vector_index_status

    return vector_index_status(start_time, collection=collection)


# =============================================================================
# Graph Commands
# =============================================================================


@graph_app.command("create")
def graph_create_cmd(
    name: str = typer.Option(..., "--name", "-n", help="Graph name"),
    edge_defs: str = typer.Option(
        ...,
        "--edge-defs",
        "-e",
        help='Edge definitions as JSON (e.g., \'[{"collection":"edges","from":["A"],"to":["B"]}]\')',
    ),
) -> None:
    """Create a named graph.

    Examples:
        hades database graph create --name my_graph --edge-defs '[{"collection":"edges","from":["A"],"to":["B"]}]'
    """
    start_time = time.time()

    try:
        from core.cli.commands.database import graph_create

        response = graph_create(name, edge_defs, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.graph.create",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@graph_app.command("list")
def graph_list_cmd() -> None:
    """List all named graphs."""
    start_time = time.time()

    try:
        from core.cli.commands.database import graph_list

        response = graph_list(start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.graph.list",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@graph_app.command("drop")
def graph_drop_cmd(
    name: str = typer.Option(..., "--name", "-n", help="Graph name to drop"),
    drop_collections: bool = typer.Option(False, "--drop-collections", help="Also drop the graph's collections"),
    force: bool = typer.Option(
        False,
        "--force",
        "-y",
        help="Skip interactive confirmation (still requires HADES_DESTRUCTIVE_OPS=enabled)",
    ),
) -> None:
    """Drop a named graph.

    Requires HADES_DESTRUCTIVE_OPS=enabled environment variable.
    Interactive confirmation is designed to require human involvement when using Claude Code.

    Examples:
        hades db graph drop --name my_graph
        hades db graph drop --name my_graph --drop-collections
    """
    start_time = time.time()

    try:
        from core.cli.destructive import check_destructive_allowed

        # Check if destructive operation is allowed
        blocked = check_destructive_allowed(
            command="database.graph.drop",
            operation_desc=f"drop graph {name}" + (" and collections" if drop_collections else ""),
            confirm_text=f"DROP {name}",
            start_time=start_time,
            force=force,
        )
        if blocked:
            print_response(blocked)
            raise typer.Exit(1) from None

        from core.cli.commands.database import graph_drop

        response = graph_drop(name, drop_collections, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.graph.drop",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@graph_app.command("traverse")
def graph_traverse_cmd(
    start: str = typer.Option(..., "--start", "-s", help="Start vertex document ID (e.g., collection/key)"),
    graph: str = typer.Option(..., "--graph", "-g", help="Graph name"),
    direction: str = typer.Option("outbound", "--direction", "-d", help="Traversal direction (outbound, inbound, any)"),
    min_depth: int = typer.Option(1, "--min", help="Minimum traversal depth"),
    max_depth: int = typer.Option(3, "--max", help="Maximum traversal depth"),
    limit: int = typer.Option(100, "--limit", "-n", help="Maximum number of results"),
) -> None:
    """Traverse a named graph from a start vertex.

    Examples:
        hades database graph traverse --start nodes/1 --graph my_graph
        hades database graph traverse --start nodes/1 --graph my_graph --direction any --max 5
    """
    start_time = time.time()

    try:
        from core.cli.commands.database import graph_traverse

        response = graph_traverse(start, graph, direction, min_depth, max_depth, limit, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.graph.traverse",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@graph_app.command("shortest-path")
def graph_shortest_path_cmd(
    from_id: str = typer.Option(..., "--from", help="Source vertex document ID"),
    to_id: str = typer.Option(..., "--to", help="Target vertex document ID"),
    graph: str = typer.Option(..., "--graph", "-g", help="Graph name"),
    direction: str = typer.Option("any", "--direction", "-d", help="Edge direction (outbound, inbound, any)"),
) -> None:
    """Find shortest path between two vertices.

    Examples:
        hades database graph shortest-path --from nodes/1 --to nodes/5 --graph my_graph
    """
    start_time = time.time()

    try:
        from core.cli.commands.database import graph_shortest_path

        response = graph_shortest_path(from_id, to_id, graph, direction, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.graph.shortest-path",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@graph_app.command("neighbors")
def graph_neighbors_cmd(
    start: str = typer.Option(..., "--start", "-s", help="Start vertex document ID"),
    graph: str = typer.Option(..., "--graph", "-g", help="Graph name"),
    direction: str = typer.Option("outbound", "--direction", "-d", help="Edge direction (outbound, inbound, any)"),
    limit: int = typer.Option(100, "--limit", "-n", help="Maximum number of results"),
) -> None:
    """Get immediate neighbors of a vertex.

    Examples:
        hades database graph neighbors --start nodes/1 --graph my_graph
    """
    start_time = time.time()

    try:
        from core.cli.commands.database import graph_neighbors

        response = graph_neighbors(start, graph, direction, limit, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.graph.neighbors",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@graph_app.command("materialize")
def graph_materialize_cmd(
    edge: str | None = typer.Option(None, "--edge", "-e", help="Only materialize this edge collection"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Count edges without inserting"),
    register_graphs: bool = typer.Option(
        False, "--register", "-r", help="Register named graphs (drops and recreates existing definitions)"
    ),
) -> None:
    """Materialize NL graph edges from cross-reference fields.

    Scans NL vertex collections, extracts embedded references (axiom_basis,
    depends_on, inherits_from, etc.), and creates native ArangoDB edges.

    Examples:
        hades --database NL db graph materialize --dry-run
        hades --database NL db graph materialize
        hades --database NL db graph materialize --edge nl_axiom_basis_edges
        hades --database NL db graph materialize --register
    """
    start_time = time.time()

    try:
        from core.cli.commands.database import graph_materialize

        response = graph_materialize(start_time, edge=edge, dry_run=dry_run, register_graphs=register_graphs)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="database.graph.materialize",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Embedding Service Commands
# =============================================================================


@embed_service_app.command("status")
def embedding_service_status() -> None:
    """Check embedding service health and status."""
    start_time = time.time()

    try:
        from core.cli.commands.embedding import service_status

        response = service_status(start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="embedding.service.status",
            code=ErrorCode.SERVICE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@embed_service_app.command("start")
def embedding_service_start() -> None:
    """Start the embedding service daemon."""
    start_time = time.time()

    try:
        from core.cli.commands.embedding import service_start

        response = service_start(start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="embedding.service.start",
            code=ErrorCode.SERVICE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@embed_service_app.command("stop")
def embedding_service_stop(
    token: str = typer.Option(None, "--token", "-t", help="Shutdown token (if configured)"),
) -> None:
    """Stop the embedding service daemon."""
    start_time = time.time()

    try:
        from core.cli.commands.embedding import service_stop

        response = service_stop(start_time, token=token)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="embedding.service.stop",
            code=ErrorCode.SERVICE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@embed_app.command("text")
def embedding_text(
    text: str = typer.Argument(..., help="Text to embed", metavar="TEXT"),
    task: str = typer.Option(
        "retrieval.passage", "--task", "-t", help="Task type (retrieval.passage, retrieval.query)"
    ),
    raw: bool = typer.Option(False, "--raw", "-r", help="Output raw embedding array only"),
) -> None:
    """Embed a text string using the embedding service."""
    start_time = time.time()

    try:
        from core.cli.commands.embedding import embed_text

        output_format = "raw" if raw else "json"
        response = embed_text(text, start_time, task=task, output_format=output_format)

        # Always print errors, only print success in non-raw mode
        if not response.success:
            print_response(response)
            raise typer.Exit(1) from None
        elif output_format != "raw":
            print_response(response)

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="embedding.text",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@embed_gpu_app.command("status")
def embedding_gpu_status() -> None:
    """Show GPU status and memory usage."""
    start_time = time.time()

    try:
        from core.cli.commands.embedding import gpu_status

        response = gpu_status(start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="embedding.gpu.status",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@embed_gpu_app.command("list")
def embedding_gpu_list() -> None:
    """List available GPUs."""
    start_time = time.time()

    try:
        from core.cli.commands.embedding import gpu_list

        response = gpu_list(start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="embedding.gpu.list",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Persephone Task Commands
# =============================================================================


@task_app.command("create")
@cli_command("task.create", ErrorCode.TASK_ERROR)
def task_create_cmd(
    title: str = typer.Argument(..., help="Task title", metavar="TITLE"),
    description: str = typer.Option(None, "--description", "-d", help="Detailed description"),
    priority: str = typer.Option("medium", "--priority", "-p", help="Priority: critical|high|medium|low"),
    type_: str = typer.Option("task", "--type", "-t", help="Type: task|bug|epic"),
    labels: str = typer.Option(None, "--labels", "-l", help="Comma-separated labels"),
    parent: str = typer.Option(None, "--parent", help="Parent task key (for epics)"),
    acceptance: str = typer.Option(None, "--acceptance", "-a", help="Acceptance criteria"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Create a new task.

    Examples:
        hades task create "Fix login bug" --priority high --type bug
        hades task create "Implement search" --description "Add semantic search" --labels "feature,search"
        hades task create "Subtask" --parent task_abc123
    """
    from core.cli.commands.persephone import task_create

    label_list = [s.strip() for s in labels.split(",") if s.strip()] if labels else None
    return task_create(
        title,
        start_time,
        description=description,
        priority=priority,
        type_=type_,
        labels=label_list,
        parent_key=parent,
        acceptance=acceptance,
    )


@task_app.command("list")
@cli_command("task.list", ErrorCode.TASK_ERROR)
def task_list_cmd(
    status: str = typer.Option(
        None, "--status", "-s", help="Filter by status: open|in_progress|in_review|closed|blocked"
    ),
    priority: str = typer.Option(None, "--priority", "-p", help="Filter by priority: critical|high|medium|low"),
    type_: str = typer.Option(None, "--type", "-t", help="Filter by type: task|bug|epic"),
    parent: str = typer.Option(None, "--parent", help="Filter by parent task key"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum results"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """List tasks with optional filters.

    Examples:
        hades task list
        hades task list --status open --priority high
        hades task list --type bug --limit 10
    """
    from core.cli.commands.persephone import task_list

    return task_list(
        start_time,
        status=status,
        priority=priority,
        type_=type_,
        parent_key=parent,
        limit=limit,
    )


@task_app.command("show")
@cli_command("task.show", ErrorCode.TASK_ERROR)
def task_show_cmd(
    key: str = typer.Argument(..., help="Task key (e.g., task_abc123)", metavar="KEY"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Show details of a single task.

    Examples:
        hades task show task_abc123
    """
    from core.cli.commands.persephone import task_show

    return task_show(key, start_time)


@task_app.command("update")
@cli_command("task.update", ErrorCode.TASK_ERROR)
def task_update_cmd(
    key: str = typer.Argument(..., help="Task key", metavar="KEY"),
    title: str = typer.Option(None, "--title", help="New title"),
    description: str = typer.Option(None, "--description", "-d", help="New description"),
    status: str = typer.Option(None, "--status", "-s", help="New status: open|in_progress|in_review|closed|blocked"),
    priority: str = typer.Option(None, "--priority", "-p", help="New priority: critical|high|medium|low"),
    type_: str = typer.Option(None, "--type", "-t", help="New type: task|bug|epic"),
    labels: str = typer.Option(None, "--labels", "-l", help="Replace labels (comma-separated)"),
    acceptance: str = typer.Option(None, "--acceptance", "-a", help="New acceptance criteria"),
    minor: bool = typer.Option(None, "--minor/--no-minor", help="Mark as minor task"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Update fields on an existing task.

    Only specified fields are changed; others are left as-is.

    Examples:
        hades task update task_abc123 --status in_progress
        hades task update task_abc123 --priority critical --labels "urgent,p0"
    """
    from core.cli.commands.persephone import task_update

    fields: dict = {}
    if title is not None:
        fields["title"] = title
    if description is not None:
        fields["description"] = description
    if status is not None:
        fields["status"] = status
    if priority is not None:
        fields["priority"] = priority
    if type_ is not None:
        fields["type"] = type_
    if labels is not None:
        fields["labels"] = [s.strip() for s in labels.split(",") if s.strip()]
    if acceptance is not None:
        fields["acceptance"] = acceptance
    if minor is not None:
        fields["minor"] = minor

    if not fields:
        return error_response(
            command="task.update",
            code=ErrorCode.VALIDATION_ERROR,
            message="No fields to update. Specify at least one option.",
            start_time=start_time,
        )

    return task_update(key, start_time, **fields)


@task_app.command("close")
@cli_command("task.close", ErrorCode.TASK_ERROR)
def task_close_cmd(
    key: str = typer.Argument(..., help="Task key to close", metavar="KEY"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Close a task (shorthand for update --status closed, no guards).

    For guarded closure (reviewer != implementer), use 'hades task approve'.

    Examples:
        hades task close task_abc123
    """
    from core.cli.commands.persephone import task_close

    return task_close(key, start_time)


@task_app.command("usage")
@cli_command("task.usage", ErrorCode.TASK_ERROR)
def task_usage_cmd(
    new_session: bool = typer.Option(False, "--new-session", help="Force a new session"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Get session briefing: current session, active tasks, and ready work.

    Auto-detects the calling agent (Claude Code, Cursor, etc.) and
    creates or resumes a session. Returns in-progress tasks, reviewable
    tasks, and open work sorted by priority.

    Examples:
        hades task usage
        hades task usage --new-session
    """
    from core.cli.commands.persephone import task_usage

    return task_usage(start_time, new_session=new_session)


# --- Workflow commands (Phase 3: state machine) ---


@task_app.command("start")
@cli_command("task.transition", ErrorCode.TASK_ERROR)
def task_start_cmd(
    key: str = typer.Argument(..., help="Task key", metavar="KEY"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Start working on a task (open → in_progress).

    Creates a session-task edge and enforces dependency guards.

    Examples:
        hades task start task_abc123
    """
    from core.cli.commands.persephone import task_transition

    return task_transition(key, "in_progress", start_time)


@task_app.command("review")
@cli_command("task.transition", ErrorCode.TASK_ERROR)
def task_review_cmd(
    key: str = typer.Argument(..., help="Task key", metavar="KEY"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Submit a task for review (in_progress → in_review).

    Examples:
        hades task review task_abc123
    """
    from core.cli.commands.persephone import task_transition

    return task_transition(key, "in_review", start_time)


@task_app.command("approve")
@cli_command("task.transition", ErrorCode.TASK_ERROR)
def task_approve_cmd(
    key: str = typer.Argument(..., help="Task key", metavar="KEY"),
    human: bool = typer.Option(False, "--human", help="Human auditor override (bypasses reviewer guard)"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Approve and close a task (in_review → closed).

    By default, the approving session must be different from the implementing
    session. Use --human when you (the human auditor) are approving directly.

    Examples:
        hades task approve task_abc123
        hades task approve task_abc123 --human
    """
    from core.cli.commands.persephone import task_transition

    return task_transition(key, "closed", start_time, human_override=human)


@task_app.command("block")
@cli_command("task.transition", ErrorCode.TASK_ERROR)
def task_block_cmd(
    key: str = typer.Argument(..., help="Task key", metavar="KEY"),
    reason: str = typer.Option(..., "--reason", "-r", help="Why this task is blocked"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Block a task (in_progress → blocked).

    Examples:
        hades task block task_abc123 --reason "waiting on embedder fix"
    """
    from core.cli.commands.persephone import task_transition

    return task_transition(key, "blocked", start_time, block_reason=reason)


@task_app.command("unblock")
@cli_command("task.transition", ErrorCode.TASK_ERROR)
def task_unblock_cmd(
    key: str = typer.Argument(..., help="Task key", metavar="KEY"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Unblock a task (blocked → in_progress).

    Examples:
        hades task unblock task_abc123
    """
    from core.cli.commands.persephone import task_transition

    return task_transition(key, "in_progress", start_time)


@task_app.command("dep")
@cli_command("task.dep", ErrorCode.TASK_ERROR)
def task_dep_cmd(
    key: str = typer.Argument(..., help="Task key", metavar="KEY"),
    blocked_by: str = typer.Option(None, "--blocked-by", "-b", help="Task key that blocks this one"),
    remove: str = typer.Option(None, "--remove", help="Remove dependency on this task key"),
    start_time: float = typer.Option(0.0, hidden=True),
) -> CLIResponse:
    """Manage task dependencies.

    Examples:
        hades task dep task_abc --blocked-by task_xyz
        hades task dep task_abc --remove task_xyz
        hades task dep task_abc    # show blockers
    """
    from core.cli.commands.persephone import task_blocked, task_dep_add, task_dep_remove

    if blocked_by:
        return task_dep_add(key, blocked_by, start_time)
    elif remove:
        return task_dep_remove(key, remove, start_time)
    else:
        return task_blocked(key, start_time)


# =============================================================================
# Version and Help
# =============================================================================


if __name__ == "__main__":
    app()
