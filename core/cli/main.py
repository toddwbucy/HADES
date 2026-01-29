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
) -> None:
    """HADES Knowledge Base CLI - AI model interface for semantic search over academic papers."""
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
    help="ArXiv paper search, abstract search, sync, and ingestion.",
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


# =============================================================================
# Top-Level Commands (Standalone Tools)
# =============================================================================


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

        # Handle resume-only mode (no inputs required)
        actual_inputs = inputs or []

        response = ingest(
            actual_inputs,
            document_id=id,
            force=force,
            batch=batch,
            resume=resume,
            start_time=start_time,
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


@arxiv_app.command("search")
@cli_command("arxiv.search", ErrorCode.SEARCH_FAILED)
def arxiv_search(
    query: str = typer.Argument(..., help="Search query for arxiv papers", metavar="QUERY"),
    max_results: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    categories: str = typer.Option(
        None, "--categories", "-c", help="Comma-separated arxiv categories (e.g., cs.AI,cs.CL)"
    ),
    start_time: float = typer.Option(0.0, hidden=True),  # Injected by decorator
) -> CLIResponse:
    """Search arxiv for papers matching a query.

    Returns paper metadata including abstracts for review before ingestion.
    """
    from core.cli.commands.arxiv import search_arxiv

    return search_arxiv(query, max_results, categories, start_time)


@arxiv_app.command("info")
@cli_command("arxiv.info", ErrorCode.PAPER_NOT_FOUND)
def arxiv_info(
    arxiv_id: str = typer.Argument(..., help="ArXiv paper ID (e.g., 2401.12345)", metavar="ARXIV_ID"),
    start_time: float = typer.Option(0.0, hidden=True),  # Injected by decorator
) -> CLIResponse:
    """Get detailed metadata for a specific arxiv paper."""
    from core.cli.commands.arxiv import get_paper_info

    return get_paper_info(arxiv_id, start_time)


@arxiv_app.command("abstract")
def arxiv_abstract(
    query: str = typer.Argument(..., help="Search query for abstracts", metavar="QUERY"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by arxiv category (e.g., cs.AI)"),
    hybrid: bool = typer.Option(False, "--hybrid", "-H", help="Combine semantic search with keyword matching"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Search the synced abstract database (2.8M papers).

    Performs semantic search over all synced abstracts and shows whether
    each paper has been fully ingested locally.

    Examples:
        hades arxiv abstract "transformer attention" --limit 20
        hades arxiv abstract "neural networks" --category cs.LG
        hades arxiv abstract "BERT fine-tuning" --hybrid  # semantic + keyword
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.arxiv import search_abstracts

        response = search_abstracts(query, limit, start_time, category=category, hybrid=hybrid)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="arxiv.abstract",
            code=ErrorCode.SEARCH_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@arxiv_app.command("bulk-search")
def arxiv_bulk_search(
    queries: list[str] = typer.Argument(..., help="Search queries (multiple allowed)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results per query"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by arxiv category (e.g., cs.AI)"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Search abstracts with multiple queries in a single optimized pass.

    More efficient than running multiple searches because:
    - Loads embedding model once (not N times)
    - Single pass over 2.8M embeddings (not N passes)
    - Uses matrix multiplication for batch similarity

    Examples:
        hades arxiv bulk-search "attention" "transformer" "BERT"
        hades arxiv bulk-search "neural networks" "deep learning" --limit 5
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.arxiv import search_abstracts_bulk

        response = search_abstracts_bulk(queries, limit, start_time, category=category)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="arxiv.bulk-search",
            code=ErrorCode.SEARCH_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@arxiv_app.command("similar")
def arxiv_similar(
    arxiv_id: str = typer.Argument(..., help="ArXiv paper ID to find similar papers for", metavar="ARXIV_ID"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by arxiv category (e.g., cs.AI)"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Find papers similar to a given paper.

    Uses the paper's existing embedding to find semantically similar papers
    without needing to generate a new query embedding.

    Examples:
        hades arxiv similar 2401.12345 --limit 20
        hades arxiv similar 2401.12345 --category cs.LG
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.arxiv import find_similar

        response = find_similar(arxiv_id, limit, start_time, category=category)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="arxiv.similar",
            code=ErrorCode.SEARCH_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@arxiv_app.command("refine")
def arxiv_refine(
    query: str = typer.Argument(..., help="Original search query", metavar="QUERY"),
    positive: list[str] = typer.Option(..., "--positive", "-p", help="Relevant paper IDs (positive exemplars)"),
    negative: list[str] | None = typer.Option(
        None, "--negative", "-x", help="Irrelevant paper IDs (negative exemplars)"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results"),
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by arxiv category (e.g., cs.AI)"),
    alpha: float = typer.Option(1.0, "--alpha", help="Weight for original query (default 1.0)"),
    beta: float = typer.Option(0.75, "--beta", help="Weight for positive exemplars (default 0.75)"),
    gamma: float = typer.Option(0.15, "--gamma", help="Weight for negative exemplars (default 0.15)"),
    gpu: int | None = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Refine search using relevance feedback (Rocchio algorithm).

    After an initial search, mark relevant papers as positive exemplars to
    refine the query. The refined search uses a weighted combination of
    the original query and the exemplar embeddings.

    Rocchio formula: q' = alpha*query + beta*mean(positive) - gamma*mean(negative)

    Examples:
        # Refine with positive exemplars only
        hades arxiv refine "transformer attention" -p 2401.12345 -p 2401.67890

        # Refine with both positive and negative exemplars
        hades arxiv refine "neural embeddings" -p 2401.12345 -x 2402.99999

        # Custom weights for more aggressive refinement
        hades arxiv refine "late chunking" -p 2401.12345 --alpha 0.5 --beta 1.0
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.arxiv import refine_search

        response = refine_search(
            query=query,
            positive_ids=positive,
            limit=limit,
            start_time=start_time,
            negative_ids=negative,
            category=category,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="arxiv.refine",
            code=ErrorCode.SEARCH_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


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
    paper: str = typer.Option(None, "--paper", "-p", help="Filter results to a specific paper (arxiv ID)"),
    context: int = typer.Option(0, "--context", "-c", help="Include N adjacent chunks for context"),
    cite: bool = typer.Option(False, "--cite", help="Output minimal citation format (arxiv_id, title, quote)"),
    chunks_only: bool = typer.Option(False, "--chunks", help="Get all chunks for --paper (no semantic search)"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Semantic search over the knowledge base.

    Returns relevant text chunks with similarity scores.

    Examples:
        hades database query "attention mechanism"              # Search all papers
        hades database query "Newton-Schulz" --paper 2505.23735 # Search within paper
        hades database query "attention" --context 1            # Include ±1 adjacent chunks
        hades db query "attention" --cite --limit 3             # Citation format, top 3
        hades database query --paper 2505.23735 --chunks        # Get all chunks (no search)
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.database import get_paper_chunks, semantic_query

        # Mode 1: Get all chunks for a paper (no semantic search)
        if chunks_only and paper:
            response = get_paper_chunks(paper, limit, start_time)
        # Mode 2: Semantic search (optionally filtered by paper)
        elif search_text:
            response = semantic_query(
                search_text,
                limit,
                start_time,
                paper_filter=paper,
                context=context,
                cite_only=cite,
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
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of papers to list"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by arxiv category"),
) -> None:
    """List papers stored in the database."""
    start_time = time.time()

    try:
        from core.cli.commands.database import list_stored_papers

        response = list_stored_papers(limit, category, start_time)
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
    start_time: float = typer.Option(0.0, hidden=True),  # Injected by decorator
) -> CLIResponse:
    """Show database statistics."""
    from core.cli.commands.database import get_stats

    return get_stats(start_time)


@db_app.command("check")
@cli_command("database.check", ErrorCode.DATABASE_ERROR)
def database_check(
    arxiv_id: str = typer.Argument(..., help="ArXiv paper ID to check", metavar="ARXIV_ID"),
    start_time: float = typer.Option(0.0, hidden=True),  # Injected by decorator
) -> CLIResponse:
    """Check if a paper exists in the database."""
    from core.cli.commands.database import check_paper_exists

    return check_paper_exists(arxiv_id, start_time)


@db_app.command("purge")
def database_purge(
    arxiv_id: str = typer.Argument(..., help="ArXiv paper ID to purge", metavar="ARXIV_ID"),
    force: bool = typer.Option(
        False,
        "--force",
        "-y",
        help="Skip interactive confirmation (still requires HADES_DESTRUCTIVE_OPS=enabled)",
    ),
) -> None:
    """Remove all data for a paper from all collections (metadata, chunks, embeddings).

    Requires HADES_DESTRUCTIVE_OPS=enabled environment variable.
    Interactive confirmation is designed to require human involvement when using Claude Code.
    """
    start_time = time.time()

    try:
        from core.cli.destructive import check_destructive_allowed

        # Check if destructive operation is allowed
        blocked = check_destructive_allowed(
            command="database.purge",
            operation_desc=f"purge paper {arxiv_id}",
            confirm_text=f"PURGE {arxiv_id}",
            start_time=start_time,
            force=force,
        )
        if blocked:
            print_response(blocked)
            raise typer.Exit(1) from None

        from core.cli.commands.database import purge_paper

        response = purge_paper(arxiv_id, start_time)
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
# Version and Help
# =============================================================================


if __name__ == "__main__":
    app()
