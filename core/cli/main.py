"""HADES CLI - Main entry point.

AI-focused CLI for interacting with the HADES knowledge base.
All commands output JSON for predictable parsing by AI models.

Usage:
    hades arxiv search "query"     # Search arxiv API for papers
    hades arxiv info <arxiv_id>    # Get paper metadata from arxiv
    hades abstract search "query"  # Search 2.8M synced abstracts
    hades abstract ingest <id>...  # Ingest papers from abstract search
    hades ingest <arxiv_id>...     # Download, process, store papers
    hades query "search text"      # Semantic search over stored chunks
    hades list                     # List papers in database
    hades stats                    # Database statistics
    hades check <arxiv_id>         # Check if paper exists in DB
"""

from __future__ import annotations

import os
import time

import typer

from core.cli.output import (
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
    add_completion=False,
    rich_markup_mode=None,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit", is_eager=True),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index for embedding commands (e.g., 0, 1, 2)"),
) -> None:
    """HADES Knowledge Base CLI - AI model interface for semantic search over academic papers."""
    if version:
        from importlib.metadata import version as get_version
        try:
            print(f"hades {get_version('hades')}")
        except Exception:
            print("hades 0.1.0")
        raise typer.Exit()

    _global_gpu_callback(ctx, gpu)

    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()

# Create subcommand groups
arxiv_app = typer.Typer(
    name="arxiv",
    help="ArXiv paper search and metadata.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
app.add_typer(arxiv_app, name="arxiv")

abstract_app = typer.Typer(
    name="abstract",
    help="Search synced abstracts (2.8M) and ingest papers.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
app.add_typer(abstract_app, name="abstract")

# Embedding service commands
embedding_app = typer.Typer(
    name="embedding",
    help="Embedding service management and direct text embedding.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
app.add_typer(embedding_app, name="embedding")

embedding_service_app = typer.Typer(
    name="service",
    help="Manage the embedding service daemon.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
embedding_app.add_typer(embedding_service_app, name="service")

embedding_gpu_app = typer.Typer(
    name="gpu",
    help="GPU status and management.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
embedding_app.add_typer(embedding_gpu_app, name="gpu")


# =============================================================================
# ArXiv Commands
# =============================================================================


@arxiv_app.command("search")
def arxiv_search(
    query: str = typer.Argument(..., help="Search query for arxiv papers", metavar="QUERY"),
    max_results: int = typer.Option(10, "--max", "-m", help="Maximum number of results"),
    categories: str = typer.Option(None, "--categories", "-c", help="Comma-separated arxiv categories (e.g., cs.AI,cs.CL)"),
) -> None:
    """Search arxiv for papers matching a query.

    Returns paper metadata including abstracts for review before ingestion.
    """
    start_time = time.time()

    try:
        from core.cli.commands.arxiv import search_arxiv

        response = search_arxiv(query, max_results, categories, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="arxiv.search",
            code=ErrorCode.SEARCH_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@arxiv_app.command("info")
def arxiv_info(
    arxiv_id: str = typer.Argument(..., help="ArXiv paper ID (e.g., 2401.12345)", metavar="ARXIV_ID"),
) -> None:
    """Get detailed metadata for a specific arxiv paper."""
    start_time = time.time()

    try:
        from core.cli.commands.arxiv import get_paper_info

        response = get_paper_info(arxiv_id, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="arxiv.info",
            code=ErrorCode.PAPER_NOT_FOUND,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Abstract Commands (search 2.8M synced abstracts)
# =============================================================================


@abstract_app.command("search")
def abstract_search(
    query: str = typer.Argument(..., help="Search query for abstracts", metavar="QUERY"),
    limit: int = typer.Option(10, "--limit", "-n", "--top-k", "-k", help="Maximum number of results"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by arxiv category (e.g., cs.AI)"),
    hybrid: bool = typer.Option(False, "--hybrid", "-H", help="Combine semantic search with keyword matching"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Search the synced abstract database (2.8M papers).

    Performs semantic search over all synced abstracts and shows whether
    each paper has been fully ingested locally.

    Examples:
        hades abstract search "transformer attention" --limit 20
        hades abstract search "neural networks" --category cs.LG
        hades abstract search "BERT fine-tuning" --hybrid  # semantic + keyword
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.abstract import search_abstracts

        response = search_abstracts(query, limit, start_time, category=category, hybrid=hybrid)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="abstract.search",
            code=ErrorCode.SEARCH_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@abstract_app.command("search-bulk")
def abstract_search_bulk(
    queries: list[str] = typer.Argument(..., help="Search queries (multiple allowed)"),
    limit: int = typer.Option(10, "--limit", "-n", "--top-k", "-k", help="Maximum results per query"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by arxiv category (e.g., cs.AI)"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Search abstracts with multiple queries in a single optimized pass.

    More efficient than running multiple searches because:
    - Loads embedding model once (not N times)
    - Single pass over 2.8M embeddings (not N passes)
    - Uses matrix multiplication for batch similarity

    Examples:
        hades abstract search-bulk "attention" "transformer" "BERT"
        hades abstract search-bulk "neural networks" "deep learning" --limit 5
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.abstract import search_abstracts_bulk

        response = search_abstracts_bulk(queries, limit, start_time, category=category)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="abstract.search-bulk",
            code=ErrorCode.SEARCH_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@abstract_app.command("ingest")
def abstract_ingest(
    arxiv_ids: list[str] = typer.Argument(..., help="ArXiv paper IDs to ingest"),
    force: bool = typer.Option(False, "--force", help="Force reprocessing even if already exists"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Ingest papers identified from abstract search.

    Downloads PDFs, extracts text, generates embeddings, and stores in ArangoDB.
    Use after 'hades abstract search' to download interesting papers.

    Examples:
        hades abstract ingest 2401.12345 2401.67890
        hades abstract ingest 2401.12345 --gpu 2
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.abstract import ingest_from_abstract

        response = ingest_from_abstract(arxiv_ids, force, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="abstract.ingest",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@abstract_app.command("similar")
def abstract_similar(
    arxiv_id: str = typer.Argument(..., help="ArXiv paper ID to find similar papers for", metavar="ARXIV_ID"),
    limit: int = typer.Option(10, "--limit", "-n", "--top-k", "-k", help="Maximum number of results"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by arxiv category (e.g., cs.AI)"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Find papers similar to a given paper.

    Uses the paper's existing embedding to find semantically similar papers
    without needing to generate a new query embedding.

    Examples:
        hades abstract similar 2401.12345 --limit 20
        hades abstract similar 2401.12345 --category cs.LG
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.abstract import find_similar

        response = find_similar(arxiv_id, limit, start_time, category=category)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="abstract.similar",
            code=ErrorCode.SEARCH_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@abstract_app.command("refine")
def abstract_refine(
    query: str = typer.Argument(..., help="Original search query", metavar="QUERY"),
    positive: list[str] = typer.Option(
        ..., "--positive", "-p", help="Relevant paper IDs (positive exemplars)"
    ),
    negative: list[str] | None = typer.Option(
        None, "--negative", "-x", help="Irrelevant paper IDs (negative exemplars)"
    ),
    limit: int = typer.Option(10, "--limit", "-n", "--top-k", "-k", help="Maximum number of results"),
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
        hades abstract refine "transformer attention" -p 2401.12345 -p 2401.67890

        # Refine with both positive and negative exemplars
        hades abstract refine "neural embeddings" -p 2401.12345 -x 2402.99999

        # Custom weights for more aggressive refinement
        hades abstract refine "late chunking" -p 2401.12345 --alpha 0.5 --beta 1.0
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.abstract import refine_search

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
            command="abstract.refine",
            code=ErrorCode.SEARCH_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Ingest Commands
# =============================================================================


@app.command("ingest")
def ingest(
    arxiv_ids: list[str] = typer.Argument(None, help="ArXiv paper IDs to ingest"),
    file: str = typer.Option(None, "--file", "-f", help="Path to local PDF file"),
    force: bool = typer.Option(False, "--force", help="Force reprocessing even if already exists"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Ingest papers into the knowledge base.

    Downloads PDFs, extracts text, generates embeddings, and stores in ArangoDB.

    Examples:
        hades ingest 2401.12345 2401.67890
        hades ingest --file /path/to/paper.pdf
        hades ingest 2401.12345 --gpu 2
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.ingest import ingest_papers

        if file:
            response = ingest_papers(pdf_paths=[file], force=force, start_time=start_time)
        elif arxiv_ids:
            response = ingest_papers(arxiv_ids=arxiv_ids, force=force, start_time=start_time)
        else:
            response = error_response(
                command="ingest",
                code=ErrorCode.CONFIG_ERROR,
                message="Provide either arxiv IDs or --file path",
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
            command="ingest",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Query Commands
# =============================================================================


@app.command("query")
def query(
    search_text: str = typer.Argument(None, help="Search query text", metavar="SEARCH_TEXT"),
    limit: int = typer.Option(10, "--limit", "-n", "--top-k", "-k", help="Maximum number of results"),
    paper: str = typer.Option(None, "--paper", "-p", help="Filter results to a specific paper (arxiv ID)"),
    context: int = typer.Option(0, "--context", "-c", help="Include N adjacent chunks for context"),
    cite: bool = typer.Option(False, "--cite", help="Output minimal citation format (arxiv_id, title, quote)"),
    chunks_only: bool = typer.Option(False, "--chunks", help="Get all chunks for --paper (no semantic search)"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Semantic search over the knowledge base.

    Returns relevant text chunks with similarity scores.

    Examples:
        hades query "attention mechanism"              # Search all papers
        hades query "Newton-Schulz" --paper 2505.23735 # Search within paper
        hades query "attention" --context 1            # Include ±1 adjacent chunks
        hades query "attention" --cite --top-k 3       # Citation format, top 3
        hades query --paper 2505.23735 --chunks        # Get all chunks (no search)
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.query import get_paper_chunks, semantic_query

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
                command="query",
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
            command="query",
            code=ErrorCode.QUERY_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Sync Commands
# =============================================================================


# Create sync subcommand group
sync_app = typer.Typer(
    name="sync",
    help="Sync abstracts from arxiv (incremental or manual).",
    no_args_is_help=False,
    invoke_without_command=True,
    rich_markup_mode=None,
)
app.add_typer(sync_app, name="sync")


@sync_app.callback(invoke_without_command=True)
def sync_default(
    ctx: typer.Context,
    from_date: str = typer.Option(None, "--from", "-f", help="Start date (YYYY-MM-DD, default: 7 days ago)"),
    categories: str = typer.Option(None, "--categories", "-c", help="Comma-separated arxiv categories (e.g., cs.AI,cs.CL)"),
    max_results: int = typer.Option(1000, "--max", "-m", help="Maximum papers to sync"),
    batch_size: int = typer.Option(8, "--batch", "-b", help="Batch size for embedding (default 8 for 16GB GPU)"),
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Sync only papers newer than last sync"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Sync recent abstracts from arxiv for semantic search.

    Fetches metadata and abstracts, embeds them with Jina, and stores
    for fast semantic search - WITHOUT downloading full PDFs.

    Use this to keep your abstract database current, then use 'hades ingest'
    to download full papers you're interested in.

    Examples:
        hades sync --gpu 2 --batch 8
        hades sync --incremental                    # Sync since last sync
        hades sync --from 2025-01-01 --categories cs.AI,cs.CL --gpu 0
    """
    # Only run sync if no subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.sync import sync_abstracts

        response = sync_abstracts(
            from_date, categories, max_results, batch_size, start_time, incremental=incremental
        )
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="sync",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@sync_app.command("status")
def sync_status() -> None:
    """Show sync status including last sync time and history.

    Examples:
        hades sync status
    """
    start_time = time.time()

    try:
        from core.cli.commands.sync import get_sync_status

        response = get_sync_status(start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="sync.status",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Database Management Commands
# =============================================================================


@app.command("list")
def list_papers(
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of papers to list"),
    category: str = typer.Option(None, "--category", "-c", help="Filter by arxiv category"),
) -> None:
    """List papers stored in the database."""
    start_time = time.time()

    try:
        from core.cli.commands.db import list_stored_papers

        response = list_stored_papers(limit, category, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="list",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@app.command("stats")
def stats() -> None:
    """Show database statistics."""
    start_time = time.time()

    try:
        from core.cli.commands.db import get_stats

        response = get_stats(start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="stats",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


@app.command("check")
def check(
    arxiv_id: str = typer.Argument(..., help="ArXiv paper ID to check", metavar="ARXIV_ID"),
) -> None:
    """Check if a paper exists in the database."""
    start_time = time.time()

    try:
        from core.cli.commands.db import check_paper_exists

        response = check_paper_exists(arxiv_id, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="check",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None


# =============================================================================
# Embedding Service Commands
# =============================================================================


@embedding_service_app.command("status")
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


@embedding_service_app.command("start")
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


@embedding_service_app.command("stop")
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


@embedding_app.command("text")
def embedding_text(
    text: str = typer.Argument(..., help="Text to embed", metavar="TEXT"),
    task: str = typer.Option("retrieval.passage", "--task", "-t", help="Task type (retrieval.passage, retrieval.query)"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Output raw embedding array only"),
) -> None:
    """Embed a text string using the embedding service."""
    start_time = time.time()

    try:
        from core.cli.commands.embedding import embed_text

        output_format = "raw" if raw else "json"
        response = embed_text(text, start_time, task=task, output_format=output_format)
        if output_format != "raw":
            print_response(response)
        if not response.success:
            raise typer.Exit(1) from None

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


@embedding_gpu_app.command("status")
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


@embedding_gpu_app.command("list")
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
