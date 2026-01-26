"""HADES CLI - Main entry point.

AI-focused CLI for interacting with the HADES knowledge base.
All commands output JSON for predictable parsing by AI models.

Usage:
    hades arxiv search "query"     # Search arxiv for papers
    hades arxiv info <arxiv_id>    # Get paper metadata
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

# Create subcommand groups
arxiv_app = typer.Typer(
    name="arxiv",
    help="ArXiv paper search and metadata.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
app.add_typer(arxiv_app, name="arxiv")


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


@app.command("sync")
def sync(
    from_date: str = typer.Option(None, "--from", "-f", help="Start date (YYYY-MM-DD, default: 7 days ago)"),
    categories: str = typer.Option(None, "--categories", "-c", help="Comma-separated arxiv categories (e.g., cs.AI,cs.CL)"),
    max_results: int = typer.Option(1000, "--max", "-m", help="Maximum papers to sync"),
    batch_size: int = typer.Option(8, "--batch", "-b", help="Batch size for embedding (default 8 for 16GB GPU)"),
    gpu: int = typer.Option(None, "--gpu", "-g", help="GPU device index to use (e.g., 0, 1, 2)"),
) -> None:
    """Sync recent abstracts from arxiv for semantic search.

    Fetches metadata and abstracts, embeds them with Jina, and stores
    for fast semantic search - WITHOUT downloading full PDFs.

    Use this to keep your abstract database current, then use 'hades ingest'
    to download full papers you're interested in.

    Examples:
        hades sync --gpu 2 --batch 8
        hades sync --from 2025-01-01 --categories cs.AI,cs.CL --gpu 0
    """
    _set_gpu(gpu)
    start_time = time.time()

    try:
        from core.cli.commands.sync import sync_abstracts

        response = sync_abstracts(from_date, categories, max_results, batch_size, start_time)
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
# Version and Help
# =============================================================================


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit"),
) -> None:
    """HADES Knowledge Base CLI.

    An AI-focused interface for managing and querying the HADES semantic knowledge base.
    All commands output JSON for predictable parsing.
    """
    if version:
        from importlib.metadata import version as pkg_version

        try:
            v = pkg_version("hades")
        except Exception:
            v = "0.1.0-dev"
        print(f'{{"version": "{v}"}}')
        raise typer.Exit()

    # If no subcommand provided, show help
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())


if __name__ == "__main__":
    app()
