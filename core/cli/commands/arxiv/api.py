"""ArXiv API commands for HADES CLI.

Provides direct interaction with the ArXiv API for searching papers
and fetching metadata. These are live API calls, not searches over
the synced abstract database.
"""

from __future__ import annotations

from typing import Any

from defusedxml import ElementTree as ET

from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    progress,
    success_response,
)
from core.tools.arxiv.arxiv_api_client import ArXivAPIClient, ArXivMetadata


def search_arxiv(
    query: str,
    max_results: int,
    categories: str | None,
    start_time: float,
) -> CLIResponse:
    """Search arxiv for papers matching a query.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        categories: Comma-separated arxiv categories (e.g., "cs.AI,cs.CL")
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with search results
    """
    # Validate max_results
    if max_results <= 0:
        return error_response(
            command="arxiv.search",
            code=ErrorCode.CONFIG_ERROR,
            message="max_results must be >= 1",
            start_time=start_time,
        )

    progress(f"Searching arxiv for: {query}")

    client = ArXivAPIClient(rate_limit_delay=0.5)

    try:
        # Build search query
        search_query = query

        # Add category filter if specified (filter out empty tokens)
        if categories:
            cat_list = [c.strip() for c in categories.split(",") if c.strip()]
            if cat_list:
                cat_query = " OR ".join(f"cat:{cat}" for cat in cat_list)
                search_query = f"({query}) AND ({cat_query})"

        # Use arxiv API search endpoint
        params = {
            "search_query": f"all:{search_query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = client._make_request(client.api_base_url, params)
        root = ET.fromstring(response.content)

        # Parse results
        results = []
        entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for entry in entries:
            try:
                metadata = client._parse_entry(entry)
                results.append(_metadata_to_dict(metadata))
            except Exception as e:
                # Log and skip entries that fail to parse
                progress(f"Skipping entry due to parse error: {e}")
                continue

        progress(f"Found {len(results)} papers")

        return success_response(
            command="arxiv.search",
            data={"query": query, "results": results},
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="arxiv.search",
            code=ErrorCode.SEARCH_FAILED,
            message=f"Search failed: {str(e)}",
            start_time=start_time,
        )
    finally:
        client.close()


def get_paper_info(arxiv_id: str, start_time: float) -> CLIResponse:
    """Get detailed metadata for a specific arxiv paper.

    Args:
        arxiv_id: ArXiv paper ID (e.g., "2401.12345")
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with paper metadata
    """
    progress(f"Fetching metadata for: {arxiv_id}")

    client = ArXivAPIClient(rate_limit_delay=0.5)

    try:
        # Validate ID format
        if not client.validate_arxiv_id(arxiv_id):
            return error_response(
                command="arxiv.info",
                code=ErrorCode.INVALID_ARXIV_ID,
                message=f"Invalid arxiv ID format: {arxiv_id}",
                start_time=start_time,
            )

        # Fetch metadata
        metadata = client.get_paper_metadata(arxiv_id)

        if metadata is None:
            return error_response(
                command="arxiv.info",
                code=ErrorCode.PAPER_NOT_FOUND,
                message=f"Paper not found: {arxiv_id}",
                start_time=start_time,
            )

        return success_response(
            command="arxiv.info",
            data=_metadata_to_dict(metadata),
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="arxiv.info",
            code=ErrorCode.NETWORK_ERROR,
            message=f"Failed to fetch paper info: {str(e)}",
            start_time=start_time,
        )
    finally:
        client.close()


def _metadata_to_dict(metadata: ArXivMetadata) -> dict[str, Any]:
    """Convert ArXivMetadata to a JSON-serializable dictionary."""
    return {
        "arxiv_id": metadata.arxiv_id,
        "title": metadata.title,
        "abstract": metadata.abstract,
        "authors": metadata.authors,
        "categories": metadata.categories,
        "primary_category": metadata.primary_category,
        "published": metadata.published.isoformat() if metadata.published else None,
        "updated": metadata.updated.isoformat() if metadata.updated else None,
        "doi": metadata.doi,
        "journal_ref": metadata.journal_ref,
        "pdf_url": metadata.pdf_url,
        "has_latex": metadata.has_latex,
    }
