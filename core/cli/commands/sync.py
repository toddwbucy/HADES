"""Sync command for batch abstract ingestion from arxiv.

This command fetches metadata and abstracts from arxiv, embeds them,
and stores them for fast semantic search - without downloading PDFs.

Supports incremental sync via watermark tracking in the sync_metadata collection.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any

from core.cli.config import get_arango_config, get_config
from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    progress,
    success_response,
)

# Sync metadata collection and document key
SYNC_METADATA_COLLECTION = "sync_metadata"
SYNC_WATERMARK_KEY = "abstracts"


def get_sync_status(start_time: float) -> CLIResponse:
    """Get the current sync status including last sync time and history.

    Args:
        start_time: Start time for duration calculation

    Returns:
        CLIResponse with sync status
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="sync.status",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    try:
        metadata = _get_sync_metadata(config)

        if metadata is None:
            return success_response(
                command="sync.status",
                data={
                    "last_sync": None,
                    "total_synced": 0,
                    "sync_history": [],
                    "message": "No sync history found. Run 'hades sync' to begin.",
                },
                start_time=start_time,
            )

        return success_response(
            command="sync.status",
            data={
                "last_sync": metadata.get("last_sync"),
                "total_synced": metadata.get("total_synced", 0),
                "sync_history": metadata.get("sync_history", [])[-10:],  # Last 10 syncs
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="sync.status",
            code=ErrorCode.DATABASE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def _get_sync_metadata(config: Any) -> dict[str, Any] | None:
    """Fetch sync metadata from the database.

    Returns None if no metadata exists (first sync).
    """
    from core.database.arango.optimized_client import (
        ArangoHttp2Client,
        ArangoHttp2Config,
        ArangoHttpError,
    )

    arango_config = get_arango_config(config, read_only=True)
    client_config = ArangoHttp2Config(
        database=arango_config["database"],
        socket_path=arango_config.get("socket_path"),
        base_url=f"http://{arango_config['host']}:{arango_config['port']}",
        username=arango_config["username"],
        password=arango_config["password"],
    )

    client = ArangoHttp2Client(client_config)

    try:
        try:
            doc = client.get_document(SYNC_METADATA_COLLECTION, SYNC_WATERMARK_KEY)
            return doc
        except ArangoHttpError as e:
            if e.status_code == 404:
                return None
            raise
    finally:
        client.close()


def _update_sync_metadata(
    config: Any,
    added: int,
    updated: int,
    sync_date: datetime,
) -> None:
    """Update sync metadata after a successful sync.

    Creates the sync_metadata collection if it doesn't exist.
    """
    from core.database.arango.optimized_client import (
        ArangoHttp2Client,
        ArangoHttp2Config,
        ArangoHttpError,
    )

    arango_config = get_arango_config(config, read_only=False)
    client_config = ArangoHttp2Config(
        database=arango_config["database"],
        socket_path=arango_config.get("socket_path"),
        base_url=f"http://{arango_config['host']}:{arango_config['port']}",
        username=arango_config["username"],
        password=arango_config["password"],
    )

    client = ArangoHttp2Client(client_config)

    try:
        # Ensure collection exists
        try:
            client.request(
                "POST",
                f"/_db/{arango_config['database']}/_api/collection",
                json={"name": SYNC_METADATA_COLLECTION},
            )
        except ArangoHttpError as e:
            # 409 = collection already exists, which is fine
            if e.status_code != 409:
                raise

        # Get existing metadata or create new
        existing = None
        try:
            existing = client.get_document(SYNC_METADATA_COLLECTION, SYNC_WATERMARK_KEY)
        except ArangoHttpError as e:
            if e.status_code != 404:
                raise

        now_iso = datetime.now(UTC).isoformat()
        sync_entry = {
            "date": sync_date.strftime("%Y-%m-%d"),
            "added": added,
            "updated": updated,
            "timestamp": now_iso,
        }

        if existing:
            # Update existing document
            total_synced = existing.get("total_synced", 0) + added
            history = existing.get("sync_history", [])
            history.append(sync_entry)
            # Keep last 100 entries
            history = history[-100:]

            # Use REPLACE to update the document
            client.request(
                "PUT",
                f"/_db/{arango_config['database']}/_api/document/{SYNC_METADATA_COLLECTION}/{SYNC_WATERMARK_KEY}",
                json={
                    "_key": SYNC_WATERMARK_KEY,
                    "last_sync": now_iso,
                    "total_synced": total_synced,
                    "sync_history": history,
                },
            )
        else:
            # Create new document
            client.insert_documents(
                SYNC_METADATA_COLLECTION,
                [
                    {
                        "_key": SYNC_WATERMARK_KEY,
                        "last_sync": now_iso,
                        "total_synced": added,
                        "sync_history": [sync_entry],
                    }
                ],
            )

    finally:
        client.close()


def _get_last_sync_date(config: Any) -> datetime | None:
    """Get the date of the last successful sync.

    Returns None if no previous sync exists.
    """
    metadata = _get_sync_metadata(config)
    if metadata and metadata.get("last_sync"):
        # Parse ISO format timestamp
        last_sync_str = metadata["last_sync"]
        # Handle both with and without timezone
        try:
            return datetime.fromisoformat(last_sync_str.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def sync_abstracts(
    from_date: str | None,
    categories: str | None,
    max_results: int,
    batch_size: int,
    start_time: float,
    incremental: bool = False,
) -> CLIResponse:
    """Sync abstracts from arxiv to the database.

    Fetches metadata and abstracts, generates embeddings, and stores them
    for semantic search - without downloading full PDFs.

    Args:
        from_date: Start date in YYYY-MM-DD format (default: 7 days ago)
        categories: Comma-separated arxiv categories (e.g., "cs.AI,cs.CL")
        max_results: Maximum number of papers to fetch
        batch_size: Batch size for embedding generation
        start_time: Start time for duration calculation
        incremental: If True, sync only papers newer than last sync watermark

    Returns:
        CLIResponse with sync results
    """
    try:
        config = get_config()
    except ValueError as e:
        return error_response(
            command="sync",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )

    # Determine start date
    if incremental:
        # Use watermark from last sync
        last_sync = _get_last_sync_date(config)
        if last_sync is None:
            progress("No previous sync found, using default (7 days ago)...")
            start_date = datetime.now() - timedelta(days=7)
        else:
            # Convert to naive datetime for comparison
            if last_sync.tzinfo is not None:
                start_date = last_sync.replace(tzinfo=None)
            else:
                start_date = last_sync
            progress(f"Incremental sync from {start_date.strftime('%Y-%m-%d %H:%M')}...")
    elif from_date:
        try:
            start_date = datetime.strptime(from_date, "%Y-%m-%d")
        except ValueError:
            return error_response(
                command="sync",
                code=ErrorCode.CONFIG_ERROR,
                message=f"Invalid date format: {from_date}. Use YYYY-MM-DD",
                start_time=start_time,
            )
    else:
        start_date = datetime.now() - timedelta(days=7)

    # Validate batch_size
    if batch_size <= 0:
        return error_response(
            command="sync",
            code=ErrorCode.CONFIG_ERROR,
            message="batch_size must be >= 1",
            start_time=start_time,
        )

    progress(f"Syncing abstracts from {start_date.strftime('%Y-%m-%d')}...")

    try:
        # Fetch papers from arxiv
        papers = _fetch_recent_papers(start_date, categories, max_results)

        if not papers:
            return success_response(
                command="sync",
                data={"synced": 0, "skipped": 0, "message": "No new papers found"},
                start_time=start_time,
            )

        progress(f"Found {len(papers)} papers, checking for duplicates...")

        # Filter out already-synced papers
        new_papers = _filter_existing(papers, config)

        if not new_papers:
            return success_response(
                command="sync",
                data={
                    "synced": 0,
                    "skipped": len(papers),
                    "message": "All papers already in database",
                },
                start_time=start_time,
            )

        progress(f"Embedding {len(new_papers)} new abstracts...")

        # Embed abstracts and store
        synced = _embed_and_store_abstracts(new_papers, config, batch_size)

        # Update sync metadata watermark
        try:
            _update_sync_metadata(config, added=synced, updated=0, sync_date=start_date)
        except Exception as meta_err:
            progress(f"Warning: Failed to update sync metadata: {meta_err}")

        return success_response(
            command="sync",
            data={
                "synced": synced,
                "skipped": len(papers) - len(new_papers),
                "mode": "incremental" if incremental else "manual",
                "from_date": start_date.strftime("%Y-%m-%d"),
                "papers": [
                    {
                        "arxiv_id": p["arxiv_id"],
                        "title": p["title"],
                        "categories": p.get("categories", []),
                    }
                    for p in new_papers[:10]  # Show first 10
                ],
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="sync",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )


def _fetch_recent_papers(
    start_date: datetime,
    categories: str | None,
    max_results: int,
) -> list[dict[str, Any]]:
    """Fetch recent papers from arxiv API.

    ArXiv API has pagination limits (~10000 results per query), so we
    query month by month to avoid hitting those limits.
    """

    from core.tools.arxiv.arxiv_api_client import ArXivAPIClient

    client = ArXivAPIClient(rate_limit_delay=0.5)

    try:
        papers = []
        current_date = start_date
        today = datetime.now()

        # Query month by month to avoid pagination limits
        while current_date <= today and len(papers) < max_results:
            # Calculate month end
            if current_date.month == 12:
                month_end = datetime(current_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = datetime(current_date.year, current_date.month + 1, 1) - timedelta(days=1)

            if month_end > today:
                month_end = today

            progress(f"Fetching papers from {current_date.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')}...")

            # Build date query for this month
            date_str = current_date.strftime("%Y%m%d")
            end_str = month_end.strftime("%Y%m%d")
            date_query = f"submittedDate:[{date_str}0000 TO {end_str}2359]"

            if categories:
                cat_list = [c.strip() for c in categories.split(",")]
                cat_query = " OR ".join(f"cat:{cat}" for cat in cat_list)
                search_query = f"({cat_query}) AND {date_query}"
            else:
                search_query = date_query

            # Fetch papers for this month (limit to 5000 per month to stay safe)
            month_papers = _fetch_month_papers(client, search_query, min(5000, max_results - len(papers)))
            papers.extend(month_papers)

            progress(f"Found {len(month_papers)} papers for {current_date.strftime('%Y-%m')}")

            # Move to next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

        return papers[:max_results]

    finally:
        client.close()


def _fetch_month_papers(
    client: Any,
    search_query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """Fetch papers for a single month query."""
    from defusedxml import ElementTree as ET

    papers = []
    start = 0
    batch_size = 100  # arxiv max per request

    while len(papers) < max_results:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = client._make_request(client.api_base_url, params)
            root = ET.fromstring(response.content)

            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            if not entries:
                break

            for entry in entries:
                try:
                    metadata = client._parse_entry(entry)
                    papers.append(
                        {
                            "arxiv_id": metadata.arxiv_id,
                            "title": metadata.title,
                            "abstract": metadata.abstract,
                            "authors": metadata.authors,
                            "categories": metadata.categories,
                            "primary_category": metadata.primary_category,
                            "published": metadata.published.isoformat() if metadata.published else None,
                            "updated": metadata.updated.isoformat() if metadata.updated else None,
                        }
                    )
                except Exception:
                    continue

            start += batch_size
            if len(entries) < batch_size:
                break

            # Rate limit
            time.sleep(0.5)

        except Exception as e:
            # If we hit pagination limits or server errors, stop for this month
            progress(f"Warning: Stopped at {start} papers due to: {e}")
            break

    return papers


def _filter_existing(
    papers: list[dict[str, Any]],
    config: Any,
) -> list[dict[str, Any]]:
    """Filter out papers that already exist in the database."""
    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

    if not papers:
        return []

    # Use read-write socket - cursor operations require write access
    arango_config = get_arango_config(config, read_only=False)
    client_config = ArangoHttp2Config(
        database=arango_config["database"],
        socket_path=arango_config.get("socket_path"),
        base_url=f"http://{arango_config['host']}:{arango_config['port']}",
        username=arango_config["username"],
        password=arango_config["password"],
    )

    client = ArangoHttp2Client(client_config)

    try:
        # Build list of base IDs to check (strip version suffixes)
        check_ids = []
        for p in papers:
            arxiv_id = p["arxiv_id"]
            base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            check_ids.append(base_id)

        # Query only for the specific IDs we're checking (not all 2.8M!)
        # This avoids chunked transfer encoding issues with large result sets
        existing_ids = set()
        try:
            results = client.query(
                "FOR doc IN arxiv_papers FILTER doc.arxiv_id IN @ids RETURN doc.arxiv_id",
                bind_vars={"ids": check_ids},
            )
            existing_ids.update(r for r in results if r)
        except Exception as e:
            # Collection may not exist yet - treat as empty
            progress(f"Note: Could not query existing papers: {e}")

        # Filter out existing
        new_papers = []
        for p in papers:
            arxiv_id = p["arxiv_id"]
            base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
            if base_id not in existing_ids:
                new_papers.append(p)

        return new_papers

    finally:
        client.close()


def _embed_and_store_abstracts(
    papers: list[dict[str, Any]],
    config: Any,
    batch_size: int,
) -> int:
    """Embed abstracts and store in database.

    Uses the existing database schema:
    - arxiv_papers: metadata (arxiv_id, title, authors, categories, etc.)
    - arxiv_abstracts: abstract text (arxiv_id, title, abstract)
    - arxiv_embeddings: embeddings (arxiv_id, combined_embedding)
    """
    from datetime import UTC, datetime

    from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config
    from core.embedders.embedders_jina import JinaV4Embedder

    # Initialize embedder
    embedder = JinaV4Embedder(
        {
            "device": config.device,
            "use_fp16": True,
        }
    )

    arango_config = get_arango_config(config, read_only=False)
    client_config = ArangoHttp2Config(
        database=arango_config["database"],
        socket_path=arango_config.get("socket_path"),
        base_url=f"http://{arango_config['host']}:{arango_config['port']}",
        username=arango_config["username"],
        password=arango_config["password"],
    )

    client = ArangoHttp2Client(client_config)

    try:
        synced = 0
        now_iso = datetime.now(UTC).isoformat()

        # Process in batches
        for i in range(0, len(papers), batch_size):
            batch = papers[i : i + batch_size]
            progress(f"Processing batch {i // batch_size + 1} ({len(batch)} papers)...")

            # Extract abstracts for embedding
            abstracts = [p["abstract"] for p in batch]

            # Generate embeddings using Jina
            embeddings = embedder.embed_texts(abstracts, task="retrieval")

            # Prepare documents matching existing schema
            paper_docs = []
            abstract_docs = []
            embedding_docs = []

            for j, paper in enumerate(batch):
                # Use arxiv_id directly as key (matching existing format)
                arxiv_id = paper["arxiv_id"]
                # Remove version suffix for key (e.g., "2501.12345v1" -> "2501_12345")
                base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sanitized_key = base_id.replace(".", "_").replace("/", "_")

                # Parse year_month from arxiv_id (YYMM.NNNNN format)
                if "." in base_id:
                    yymm = base_id.split(".")[0]
                    year = 2000 + int(yymm[:2])
                    month = int(yymm[2:4])
                else:
                    year = 2024
                    month = 1
                    yymm = "2401"

                # arxiv_papers document (metadata)
                paper_docs.append(
                    {
                        "_key": sanitized_key,
                        "arxiv_id": base_id,
                        "authors": paper["authors"],
                        "categories": paper["categories"],
                        "primary_category": paper["primary_category"],
                        "year": year,
                        "month": month,
                        "year_month": f"{year}{month:02d}",
                        "created_at": now_iso,
                    }
                )

                # arxiv_abstracts document
                abstract_docs.append(
                    {
                        "_key": sanitized_key,
                        "arxiv_id": base_id,
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                    }
                )

                # arxiv_embeddings document (matching existing schema)
                embedding_docs.append(
                    {
                        "_key": sanitized_key,
                        "arxiv_id": base_id,
                        "combined_embedding": embeddings[j].tolist(),
                        "abstract_embedding": [],  # Empty to match existing schema
                        "title_embedding": [],  # Empty to match existing schema
                    }
                )

            # Insert documents (skip duplicates - they raise unique constraint errors)
            try:
                duplicates = 0
                for doc in paper_docs:
                    try:
                        client.insert_documents("arxiv_papers", [doc])
                    except Exception:
                        duplicates += 1  # Likely duplicate key

                for doc in abstract_docs:
                    try:
                        client.insert_documents("arxiv_abstracts", [doc])
                    except Exception:
                        pass  # Skip if exists

                for doc in embedding_docs:
                    try:
                        client.insert_documents("arxiv_embeddings", [doc])
                    except Exception:
                        pass  # Skip if exists

                synced += len(batch) - duplicates
            except Exception as e:
                progress(f"Warning: Failed to store batch: {e}")

        return synced

    finally:
        client.close()
