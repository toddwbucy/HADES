"""Utility script to inspect ArangoDB ingest progress using the memory client."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import sys

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from core.database.database_factory import DatabaseFactory
from core.database.arango import MemoryServiceError


EMBEDDABLE_ABSTRACT_MIN_LEN = 655

def _count_documents(db, collection: str) -> int:
    """Return LENGTH(collection) or 0 if the collection is missing."""
    try:
        result = db.execute_query(f"RETURN LENGTH({collection})")
        return int(result[0]) if result else 0
    except MemoryServiceError as exc:
        # Collection may not exist in a given deployment; treat as zero
        print(f"[warn] Could not count collection '{collection}': {exc.details()}")
        return 0
    except Exception as exc:  # pragma: no cover - diagnostics
        print(f"[warn] Unexpected error counting '{collection}': {exc}")
        return 0


def _print_recent_papers(db) -> None:
    """Display the most recent processed metadata records."""
    print("\nRecent Papers")
    print("-" * 60)
    try:
        rows = db.execute_query(
            """
            FOR doc IN arxiv_metadata
                SORT doc.processing_timestamp DESC NULLS LAST
                LIMIT 5
                LET emb = FIRST(
                    FOR e IN arxiv_abstract_embeddings
                        FILTER e.arxiv_id == doc.arxiv_id
                        LIMIT 1
                        RETURN e
                )
                LET dim = (
                    !IS_NULL(emb) && !IS_NULL(emb.embedding) && IS_LIST(emb.embedding)
                        ? LENGTH(emb.embedding)
                        : emb.embedding_dim
                )
                RETURN {
                    arxiv_id: doc.arxiv_id,
                    processing_timestamp: doc.processing_timestamp,
                    title: doc.title,
                    embedding_dim: dim
                }
            """
        )
    except MemoryServiceError:
        print("No arxiv_metadata collection available yet.")
        return

    if not rows:
        print("No papers processed yet.")
        return

    for rec in rows:
        arxiv_id = rec.get("arxiv_id", "<unknown>")
        processed = rec.get("processing_timestamp", "<unknown>")
        title = rec.get("title", "<untitled>")
        print(f"ID: {arxiv_id:20} | {processed} | {title[:80]}")
        dim = rec.get("embedding_dim")
        if dim:
            print(f"  ↳ embedding dimension: {dim}")
        else:
            print("  ↳ no embedding record found")


def _print_recent_activity(db) -> None:
    """Show metadata records processed in the last minute."""
    print("\nRecent Activity (last minute)")
    print("-" * 60)
    since = datetime.now(timezone.utc) - timedelta(minutes=1)
    since_ms = int(since.timestamp() * 1000)
    try:
        result = db.execute_query(
            """
            FOR doc IN arxiv_metadata
                FILTER DATE_TIMESTAMP(doc.processing_timestamp) >= @ts
                COLLECT WITH COUNT INTO count
                RETURN count
            """,
            bind_vars={"ts": since_ms},
        )
        recent = int(result[0]) if result else 0
        print(f"Metadata records processed: {recent}")
        print(f"Rate: ~{recent:.0f} records/minute")
    except MemoryServiceError as exc:
        print(f"Unable to query recent activity: {exc.details()}")


def _print_unprocessed_metadata(db) -> None:
    """Show candidate metadata entries that still need embeddings (legacy workflow)."""
    legacy_embeddings = _count_documents(db, "arxiv_abstract_embeddings")
    legacy_metadata = _count_documents(db, "arxiv_metadata")
    if legacy_metadata == 0 or legacy_embeddings == 0:
        return

    print("\nLegacy Metadata Backlog")
    print("-" * 60)
    try:
        rows = db.execute_query(
            """
            LET processed = (
                FOR emb IN arxiv_abstract_embeddings
                    RETURN DISTINCT emb.arxiv_id
            )
                FOR doc IN arxiv_metadata
                    FILTER doc.abstract != null
                    FILTER doc.abstract_length >= @min_len
                    FILTER doc.arxiv_id NOT IN processed
                    SORT doc.abstract_length ASC
                    LIMIT 10
                    RETURN {id: doc.arxiv_id, length: doc.abstract_length}
                """,
                bind_vars={"min_len": EMBEDDABLE_ABSTRACT_MIN_LEN},
            )
    except MemoryServiceError as exc:
        print(f"Unable to compute backlog: {exc.details()}")
        return

    if not rows:
        print("All eligible metadata rows have embeddings.")
        return

    for idx, rec in enumerate(rows, start=1):
        print(f"{idx:2}. ID: {rec['id']:20} | Length: {rec['length']:4}")


def main() -> None:
    password = os.environ.get("ARANGO_PASSWORD")
    if not password:
        print("[warn] ARANGO_PASSWORD not set; relying on factory defaults", file=sys.stderr)

    try:
        db = DatabaseFactory.get_arango_memory_service(
            database="arxiv_repository",
            password=password,
        )
    except ValueError as exc:
        print(f"ERROR: Configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
    except MemoryServiceError as exc:
        print(f"ERROR: Unable to initialise memory service: {exc.details()}", file=sys.stderr)
        sys.exit(2)

    print("=" * 60)
    print("CURRENT DATABASE STATE")
    print("=" * 60)

    metadata_count = _count_documents(db, "arxiv_metadata")
    embeddings_count = _count_documents(db, "arxiv_abstract_embeddings")
    structures_count = _count_documents(db, "arxiv_structures")

    print(f"arxiv_metadata:             {metadata_count:,}")
    print(f"arxiv_abstract_embeddings:  {embeddings_count:,}")
    print(f"arxiv_structures:           {structures_count:,}")

    _print_recent_papers(db)
    _print_recent_activity(db)
    _print_unprocessed_metadata(db)


if __name__ == "__main__":
    main()
