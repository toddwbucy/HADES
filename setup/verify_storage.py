#!/usr/bin/env python3
"""
Verify that records are being stored properly in the database.
"""

import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Add project root to path (parent of setup/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database.database_factory import DatabaseFactory


def verify_recent_records():
    """Check recently stored records."""
    password = os.environ.get('ARANGO_PASSWORD')
    if not password:
        print(
            "ERROR: ARANGO_PASSWORD environment variable not set. "
            "Export the database password before running verify_storage.",
            file=sys.stderr,
        )
        sys.exit(1)

    db = DatabaseFactory.get_arango_memory_service(
        database='academy_store',
        password=password,
    )

    print("Database Storage Verification")
    print("=" * 60)

    # Get counts
    papers_count = db.execute_query("RETURN LENGTH(arxiv_metadata)")[0]
    embeddings_count = db.execute_query("RETURN LENGTH(arxiv_abstract_embeddings)")[0]
    structures_count = db.execute_query("RETURN LENGTH(arxiv_structures)")[0]

    print("\nCollection Counts:")
    print(f"  arxiv_metadata:            {papers_count:,}")
    print(f"  arxiv_abstract_embeddings: {embeddings_count:,}")
    print(f"  arxiv_structures:          {structures_count:,}")

    # Check if counts are aligned (they should be close)
    print("\nConsistency Check:")
    if papers_count:
        avg_embeddings = embeddings_count / papers_count
        print(f"  Average embeddings per paper: {avg_embeddings:.2f}")
    else:
        print("  ⚠️  No papers found")

    # Get a recent record to verify structure
    print("\nSample Recent Record:")
    try:
        # Get most recent from metadata
        rows = db.execute_query('''
            FOR doc IN arxiv_metadata
                SORT doc.processing_timestamp DESC NULLS LAST
                LIMIT 1
                LET emb = FIRST(
                    FOR e IN arxiv_abstract_embeddings
                        FILTER e.arxiv_id == doc.arxiv_id
                        LIMIT 1
                        RETURN e
                )
                LET chunk_total = LENGTH(
                    FOR c IN arxiv_abstract_chunks
                        FILTER c.arxiv_id == doc.arxiv_id
                        RETURN 1
                )
                LET dim = (
                    !IS_NULL(emb) && !IS_NULL(emb.embedding) && IS_LIST(emb.embedding)
                        ? LENGTH(emb.embedding)
                        : emb.embedding_dim
                )
                RETURN {
                    arxiv_id: doc.arxiv_id,
                    title: doc.title,
                    processed_at: doc.processing_timestamp,
                    embedding_dim: dim,
                    chunk_total: chunk_total
                }
        ''')
        recent_meta = rows[0] if rows else None

        if recent_meta:
            arxiv_id = recent_meta.get('arxiv_id')
            print(f"  ArXiv ID: {arxiv_id}")
            print(f"  Title: {recent_meta.get('title', 'N/A')[:80]}...")
            print(f"  Processed: {recent_meta.get('processed_at', 'N/A')}")

            dim = recent_meta.get('embedding_dim')
            if dim:
                print(f"  ✅ Has embedding (dim: {dim})")
            else:
                print("  ❌ No embedding found")

            chunk_total = recent_meta.get('chunk_total', 0)
            print(f"  ✅ Has {chunk_total} chunks")

    except Exception as e:
        print(f"  Error checking recent record: {e}")

    # Check processing rate over last minute
    print("\nRecent Processing Activity:")
    try:
        one_min_ago = datetime.now(UTC) - timedelta(minutes=1)
        one_min_ago_ms = int(one_min_ago.timestamp() * 1000)
        cursor = db.execute_query('''
            FOR doc IN arxiv_metadata
                FILTER DATE_TIMESTAMP(doc.processing_timestamp) >= @time
                COLLECT WITH COUNT INTO count
                RETURN count
        ''', bind_vars={'time': one_min_ago_ms})
        recent_count = cursor[0] if cursor else 0
        print(f"  Records in last minute: {recent_count}")
        print(f"  Rate: ~{recent_count:.0f} records/minute")

    except Exception as e:
        print(f"  Could not check recent activity: {e}")

if __name__ == "__main__":
    verify_recent_records()
