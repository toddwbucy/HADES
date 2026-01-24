#!/usr/bin/env python3
"""
Process arxiv PDFs through the full HADES pipeline.

This script:
1. Fetches metadata from arxiv API
2. Processes PDF through DocumentProcessor (extraction + embedding)
3. Stores results in ArangoDB

Usage:
    poetry run python tests/functional/process_arxiv_pdf.py <pdf_path> [--dry-run]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config
from core.processors.document_processor import DocumentProcessor, ProcessingConfig


@dataclass
class ArxivMetadata:
    """Metadata from arxiv API."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str


def extract_arxiv_id(filename: str) -> str:
    """Extract arxiv ID from filename like 'ATLAS_2505.23735.pdf'."""
    match = re.search(r'(\d{4}\.\d{4,5})', filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract arxiv ID from filename: {filename}")


def fetch_arxiv_metadata(arxiv_id: str) -> ArxivMetadata:
    """Fetch metadata from arxiv API."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    print(f"  Fetching metadata from: {url}")

    with urllib.request.urlopen(url, timeout=30) as response:
        xml_data = response.read().decode('utf-8')

    # Parse XML
    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

    entry = root.find('atom:entry', ns)
    if entry is None:
        raise ValueError(f"No entry found for arxiv ID: {arxiv_id}")

    # Extract fields
    title = entry.find('atom:title', ns)
    title = title.text.strip().replace('\n', ' ') if title is not None else ''

    abstract = entry.find('atom:summary', ns)
    abstract = abstract.text.strip() if abstract is not None else ''

    authors = []
    for author in entry.findall('atom:author', ns):
        name = author.find('atom:name', ns)
        if name is not None:
            authors.append(name.text)

    categories = []
    primary_category = ''
    for cat in entry.findall('atom:category', ns):
        term = cat.get('term', '')
        if term:
            categories.append(term)
            if not primary_category:
                primary_category = term

    # Also check arxiv:primary_category
    prim_cat = entry.find('arxiv:primary_category', ns)
    if prim_cat is not None:
        primary_category = prim_cat.get('term', primary_category)

    published = entry.find('atom:published', ns)
    published = published.text if published is not None else ''

    updated = entry.find('atom:updated', ns)
    updated = updated.text if updated is not None else ''

    return ArxivMetadata(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        primary_category=primary_category,
        published=published,
        updated=updated,
    )


def generate_document_key(arxiv_id: str) -> str:
    """Generate ArangoDB document key from arxiv ID.

    Format: YYYYMM_NNNNN (e.g., 2505.23735 -> 202505_23735)
    """
    parts = arxiv_id.split('.')
    if len(parts) != 2:
        raise ValueError(f"Invalid arxiv ID format: {arxiv_id}")

    yymm = parts[0]  # e.g., "2505"
    num = parts[1]   # e.g., "23735"

    # Convert YY to YYYY (assuming 20XX for now)
    year = f"20{yymm[:2]}"
    month = yymm[2:]

    return f"{year}{month}_{num}"


def process_pdf(pdf_path: Path, config: ProcessingConfig) -> dict:
    """Process PDF through DocumentProcessor."""
    print(f"  Initializing DocumentProcessor...")
    processor = DocumentProcessor(config)

    print(f"  Processing PDF: {pdf_path.name}")
    start_time = time.time()

    result = processor.process_document(pdf_path)

    elapsed = time.time() - start_time
    print(f"  Processing completed in {elapsed:.1f}s")
    print(f"    - Success: {result.success}")
    print(f"    - Chunks: {len(result.chunks)}")
    print(f"    - Extraction time: {result.extraction_time:.1f}s")
    print(f"    - Embedding time: {result.embedding_time:.1f}s")

    if result.errors:
        print(f"    - Errors: {result.errors}")

    processor.cleanup()

    return result


def store_in_arango(
    metadata: ArxivMetadata,
    processing_result,
    client: ArangoHttp2Client,
    dry_run: bool = False,
) -> bool:
    """Store paper metadata and embeddings in ArangoDB."""
    doc_key = generate_document_key(metadata.arxiv_id)

    # Parse year/month from arxiv_id
    yymm = metadata.arxiv_id.split('.')[0]
    year = 2000 + int(yymm[:2])
    month = int(yymm[2:])

    # Prepare paper document
    paper_doc = {
        "_key": doc_key,
        "arxiv_id": metadata.arxiv_id,
        "title": metadata.title,
        "abstract": metadata.abstract,
        "authors": metadata.authors,
        "categories": metadata.categories,
        "primary_category": metadata.primary_category,
        "year": year,
        "month": month,
        "year_month": f"{year}{month:02d}",
        "published": metadata.published,
        "updated": metadata.updated,
        "created_at": datetime.now(UTC).isoformat(),
        "pdf_processed": True,
        "chunk_count": len(processing_result.chunks),
    }

    # Prepare embedding document
    # Combine all chunk embeddings into one (average) for combined_embedding
    # Or use the first chunk's embedding as representative
    combined_embedding = []
    if processing_result.chunks:
        import numpy as np
        # Average all chunk embeddings
        embeddings = [chunk.embedding for chunk in processing_result.chunks]
        combined_embedding = np.mean(embeddings, axis=0).tolist()

    embedding_doc = {
        "_key": doc_key,
        "arxiv_id": metadata.arxiv_id,
        "combined_embedding": combined_embedding,
        "title_embedding": [],  # Could embed title separately
        "abstract_embedding": [],  # Could embed abstract separately
        "chunk_count": len(processing_result.chunks),
        "embedding_dim": len(combined_embedding),
        "created_at": datetime.now(UTC).isoformat(),
    }

    if dry_run:
        print(f"  [DRY RUN] Would store paper: {doc_key}")
        print(f"    Title: {metadata.title[:60]}...")
        print(f"    Embedding dim: {len(combined_embedding)}")
        return True

    # Store paper using AQL upsert
    print(f"  Storing paper document: {doc_key}")
    try:
        client.query('''
            UPSERT { _key: @key }
            INSERT @doc
            UPDATE @doc
            IN arxiv_papers
        ''', bind_vars={'key': doc_key, 'doc': paper_doc})
        print(f"    ✓ Paper stored")
    except Exception as e:
        print(f"    Error: {e}")
        return False

    # Store embedding using AQL upsert
    print(f"  Storing embedding document: {doc_key}")
    try:
        client.query('''
            UPSERT { _key: @key }
            INSERT @doc
            UPDATE @doc
            IN arxiv_embeddings
        ''', bind_vars={'key': doc_key, 'doc': embedding_doc})
        print(f"    ✓ Embedding stored ({len(combined_embedding)} dims)")
    except Exception as e:
        print(f"    Error: {e}")
        return False

    return True


def verify_storage(arxiv_id: str, client: ArangoHttp2Client) -> bool:
    """Verify the paper was stored correctly."""
    doc_key = generate_document_key(arxiv_id)

    print(f"  Verifying storage for {doc_key}...")

    # Check paper
    result = client.query(f'''
        FOR doc IN arxiv_papers
            FILTER doc._key == "{doc_key}"
            RETURN {{
                arxiv_id: doc.arxiv_id,
                title: LEFT(doc.title, 50),
                chunk_count: doc.chunk_count
            }}
    ''')

    if not result:
        print(f"    ✗ Paper not found")
        return False
    print(f"    ✓ Paper found: {result[0]}")

    # Check embedding
    result = client.query(f'''
        FOR doc IN arxiv_embeddings
            FILTER doc._key == "{doc_key}"
            RETURN {{
                arxiv_id: doc.arxiv_id,
                embedding_dim: LENGTH(doc.combined_embedding)
            }}
    ''')

    if not result:
        print(f"    ✗ Embedding not found")
        return False
    print(f"    ✓ Embedding found: {result[0]}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Process arxiv PDF through HADES pipeline")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--dry-run", action="store_true", help="Don't store in database")
    parser.add_argument("--device", default="cuda", help="Device for embeddings (cuda/cpu)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"=" * 60)
    print(f"Processing: {pdf_path.name}")
    print(f"=" * 60)

    # Step 1: Extract arxiv ID and fetch metadata
    print("\n[1/4] Fetching arxiv metadata...")
    arxiv_id = extract_arxiv_id(pdf_path.name)
    print(f"  Arxiv ID: {arxiv_id}")

    metadata = fetch_arxiv_metadata(arxiv_id)
    print(f"  Title: {metadata.title[:70]}...")
    print(f"  Authors: {len(metadata.authors)} authors")
    print(f"  Categories: {metadata.categories}")

    # Step 2: Process PDF
    print("\n[2/4] Processing PDF...")
    config = ProcessingConfig(
        use_gpu=args.device == "cuda",
        device=args.device,
        chunking_strategy="late",
        chunk_size_tokens=512,
        chunk_overlap_tokens=50,
        use_ocr=False,
        extract_tables=True,
        extract_equations=True,
        use_ramfs_staging=True,
    )

    result = process_pdf(pdf_path, config)

    if not result.success:
        print(f"Error: Processing failed: {result.errors}")
        sys.exit(1)

    # Step 3: Store in ArangoDB
    print("\n[3/4] Storing in ArangoDB...")

    arango_config = ArangoHttp2Config(
        socket_path='/run/arangodb3/arangodb.sock',
        database='arxiv_datastore',
        username='root',
        password=os.environ.get('ARANGO_PASSWORD'),
        read_timeout=60.0,
        write_timeout=60.0,
    )

    client = ArangoHttp2Client(arango_config)

    success = store_in_arango(metadata, result, client, dry_run=args.dry_run)

    if not success:
        print("Error: Failed to store in database")
        client.close()
        sys.exit(1)

    # Step 4: Verify
    if not args.dry_run:
        print("\n[4/4] Verifying storage...")
        verify_storage(arxiv_id, client)

    client.close()

    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print(f"  Arxiv ID: {arxiv_id}")
    print(f"  Chunks: {len(result.chunks)}")
    print(f"  Total time: {result.total_processing_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
