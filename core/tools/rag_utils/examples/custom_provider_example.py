#!/usr/bin/env python3
"""
Custom Provider Example
=======================

Demonstrates creating custom DocumentProvider and CitationStorage implementations
for the Academic Citation Toolkit. Shows how to extend the toolkit for any
academic corpus or storage system.
"""

import json
import os
import sqlite3

# Use proper relative import from parent package
from ..academic_citation_toolkit import (
    BibliographyEntry,
    CitationStorage,
    DocumentProvider,
    InTextCitation,
    UniversalBibliographyExtractor,
)


class WebAPIDocumentProvider(DocumentProvider):
    """
    Example custom provider for web API document sources.

    This could be adapted for:
    - SSRN API
    - PubMed API
    - Harvard Law Library API
    - Any academic database API
    """

    def __init__(self, api_base_url: str, api_key: str | None = None):
        """
        Create a WebAPIDocumentProvider configured to call the remote document API.

        Parameters:
            api_base_url (str): Base URL of the document API; trailing slash is removed automatically.
            api_key (Optional[str]): Optional bearer token used to set the `Authorization: Bearer <token>` header for requests.
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def get_document_text(self, document_id: str) -> str | None:
        """
        Fetch the full text for a document from the provider's web API.

        Attempts an HTTP GET to the provider endpoint for the given document identifier and returns the document's full text when available. On success returns the value of the API's `full_text` field, falls back to `content`, and if neither field is present returns an empty string. If the HTTP response is not successful or an error occurs while fetching/parsing, returns None.

        Parameters:
            document_id (str): Provider-specific identifier for the document to retrieve.

        Returns:
            Optional[str]: The document full text on success, an empty string if the response contains no text fields, or None on HTTP error or exception.
        """
        try:
            import requests

            url = f"{self.api_base_url}/documents/{document_id}/fulltext"
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                return data.get('full_text', data.get('content', ''))
            else:
                print(f"API Error {response.status_code} for document {document_id}")
                return None

        except Exception as e:
            print(f"Error fetching document {document_id}: {e}")
            return None

    def get_document_chunks(self, document_id: str) -> list[str]:
        """
        Return the document split into non-empty paragraph chunks.

        Retrieves the full text for the given document_id and splits it into paragraph-like chunks using double-newline separators. Leading and trailing whitespace is trimmed from each chunk; empty chunks are omitted.

        Parameters:
            document_id (str): Provider-specific identifier of the document to fetch.

        Returns:
            List[str]: A list of non-empty paragraph chunks (strings). If the document text cannot be retrieved, returns an empty list.
        """
        full_text = self.get_document_text(document_id)
        if not full_text:
            return []

        # Simple paragraph-based chunking
        chunks = [chunk.strip() for chunk in full_text.split('\n\n') if chunk.strip()]
        return chunks

class SQLiteCitationStorage(CitationStorage):
    """
    Example custom storage implementation using SQLite.

    This pattern can be adapted for:
    - PostgreSQL
    - MySQL
    - MongoDB
    - Any database system
    """

    def __init__(self, db_path: str):
        """
        Initialize the SQLiteCitationStorage.

        Parameters:
            db_path (str): Filesystem path to the SQLite database file. If the file does not exist, the database schema will be created at this path.
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """
        Initialize the SQLite database schema used by SQLiteCitationStorage.

        Creates the required tables and indexes if they do not already exist:
        - bibliography_entries: stores extracted bibliography records with a UNIQUE(source_paper_id, entry_number) constraint and a `created_at` timestamp default.
        - in_text_citations: stores extracted in-text citation occurrences with a `created_at` timestamp default.

        This operation is idempotent (uses CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS) and commits the schema changes. The database connection is always closed when finished.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            # Bibliography entries table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS bibliography_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper_id TEXT NOT NULL,
                entry_number TEXT,
                raw_text TEXT NOT NULL,
                title TEXT,
                authors TEXT,
                venue TEXT,
                year INTEGER,
                arxiv_id TEXT,
                doi TEXT,
                pmid TEXT,
                ssrn_id TEXT,
                url TEXT,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_paper_id, entry_number)
            )
            """)

            # In-text citations table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS in_text_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_paper_id TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                citation_type TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                context TEXT,
                bibliography_ref TEXT,
                confidence REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bib_source_paper ON bibliography_entries(source_paper_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bib_arxiv_id ON bibliography_entries(arxiv_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_bib_doi ON bibliography_entries(doi)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cite_source_paper ON in_text_citations(source_paper_id)")

            conn.commit()
        finally:
            conn.close()

    def store_bibliography_entries(self, entries: list[BibliographyEntry]) -> bool:
        """
        Store a list of BibliographyEntry objects into the configured SQLite database.

        If `entries` is empty this is a no-op and returns True. Each entry is inserted using
        `INSERT OR REPLACE` into the `bibliography_entries` table; `entry.authors` is
        serialized to JSON when present. The function commits the transaction on success
        and rolls back on any error.

        Parameters:
            entries (List[BibliographyEntry]): Bibliography entries to persist.

        Returns:
            bool: True if all entries were stored successfully (or the input list was empty),
                  False if an error occurred and the transaction was rolled back.
        """
        if not entries:
            return True

        conn = sqlite3.connect(self.db_path)
        try:
            for entry in entries:
                authors_str = json.dumps(entry.authors) if entry.authors else None

                conn.execute("""
                INSERT OR REPLACE INTO bibliography_entries
                (source_paper_id, entry_number, raw_text, title, authors, venue,
                 year, arxiv_id, doi, pmid, ssrn_id, url, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.source_paper_id, entry.entry_number, entry.raw_text,
                    entry.title, authors_str, entry.venue, entry.year,
                    entry.arxiv_id, entry.doi, entry.pmid, entry.ssrn_id,
                    entry.url, entry.confidence
                ))

            conn.commit()
            print(f"✅ Stored {len(entries)} bibliography entries in SQLite")
            return True

        except Exception as e:
            print(f"❌ Error storing bibliography entries: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def store_citations(self, citations: list[InTextCitation]) -> bool:
        """
        Store a list of in-text citation records into the configured SQLite database.

        Parameters:
            citations (List[InTextCitation]): In-text citation objects to persist; each item should contain source_paper_id, raw_text, citation_type, start_pos, end_pos, context, bibliography_ref, and confidence.

        Returns:
            bool: True if all citations were stored successfully or if `citations` is empty; False if a database error occurred.

        Side effects:
            Writes rows to the `in_text_citations` table in the SQLite database at `self.db_path`.
        """
        if not citations:
            return True

        conn = sqlite3.connect(self.db_path)
        try:
            for citation in citations:
                conn.execute("""
                INSERT INTO in_text_citations
                (source_paper_id, raw_text, citation_type, start_pos, end_pos,
                 context, bibliography_ref, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    citation.source_paper_id, citation.raw_text, citation.citation_type,
                    citation.start_pos, citation.end_pos, citation.context,
                    citation.bibliography_ref, citation.confidence
                ))

            conn.commit()
            print(f"✅ Stored {len(citations)} in-text citations in SQLite")
            return True

        except Exception as e:
            print(f"❌ Error storing citations: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_bibliography_stats(self) -> dict:
        """
        Return statistics about bibliography entries stored in the SQLite backend.

        Returns a dict containing:
        - total_entries (int): total number of bibliography records.
        - entries_by_paper (dict[str, int]): mapping from source_paper_id to number of entries for that paper.
        - with_arxiv_id (int): count of entries that have a non-null arxiv_id.
        - with_doi (int): count of entries that have a non-null doi.
        - confidence_distribution (dict): counts partitioned by confidence score:
            - high (int): confidence >= 0.8
            - medium (int): 0.6 <= confidence < 0.8
            - low (int): confidence < 0.6

        The function always closes the database connection before returning.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Total entries
            cursor.execute("SELECT COUNT(*) FROM bibliography_entries")
            total_entries = cursor.fetchone()[0]

            # Entries by source paper
            cursor.execute("""
            SELECT source_paper_id, COUNT(*)
            FROM bibliography_entries
            GROUP BY source_paper_id
            ORDER BY COUNT(*) DESC
            """)
            by_paper = cursor.fetchall()

            # Entries with identifiers
            cursor.execute("SELECT COUNT(*) FROM bibliography_entries WHERE arxiv_id IS NOT NULL")
            with_arxiv = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM bibliography_entries WHERE doi IS NOT NULL")
            with_doi = cursor.fetchone()[0]

            # Confidence distribution
            cursor.execute("""
            SELECT
                SUM(CASE WHEN confidence >= 0.8 THEN 1 ELSE 0 END) as high,
                SUM(CASE WHEN confidence >= 0.6 AND confidence < 0.8 THEN 1 ELSE 0 END) as medium,
                SUM(CASE WHEN confidence < 0.6 THEN 1 ELSE 0 END) as low
            FROM bibliography_entries
            """)
            confidence_dist = cursor.fetchone()

            return {
                'total_entries': total_entries,
                'entries_by_paper': dict(by_paper),
                'with_arxiv_id': with_arxiv,
                'with_doi': with_doi,
                'confidence_distribution': {
                    'high': confidence_dist[0],
                    'medium': confidence_dist[1],
                    'low': confidence_dist[2]
                }
            }

        finally:
            conn.close()

class MockAPIDocumentProvider(DocumentProvider):
    """Mock API provider for demonstration purposes."""

    def __init__(self):
        # Sample academic paper content
        """
        Initialize the mock provider and populate self.sample_papers with example paper texts.

        Creates an in-memory mapping self.sample_papers where keys are document IDs ("paper_001", "paper_002")
        and values are multi-section markdown-like paper texts (abstract, introduction, sections, and a References
        list). These samples are intended for testing or demonstration of document retrieval and chunking.
        """
        self.sample_papers = {
            "paper_001": """
# Advanced Machine Learning Techniques

## Abstract
This paper presents novel approaches to machine learning with applications to natural language processing.

## Introduction
Recent advances in machine learning have shown promising results. Building on prior work, we propose new techniques.

## Related Work
The field has been advanced by several key contributions. Hinton et al. [1] introduced deep learning concepts. LeCun et al. [2] developed convolutional neural networks. More recently, transformer architectures have emerged [3].

## Methodology
Our approach combines several existing techniques in novel ways.

## Results
We demonstrate significant improvements over baseline methods on standard benchmarks.

## Conclusion
This work presents meaningful contributions to the field of machine learning.

## References

[1] G. E. Hinton, S. Osindero, and Y. W. Teh. A fast learning algorithm for deep belief nets. Neural computation, 18(7):1527-1554, 2006.

[2] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.

[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[4] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805, 2018.

[5] T. Brown, B. Mann, N. Ryder, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.
            """.strip(),

            "paper_002": """
# Reinforcement Learning in Complex Environments

## Abstract
We explore applications of reinforcement learning to complex, multi-agent environments.

## Introduction
Reinforcement learning has shown remarkable success in various domains. This work extends previous approaches to more complex scenarios.

## Background
Key developments in reinforcement learning include foundational work by Sutton and Barto [1]. Deep reinforcement learning was pioneered by Mnih et al. [2]. Multi-agent systems have been explored by Tampuu et al. [3].

## Approach
Our method builds on established reinforcement learning principles while introducing novel multi-agent coordination mechanisms.

## Experiments
We evaluate our approach on several benchmark environments and demonstrate improved performance.

## Discussion
The results suggest that our approach is effective for complex multi-agent scenarios.

## References

[1] R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT press, 2018.

[2] V. Mnih, K. Kavukcuoglu, D. Silver, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.

[3] A. Tampuu, T. Matiisen, D. Kodelja, I. Kuzovkin, K. Korjus, J. Aru, J. Aru, and R. Vicente. Multiagent cooperation and competition with deep reinforcement learning. arXiv:1511.09729, 2015.

[4] R. Lowe, Y. Wu, A. Tamar, J. Harb, O. P. Abbeel, and I. Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. Advances in neural information processing systems, 30, 2017.
            """.strip()
        }

    def get_document_text(self, document_id: str) -> str | None:
        """
        Get the sample paper text for a mock document ID.

        Looks up the provided document_id in the provider's in-memory sample_papers mapping and returns the stored text. Returns None if no sample exists for the given ID.

        Parameters:
            document_id (str): Identifier of the sample paper to retrieve.

        Returns:
            Optional[str]: The full sample paper text, or None if not found.
        """
        return self.sample_papers.get(document_id)

    def get_document_chunks(self, document_id: str) -> list[str]:
        """
        Return the document split into paragraph chunks.

        Retrieves the full document text for the given document_id and splits it into paragraphs using a double-newline delimiter. Each returned chunk is stripped of surrounding whitespace and empty chunks are omitted. Returns an empty list if the document is not found or contains no text.
        """
        text = self.get_document_text(document_id)
        if not text:
            return []
        return [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]

def main():
    """
    Run a demonstration of custom DocumentProvider and CitationStorage implementations.

    Performs an end-to-end example that:
    - Instantiates MockAPIDocumentProvider, SQLiteCitationStorage (at /tmp/custom_citations.db), and UniversalBibliographyExtractor.
    - Processes two sample papers ("paper_001", "paper_002"), extracts bibliography entries, prints short previews, and aggregates results.
    - Persists found bibliography entries to the SQLite database and prints basic database statistics (total entries, per-paper counts, ArXiv/DOI presence, and a simple confidence distribution).
    - Prints examples of extension ideas and the locations of created files.

    Side effects:
    - May remove an existing file at /tmp/custom_citations.db and create a new SQLite database with tables `bibliography_entries` and `in_text_citations`.
    - Writes output to stdout.

    Returns:
        None
    """

    print("🔧 Custom Provider & Storage Example")
    print("=" * 50)
    print("Demonstrating how to extend the Academic Citation Toolkit")
    print("with custom DocumentProvider and CitationStorage implementations.")
    print()

    # Setup SQLite database
    db_path = "/tmp/custom_citations.db"
    if os.path.exists(db_path):
        os.remove(db_path)  # Start fresh

    # Create custom components
    provider = MockAPIDocumentProvider()
    storage = SQLiteCitationStorage(db_path)
    extractor = UniversalBibliographyExtractor(provider)

    print("🏗️  Created custom components:")
    print("   DocumentProvider: MockAPIDocumentProvider")
    print(f"   CitationStorage: SQLiteCitationStorage ({db_path})")
    print("   Extractor: UniversalBibliographyExtractor")
    print()

    # Process sample papers
    sample_papers = ["paper_001", "paper_002"]
    all_entries = []

    for paper_id in sample_papers:
        print(f"📄 Processing: {paper_id}")

        # Extract bibliography
        entries = extractor.extract_paper_bibliography(paper_id)

        if entries:
            print(f"   ✅ Found {len(entries)} bibliography entries")

            # Show sample entries
            for i, entry in enumerate(entries[:2], 1):
                title_preview = entry.title[:50] + "..." if entry.title and len(entry.title) > 50 else entry.title or "No title"
                print(f"     {i}. [{entry.entry_number}] {title_preview}")
                if entry.authors:
                    authors_preview = ', '.join(entry.authors[:2])
                    if len(entry.authors) > 2:
                        authors_preview += f" (and {len(entry.authors) - 2} more)"
                    print(f"        Authors: {authors_preview}")
                print(f"        Confidence: {entry.confidence:.2f}")

            if len(entries) > 2:
                print(f"     ... and {len(entries) - 2} more entries")

            all_entries.extend(entries)
        else:
            print("   ❌ No bibliography entries found")

        print()

    # Store results
    if all_entries:
        print("💾 Storing results in SQLite database...")
        success = storage.store_bibliography_entries(all_entries)

        if success:
            # Show database statistics
            stats = storage.get_bibliography_stats()
            print("📊 Database Statistics:")
            print(f"   Total entries: {stats['total_entries']}")
            print(f"   Entries by paper: {stats['entries_by_paper']}")
            print(f"   With ArXiv IDs: {stats['with_arxiv_id']}")
            print(f"   With DOIs: {stats['with_doi']}")

            conf_dist = stats['confidence_distribution']
            print("   Confidence distribution:")
            print(f"     High (≥0.8): {conf_dist['high']}")
            print(f"     Medium (0.6-0.8): {conf_dist['medium']}")
            print(f"     Low (<0.6): {conf_dist['low']}")

    print("\n🌟 Custom Implementation Benefits:")
    print("✅ Works with any document source (APIs, databases, files)")
    print("✅ Stores data in any backend (SQLite, PostgreSQL, MongoDB, etc.)")
    print("✅ Maintains same interface - easy to swap implementations")
    print("✅ Full control over data storage schema and querying")
    print("✅ Can add custom business logic and validation")

    print("\n📈 Extension Examples:")
    print("• WebAPIDocumentProvider → Connect to SSRN, PubMed, any API")
    print("• PostgreSQLCitationStorage → Enterprise database integration")
    print("• CachedDocumentProvider → Add caching layer for performance")
    print("• ValidatedCitationStorage → Add data validation and cleaning")
    print("• MultiSourceProvider → Aggregate multiple document sources")

    print("\n📂 Files Created:")
    print(f"   SQLite database: {db_path}")
    print("   Tables: bibliography_entries, in_text_citations")

    # Show database file size
    if os.path.exists(db_path):
        size_bytes = os.path.getsize(db_path)
        print(f"   Database size: {size_bytes:,} bytes")

if __name__ == "__main__":
    main()
