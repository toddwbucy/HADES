#!/usr/bin/env python3
"""
ArXiv Citation Extraction Example
=================================

Demonstrates using the Academic Citation Toolkit with ArXiv papers stored in ArangoDB.
This is our current use case for the word2vec evolution study.
"""

import os

from arango import ArangoClient

# Use proper relative import from parent package
from ..academic_citation_toolkit import create_arxiv_citation_toolkit


def main():
    """
    Extract and store bibliography entries for a small set of ArXiv papers.

    This is an example command-line entry point that:
    - Requires the ARANGO_PASSWORD environment variable; exits early if not set.
    - Connects to a hard-coded ArangoDB host and builds an ArXiv citation toolkit
      via create_arxiv_citation_toolkit(client).
    - Extracts bibliography entries for a predefined set of "core" paper IDs,
      reports per-paper and aggregated confidence and identifier statistics,
      and attempts to persist the collected entries using storage.store_bibliography_entries().
    - Prints progress, summary statistics, and next steps to stdout.

    No return value. Side effects: reads environment, connects to ArangoDB, prints to stdout,
    and writes bibliography entries to the ArangoDB collection (via the storage component).
    """

    print("🕸️ ArXiv Citation Extraction Example")
    print("=" * 50)
    print("Extracting citations from word2vec evolution papers using the")
    print("Universal Academic Citation Toolkit with ArangoDB backend.")
    print()

    # Connect to ArangoDB
    arango_password = os.getenv('ARANGO_PASSWORD')
    if not arango_password:
        print("❌ Error: ARANGO_PASSWORD environment variable not set")
        return

    client = ArangoClient(hosts='http://192.168.1.69:8529')

    # Create toolkit for ArXiv papers
    extractor, storage = create_arxiv_citation_toolkit(client)

    # Our core papers for word2vec evolution study
    core_papers = {
        '1301_3781': 'Efficient Estimation of Word Representations in Vector Space',
        '1405_4053': 'Distributed Representations of Sentences and Documents',
        '1803_09473': 'code2vec: Learning Distributed Representations of Code'
    }

    print(f"📚 Processing {len(core_papers)} core papers:")
    for paper_id, title in core_papers.items():
        print(f"  • {paper_id}: {title}")
    print()

    # Extract bibliography from each paper
    all_entries = []

    for paper_id, _title in core_papers.items():
        print(f"📄 Processing: {paper_id}")

        # Extract bibliography
        entries = extractor.extract_paper_bibliography(paper_id)

        if entries:
            print(f"  ✅ Found {len(entries)} bibliography entries")

            # Show confidence distribution
            high_conf = [e for e in entries if e.confidence >= 0.8]
            med_conf = [e for e in entries if 0.6 <= e.confidence < 0.8]
            low_conf = [e for e in entries if e.confidence < 0.6]

            print(f"     High confidence (≥0.8): {len(high_conf)}")
            print(f"     Medium confidence (0.6-0.8): {len(med_conf)}")
            print(f"     Low confidence (<0.6): {len(low_conf)}")

            # Show sample entries
            print("     Sample entries:")
            for i, entry in enumerate(entries[:3], 1):
                title_preview = entry.title[:40] + "..." if entry.title and len(entry.title) > 40 else entry.title or "No title"
                print(f"       {i}. [{entry.entry_number}] {title_preview}")
                if entry.arxiv_id:
                    print(f"          ArXiv: {entry.arxiv_id}")
                if entry.authors:
                    authors_preview = ', '.join(entry.authors[:2])
                    if len(entry.authors) > 2:
                        authors_preview += f" (and {len(entry.authors) - 2} more)"
                    print(f"          Authors: {authors_preview}")
                print(f"          Confidence: {entry.confidence:.2f}")

            if len(entries) > 3:
                print(f"       ... and {len(entries) - 3} more entries")

            all_entries.extend(entries)
        else:
            print("  ❌ No bibliography entries found")

        print()

    # Summary statistics
    print("📊 Summary Statistics:")
    print(f"  Total papers processed: {len(core_papers)}")
    print(f"  Total bibliography entries: {len(all_entries)}")

    if all_entries:
        # Confidence breakdown
        confidence_bins = {
            'High (≥0.8)': len([e for e in all_entries if e.confidence >= 0.8]),
            'Medium (0.6-0.8)': len([e for e in all_entries if 0.6 <= e.confidence < 0.8]),
            'Low (<0.6)': len([e for e in all_entries if e.confidence < 0.6])
        }

        print("  Confidence distribution:")
        for category, count in confidence_bins.items():
            percentage = (count / len(all_entries)) * 100
            print(f"    {category}: {count} ({percentage:.1f}%)")

        # Identifier statistics
        arxiv_count = len([e for e in all_entries if e.arxiv_id])
        doi_count = len([e for e in all_entries if e.doi])

        print(f"  Entries with ArXiv IDs: {arxiv_count}")
        print(f"  Entries with DOIs: {doi_count}")
        print(f"  Entries with strong identifiers: {arxiv_count + doi_count}")

    # Store results
    print("\n💾 Storing Results:")
    if all_entries:
        success = storage.store_bibliography_entries(all_entries)
        if success:
            print("✅ Successfully stored all bibliography entries in ArangoDB!")
            print("   Collection: bibliography_entries")
            print(f"   Total entries stored: {len(all_entries)}")
        else:
            print("❌ Error storing bibliography entries")
    else:
        print("ℹ️  No entries to store")

    print("\n🎉 ArXiv citation extraction complete!")
    print("Next steps:")
    print("1. Resolve these bibliography entries to papers in PostgreSQL database")
    print("2. Map in-text citations [1], [2], etc. to bibliography entries")
    print("3. Build citation network graph for word2vec evolution analysis")

if __name__ == "__main__":
    main()
