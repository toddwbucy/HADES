#!/usr/bin/env python3
"""
Filesystem Citation Extraction Example
======================================

Demonstrates using the Academic Citation Toolkit with local PDF/text files.
Shows how the same toolkit works with any document source.
"""

import os

# Prefer package-relative import; provide a helpful message if run as a script
try:
    from ..academic_citation_toolkit import create_filesystem_citation_toolkit
except ImportError as e:
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        raise SystemExit(
            "This example must be run as a module:\n"
            "  python -m tools.rag_utils.examples.filesystem_example"
        ) from e
    raise

def create_sample_paper(file_path: str):
    """
    Create a small synthetic academic paper and write it to the given path.

    The file contains a multi-section example (title, Abstract, Introduction, Related Work, Methodology,
    Results, Conclusion) and a References section with five bibliographic entries. The file is written
    using UTF-8 encoding; any existing file at file_path will be overwritten.
    """
    sample_content = """
# Advances in Natural Language Processing

## Abstract
This paper presents novel approaches to natural language processing using deep learning techniques.

## Introduction
Natural language processing has seen significant advances in recent years. Our approach builds on previous work in the field.

## Related Work
Several important papers have contributed to this field. The foundational work by Mikolov et al. [1] introduced word embeddings. Subsequently, attention mechanisms were explored by Vaswani et al. [2]. More recently, transformer architectures have been refined [3].

## Methodology
Our approach combines several techniques from the literature to achieve improved performance.

## Results
We demonstrate significant improvements over baseline methods.

## Conclusion
This work presents a novel contribution to natural language processing.

## References

[1] T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781, 2013.

[2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention Is All You Need. Advances in Neural Information Processing Systems, 2017.

[3] J. Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805, 2018.

[4] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems, 33:1877-1901, 2020.

[5] OpenAI. GPT-4 Technical Report. arXiv:2303.08774, 2023.
"""

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_content.strip())

def main():
    """
    Demonstrates extracting bibliographic entries from a local text file and storing results as JSON.

    This example:
    - Creates /tmp/sample_papers and /tmp/citation_results (if missing) and writes a sample paper to
      /tmp/sample_papers/sample_nlp_paper.txt.
    - Instantiates a filesystem-backed citation toolkit (via create_filesystem_citation_toolkit),
      extracts bibliography entries for the sample paper (paper_id "sample_nlp_paper"), prints a
      concise per-entry summary (title, arXiv/DOI, author preview, venue, year, confidence, raw text
      snippet), and attempts to store the extracted entries to output_dir/bibliography.json.
    - On successful storage, loads the JSON to print a small storage summary (entry count, file size,
      and a sample stored entry).

    Side effects:
    - Creates filesystem paths and files described above.
    - Prints progress and diagnostic information to stdout.
    """

    print("📁 Filesystem Citation Extraction Example")
    print("=" * 50)
    print("Demonstrating the Universal Academic Citation Toolkit")
    print("with local text files and JSON output storage.")
    print()

    # Setup directories
    input_dir = "/tmp/sample_papers"
    output_dir = "/tmp/citation_results"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create sample paper
    sample_paper_path = f"{input_dir}/sample_nlp_paper.txt"
    create_sample_paper(sample_paper_path)

    print(f"📝 Created sample paper: {sample_paper_path}")
    print()

    # Create toolkit for filesystem
    extractor, storage = create_filesystem_citation_toolkit(
        file_path=input_dir,
        output_path=output_dir
    )

    # Extract bibliography from sample paper
    paper_id = "sample_nlp_paper"  # filename without extension

    print(f"📄 Processing paper: {paper_id}")

    # Extract bibliography
    entries = extractor.extract_paper_bibliography(paper_id)

    if entries:
        print(f"  ✅ Found {len(entries)} bibliography entries")

        # Show all entries
        print("  📚 Bibliography entries:")
        for i, entry in enumerate(entries, 1):
            print(f"    {i}. [{entry.entry_number}] {entry.title or 'No title'}")

            if entry.arxiv_id:
                print(f"       ArXiv ID: {entry.arxiv_id}")
            if entry.doi:
                print(f"       DOI: {entry.doi}")
            if entry.authors:
                authors_preview = ', '.join(entry.authors[:3])
                if len(entry.authors) > 3:
                    authors_preview += f" (and {len(entry.authors) - 3} more)"
                print(f"       Authors: {authors_preview}")
            if entry.venue:
                print(f"       Venue: {entry.venue}")
            if entry.year:
                print(f"       Year: {entry.year}")

            print(f"       Confidence: {entry.confidence:.2f}")
            print(f"       Raw text: {entry.raw_text[:100]}...")
            print()

        # Store results to JSON
        success = storage.store_bibliography_entries(entries)

        if success:
            print(f"  💾 Stored bibliography entries to: {output_dir}/bibliography.json")

            # Show what was stored
            import json
            with open(f"{output_dir}/bibliography.json") as f:
                stored_data = json.load(f)

            print("  📊 Storage summary:")
            print(f"     Entries stored: {len(stored_data)}")
            print(f"     JSON file size: {os.path.getsize(f'{output_dir}/bibliography.json')} bytes")

            # Show sample stored entry
            if stored_data:
                sample_entry = stored_data[0]
                print("     Sample stored entry:")
                for key, value in sample_entry.items():
                    if key == 'raw_text':
                        print(f"       {key}: {str(value)[:60]}...")
                    else:
                        print(f"       {key}: {value}")
        else:
            print("  ❌ Error storing bibliography entries")
    else:
        print("  ❌ No bibliography entries found")

    print("\n🌍 Universal Nature Demonstrated:")
    print("This same code works with:")
    print("  • ArXiv papers (via ArangoDB)")
    print("  • Local PDF/text files (via filesystem)")
    print("  • SSRN papers (via custom API provider)")
    print("  • PubMed articles (via custom API provider)")
    print("  • Harvard Law Library (via custom provider)")
    print("  • Any academic corpus!")

    print("\n📂 Output files created:")
    print(f"  Input paper: {sample_paper_path}")
    print(f"  Bibliography JSON: {output_dir}/bibliography.json")

    print("\n🎯 Next Steps:")
    print("1. Implement custom DocumentProvider for your specific corpus")
    print("2. Choose appropriate CitationStorage for your needs")
    print("3. Process papers in batch for efficiency")
    print("4. Resolve bibliography entries to build citation networks")

if __name__ == "__main__":
    main()
