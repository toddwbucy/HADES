#!/usr/bin/env python3
"""
Batch process multiple arxiv PDFs through the HADES pipeline.

Usage:
    poetry run python tests/functional/batch_process.py <pdf1> <pdf2> ...
"""

import subprocess
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: batch_process.py <pdf1> <pdf2> ...")
        sys.exit(1)

    pdfs = sys.argv[1:]
    results = []
    total_start = time.time()

    print("=" * 60)
    print(f"Batch Processing {len(pdfs)} PDFs")
    print("=" * 60)

    for i, pdf in enumerate(pdfs, 1):
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            print(f"\n[{i}/{len(pdfs)}] SKIP: {pdf} (not found)")
            results.append((pdf, False, "File not found"))
            continue

        print(f"\n[{i}/{len(pdfs)}] Processing: {pdf_path.name}")
        print("-" * 40)

        start = time.time()
        result = subprocess.run(
            ["poetry", "run", "python", "tests/functional/process_arxiv_pdf.py", str(pdf_path)],
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"✓ Success in {elapsed:.1f}s")
            results.append((pdf, True, f"{elapsed:.1f}s"))
        else:
            print(f"✗ Failed after {elapsed:.1f}s")
            # Print last few lines of error
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-5:]:
                print(f"  {line}")
            results.append((pdf, False, result.stderr[-200:] if result.stderr else "Unknown error"))

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)

    success_count = sum(1 for _, success, _ in results if success)
    print(f"\nTotal: {success_count}/{len(results)} successful")
    print(f"Total time: {total_elapsed:.1f}s")

    print("\nResults:")
    for pdf, success, msg in results:
        status = "✓" if success else "✗"
        print(f"  {status} {Path(pdf).name}: {msg}")

    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
