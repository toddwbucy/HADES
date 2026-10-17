#!/usr/bin/env python3
"""CPU-only Docling layout experiment; never connects to a HADES database.

Run with the services Python environment and a private output directory. Each
case names a local PDF and one physical, one-based page. Source files stay intact.
The output contains source text and paths and is intended to remain local.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=3, action="append", required=True,
                        metavar=("LABEL", "PDF", "PAGE"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions

    options = PdfPipelineOptions(
        do_ocr=False, do_table_structure=False,
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=4),
    )
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
    })
    manifest = {"kind": "hades-pdf-layout-probe", "version": 1,
                "docling_version": importlib.metadata.version("docling"),
                "device": "cpu", "ocr": False, "table_structure": False,
                "formula_enrichment": False, "cases": []}
    for index, (label, filename, raw_page) in enumerate(args.case):
        path = Path(filename).resolve(strict=True)
        page = int(raw_page)
        if page < 1 or path.suffix.lower() != ".pdf":
            parser.error("cases require a local PDF and a positive physical page number")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        result = converter.convert(path, page_range=(page, page))
        if result.status.value != "success":
            raise RuntimeError(f"{label}: conversion did not complete: {result.status}")
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != after:
            raise RuntimeError(f"source changed during extraction: {label}")
        output = f"case-{index}.docling.json"
        (args.output / output).write_text(json.dumps(result.document.export_to_dict(), indent=2))
        manifest["cases"].append({"label": label, "pdf": str(path), "page": page,
                                  "source_sha256": before, "docling": output})
        print(f"{label}: page {page}, {len(result.document.texts)} text elements", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
