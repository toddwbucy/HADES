# Guided reading: PDF layout probe

Date: 2026-10-17. Base revision: `630cb9a459a37011c324d3810567a7631d6a02e2`
(PR #207). Work branch: `feat/guided-reading-foundation`.

This experiment checks the PDF prerequisite for issue #206. It does not implement
the reader, shared focus, proposals, acceptance, or semantic retrieval. No database
was accessed and no live service was changed. The PDFs stayed byte-identical.

## Method and results

`scripts/probe_pdf_layout.py` extracts selected physical, one-based pages on CPU.
`scripts/probe_pdf_layout.cjs` renders the original PDFs with PDF.js, overlays
Docling text boxes, and captures a selection from the browser text layer.
Browser requests are intercepted and served from local files. No listener is
opened. Raw extracts, quotes, and screenshots remain private experiment artifacts.

Versions exercised: Docling 2.126.0, docling-core 2.96.0, PDF.js 5.4.296,
Playwright 1.50.1, Chromium 131.0.6778.204. OCR, table structure recognition, and
formula enrichment were disabled. These are probe settings, not new defaults.

| Case | Source and physical page | Text elements | Unique normalized text matches | Empty text elements | Selected text runs |
|---|---|---:|---:|---:|---:|
| Two columns | Mikolov et al., *Recurrent neural network based language model* (2010), p. 2 | 25 | 8 | 7 | 3 |
| Equations | Behrouz et al., *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663, p. 6 | 14 | 5 | 5 | 13 |
| Figure caption | Same Titans PDF, p. 8, Figure 2 | 11 | 7 | 2 | 14 |

PDF SHA-256 identities:

- Mikolov: `1ac51c2f2e8f009b0312bdbc56a28415c6024887065f96c02af2e422fbc5ad37`
- Titans: `a65e4a7d02784df1a040b487127e6dd09fff4474e5caf94d93263af3d50cfbc2`

The matching column compares each Docling text element with PDF.js text runs
after NFKC and whitespace normalization. It is a diagnostic, not anchor
validation or a measured accuracy rate. Nonempty unmatched elements numbered
10, 4, and 2 respectively. No tested nonempty element had multiple matches.

Visual inspection found useful paragraph and equation boxes on all three pages.
The two-column page retained separate paragraph boxes. The selected prose did
not include the neighboring column, although native selection paint extended
across the gutter. The interface must distinguish a passage's highlight geometry
from the browser's selection paint.

The equation page had five formula boxes with empty extracted text. Formula
recognition was disabled, so this is not evidence that the source lacks content.
The reader must preserve access to the rendered equation and report unavailable
extracted text rather than supplying an invented transcription.

The Figure 2 caption was one Docling element linked from the picture. PDF.js
split it across fourteen selected text runs. The browser selection preserved the
caption across these runs. This checks a multiline, mixed-format caption, not a
caption divided across pages or multiple Docling elements. Those cases remain
untested, as do scans, rotated pages, and repeated identical quotations.

## Consequences for anchors

Use the user's exact browser quote, source hash, physical page, prefix, and
suffix. Keep Docling element IDs, boxes, coordinate origin, and extraction version
as navigation aids. Re-extraction must not redefine an accepted quote's identity.
Figure anchors also need a page, box, and caption reference.

The scripts deliberately do not claim to implement a durable anchor resolver.
Browser `Selection.toString()`, DOM range text, and PDF.js text runs differ in
line-break handling. Before storing accepted anchors, define and version one
page-text serialization, retain the exact displayed quote, and test re-resolution
after reload. Detect zero or multiple matches and require human resolution.
Do not silently normalize mathematical symbols into another meaning.

The automated probe selects spans using a box and creates a DOM range. Manual
mouse/keyboard selection and a full accept/reopen interaction are still required.
Spoken paragraph numbers need visible numbering and an explicit choice when the
user's paragraph count differs from the extractor's.

## Existing extraction adapter

The inspected backend used `DocumentConverter.convert_single`, which the installed
Docling version does not expose. This branch switches to `convert`, rejects
incomplete conversions, reads the page-count method, and resolves table/picture
captions through their document. Four model-free adapter tests exercise real
Docling document types. A separate CPU run through `DoclingExtractor.extract`
returned 24,596 text characters, four tables, two pictures, and four pages for the
Mikolov PDF, with fallback disabled. Table recognition remained disabled for this
run, so table counts do not establish table-content quality.

This is local backend validation, not a live gRPC or ingestion test. Existing
equation extraction still needs to handle current Docling `formula` labels.
`ExtractionResult` and `proto/persephone/extraction/extraction.proto` do not yet
transport layout provenance. Adding that data remains Python/proto work, not a
Rust extraction rewrite. PyMuPDF was unavailable in the tested environment, so
fallback was not validated.

Validation: all 14 tests across `test_docling_backend.py` and
`test_extraction_ownership.py` passed, including private Unix-socket fixtures.
Both probe scripts passed syntax checks, and `git diff --check` passed.

## Reproduce

Use the services environment, a fresh private output directory, local PDFs, and
a local PDF.js distribution containing `pdf.mjs`, `pdf.worker.mjs`,
`pdf_viewer.css`, and its `package.json`. Install Playwright and a compatible
browser separately. Docling may download its layout model on the first run.
The dependency cache, source paths, extracts, and screenshots are local artifacts.

```bash
CUDA_VISIBLE_DEVICES='' services/.venv/bin/python scripts/probe_pdf_layout.py \
  --output /tmp/hades-layout-new-run \
  --case two-column /path/to/mikolov.pdf 2 \
  --case equations /path/to/titans.pdf 6 \
  --case figure-caption /path/to/titans.pdf 8

PLAYWRIGHT_MODULE=/path/to/playwright CHROMIUM_PATH=/path/to/chrome \
  node scripts/probe_pdf_layout.cjs /tmp/hades-layout-new-run /path/to/pdfjs

services/.venv/bin/python -m pytest services/tests/test_docling_backend.py -q
```

The output directory contains the source manifest, Docling JSON per case,
`browser-report.json`, and screenshots. Keep it private: it includes original
text and machine-specific paths. The scripts are diagnostic tools, not importers.

## Next implementation boundary

The samples support proceeding with a quote-first reader, with the limitations
above visible. The next coherent slice is a source selection and semantic label,
an agent proposal with existing graph endpoints and supporting passages, and
human acceptance of an exact revision. HADES core must own the revisioned records
and retry-safe accepted write. The viewer and MCP must share those operations.
Persist model origin, human edits, and reviewer attribution separately.
IS/IS-NOT boundaries remain source-contextual and unresolved where unstated.
