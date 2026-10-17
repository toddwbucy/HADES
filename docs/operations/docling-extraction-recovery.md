# Docling extraction and explicit recovery

Docling 2.126.0 and docling-core 2.96.0 are the tested converter pair. Install
`services/pyproject.toml` in a dedicated environment. CI installs the hashed CPU
requirements and runs actual Docling document types and converters, not only mocks.
The extraction service remains Python. The source files remain unchanged.

## Conversion policy

The backend calls `DocumentConverter.convert`, accepts only complete `success`,
and resolves table and figure captions against the returned document. Conversion
failures produce an INTERNAL gRPC error, which becomes a failed document in the
Rust ingest envelope. There is no automatic text-only fallback. The Rust client
also rejects `pymupdf_fallback` responses from older services.

Remove `HADES_EXTRACTOR_FALLBACK=true` from an old service configuration before
restarting the upgraded service. That setting now refuses startup with an explicit
error. Upgrade/restart is an operator deployment step, not an ingest side effect.

Formula enrichment defaults to off to avoid an extra model's memory and inference
cost on small machines. Set `HADES_EXTRACTOR_FORMULA_ENRICHMENT=true` to elect it.
Current `formula` labels are returned even when recognition yields empty text.
With enrichment off, `latex` remains empty rather than claiming plain text is
recognized LaTeX. Metadata records this setting, OCR, converter versions, and the
completed conversion status. `extract_equations` controls returned fields, not the
service's enrichment policy. Per-request OCR that contradicts service policy fails.

## Format routing and limits

The pinned Docling extension registry supplies the extractor's format set. A test
checks every registry extension against the Rust document routing table. The
Markdown spelling `.markdown` aliases `.md`. Markdown and plain text now pass through
Docling, so their derived text may differ from the original source bytes. The
mechanical graphable importer does not use this conversion path and is unchanged.

Actual model-free converters are tested over gRPC for DOCX, PPTX, XLSX, HTML,
Markdown, text, CSV, AsciiDoc, VTT, and Docling JSON. A separate opt-in test exercises
a real PDF. Registry agreement verifies routing, not successful conversion of
every possible file in every format. Legacy Office, media, and specialized XML
formats retain upstream dependencies such as LibreOffice, audio extras, or XBRL
extras. Missing prerequisites fail visibly. JSON must be Docling document JSON,
not arbitrary application configuration. Unsupported content is not accepted merely
because its extension is recognized.

ArXiv archives retain their native backend. METS bundles use the pinned Docling
archive-format probe before routing to Docling. RST and source-code extraction keep
the native text path. Dedicated code analyzers still take precedence in the Rust
router. PDF/image OCR and formula policy share the configured device. The default is
`auto`, with no workstation-specific GPU ordinal. Set device visibility locally.

## Inventory before recovery

Use the existing supported read operations. Select the database explicitly and
provide credentials through the usual environment/configuration, never in commands.
Run `hades --db DATABASE db databases`, then `db stats` and `db collections` for
each authorized database. Identify the metadata collection of each document profile.

Count recorded fallback rows and distinguish missing provenance:

```bash
hades --db DATABASE db aql '
FOR d IN @@collection
  COLLECT extractor = NOT_NULL(d.extractor_metadata.extractor,
                              d.metadata.extractor, "unknown") WITH COUNT INTO n
  RETURN {extractor, count: n}' \
  --bind '{"@collection":"documents"}'
```

List exact recovery candidates without extracting their text:

```bash
hades --db DATABASE db aql '
FOR d IN @@collection
  FILTER d.extractor_metadata.extractor == "pymupdf_fallback"
      OR d.metadata.extractor == "pymupdf_fallback"
  RETURN KEEP(d, "_key", "_id", "_rev", "source_path", "source_rel",
                 "content_hash", "extractor_metadata", "metadata")' \
  --bind '{"@collection":"documents"}'
```

A zero fallback count proves only that no rows explicitly carry that marker.
Missing metadata cannot establish which extractor ran. Check the actual deployed
unit and retained journal as a separate evidence source.

## Explicit re-extraction

1. Review candidates, the original files, and their database/profile identities.
   Archive the old document/chunks/embeddings with supported `db export` or the
   existing backup procedure before replacement. Derived text and vectors change.
2. Check the upgraded extraction service with a real PDF and assert
   `metadata.extractor == "docling"`. Verify the configured embedder is available.
3. For one reviewed candidate, use its original source path, exact key, profile,
   and original root when `source_rel` was recorded:

   ```bash
   hades --db DATABASE ingest /original/root/paper.pdf \
     --collection PROFILE --id STORED_KEY --root /original/root --force
   ```

   For a standalone source without `source_rel`, omit `--root`. Missing or ambiguous
   source identities require operator resolution. Do not invent a new key or blindly
   process every row. The source identity collision guard remains active under force.
4. Inspect the per-document ingest envelope and read the same metadata row with
   `db get`. Confirm Docling provenance and expected chunks, then run semantic
   retrieval with the correct profile. Successful extraction alone is not indexing.

Without explicit `--force`, transaction-level replacement preserves a stored
fallback document and reports the required election. The replacement transaction
covers metadata, chunks, and embeddings, including deletion of obsolete derived
rows. A failed replacement leaves that prior data intact. This operation does not
retain earlier derived versions automatically, does not repair authored references,
and never rewrites authored interpretation or graphable-import collections. Source
metadata updates still occur after the content transaction, so inspect any reported
post-commit metadata failure before retrying.

## Reproduce the real-PDF check

Generate protobuf stubs, use a private environment/model cache, and provide a local
PDF. This starts a private CPU gRPC server, verifies Docling/version metadata, and
checks the source hash before and after. It does not ingest or restart services.

```bash
make -C services proto-gen PROTO_DIR=../proto PYTHON=/path/to/venv/bin/python
CUDA_VISIBLE_DEVICES='' HF_HOME=/private/model-cache \
HADES_TEST_PDF=/path/to/original.pdf \
/path/to/venv/bin/python -m pytest services/tests/test_docling_formats.py -v
```

The first PDF conversion downloads Docling's public model artifacts if absent.
After populating the cache, use `HF_HUB_OFFLINE=1`. Offline/missing-model errors
must fail the test, not select another extractor.

## Read-only deployed audit, 2026-10-17

At reviewed main revision `35a628e`, the installed versions matched the pin. The
user-level unit was `hades-extractor.service`. Neither inspected journal exposed
retained entries, so no historical fallback event count can be established.

The supported CLI enumerated 45 accessible databases. Every nonempty non-system
document collection was inspected for current and legacy extractor metadata. No
row explicitly recorded `pymupdf_fallback`; 651 source/provenance-bearing rows
had no named extractor. This does not prove that they used Docling.

A real PDF sent to the deployed service failed with `Docling failed and PyMuPDF
not available for fallback`. The same original PDF succeeded through the fixed
private CPU gRPC service with the pinned converter pair. Its source hash was
unchanged. No production re-ingestion, package upgrade, or service restart was
performed. Deployment and any reviewed recovery remain operator steps.
