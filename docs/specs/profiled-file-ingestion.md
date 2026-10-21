# Profile-aware foreground file ingestion

Status: foreground document ingestion for
[#225](https://github.com/toddwbucy/HADES/issues/225). Public database setup and
frontend activation remain gated; background ingestion still refuses managed
databases until jobs carry profile pins through admission, startup and retry.

## Entry points and policy

`hades --db WORKSPACE ingest notes.md` now selects the saved model profile for a
managed database. Multiple named document files and documents-only directories
use the same core `ProfiledDocuments::ingest_file` operation. CLI/service/MCP
search can retrieve these documents through the existing verified query path.
The explicit [registered provider routes](profiled-search-admission.md) are
required. The configured extraction endpoint remains an operational setting.

Admission requires a ready conventional document profile, existing schema and
an independently verified registered provider. It resolves the model before
connecting to extraction and never creates schema as an ingestion side effect.
Global embedding model/task, chunk sizes, late-chunking settings and the legacy
embedder socket cannot override the saved profile. Every file retains one profile
snapshot and serving observation. Profile drift at commit, or a changed provider
observation returned by planning/inference, fails that item.

The CLI rejects `--task`, `--claims`, unsupported collection profiles and
`--concurrency` outside 1–4. Code and unrouted named files are rejected before
starting the batch. Mixed trees containing code remain refused by their code
phase before document writes. Code support and feature-aware mixed-tree routing
are separate work; no code is silently embedded as prose.

## Source identity and replacement

File identities preserve the complete UTF-8 name and extension. Without a root,
the identity uses the canonical absolute path. With `--root`, it uses the exact
relative path, so `a/README.md`, `b/README.md` and `a/README.txt` remain distinct.
Sources outside the supplied root fail. A directory ingest supplies its own root.
`--id` provides an explicit logical identity for one input, without lossy key
normalization. All database keys are SHA-256-derived from these identities.

```bash
hades --db WORKSPACE ingest papers/a.md papers/b.txt --root papers
hades --db WORKSPACE ingest papers/a.md --root papers --force
```

Existing sources require `--force`, including unchanged files. There is no
automatic replacement or hash-only skip. A reused explicit ID cannot overwrite
a different source file even with force. Root-relative identities can be moved
with their corpus; absolute identities change when their path changes.

The source revision is pinned before extraction. Replacement validates existing
provenance and complete child records, then commits the source, file metadata,
passages and vectors together under the original profile guard. It refuses
unknown fields, foreign/orphan children, concurrent source changes and profile
drift. Text-only preparation cannot discard file provenance or adopt a
file-ingested source. Failed transactions preserve the prior document. Lost
commit acknowledgements remain uncertain; inspect stored state before an explicit
retry. A batch may contain both successful and failed documents and reports each
outcome with a nonzero exit when any item fails.

## Snapshots, metadata and bounds

The reader opens a regular file through a fixed descriptor, snapshots at most
8 MiB and sends those exact bytes to the extractor with a basename. The extractor
never reopens the CLI's mutable source path. Empty files, unsupported formats,
invalid identities and oversized inputs fail explicitly. Symlink targets must
also route as documents. Prepare valid smaller documents when needed; do not
split binary documents by byte offset to satisfy the limit.

Stored `file_ingestion` metadata has its own versioned format and includes:

- Canonical path, optional relative path and explicit ID.
- SHA-256 and byte length of the exact submitted file snapshot.
- Source Git observation, extractor metadata and reported source type.
- User `--metadata` under a separate `metadata` object. Existing keys survive
  replacement; supplied values replace matching keys. Keys such as `_key` or
  `embedding_binding` here cannot alter the authoritative record fields.

The existing `source_sha256` describes extracted text, which can differ from
input bytes for PDFs or other formats. Git is an admission-time observation,
not an atomic Git/content snapshot. File metadata is bounded to 64 KiB total;
extracted text remains limited to 2 MiB, prepared rows to 16 MiB and extraction
protocol messages to 16 MiB. Four files may run concurrently per process.
Blocking readers and transaction cleanup retain admission when their awaiting
task is cancelled. Remote extraction/inference cancellation depends on the service.

Degraded `pymupdf_fallback`, empty/oversized text, code responses and invalid
metadata fail before embedding. This path requests text only; it does not provide
visual embedding or separately store images/tables.

`--batch`, `--resume` and `--reset` are refused for managed databases. Foreground
multi-file processing neither reads nor writes `.hades-batch-state.json`.
Legacy checkpoints cannot prove a profile revision or corpus generation. Legacy
databases retain their existing pipeline and checkpoint behavior.

## Validation and remaining work

`profiled_files` tests snapshots, identity/relocation, metadata, explicit
replacement, rollback, cancellation, extractor errors, limits, source/profile
drift, provider reload and malformed provenance. Its strict runner also executes
`profiled_ingestion`, invoking the real CLI for named files, document-only trees,
policy refusal, untouched checkpoints and partial failure. Writes use `with_temp_db`.

```bash
cargo fetch --locked
python3 scripts/test_isolated_database.py --docker --contract profiled_files
```

The documented native `--arangod` alternative is supported. Both targets run in
default isolated CI. The fixed real-Jina qualification also uses file ingestion
with a private text-passthrough extractor; it qualifies embedding/storage
integration, not Docling extraction quality.
The [ThinkPad qualification](../evidence/jina-file-ingestion-thinkpad-2026-10-21.json)
passed all six expected retrieval answers and the long Unicode token-recount gates.

Job pins, broader workloads and mixed-tree routing, legacy adoption/migration,
backup/reconciliation tooling, public setup authorization and frontend activation
remain follow-up work. No installed services or active corpora are changed.
