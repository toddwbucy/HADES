# Ingestion atomicity remediation (#40, in progress)

The baseline schema-rejection fixture is retained in
`ingestion-failed-replacement-baseline.json`: failed replacement removed the old
chunks and symbols. The current isolated implementation preserves that graph.

## Implemented boundary

Parsed and fallback file ingestion prepare documents before starting a database
transaction. The transaction checks the file revision observed before preparation,
then replaces chunks, symbols, embeddings, outgoing definition/call/import/impl
edges as applicable, and file metadata together. A competing writer invalidates
the prepared revision and produces a retryable error. In-memory import indexes
are updated only after persistence is acknowledged. Inbound edges to moved
symbols are remapped within the same file transaction, before commit.

The implementation uses the [ArangoDB stream transaction API](https://docs.arango.ai/arangodb/stable/develop/http-api/transactions/stream-transactions/)
and Document API batch writes; the existing Import API is not used inside the
transaction. Each response is checked for per-document failures. All eight
codebase collections receive exclusive locks during persistence; model work is
outside this interval. This serializes file commits and may reduce write
throughput. A five-second lock wait, 60-second operation deadline and 32 MiB
server transaction budget bound this stage; oversized writes fail atomically.

An independent task retains creation/abort ownership if a caller disappears.
Cancellation before commit, callback errors and panics attempt a bounded abort.
A lost transaction-creation response or unavailable abort requires server idle
expiry as fallback. A lost commit response is reported as an uncertain outcome;
callers must inspect committed state before retrying. Cancellation after commit
has been sent cannot promise rollback.

## Evidence and remaining work

- Private transaction contracts verify commit, rollback, callback panic,
  cancellation during creation and lock release after cancellation.
- The schema-rejection regression compares all eight collection contents before
  and after failure, then verifies successful retry and graph validation.
- The maintained disposable database and CLI suites pass with this implementation.
- Competing prepared updates admit exactly one commit; rejected fallback writes
  preserve both merged metadata and existing chunks. Both private contracts pass.

LSP enrichment now commits its prepared symbols, edges, success metadata and
recomputed counts as one separate atomic stage. File revisions are captured
before language-server work; stale preparations are rejected. Each captured source
must match its stored content hash before analysis and is checked again inside
the transaction before writes and before commit. All captured inputs participate,
including a file for which the analyzer returns no extraction. A metadata-schema
fault after symbol insertion rolls back the entire stage, and retry succeeds.
Structural file commits that preceded enrichment remain durable.

Embedding preparation errors now return a failed file result with its embedding
diagnostic before starting persistence. A malformed-response fixture confirms
all previously committed graph contents survive and retry succeeds. Source-only
ingestion when no embedding backend connects retains its existing explicit
warning behavior; it does not fabricate vectors.

Inbound-remap writes use the transaction-scoped Document API and validate every
batch response. Each edge collection is snapshotted before its rewrites, preserving
overlapping `A → B → C` moves and canonical keys. Snapshot response and server
query memory are each limited to 32 MiB; exceeding a limit aborts replacement.
A strict edge-schema rejection after new chunk/symbol writes preserves the old
file revision, chunk, symbol and inbound edge exactly; retry succeeds. Existing
canonical-collapse and overlapping-chain regressions pass, including a chain
across the former 2,000-entry read boundary. The full maintained isolated suite
and core/CLI all-target Clippy pass at this stage.

Cross-file Python/Rust/C++/Tree-sitter relationships now commit in one separate
transaction. The file replacement returns its new revision from the transactional
write response; relationship persistence verifies these exact revisions under
exclusive locks, checks endpoint existence, and validates each Document API batch.
Errors propagate instead of logging and reporting generated edge counts as stored.
A rejected call-edge batch rolls back earlier import-edge writes; retry, stale
preparation rejection and missing-endpoint rejection have isolated regression
coverage. The full maintained suite passes after this change.

This does **not** close #40. Earlier file replacements remain committed if the
relationship stage fails. Parsed file commits carry `relationships_pending: true`;
the successful relationship transaction clears it, even when no edges resolve.
Pending files bypass the unchanged-source skip. A private CLI regression rejects
a call-edge write, observes pending files and zero stored relationship counts in
the failure envelope, removes the fault, retries without `--force`, validates the
graph, and verifies the next repeat skips completed files. Lower-fidelity analysis
cannot silently skip a pending higher-fidelity file. File failures defer the
relationship stage and leave prepared files pending; language-server enrichment
is deferred when that stage fails.

Unchanged files that pass the normal content gate now contribute analyzed symbols
to the relationship index. Their pre-analysis file revision is verified by the
relationship transaction, so concurrent replacements reject stale resolution.
A CLI regression confirmed that a consumer-only body edit previously removed its
call edge; it now retains both call and import edges to the unchanged provider.
Explicit raw-text downgrade clears the pending marker in the fallback transaction;
a rejected chunk write retains the old metadata and marker, and successful retry
clears it before subsequent incremental skipping.

A process-level CLI test blocks a replacement embedding request on a private mock,
kills and reaps only that CLI process, compares all eight graph collections to
their prior contents, then retries and validates retrieval of the changed text.
This proves interruption during preparation preserves committed data. It does
not by itself establish cleanup after a stream transaction has begun. A separate
transaction-writer subprocess test deletes an existing document and inserts a
new one under an exclusive stream transaction, signals that both writes finished,
then is killed and reaped. With no client cleanup task left alive, the next
exclusive transaction must recover the exact original document, find the partial
insert absent, and commit a new write after server expiry. The private ArangoDB
uses its normal 60-second streaming idle timeout; the test permits a 100-second
cleanup window. This exercises the shared transaction primitive, not a full CLI
process interrupted at every persistence request.

When parsed ingestion preserves higher-fidelity stored analysis, it loads durable
symbol targets through a bounded snapshot and validates their canonical keys and
file revision. Stored outgoing-call metadata is excluded from this index; existing
outgoing relationships remain preserved. The same target loading applies when
registered-language analysis fails and raw fallback preserves the old graph.
A private resolver fixture verifies target resolution, exclusion of outgoing
calls, and stale-snapshot rejection; the existing Go preservation fixture also
exercises this path.

Explicitly unparsed files also load preserved targets using the stored file's
language. A CLI fixture first ingests an unusual-extension provider under a Python
language override, then preserves it through `--unparsed-ext`; a consumer-only
edit retains the same call/import edges and the provider's semantic metadata.
Missing or unsupported stored language fails explicitly instead of silently
omitting its targets.

Remaining review includes full-ingest process death during persistence and
additional analyzer failure coverage. Source-hash checks detect observed drift but do not lock the
filesystem: edit-and-restore races, changes after the final check, and analyzer
inputs outside the captured source list are not excluded by this contract. The transaction protects the prepared database
replacement, not filesystem reads or the entire multi-file ingest job. No live
service, database or installed binary has been changed.
