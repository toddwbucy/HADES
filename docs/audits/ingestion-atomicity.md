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
before language-server work; stale preparations are rejected. A metadata-schema
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

This does **not** close #40. Remaining review includes the cross-file relationship
storage phase, end-to-end cancellation, physical source changes during
external analysis, and additional fallback/analyzer failure coverage. The transaction protects the prepared database
replacement, not filesystem reads or the entire multi-file ingest job. No live
service, database or installed binary has been changed.
