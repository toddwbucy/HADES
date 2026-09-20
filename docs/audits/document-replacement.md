# Document replacement failure (#88)

At `a2236f3`, a private end-to-end pipeline fixture confirms that rejected
replacement storage does not preserve the last complete document. The real
`Pipeline::process_document` uses a private gRPC extraction peer and a private
HTTP fixed-vector embedder. Only synthetic records enter the disposable database.

The successful overwrite control stores replacement metadata, chunk and vector.
A second case seeds an old complete document, then installs a strict embedding
schema requiring a field absent from newly generated rows. ArangoDB rejects the
replacement vector with error 1620. The pipeline correctly returns failure, but
subsequent reads show replacement metadata and chunk text and no old embedding.
The failure therefore leaves a partial replacement, not the old complete state.

`pipeline/orchestrator.rs::store` separately commits two deletes and three
imports. Its comment accepts an orphan-embedding window between the deletes;
the executed case covers the later embedding-import failure. This is separate
from the atomic code-file ingestion path and the WeaverTools adapter.

The historical probe is `repros/document_replacement_baseline.rs`, executed as
`crates/hades-core/tests/document_replacement_db.rs` with the shared embedding
mock and the strict isolated runner's contract list extended for that target.
It asserts the baseline defect and must not become a passing regression that
requires the defect to remain. The retained JSON records selected source hashes.

The run used ArangoDB 3.12.11, a fresh private Unix socket without TCP, one CPU,
lowered priority, 8 GiB child limits and a 300-second command deadline. The test
passed its baseline observations; the owned database process exited zero and
private peer tasks were stopped. No production service, database, model or
backup changed. No production data loss or retrieval-quality result is claimed.

Remediation and acceptance remain in #88: atomic per-document storage, ordinary
and legacy foreign-key cleanup, duplicate rejection, cancellation and concurrent
replacement contracts, with explicit commit-response ambiguity. The reproduction
alone does not close that issue.

## Transactional remediation in progress

Prepared metadata, chunks and vectors now enter one existing `transaction::run`
operation with exclusive locks on the three profile collections. Deletes use
transaction-bound AQL and writes use the Document API, whose per-row identity
and revision acknowledgments are checked before commit. The Import API is not
transactional and is no longer used by this storage path. External extraction
and embedding complete before lock acquisition.

The expanded real database fixture passes: successful overwrite removes legacy
foreign-key-only rows; a rejected embedding restores the exact previous JSON
rows, including revisions; unrelated rows survive; overwrite=false rejects an
existing document and preserves its metadata revision. The source document may
have changed during preparation: this is atomic last-committer replacement,
not source-revision conflict detection. Exclusive collection locks serialize
writes at storage time. Cancellation and concurrent-writer pipeline fixtures
remain required before the issue can close.

The shared transaction owner bounds lock acquisition to five seconds, operations
to 60 seconds and server transaction size to 32 MiB. Oversized replacements fail
instead of falling back to partial writes. Cancellation before commit requests
abort; a lost commit response leaves an uncertain outcome that requires durable
state reconciliation before retry. This is one-document atomicity, not a whole
batch transaction, source snapshot or production deployment.
