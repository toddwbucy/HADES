# Ingestion and schema-application contracts

## Scope and evidence

Source trace for epic #12 pinned to `2bb0e6a13cdc705d6708010047d487cd37dbc911`,
including the metadata/checkpoint fix proposed in PR #138. This revision is an
audit branch, not deployed provenance. The adjacent manifest pins inspected
sources; schema and document-metadata outcome reports preserve focused runtime
evidence separately. These two leaves bring combined source maps to 68/80,
including pending maps. Remaining: seven codebase commands, four graph-embed
commands and daemon. This is not full runtime or security certification.

| Leaf | Route and output |
| --- | --- |
| `ingest` | One directory routes to unified code/document phases; explicit inputs route to document batch processing. JSON envelope includes per-item or per-phase outcomes. A failed item makes outer success false and exit nonzero. Setup errors can precede an envelope. |
| `schema apply` | Reads and validates YAML, then either emits an offline JSON operation plan or executes ordered operations and prints applied:true. Import confirmation is required at all four document-write stages after PR #136. |

## Ingestion selection and setup

Directory mode rejects id, batch, resume, reset, claims and root flags. The given
directory is the root. Discovery uses ignore walking, does not follow directory
symlinks, respects `.hadesignore` and configured skip directories, sorts results,
and fails on walk errors. Known code extensions win; document routing accepts
md/markdown/txt/text/rst/pdf/tex/gz. Otherwise explicit unparsed extensions and
extensionless shebangs may select code; unclaimed files are listed. Listing
unrouted files alone does not make the overall result fail.

Code runs first. A returned code-phase failure is retained while the document
phase runs; a code setup error propagated with `?` can terminate before it.
Document setup failure is retained alongside any code results. This route does
not provide a transaction spanning phases. Code-phase internal fidelity and
relationship semantics remain part of the separate codebase audit.

Explicit single code files are rejected; the same single-file guard does not
apply to multiple named inputs. Explicit inputs reject unparsed-ext. Claims are
accepted on this path but the document handler's `_claims` argument is unused.
ID requires a single input; resume permits no inputs but does not reconstruct
paths from the checkpoint. Metadata must be a JSON object. Profiles come from a
static registry despite help referring to runtime schema; the default consults
`HADES_DEFAULT_COLLECTION`, falling back to default for unknown names.

The adapter ensures three profile collections before connecting extraction and
embedding services. Earlier collection creation can remain if connection fails.
CLI concurrency is nonzero; configured concurrency is clamped to at least one.
Rate limiting spaces item starts; the processor calls acquire, not the limiter's
backoff/retry methods. These settings do not establish whole-ingest time or size
bounds. Chunking uses whitespace tokens, not the model tokenizer; overlap at or
above size reduces the step to one, and size zero yields an empty-chunk failure.

## Identity, storage and metadata

Explicit ID wins; otherwise a root-relative extensionless path is normalized,
with out-of-root inputs falling back to a stem with a warning. Normalization
strips a trailing version suffix and replaces dots/slashes with underscores;
this is not a collision-free encoding. Paths are canonicalized before extraction.
Existing relative identities are compared where both are available; otherwise
absolute identities are compared only when neither side has a relative identity.
Mixed legacy identity shapes are tolerated. Identity mismatch fails even under
force. Matching UTF-8 content hashes skip unchanged files unless forced; files
without a readable text hash are processed again.

The pipeline extracts, chunks and embeds before entering a database transaction.
It replaces document metadata, chunks and embeddings together, checking document
acknowledgment cardinality, key and revision; overwrite cleans old chunk/vector
rows inside that transaction. The CLI subsequently adds source identity, hash
and custom metadata in a separate PATCH. At this pin, rejection fails the item
and explicitly discloses previously committed content (#137/PR #138).

Custom metadata protects only `_key`, status and source; other identity/hash
fields may be overridden. Stored task/model identity is not part of the unchanged
content skip decision. Concurrent same-key writers, crash windows between the
two writes, metadata overrides and malformed PATCH acknowledgments still need
separate validation. The successful stage outcome does not prove atomic whole
ingestion or corpus freshness under every configuration change.

## Batch checkpoints and result completeness

Batch/resume/multiple inputs enable `.hades-batch-state.json` in the working
directory. Keys are caller-provided path strings, not database/profile/model or
content identities. Resume skips completed entries and retries failed entries at
this pin (PR #138); reset conflicts with resume. Empty resume input can clear a
checkpoint without reconstructing its prior work. Save uses a fixed temporary
filename and rename, with no interprocess checkpoint lock. Per-item save errors
warn and continue; final save/clear errors propagate.

Results accumulate in memory. Normal errors are recorded per item. The opportunistic
`while let Some(Ok(...))` drain can consume a join error without recording it,
whereas the final drain explicitly reports panics. This is a source candidate
requiring a focused test, not a proven production incident. Success accounting
and checkpoint scope need further adversarial/concurrency validation.

## Schema application

Dry-run validates and plans without querying the database. Core dry-run returns
zero counters instead of the CLI plan. Validation checks declaration/key
uniqueness and several reference/type constraints, but does not establish valid
server identifiers, positive feature dimensions or unique graph names.

The in-use guard checks declared non-universal collections; implicit schema
metadata is not inspected unless declared, universal codebase collections are
exempt, and force skips the guard. Missing/nonnumeric count results become zero.
No transaction couples guard reads with later writes.

Execution ensures collections, imports seeds in sorted collection order, writes
edge and graph metadata, writes schema_meta, then creates graphs. Existing-object
conflicts are accepted without type/definition readback. Import replies require
error:false, unsigned counters, no errors/empty/ignored rows and checked
created+updated equal to submitted count. Failure names the stage and warns about
earlier effects. Real-backend tests verify create/replace and one complete=true
batch rejection; they do not prove whole-operation rollback.

Empty YAML still writes metadata; omitted old definitions remain. Checksums cover
relation order rather than complete schema content. Reconciliation, guard races,
existing-object mismatch and recovery remain open scope. See the schema outcome
report for baseline and remediation evidence; no commands were deployed here.
