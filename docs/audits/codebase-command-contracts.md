# Codebase command contracts

Epic #12 source map at `13dff20c2811314cece05313bc0c12315c1311fd`.
The manifest pins the seven codebase leaves and reviewed entry/persistence sources.
This completes the native CLI entry-point map, not the full implementation audit,
parser/resolver conformance suite, deployed-server assessment or retrieval evaluation.

## Routing and outcomes

| Command | Behavior and important limits |
|---|---|
| `ingest` | Discovers files, prepares analysis/vectors, replaces files, then resolves relationships and enriches with LSP in separate stages |
| `update` | Delegates to ingestion defaults; it is not a separate incremental-update implementation |
| `stats` | Sequential counts of eight fixed collections; missing collections count as zero, other errors propagate; no common snapshot |
| `validate` | Seventeen declared invariants, fourteen database queries and three design/unit-test assumptions; inspect `summary.failed`, since violations do not make the outer command fail |
| `prune-orphans` | Cascade-ordered deletion across symbols, chunks, embeddings and four edge collections; dry-run counts are lower bounds |
| `drift` | Compares discovered files/content hashes with records attributed to the selected root; reports differences without a failing exit status |
| `retire` | Deletes explicit file keys and dependent records; additional authored edges require confirmation and are removed in subsequent operations |

The ingest CLI prints a failure-aware summary and exits nonzero on its typed
failure result; other errors propagate. Successful skipped files count as completed.
Relationship, enrichment and individual-file failures have ordered summary precedence.
A failing run can still have committed earlier files. Statistics output precedes
stderr diagnostics, so diagnostic-write failure can follow emitted JSON.

## Ingest discovery, identity and admission

Ingestion canonicalizes the root and connects to an optional embedder, then creates
schema before discovery, identity validation and analyzer preflight. An empty run
can therefore create collections or indices. Existing collection types and named
graph definitions are not compared with the desired definitions. Graph creation
conflict means already present; selected permission/unavailable statuses are nonfatal.
Index creation acknowledgments do not prove an existing definition matches intent.

Discovery honors ignore files including `.hadesignore`, skips configured directories,
does not follow directory symlinks and sorts results. Known extensions are included;
`--unparsed-ext` adds raw-text extensions. Tree-walk language override adds extensionless
files, while explicit-file override can select other extensions. Extensionless
shebang probes read at most 256 bytes; unreadable/non-UTF8 probes return no match.
Unified routing gives declared code/document routes priority over fallback classification.

Identity is scoped to canonical root and relative path. Preflight rejects legacy
identities anywhere in the database and checks requested identities in batches.
Explicit roots and paths must be UTF-8. Writer revisions are captured before file
preparation. These checks do not freeze the source tree or database for the whole run.

Actual incremental skipping uses **content hash**, pending-relationship state and,
when an embedder is connected, denormalized embedding/chunk counts. CLI help still
mentions symbol-hash skipping; that text is stale. Model/version changes are not
part of the predicate, and stored counts are not a fresh row-by-row validation.
The fidelity guard precedes `--force`: richer stored analysis is preserved unless
analysis downgrade is allowed. A Go-only exception requires scheduled gopls and a
resolvable module; it does not apply to incoming text-only analysis or Rust.
Analyzer probing uses the workspace and explicit configuration. Downgrade permission
can permit proceeding without semantic enrichment; it is not whole-run rollback.
The batch flag controls progress reporting; the reviewed file loop is serial.

## Embedding and persistence

Parsed files use late-chunking windows unless disabled by the supported environment
flag. Window size derives from the backend token ceiling through a character/token
heuristic, with fallback on missing information; this is not tokenizer-verified
admission. Byte ranges convert to character ranges, and explicit chunk indices
preserve alignment across skipped chunks. Invalid source slices skip a window.
Raw-text ingestion embeds independent chunks.

An embedding request error or partial output rejects file replacement before its
transaction. Two further source candidates need independent fixtures: batched late
embedding retains its initial error even after successful per-window retries;
optional embedder connection failure permits preparation with no vector documents,
which may remove prior vectors on a changed-file replacement. These are not claims
of observed production loss. See [vector retention](ingestion-vector-retention.md)
for previously tested cases and their narrower scope.

Each replacement transaction locks all codebase collections, verifies the prepared
revision, purges owned records, writes replacements, remaps inbound edges and requires
a returned file revision. Parsed files replace metadata; raw files merge metadata
and purge symbols only with explicit downgrade. Writes use the transactional Document
API, not Import API. Bulk acknowledgments check array length and explicit error=true;
full returned identity/type validation is not established by that check.

Inbound remapping snapshots edges before writes and writes canonical new keys before
removing superseded keys. Matching uses qualified name and position when before/after
counts agree; equal-count definition reordering remains ambiguous. Removed or unmatched
symbols can leave inbound edges dangling. The final diagnostic warns on query failure
and defaults malformed counts to zero; it is not a successful full-integrity proof.
Historical warning-only purge helpers under `cfg(test)` are not production paths.

Cross-file relationships commit separately after successful file processing. Any file
failure defers that stage. It verifies contributing revisions, endpoint classes and
existence, batches writes and clears pending flags. Preserved-file target snapshots
validate revisions, symbol identities/ranges and metadata, omit call metadata and
contribute targets rather than new source relationships.

LSP enrichment is another atomic stage after external analyzer work. It captures file
revisions and content hashes, rechecks them inside the transaction and rechecks source
content before finishing. It adds/replaces symbols and edges, patches analyzer metadata
and recomputes symbol counts. Filesystem changes after the last hash check remain possible.
Unchanged-file enrichment does not remove obsolete semantic artifacts by itself.
Workspace/extraction failures are tracked; incomplete enrichment of rewritten files
can fail the run unless downgrade was accepted. Zero analyzed Go workspaces also fails
with discovered Go targets and a passing probe. LSP bulk writes are not subdivided
like relationship batches. Transaction failure becomes `store_failed` for outer handling.

The shared transaction owner survives caller cancellation, limits lock acquisition to
five seconds, callback work to sixty seconds and server transaction size to 32 MiB.
It catches callback panics and attempts abort with a two-second wait. A lost commit
response produces an uncertain-outcome error; neither that error nor caller cancellation
after commit submission proves rollback. See [atomicity evidence](ingestion-atomicity.md).

## Management command boundaries

Validation queries sample at most fifty violations each. Their counts are sample counts.
Three assumed invariants and missing-collection results are recorded as skipped, even
though individual records can carry pass=true. Other query failures abort the report.
Endpoint existence does not establish allowed collection classes. Deterministic keys
are not checked against every stored record by the assumed invariants.

Pruning uses separate cascade queries rather than a whole-operation transaction.
Missing-collection errors and malformed/non-u64 count values can become zero. The
chunk-collection probe treats errors other than missing as existence; partial schemas
need fixtures. A dry run does not simulate downstream orphans created by actual deletion.

Drift separates attributed stale keys from unattributed keys and other roots. Unreadable
files or missing hashes are unverifiable and prevent clean=true; unhandled files and
other roots do not. Output samples cap at 200 unless full output is selected. Database
queries and filesystem reads do not share a snapshot. Malformed attribution/count fields
and duplicate returned keys need response-contract tests.

Retirement sorts/deduplicates explicit keys, rejects an all-missing selection and permits
mixed present/missing keys. Partition/scan response identity validation is incomplete.
Codebase sweep and authored-edge deletions are separate; scan/delete races remain possible.
At this source pin authored-edge backend failures become removed=0 and command success.
[Issue #147](https://github.com/toddwbucy/HADES/issues/147) and
[PR #148](https://github.com/toddwbucy/HADES/pull/148) contain a reproduced CLI failure
and tested correction; the correction is not part of this historical source pin.
Sweep acknowledgment shape and full-operation atomicity remain separate questions.

Python `from_import` can likewise select an unrelated same-name local definition at
this pin. [Issue #149](https://github.com/toddwbucy/HADES/issues/149) and
[PR #150](https://github.com/toddwbucy/HADES/pull/150) preserve the resolver reproduction
and correction. Relative imports, re-exports, module-prefix fallback and ambiguity still
require broader conformance tests; a module map is not proof of Python import semantics.

## Coverage and remaining acceptance

These seven leaves bring documented native CLI entry-point coverage to **80/80**,
combined with the previously published command maps. No new runtime tests, database
operations, service changes or model calls were performed for this report. The manifest
pins what was read; it does not certify every transitive analyzer/resolver/SDK function.
Existing isolated lifecycle evidence must retain its original source and fixture scope.

The full epic still requires candidate reproduction/disposition, protocol and parser
conformance, deployed source/model/schema/config identity, performance and retrieval
assessment with owner plus independent reviewer labels, backup coverage and recovery
verification, and resolution or explicit acceptance of remaining high-severity findings.
Neither complete command mapping nor merged fixes authorizes production deployment.
