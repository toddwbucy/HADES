# Database embedding profiles: audit and implementation plan

Status: proposed design for review; no production behavior is changed.
Issues: [#215](https://github.com/toddwbucy/HADES/issues/215) (this audit) and
[#225](https://github.com/toddwbucy/HADES/issues/225) (implementation).

Audited on 2026-10-19 at `f930fbca1c5362f0e8e337ceeb1a1f9d21173cec`,
branch `docs/database-embedding-profile-audit`, from merged main including #224.
The original working checkout and its installation notes were preserved.
Paths and symbols below describe that revision; proposed behavior is identified
separately. This is a source audit with private mock checks, not a GPU benchmark.

## Recommendation

Make each database own a versioned embedding contract: a specific model artifact
identity, compatible task mappings, representation, and enabled ingestion policy.
Confirm the configured model with MD5 checksums of its actual inference assets.
Resolve this contract before scheduling ingestion or embedding a search query.
Keep endpoint, device, and batching in a separate host configuration.

Reuse the HTTP transport, response validation, window planner, schema operations,
job supervision, and frontend CLI boundary. Add a shared profile resolver and
verified provider contract before exposing database creation in the browser.
Start with Jina text/document ingestion and retrieval, then gate code and late
chunking explicitly. Prove the abstraction with a smaller model in a later PR.

## Current implementation and gaps

| Area | Observed implementation | Consequence for the feature |
|---|---|---|
| Configuration | [`config::load_config`](../../crates/hades-core/src/config/mod.rs) loads one YAML/env configuration. [`EmbeddingModelConfig`](../../crates/hades-core/src/config/types.rs) has name, dimension, token limit, precision, and normalization fields. | These are process configuration, not a persisted database contract. |
| Client construction | [`EmbeddingClient::connect_at`](../../crates/hades-core/src/persephone/embedding.rs) overrides only the endpoint and fills the remaining fields from defaults: Jina v4 and 2,048 dimensions. `connect(EmbeddingClientConfig)` can accept explicit alternatives. | Merely changing YAML model fields does not configure the common request paths. |
| Call sites | [`ingest::run_phase`](../../crates/hades-cli/src/commands/ingest.rs), [`codebase_ingest::run_phase`](../../crates/hades-cli/src/commands/codebase_ingest.rs), [`embed_mgmt::run_embed_text`](../../crates/hades-cli/src/commands/embed_mgmt.rs), and [`dispatch::handlers`](../../crates/hades-core/src/dispatch.rs) query/embed/smell handlers use `connect_at`. | Introduce one resolved-client factory and audit all callers; changing only the frontend or search would leave ingestion inconsistent. |
| Multiple databases | [`PoolCache::entry_for`](../../crates/hades-cli/src/commands/mcp_server.rs) clones the base config, changes the database, and caches the config/pool pair. [`daemon::run`](../../crates/hades-cli/src/commands/daemon.rs) retains its startup config. | Database pools are separated, but their embedding settings come from the same base configuration. There is no per-database model refresh/invalidation. |
| Background jobs | [`ingest_jobs::records::start_with`](../../crates/hades-core/src/ingest_jobs/records.rs) hands the child a sealed [`config::snapshot::Snapshot`](../../crates/hades-core/src/config/snapshot.rs). | Reuse the immutable handoff, but resolve and include the database profile revision first. It currently freezes global settings. The snapshot has a 64 KiB limit: do not insert an unbounded artifact manifest. |
| Model service | [`http_server.AppState`](../../services/embedding/http_server.py) directly constructs `JinaV4Embedder`; [`embedding/config.py`](../../services/embedding/config.py) changes the model path/name, not the implementation. | A backend adapter interface is needed; a different name does not supply a different model implementation. |
| Capabilities | Python `ModelInfo` advertises dimension, `max_seq_length`, tasks, and device/profile. Rust `ProviderInfo` reads only part of this metadata and has no artifact checksum or late-chunk capability field. Its single-entry fallback can read metadata despite an unmatched configured name. | Availability and advertised metadata do not establish verified compatibility. Discover and validate a versioned capability contract. |
| Identity | [`keys::model_hash`](../../crates/hades-core/src/db/keys.rs) is SHA-256 of the model identifier string. Client `validated_response_model` also accepts a Jina-specific directory-name alias. | Neither proves which model files were loaded. Keep the legacy field distinct from the new artifact MD5. |
| Creation | CLI [`db_write::run_create_database`](../../crates/hades-cli/src/commands/db_write.rs) and dispatch `db_create_database` create an ArangoDB database. Schema initialization/application and collection/index creation are separate operations. | No composed, model-bound setup receipt or readiness state exists. Both creation entry points must converge on the new workflow. |
| Schema | [`RuntimeSchema`](../../crates/hades-core/src/graph/runtime_schema.rs) reads `hades_schema`; [`schema_apply`](../../crates/hades-core/src/schema_apply.rs) supports declarative graph/collection definitions. | This describes graph structure, not a model binding. Preserve this distinction; avoid making schema reseeding erase an embedding contract. |
| Frontend | [`server::serve`](../../crates/hades-frontend/src/server.rs) discovers databases once or uses explicit `--db`, refuses an empty list, and serves graph read routes. Guided-reading writes use a configured database/socket. | Add authenticated setup and an empty-installation screen, deliberate allowlist refresh, and clear behavior for the separately configured reading database. Creation alone must not claim chat works in the new database. |

### Ingestion and capability enforcement

[`ingest_routing::route_for`](../../crates/hades-core/src/ingest_routing.rs) classifies
files by extension. Unified `ingest::run_unified` schedules code first when code
files are discovered, then documents. There is no database-level documents-only
gate. Direct `codebase ingest` provides another entry point that a UI-only gate
would miss. Unsupported extensions are reported; the explicit `--unparsed-ext`
path permits parser-free text ingestion.

Document extraction uses [`ExtractionClient`](../../crates/hades-core/src/persephone/extraction.rs)
and the Python extraction services; [`Pipeline::process_inner` / `chunk_embed_store`](../../crates/hades-core/src/pipeline/orchestrator.rs)
then chunks extracted `full_text`, embeds it, and prepares storage. Document
`TokenChunking` uses whitespace spans, not the model tokenizer. The code path
uses analyzer/AST boundaries and the `code` task. It can proceed without an
embedder when client construction fails; this is not a blanket fallback for all
HTTP or inference failures. Profile-bound embedding must not silently enter this
mode or replace prior vectors with an unembedded result.

`CollectionProfile` in [`db/collections.rs`](../../crates/hades-core/src/db/collections.rs)
currently combines physical collections and Jina task labels:

| Corpus | Corpus task | Query task |
|---|---|---|
| Documents | `retrieval.passage` | `retrieval.query` |
| Codebase, including parser-free files | `code` | `code` |

Retain the collection mapping, but move task semantics into the database's
validated workload configuration. These two Jina task pairs must remain separate
retrieval spaces even though their dimensions match. The current query handler
selects one collection profile; it is not a calibrated code-plus-document search.

Image/PDF extraction into text is not visual embedding. The current Python HTTP
request schema explicitly refuses image embedding, despite the underlying Jina
architecture's broader capabilities. The first profile must advertise text
embedding only; visual-document search remains unavailable until implemented and
validated through the entire serving path.

### Context windows, token limits, and storage

The merged #223 behavior is implemented in
[`PrechunkConfig::from_backend` / `resolve`](../../crates/hades-core/src/chunking/prechunk.rs)
and [`chunking::windows`](../../crates/hades-core/src/chunking/windows.rs):

- Read the running service's `max_seq_length`; missing metadata fails rather
  than guessing a ceiling. Subtract the safety margin and apply a configured cap.
- Convert the token budget to an initial byte estimate at two bytes per token.
  Whitespace chunk sizes and these byte windows are not tokenizer-exact budgets.
- For late chunking, spread full byte windows across the covered source span and
  select the nearest containing window for each passage. Split oversized passages
  into adjacent fragments, preserving parent and file-order indices.
- Refine the plan after a verified size refusal; discard provisional vectors
  before retry. Invalid vectors/model drift do not trigger a different mode.
- Persist the window plan on document/file metadata and selected-window
  provenance on embedding rows. Existing sources require explicit re-ingestion.

Documents default to independent chunk embedding, with an installation-wide
`document_late_chunking` opt-in. Parsed code defaults to late chunking; parser-free
code uses independent chunks. `HADES_DISABLE_LATE_CHUNKING` is a process-wide
escape hatch. These controls must become explicit database/workload choices.
See [the existing window guide](../long-document-embedding.md).

The Python backend counts ordinary inputs with its tokenizer and special tokens,
reserving eight additional tokens for the task prompt; that reserve is a margin,
not a measured count of each rendered prompt. Late chunking tokenizes prefixed
text and checks offset coverage for truncation. Neither makes the Rust planner
token-exact. The new provider interface needs bounded
token counting/offset mapping, including prompts and special tokens, for both
conventional chunks and context windows. Visual token accounting belongs to a
future visual adapter. Keep retrieval-passage size, passage overlap, encoding
window size, and window overlap as separate settings. No 32K or approximately
8K preset is a promise that a particular device can serve it.

### Retrieval and indexing

`dispatch::handlers::db_query` embeds with the selected collection's task and
performs a bounded exact scan. [`retrieval::TopK`](../../crates/hades-core/src/retrieval.rs)
rejects rows with missing/incompatible model strings or dimensions and malformed
vectors. This is a useful existing guard, but same-name changed weights and
changed prompts/pooling have no artifact/behavior identity to compare.

[`db::index::create_vector_index`](../../crates/hades-core/src/db/index.rs) accepts
dimensions and metric; index parameters must be derived from the profile.
Creating an ANN index is not a prerequisite to the current exact search, and
index training/population requirements must not block an empty database from
being ready for ingestion. Preserve bounded response/scan limits when supporting
new dimensions (`TopK` currently caps vectors at 8,192 elements).

Structural embeddings are a separate layer. `schema_apply` defaults
`feature_dim` to 2,048; [`graph::loader`](../../crates/hades-core/src/graph/loader.rs)
uses schema feature width and model-name/hash checks when loading features.
Do not pad, truncate, or average incompatible text/code spaces to satisfy that
width. A smaller-model profile needs matching feature schema and explicit
structural-training compatibility, or that feature must remain disabled.

## Proposed data contracts

Use a dedicated, versioned workspace metadata collection, proposed
`hades_workspace`, rather than mixing the profile into domain graph definitions.
One `embedding_profile` record is authoritative per database. Retain immutable
profile revisions and generation manifests for provenance; update the active
pointer with an ArangoDB revision precondition. Names below are proposed, not
existing API or schema fields.

An abbreviated example (digest values are placeholders, not a usable profile):

```yaml
format: hades-embedding-profile-v1
revision: 1
state: ready
generation: initial
model:
  id: jinaai/jina-embeddings-v4
  revision: <resolved-model-revision>
  artifact:
    kind: bundle
    manifest_format: hades-model-manifest-v1
    manifest_md5: <32-lowercase-hex-digits>
  tokenizer_identity: <manifest-tokenizer-entry-set>
representation:
  dimension: 2048
  output: dense-f32
  normalize: l2
  pooling: masked-mean
  distance: cosine
  inference_precision: fp16
  implementation: <adapter-contract-version>
capabilities:
  text: true
  image_embedding: false
  specialized_code: true
  late_chunking: true
  tokenizer_offsets: true
workloads:
  documents:
    enabled: true
    passage_task: retrieval.passage
    query_task: retrieval.query
    space_id: <artifact-plus-document-behavior-digest>
  code:
    enabled: false
    passage_task: code
    query_task: code
    space_id: <artifact-plus-code-behavior-digest>
chunking:
  documents:
    mode: conventional
    passage_tokens: 500
    passage_overlap_tokens: 200
    tokenizer_contract: <adapter-tokenizer-contract-version>
  code:
    mode: late
    passage_strategy: ast
    context_cap_tokens: null
    minimum_window_overlap_tokens: 1000
runtime_binding:
  provider_id: local-jina
```

Persist the complete manifest as a separately bounded record; profiles reference
it by checksum. Resolve actual prompt templates, task adapters, special-token
policy, pooling, representation, precision, and implementation contract into a
canonical **space identity**. An asymmetric query/passage pair is one declared
compatible space; unrelated task adapters are not. Store profile revision,
generation, artifact identity, space identity, and chunk/window provenance on
new vectors and job receipts. Unknown fields/versions and invalid ranges fail
validation rather than falling back to defaults.

The host provider registry maps `provider_id` to adapter, endpoint, credential
reference, device, batch/concurrency limits, and an observed serving-generation
identifier. No credentials or host paths belong in the portable model identity.
Changing device alone is operational; changing inference precision or an
adapter's numerical behavior is not automatically compatible. A lower serving
ceiling can change window context: record it per ingest, never silently re-embed
existing sources when a GPU profile changes.

Different databases need not keep different models resident simultaneously.
The initial registry can route to separately managed providers and report an
unloaded model as unavailable. Loading/switching a shared provider must serialize
against active work and issue a new serving generation; it must not change a
database's saved pairing. Automatic model scheduling and memory arbitration are
separate from choosing the model during setup.

### Deterministic artifact MD5

For a single-file artifact, record the file MD5. Include separate tokenizer,
configuration, or adapter assets in a bundle whenever inference needs them;
calling the weights a single file must not omit those dependencies.

For a bundle, v1 proposes a canonical UTF-8 JSON manifest with exactly
`format` and `files`; each entry contains `path`, `bytes`, and `md5`. Sort entries
by the UTF-8 bytes of their relative POSIX paths and serialize object keys in
sorted order with compact separators, no BOM, and no trailing newline. File
sizes are nonnegative base-10 integers, with no leading zeros except `0`; digests
are lowercase hex. JSON strings preserve their Unicode scalar sequence: apply
no NFC/NFD normalization or case folding. Reject invalid UTF-8 and lone
surrogates. Thus precomposed and decomposed filenames are distinct paths and
produce different manifests; do not silently rename or merge them.

Emit non-ASCII characters directly as UTF-8, including U+2028/U+2029; do not use
ASCII-only `\u` escaping. Escape quotation marks as `\"` and backslashes as
`\\`; do not escape `/`. Use `\b`, `\t`, `\n`, `\f`, and `\r` for their
respective control characters, and lowercase `\u00xx` for other U+0000–U+001F
characters. Path validation still rejects NUL and invalid relative paths.
These rules determine the bytes independently of a JSON library's defaults.
Hash those exact manifest
bytes with MD5, excluding the aggregate digest itself. Store that digest beside
the manifest, not inside its own hash input.

The [shared manifest test vectors](database-embedding-manifest-vectors.json)
include synthetic source bytes, the canonical JSON string, exact UTF-8 bytes
in hexadecimal, and expected MD5. They cover non-ASCII paths, quote/backslash
escaping, decomposed Unicode, and file ordering by UTF-8 bytes rather than
UTF-16 code units. Implementations must reproduce every vector before they
publish or verify a model identity.

Include all selected weight shards and their index, model/processor/tokenizer
configuration and vocabulary, selected adapters, and model-local inference code
loaded through `trust_remote_code`. Pin external runtime/adapter versions in the
behavior contract. Exclude unrelated cache metadata, download locks, logs, and
absolute paths. Reject duplicate or ambiguous paths, traversal, and undeclared
required assets. Resolve cache symlinks into the verified artifact snapshot;
hash their file contents under stable logical paths rather than their host-local
link destinations. A changed required file must change the recorded identity.

Initial registration is a trusted administrative operation: an authorized local
operator selects an approved artifact source and resolved revision; the HADES
registrar computes the manifest from that local snapshot and persists its
expected digest in the protected provider registry. Authorized database setup
copies that registered identity into the database profile. Neither a browser's
arbitrary digest nor a provider's self-reported digest establishes initial trust.
Changing the expected identity requires explicit authorized registration and,
for an existing database, the migration rules below.

This MD5 detects accidental drift after trusted provisioning; it is not an
authorization mechanism or authentication of an untrusted artifact source.
If untrusted initial sources are supported later, require authenticated artifact
provenance before registration; substituting SHA-256 alone does not establish
that trust. Automatic artifact acquisition is outside this initial feature.
Do not relabel the existing SHA-256-of-name `model_hash` as an artifact checksum.

### Verify what the provider loaded

The Jina adapter should load from a verified immutable artifact snapshot and
issue a new serving-generation identifier on each load/reload. Metadata and each
embedding response must identify the artifact manifest and behavior/space used.
Requests carry the expected identity; a mismatching provider rejects them before
inference. HADES independently checks the response before accepting its vectors.

Compute checksums on snapshot preparation/load, not on every query. A cache entry
must bind to that snapshot, adapter contract, and serving generation; a matching
model label or timestamp is insufficient. The implementation must prevent files
changing underneath loading, or fail verification. On reload, changed artifact,
or changed behavior, invalidate verification. Never publish a receipt for files
different from those actually loaded.

This is a HADES adapter extension, not a promise made by an arbitrary
OpenAI-shaped `/models` response. A provider without verifiable artifacts remains
unverified and cannot activate a new database pairing. A future external adapter
must define how it establishes the same contract; discovery alone is insufficient.
The first release can require the local HADES adapter. Unknown/unavailable
capability metadata does not mean capability support.

## Setup experience and provisioning

1. Choose a database name and purpose: documents, code, or both.
2. Select a registered model/provider. Show identifier, resolved revision,
   verification state, supported workflows, and available execution device.
3. Choose a preset, then advanced controls if needed. **Full** starts with Jina
   and supported late chunking; **Light** uses conventional chunking with a
   future selected smaller model. Both remain capability- and device-dependent.
4. Validate workload tasks, model identity, dimensions, tokenizer budgets, and
   runtime availability. CPU is offered only for an adapter/runtime with tested
   support; it is not inferred from a generic device dropdown.
5. Review enabled features, disabled features, chunk/context sizes, and measured
   resource information where available. Unknown cost stays unknown.
6. Create and initialize the database, save the profile, verify readiness, and
   open the workspace. Model downloads and automatic corpus ingestion are not
   part of this action.

Propose a shared `workspace_setup` operation in `hades-core`, exposed through
structured CLI JSON and scoped service dispatch. The frontend stays independent
of other HADES crates and consumes the CLI contract. Its setup writes need the
existing authentication, Host/Origin checks, bounded request handling, and an
explicit provisioning scope; do not forward arbitrary commands, endpoints,
schema paths, or secrets from browser input. Register providers through an
administrative configuration path. #191 can later supply principal attribution
without changing the model contract.

The viewer must support zero selectable databases in setup mode and add newly
created databases only after authorization and readiness checks. An explicit
startup allowlist is not permission to expose every accessible database. The
reading/chat daemon currently targets one database: show that limitation and do
not silently retarget it as a side effect of workspace creation.

### Failure and retry boundary

Creation spans database creation, collection/schema setup, and profile writes;
these are not one atomic ArangoDB transaction. Use a durable provisioning receipt
with an operation ID and exact request digest, recorded before creation in a
server-owned local state ledger that works before a first workspace exists.
Scope receipt ownership to the authorized caller/installation; serialize work
for a database name. Include this ledger in backup/recovery planning.

Track `validating -> creating -> initializing -> ready`, with explicit `failed`
and `needs_reconciliation` outcomes. After creation, write a matching receipt
marker and initializing profile inside the new database. Retries with identical
requests resume only demonstrably owned steps; a changed request under the same
operation ID is refused. Existing unrelated databases are never adopted or reset.

If the database-create response is lost before the marker is written, the name's
existence alone does not establish ownership. Report the ambiguity for explicit
reconciliation rather than taking over or deleting the database. Recheck all
required schema/profile invariants before marking ready. Failed initialization
retains diagnostics and does not appear as a usable workspace. Recovery and
concurrent creation require disposable-database tests before the UI ships.

## Enforcement and capability matrix

| Operation or capability | Required condition | Enforcement point / result |
|---|---|---|
| Mixed-tree ingestion | Ready profile; enabled content families | Shared admission before code analyzers, extraction, or vector writes; disabled files reported as `disabled_by_profile`, not silently dropped. |
| Direct code ingestion | Code enabled and a validated code-retrieval task pair | Same guard for CLI and job worker; no analyzer startup or code collection writes when disabled. |
| Documents containing code examples | Documents enabled | Treat as document text; do not schedule code analysis merely because prose contains code. |
| Unsupported source extension | Analyzer or explicit parser-free policy available | Preserve explicit unrouted reporting; no automatic extension broadening. |
| Conventional embedding | Verified text adapter and tokenizer budget | Split before embedding; no late-chunk fields sent. Explicit overflow errors, never silent truncation. |
| Late chunking | Capability supported by model and adapter, enabled for that workload | Require the validated boundary/offset contract; no silent conventional fallback. |
| Visual embedding | Implemented image input, visual token budget, compatible retrieval space | Unavailable in the current Jina HTTP path; OCR/text extraction does not satisfy this condition. |
| Semantic search | Ready active generation, verified provider, matching workload space | Check before query embedding and validate stored generation/identity before scoring. |
| Structural features | Compatible feature width, space, schema, and training artifact | Reject/disable incompatible structural operations rather than filling them with misleading vectors. |
| Raw graph/source reads and guided notes | Existing access rules | Remain usable when the embedder is offline; model unavailability must not hide source material. |

Resolve profiles in shared core operations, not only `service::handle_request`:
direct CLI ingestion bypasses that transport boundary. Reuse service policy for
authorization, then resolve database policy before worker admission. Raw
`embed text` without a database can remain an explicitly selected provider
diagnostic; it must not silently ignore its configured model, and its output
does not authorize storage into a bound database.

Jobs pin the profile revision, artifact, space, and corpus generation. Revalidate
the active generation before committing prepared work; a profile change during
inference must not commit into the new generation. Search pins one active
generation for the request. Cache by database identity plus profile revision,
provider generation, and space, not endpoint/model name alone. Unknown profile
formats, missing identities, and provider mismatch fail closed for embedding
operations while leaving non-vector reads available.

## Existing databases and compatibility changes

Do not stamp today's installed model onto existing vectors. Historical model
names, dimensions, and name hashes do not establish the model's artifact MD5.
Inventory stored identities and chunk provenance first; mark unproven databases
as `legacy_unverified`. Preserve graph/source access, but require an explicit
verified adoption or rebuild before claiming profile-bound semantic retrieval.
This changes legacy behavior and needs clear release/migration instructions.

Adoption without re-embedding requires independent provenance establishing the
original artifacts and behavior, and a complete scan for inconsistent or missing
metadata. Sampling is insufficient. If that evidence is absent, rebuild vectors
from preserved sources under the selected verified profile; do not invent old
window plans or rewrite historical exchange snapshots.

| Change | Required treatment |
|---|---|
| Endpoint/device/batching, identical artifacts and embedding behavior | Reverify provider binding; no automatic vector rewrite. Record actual runtime/window provenance. |
| Weights, tokenizer, adapters, prompts, pooling, representation, or incompatible implementation behavior | New embedding space/generation and explicit re-embedding. Equal dimensions do not avoid this. |
| Chunk boundaries, chunking mode, or desired window context | Explicit re-ingestion/re-embedding for the affected corpus; preserve old source anchors and provenance. |
| Index implementation/parameters only | Rebuild the index as needed against the existing compatible vectors; no model change implied. |
| Disabling a content family | Stop new scheduling; retain existing records. Removal is a separate explicit action. |
| Enabling a content family | Validate its task mapping/schema first; no implicit historical ingestion. |

Prepare replacement generations separately, validate complete coverage and query
compatibility, then atomically change the active generation pointer. Pause or
reject conflicting jobs while switching. Keep rollback metadata and old data
until explicit cleanup. Existing fixed chunk keys and collection assumptions
need generation-aware storage/routing before in-place model migration is offered;
`ingest --force` across a live corpus is not an atomic model migration.

Preserve document/file identities, source assets, authored nodes, relationships,
reading highlights, and exact historical references. A staging-database approach
must preserve graph endpoints and references when copying; a vector-generation
approach must isolate chunk keys as well as vectors. The initial implementation
may support only new databases and a documented offline rebuild, with automated
model switching unavailable until this migration contract is implemented.

Future cross-database search must embed each query in each database's compatible
space and keep provenance. Raw cosine scores across models, task spaces, or
different scoring methods are not directly comparable; rank fusion/calibration
requires evaluation. Federation is outside this feature.

## Proposed PR sequence for #225

Implementation begins with the [portable identity contract](embedding-identity-contract.md).
That first draft splits the schema/manifest/receipt foundation out of stage 1
for review; verified serving, database persistence, and runtime enforcement are
still required before the stage is complete or pairing can be enabled.
The following slice adds [local registration and checked Jina loading](registered-model-loading.md);
full provider receipts and database enforcement are still pending. The next
[bound-client slice](bound-embedding-client.md) implements the explicit Rust
factory and client-side receipt checks using a synthetic provider; it does not
enable database pairing or complete stage 1. The [verified Jina provider](verified-jina-provider.md)
adds the producing side for an explicit conventional retrieval contract; real-model
qualification, other capabilities, persistence, and public entry-point enforcement
are still required.

The [persistence slice](persisted-embedding-profiles.md) adds internal new-database
profile/manifest storage, immutable revisions, an atomic readiness pointer, and
a transaction-scoped prepared-write guard. It has no production callers.
The [document integration](profile-bound-documents.md) connects those primitives
for conventional text, atomic publication, and guarded exact retrieval.
Disposable databases and private synthetic providers cover the end-to-end flow,
including different dimensions and provider reloads.
[Token-aware planning](profile-token-planning.md) now uses the verified serving
tokenizer, retains counted source spans, and removes the caller-supplied-span path.
Provisioning ownership/recovery, schema/provider readiness
checks, portable job pins, and public enforcement remain pending.

The [real-Jina qualification protocol](verified-jina-qualification.md) adds an
opt-in fixed retrieval/tokenizer/reload suite and the real provider HTTP-to-Rust
document round trip on a disposable database. It records runtime-specific evidence
without activating public pairing or claiming a general quality benchmark.

[Registered provider discovery](registered-provider-discovery.md) now connects
protected registrations and an explicit installation allowlist to actual HTTP
observations. Document admission resolves its saved provider ID through that
factory and rechecks the database profile after loading. Callers no longer supply
an endpoint or observed receipt to the document path. Recoverable provisioning
and public entry-point enforcement remain pending before frontend activation.

1. **Shared profile and verification contract.** Implement schema/manifest
   canonicalization, provider receipts, an explicit resolved-client factory, and
   Jina verification. Add isolated database creation/profile persistence and one
   document ingest/query path with a synthetic adapter. Production Jina remains
   the first adapter. No enabled public entry point may bypass the binding; keep
   the feature gated until every applicable path enforces it.
2. **Complete policy enforcement.** Wire CLI, MCP, jobs, document/code pipelines,
   task mapping, tokenizer budgets, per-workload chunking, and cache invalidation.
   Add profile/generation metadata to vectors, query guards, and dimension-aware
   schema/index handling. Cover unsupported/disabled capabilities and legacy
   refusal. Publish the new-database and offline rebuild boundaries.
3. **Frontend setup.** Expose the shared setup operation with receipts, failure
   recovery, empty-installation UI, presets, capability explanations, and scoped
   database-list refresh. Browser tests use isolated databases and mock providers.
4. **Smaller model validation.** Select a candidate after measuring requirements;
   implement its adapter/task/tokenizer mapping and demonstrate conventional
   chunking. Do not require late chunking, specialized code, or images from a
   text-only candidate. Document limitations and measured tradeoffs.
5. **Explicit model migration, if required for release.** Implement staged
   generation rebuild/switch/rollback before exposing an existing database's
   model-change action. Otherwise retain the documented offline path and keep
   that action unavailable; track the automated workflow separately.

Per the owner's updated review workflow, open PRs ready for automated review,
monitor feedback and CI, and address findings before the next dependent change.
The repository owner performs merges. This audit PR stops at the plan.
#203's GPU selector repair is independent.
#167's evaluation baseline should be ready before choosing a smaller-model
default; #3 is a candidate investigation, not a required backend.

## Acceptance and evaluation plan

Use service-free fixtures for manifests and wire behavior; use disposable
databases through `with_temp_db` and the isolated runner for all write tests.

| Check | Required evidence |
|---|---|
| Artifact identity | Relocation leaves MD5 unchanged; changed shard/tokenizer/config/adapter changes it; canonical ordering is reproducible; missing/duplicate assets fail. |
| Loaded identity | Provider reload or mid-request change cannot return accepted vectors under the previous identity; files cannot mutate unnoticed during snapshot loading. |
| Database isolation | Alternate and concurrently use two databases with different dimensions/tasks/providers; requests, cached profiles, job snapshots, and stored vectors stay isolated. |
| Disabled stages | Documents-only mixed-tree and direct code requests launch zero code analyzers/workers and make no code writes; document code examples still ingest. |
| Token/context limits | Actual tokenizer counts include overhead; fitting/just-over/multi-window Unicode inputs preserve coverage, offsets, unique chunks, and provenance at multiple ceilings. |
| Feature mismatch | Unsupported late chunking, images, tasks, unavailable verification, and invalid widths fail before vector persistence; no successful fallback receipt. |
| Query compatibility | Wrong artifact or space fails even with the same name/dimension; asymmetric task pairs work; unrelated code/document spaces are never pooled as equivalent. |
| Job races | Profile change or provider reload during preparation cannot commit to the wrong generation; interruption preserves committed data and permits explicit recovery. |
| Setup recovery | Inject failure at each provisioning step, including lost create acknowledgement; retry only owned operations; no unrelated database overwrite or false readiness. |
| Legacy/migration | Unknown provenance stays unknown; rebuild preserves authored/source/reading references; generation switch and rollback are atomic to readers. |
| Frontend boundary | Zero-database setup works; unauthorized provisioning and allowlist expansion fail; new database selection does not falsely claim reading/chat retargeting. |

For conveyance efficiency, evaluate whether useful passages and relationships
remain retrievable at acceptable resource cost. Use named corpus revisions and
judged questions for documents and code separately. Measure source coverage and
retrieval quality, peak memory, ingestion time/processed tokens, and interactive
query latency separately. Report cold load and warm query costs distinctly.
Set comparison criteria before reading results. No measurements in this audit
establish an optimal overlap, an 8K/32K deployment ceiling, or quality equivalence.

## Validation performed for this audit

- Existing `embedding_client` and `embedding_contract` tests: **13 passed**, using
  types/configuration and private mock Unix sockets; no model or database.
- A private CLI mock probe configured `fixture/smaller-model`, dimension 384:
  the current CLI requested `jinaai/jina-embeddings-v4` and accepted 2,048
  dimensions. See the [reproducible probe](../audits/database_embedding_profile_probe.py)
  and [sanitized evidence](../audits/database-embedding-profile-evidence.json).
- Existing chunking tests: **38 passed**; retrieval guards: **3 passed**.
  Commands and documentation checks are recorded in the accompanying evidence
  file. These checks cover current reusable behavior, not the unimplemented
  profile design.

The initial socket test run was denied by the execution sandbox; rerunning with
permission for private temporary sockets passed. No production service was
restarted, model downloaded, corpus re-embedded, or database modified.

## Decisions remaining for review

- Confirm the proposed profile/manifest formats, local provisioning ledger, and
  initial new-database/offline-migration boundary before implementing them.
- Choose the smaller-model candidate and supported content tasks using #167;
  approximately 8K is an example, not a requirement on every candidate.
- Set measured chunk/window presets and CPU support expectations per adapter.
- Decide whether mixed code/document search initially displays separate result
  groups or needs evaluated rank fusion; do not imply a shared embedding space.
- Define acceptable independently recorded provenance for legacy adoption;
  otherwise require rebuilding. MD5 pairing itself is already an owner decision.

Begin with the shared database contract and verified Jina adapter. The frontend
should expose a working, enforced setup operation, and the smaller-model adapter
should demonstrate that capability differences are handled honestly.
