# Profile-aware public search and legacy admission

Status: shared CLI/service/MCP search integration for
[#225](https://github.com/toddwbucy/HADES/issues/225). Database creation in the
frontend remains gated. [Foreground file ingestion](profiled-file-ingestion.md)
now shares the verified document path, as do [background document jobs](profiled-ingestion-jobs.md).
Broader workloads and migration are still pending.

## Database authority

Before semantic search or a legacy vector operation, HADES checks the selected
database's writer endpoint for `hades_workspace` and `hades_setup`. Either
collection reserves the database for profile-aware operations, including an
empty metadata collection or incomplete setup. Only ArangoDB's explicit
collection-not-found response for **both** collections permits the existing
legacy path. Permission failures, a missing database, transport errors, malformed
responses, and a missing/corrupt profile pointer do not select a fallback model.

This is an admission boundary, not protection against administrator DDL. Setup
never adopts existing databases. Database replacement, metadata deletion, and
future migration must quiesce active work. The create-before-marker interval is
another reason to keep public setup and automatic database-list refresh gated.
Raw administrative AQL/CRUD remains administrative access; it is not a verified
vector-ingestion API.

## Explicit provider configuration

The installation supplies a protected registration root and an allowlist of
same-account private Unix providers in its existing `HADES_CONFIG` file:

```yaml
registered_providers:
  registry_root: /absolute/private/model-registry
  providers:
    - id: local-jina
      socket: /absolute/private/provider.sock
      timeout_seconds: 120
      max_response_bytes: 8388608
```

These are operational settings. The database's saved profile chooses the ID,
model MD5, tokenizer, task pair, dimension, and space. The existing
`embedding.service.socket`, model defaults, and `HADES_EMBEDDER_SOCKET` do not
override a bound database. No provider is inferred from a matching model name.
Missing configuration, unknown IDs, incompatible registrations, or an unavailable
provider fail explicitly. Configuration parsing rejects unknown route fields;
catalog construction validates absolute paths, unique IDs, and bounded timeouts
and responses. Registration and socket ownership checks occur on resolution.
See [registered discovery](registered-provider-discovery.md) for those rules.

Routes survive the bounded, sealed configuration handoff together with compact
job/profile pins. Foreground file ingestion and background document workers use
these same explicit routes and guarded document admission.
No installed configuration or service is automatically changed.

## Shared search behavior

`hades --db WORKSPACE db query "question"`, daemon `db.query`, and MCP `db_query`
use the same handler. A managed database resolves a ready saved profile through
its configured registered provider, rechecks the profile after verification,
embeds with the saved query task, and performs the guarded exact scan. There is
no database/provider cache that can leak one database's model into another.
An identical verified reload is compatible with stored vectors; a reload during
a request or profile change during inference invalidates that request.

Only conventional document profiles with cosine distance are currently supported.
Code collection selection, hybrid/structural ranking, and cross-encoder reranking
are rejected. Disabled documents, unsupported chunking, and incomplete schema
fail before inference. Raw source/graph reads and notes remain available without
an embedding provider. Standalone `embed text` remains a legacy provider diagnostic
and its output is not a database-bound storage receipt.

The response preserves the existing query/results envelope and adds
`embedding_profile_state: ready`, `embedding_binding`, and `scanned`. Each hit
includes `parent_key`, `chunk_key`, `text`, `source_id`, `chunk_index`,
`total_chunks`, `span`, and cosine `score`. The binding carries the profile
revision, corpus generation, provider, model, dimension, artifact MD5, space, and
serving observation.

The complete scan validates every stored vector's provenance before scoring.
Selected source text is fetched under the same profile/data transaction, checked
against its source digest and passage span, and limited to 1 MiB of combined
hits. Hits are grouped by parent for detail loading: each source is fetched and
validated once, at most one complete source is retained, and hits keep their
original ranking. Detail reads have a 16 MiB response bound. Existing row, page, query-runtime,
and transaction budgets still apply. Any failed validation or exhausted budget
discards the entire result. The process shares the existing four-search admission
budget; transaction owners retain their reservation through cancellation and
abort. These correctness checks and exclusive locks are not a throughput claim.

## Legacy operations and remaining work

Legacy databases continue to work; semantic responses identify them as
`embedding_profile_state: legacy_unverified`. This does not establish historical
artifact provenance or complete the planned legacy migration policy.

Managed or initializing databases reject these older paths before external
services or writes:

- Legacy core document ingestion, code phases of mixed-tree ingestion and direct
  code ingest. Foreground document files now use the verified profile-aware path.
- Unpinned/legacy background workers. New document jobs carry the verified
  admission through their sealed handoff; unsupported code jobs remain refused.
- Legacy vector-index creation, structural training/loading/update/export and
  structural similarity, and semantic smell reports.

The core legacy document pipeline checks admission too. Source reads, stored
structural-vector inspection, and nonsemantic smell inspection stay available.
None of these refusals removes or edits a saved profile or existing corpus.
Supported code/late-chunking flows,
index derivation, backup/reconciliation tooling, public setup authorization, and
frontend activation remain follow-up work. Do not remove metadata to bypass a
refusal or label historical vectors with today's model.

## Validation

`profiled_search` covers the shared service envelope, protected provider settings,
source text, incomplete/corrupt profiles, unsupported modes, missing routes,
provider reloads, inference/profile races, changed stored data, nonvector reads,
job refusal, and alternating/concurrent databases with different model widths.
The same isolated run executes `profiled_admission`, which runs the actual CLI
for search and ingestion/training refusals. All writes use `with_temp_db`.

```bash
cargo fetch --locked
python3 scripts/test_isolated_database.py --docker --contract profiled_search
```

The documented `--arangod /path/to/private/test/arangod` alternative is supported.
Both targets are in the default CI matrix. Service-free unit tests cover precise
absence/error handling, writer authority, configuration bounds and sealed
handoff, and reservation lifetime through cancelled transaction cleanup.
The [real Jina protocol](verified-jina-qualification.md) now runs its fixed
retrieval cases through the public handler and checks returned source text.
The [ThinkPad result after review](../evidence/jina-public-search-reviewed-thinkpad-2026-10-21.json)
passed all six retrieval/source-text checks with the approved Jina artifacts.
