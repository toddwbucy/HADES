# WeaverTools rebuild rehearsal runbook

**BLOCKED: both permitted ingestion attempts failed (stop condition B).** This is
an observed partial-rebuild runbook, not a successful rebuild or cutover plan.

Date: 2026-09-21, America/Chicago (CDT, UTC−05:00). Started 15:05:42;
the original three-hour deadline was 18:05:42. Current-main code/build revision:
`5ee16d3ce10eb6af4727594babb47efeaf8b49a4`. Source revision:
`f03142a85e1c6bf088b6299063455272df93d68d`, rooted at
`/opt/weavertools/WeaverTools`. Existing clone `/tmp/hades-epic12`, branch
`docs/rebuild-runbook`, draft PR #160. No source/configuration/code changes,
production writes, database drops or cutover were performed. Scratch data is
partial and retained for Todd's review.

## Schema gate and authorized resumption

The initial permitted schema-list and graph-list outputs matched but were
insufficient to establish full `hades_schema` equivalence. Execution stopped
before ingestion. The [independent review seat](https://github.com/toddwbucy/HADES/pull/160#pullrequestreview-5271355507)
then compared all rows using its separately permitted read access: 20 rows on
each side, 9,648 bytes of sorted JSON each, identical after excluding `_id` and
`_rev`. The review specifically confirmed all 18 edge definitions, the named
graph, and every `meta` field including relation order, feature dimension 2048,
model type `hetero_sage`, schema version 1 and checksum. This is attributed
review-seat evidence; the code seat did not run live AQL or expand its allowlist.

The reviewer also confirmed the adapter's default `127.0.0.1:8529` endpoint is
the same user-level ArangoDB instance as the CLI Unix socket. Todd authorized
resumption once GitHub tests passed. All three required gates passed on
`99e8d9b` in [run 35649401299](https://github.com/toddwbucy/HADES/actions/runs/35649401299)
before ingestion began. That CI establishes the documentation checkpoint, not a
successful corpus rebuild. The same branch, PR and database were retained.

## Commands actually executed

The following is a historical sequence, **not an instruction to rerun database
creation**. The scratch database now exists and must be retained for Todd's
review; this rehearsal neither drops nor recreates it.

```bash
cd /tmp/hades-epic12
git fetch origin
git checkout -b docs/rebuild-runbook origin/main
git rev-parse HEAD
cargo build -p hades-cli --bin hades
```

The build reused the existing cargo target cache and completed successfully;
Cargo reported 0.16 seconds. No binary was installed. Each command below ran
from that same checkout. Its elapsed time includes subprocess startup and I/O,
measured with the wall clock; these short schema-only phases say nothing about
the unexecuted ingestion duration.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades --db WeaverTools_v5 db schema list
```

Exit 0; elapsed 0.0128 seconds.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades --db scratch_rebuild_wt5 db create-database scratch_rebuild_wt5
```

Exit 0; elapsed 0.0126 seconds.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades --db scratch_rebuild_wt5 schema apply services/adapters/weavertools/schema.yaml
```

Exit 0; elapsed 0.0385 seconds.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades --db scratch_rebuild_wt5 db schema list
```

Exit 0; elapsed 0.0125 seconds.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades --db WeaverTools_v5 db graph list
```

Exit 0; elapsed 0.0103 seconds.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades --db scratch_rebuild_wt5 db graph list
```

Exit 0; elapsed 0.0098 seconds.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades --db WeaverTools_v5 db collections
```

Exit 0; elapsed 0.0182 seconds.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades --db scratch_rebuild_wt5 db collections
```

Exit 0; elapsed 0.0194 seconds.

Database creation returned `created: true`, HTTP 201. Schema application returned
`applied: true`, 35 collections created, 18 edge definitions registered, and one
named graph created, with zero collections/graphs skipped. The resulting
`hades_schema` contains 20 rows. No force option was used.

## Ingestion commands and measured outcomes

The same command was executed exactly twice, without `--force`, downgrade,
service changes, timeout changes or model/profile changes:

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades ingest /opt/weavertools/WeaverTools --db scratch_rebuild_wt5
```

| Attempt | Start (CDT) | Wall clock | Exit | Envelope success | Result |
| --- | --- | ---: | ---: | --- | --- |
| 1 | 2026-09-21T15:18:47.552214-05:00 | 668.7475 s | 1 | `false` | Rust enrichment incomplete for 160 files |
| 2 | 2026-09-21T15:30:34.507584-05:00 | 4.9374 s | 1 | `false` | Cross-file relationship endpoint no longer exists |

Attempt 1 completed 171 code file writes, 1,976 code embeddings, and all 46
documents with zero reported per-file or embedding failures. However,
rust-analyzer enriched only part of the workspace: 160 files were marked failed,
with warnings `LSP process error: LSP transport is closed`. The resulting
`enrichment_error` made the overall command exit 1. Why the transport closed was
not investigated in this bounded rehearsal; this is not a proven root cause.

Attempt 2 skipped all 171 unchanged code files and all 46 unchanged documents.
It exited 1 with this exact error:

```text
failed to atomically store cross-file relationships; earlier file replacements remain committed: request error: relationship endpoint no longer exists; retry ingestion
```

No third attempt was run. No analysis downgrade or forced rebuild was used to
manufacture success. Successful per-file counters do not override the failed
overall envelope. Earlier writes remain in scratch; the failure does not imply
whole-ingest rollback. The adapter was not run, and no downstream verification
was represented as passing.

What the records rule out: this was not a missing scratch database or schema,
an inability to reach the embedder, a reported vector-write failure, a reported
document extraction failure, or unavailability of rust-analyzer at startup.
They do **not** establish the cause of the closed LSP transport or the missing
relationship endpoint. Those are follow-up findings requiring separate scope;
no implementation repair was attempted here.

## Collection comparison at the stopped checkpoint

Live counts were read before ingestion; scratch counts were read after the
second failure. These sequential reads are not a transactionally frozen live
snapshot or an accepted rebuilt-corpus comparison. The scratch count command:

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /fastpool/venvs/cargo-target/debug/hades --db scratch_rebuild_wt5 db collections
```

Elapsed 0.0192 seconds; exit 0. Every user collection returned by the CLI is included;
system collections are outside this inventory.

| Collection | WeaverTools_v5 | scratch_rebuild_wt5 |
| --- | ---: | ---: |
| `chunks` | 1050 | 1050 |
| `codebase_calls_edges` | 3431 | 153 |
| `codebase_chunks` | 1993 | 1976 |
| `codebase_defines_edges` | 4868 | 2867 |
| `codebase_embeddings` | 1993 | 1976 |
| `codebase_files` | 188 | 171 |
| `codebase_implements_edges` | 68 | 0 |
| `codebase_imports_edges` | 1067 | 1067 |
| `codebase_symbols` | 4868 | 2867 |
| `documents` | 46 | 46 |
| `embeddings` | 1050 | 1050 |
| `hades_schema` | 20 | 20 |
| `wt_artifacts` | 3 | 0 |
| `wt_assertions` | 418 | 0 |
| `wt_asserts_edges` | 423 | 0 |
| `wt_axioms` | 5 | 0 |
| `wt_cites_edges` | 504 | 0 |
| `wt_crates` | 12 | 0 |
| `wt_declared_in_edges` | 498 | 0 |
| `wt_defines_edges` | 46 | 0 |
| `wt_documents` | 13 | 0 |
| `wt_draws_edges` | 59 | 0 |
| `wt_elects_edges` | 2 | 0 |
| `wt_floor_link_edges` | 8 | 0 |
| `wt_grounds_edges` | 87 | 0 |
| `wt_holds_edges` | 8 | 0 |
| `wt_ingest_report` | 1 | 0 |
| `wt_parent_edges` | 12 | 0 |
| `wt_party_edges` | 23 | 0 |
| `wt_reads_edges` | 1 | 0 |
| `wt_seam_edges` | 10 | 0 |
| `wt_systems` | 1 | 0 |
| `wt_terms` | 5 | 0 |
| `wt_vocabulary` | 41 | 0 |
| `wt_writes_edges` | 2 | 0 |

| Other required result | Live | Scratch |
| --- | --- | --- |
| Adapter dangling citations | Unknown: allowed reads do not expose report values | Not measured: adapter not run |
| Unrouted list | Historical live ingest envelope unavailable | 42 paths, recorded below; identical between attempts |
| Drift / validation | Not run (outside live allowlist) | Not run: ingestion failed twice |
| Default/codebase semantic queries | Not run | Not run: ingestion failed twice |
| Every embedding has model/model_hash/dimension | Not checked | Not checked; embedding counts alone are insufficient |

Live has 188 code file rows versus 171 routed code files in this rehearsal;
this difference is recorded without attributing it to an untested cause.
The exact current source revision and root are recorded above. The historical
live source/options/unrouted set were not reconstructed from count differences.

## Unrouted files

Both envelopes contain the same list; paths below are relative to
`/opt/weavertools/WeaverTools/`. No extra extension handler was selected.

| Path | Reason |
| --- | --- |
| `Cargo.lock` | no handler for extension |
| `Cargo.toml` | no handler for extension |
| `LICENSE` | no extension and no shebang |
| `crates/weaver-admin/Cargo.toml` | no handler for extension |
| `crates/weaver-analysis/Cargo.toml` | no handler for extension |
| `crates/weaver-diagnostic/Cargo.toml` | no handler for extension |
| `crates/weaver-gate/Cargo.toml` | no handler for extension |
| `crates/weaver-harness/Cargo.toml` | no handler for extension |
| `crates/weaver-internal/Cargo.toml` | no handler for extension |
| `crates/weaver-spu/Cargo.toml` | no handler for extension |
| `crates/weaver-state/Cargo.toml` | no handler for extension |
| `crates/weaver-trace/Cargo.toml` | no handler for extension |
| `crates/weaver-traits/Cargo.toml` | no handler for extension |
| `crates/weaver-types/Cargo.toml` | no handler for extension |
| `crates/weaver-web/Cargo.toml` | no handler for extension |
| `crates/weaver-web/LICENSE` | no extension and no shebang |
| `crates/weaver-web/askama.toml` | no handler for extension |
| `crates/weaver-web/assets/htmx.min.js` | no handler for extension |
| `crates/weaver-web/assets/sse.js` | no handler for extension |
| `crates/weaver-web/assets/style.css` | no handler for extension |
| `crates/weaver-web/assets/surfaces/instrument.css` | no handler for extension |
| `crates/weaver-web/deploy/config.example.toml` | no handler for extension |
| `crates/weaver-web/deploy/connector.example.toml` | no handler for extension |
| `crates/weaver-web/deploy/weaver-web.sudoers` | no handler for extension |
| `crates/weaver-web/src/surfaces/templates/instrument.html` | no handler for extension |
| `crates/weaver-web/src/surfaces/templates/record.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/agent_config.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/base.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/channel.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/channels.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/event.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/lifecycle.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/name.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/sidebar.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/trace.html` | no handler for extension |
| `crates/weaver-web/src/web/templates/trace_event.html` | no handler for extension |
| `deploy/create-agent.sh` | no handler for extension |
| `deploy/update-stack.sh` | no handler for extension |
| `docs/crates/weaver-harness/Loops/basic-inference-loop.png` | no handler for extension |
| `process/gates/census-baseline.json` | no handler for extension |
| `process/gates/lock.sh` | no handler for extension |
| `rust-toolchain.toml` | no handler for extension |

## Remaining steps — NOT EXECUTED

Do not continue this partial database under this rehearsal authorization. Todd
must decide follow-up scope for the two observed ingestion failures. No schema
read extension is needed to clear the already-cleared schema gate.

The adapter dry run, adapter write, clean drift, validation, both semantic query
profiles, exhaustive embedding metadata check and final comparisons remain
unexecuted. If separately authorized after ingestion is made complete, the
requested adapter commands are:

```bash
python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5 --repo /opt/weavertools/WeaverTools --dry-run
python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5 --repo /opt/weavertools/WeaverTools
```

These commands are **not proven by this runbook**. Adapter dangling-citation
comparison additionally needs an owner/reviewer-provided live report or explicit
permission for the read-only `db get wt_ingest_report latest` command. The live
historical unrouted comparison needs the original ingest envelope; do not infer
it from the current source tree or collection counts. These gaps remain separate
from the observed ingestion failures.

## Cutover and rollback — DOCUMENTATION ONLY, NOT EXECUTED

This section follows [Code-file identities and migration](code-file-identities.md).
It does not authorize production commands, service operations, snapshots, or
configuration changes. The schema-only scratch database is **not** a cutover target.

1. Before scheduling production, inventory canonical roots, source revisions,
   analyzer options, schema, adapter configuration, model identity, external key
   consumers, current binary and database/configuration selections. Verify backup
   coverage and a restore in a separate environment. Todd schedules a snapshot
   before any operation that risks data loss; no ZFS command was run here.
2. Rebuild into fresh approved databases using stable roots and the corrected
   binary. Rebuild domain edges and external references from source identities;
   do not copy ambiguous old keys or relabel vectors. Compare identities,
   inventories, endpoints, drift and representative retrieval; review unresolved
   mappings and checkpoint compatibility before acceptance.
3. In an authorized maintenance window, pause all writers, capture/replay source
   changes and repeat verification. Preserve the previous database, binary,
   configuration and checkpoints. Confirm the snapshot/rollback baseline before
   switching anything.
4. Switch the database selection and compatible binary together. Update
   `HADES_DATABASE` in `~/.config/hades/daemon.env` and the active MCP database
   configuration. The actual MCP configuration location and activation procedure
   were not inspected and must be identified before this plan is executable.
   Any reload/restart needs separate authorization. Verify CLI and MCP database
   identity, both retrieval profiles, integrity, and writer behavior before
   restoring normal operation.
5. If verification fails, pause writers and restore both the previous binary and
   previous database selections in daemon and MCP configuration. Reconcile writes
   made after cutover before rollback to avoid silent loss; restore compatible
   checkpoints as needed. Reverify service behavior. Keep old databases and
   snapshots until Todd completes retention and acceptance review.

No cutover, rollback, snapshot, restart, profile switch, forced ingestion, or
production write occurred. Exactly one scratch database was created and is
retained with partial ingestion data. Both ingestion processes exited; no
adapter process was started. A documentation push may launch CI but does not
authorize resumption. No code repair was made or next task started.

## Exact ingestion envelopes

The following are the complete stdout envelopes as emitted, including per-file
results and the unrouted lists. The failure messages are preserved rather than
rewritten as an assumed root cause. Local stdout/stderr and timing captures are
retained under `/tmp/hades-rebuild-rehearsal/` for this stopped run.

<details>
<summary>Attempt 1: complete stdout (SHA-256 51147b92c2ca562074115d3eed912e1d39564226059b546ce43ead965cdf62df)</summary>

```json
{
  "command": "ingest",
  "data": {
    "code": {
      "completed": 171,
      "cpp_call_edges": 0,
      "dangling_inbound_edges": 0,
      "duration_ms": 340552,
      "embedding": {
        "embedding_failure_paths": [],
        "files_embedded": 171,
        "files_with_embedding_failures": 0,
        "service_connected": true,
        "total_embeddings": 1976
      },
      "enrichment_error": "rust-analyzer enrichment incomplete for rewritten files: crates/weaver-admin/src/inventory.rs, crates/weaver-admin/src/log.rs, crates/weaver-admin/src/main.rs, crates/weaver-admin/src/sink.rs, crates/weaver-admin/src/surface.rs, crates/weaver-admin/src/unit.rs, crates/weaver-admin/src/verbs.rs, crates/weaver-admin/tests/invocation.rs, crates/weaver-admin/tests/manifest.rs, crates/weaver-analysis/src/capture.rs, crates/weaver-analysis/src/declare.rs, crates/weaver-analysis/src/field.rs, crates/weaver-analysis/src/lens.rs, crates/weaver-analysis/src/lib.rs, crates/weaver-analysis/src/main.rs, crates/weaver-analysis/src/preload.rs, crates/weaver-analysis/src/project.rs, crates/weaver-analysis/src/reading.rs, crates/weaver-analysis/src/record.rs, crates/weaver-analysis/src/signals.rs, crates/weaver-analysis/src/stream.rs, crates/weaver-analysis/tests/driver.rs, crates/weaver-analysis/tests/field.rs, crates/weaver-analysis/tests/lens.rs, crates/weaver-analysis/tests/manifest.rs, crates/weaver-analysis/tests/reading.rs, crates/weaver-analysis/tests/stream.rs, crates/weaver-diagnostic/src/event.rs, crates/weaver-diagnostic/src/failure.rs, crates/weaver-diagnostic/src/lib.rs, crates/weaver-diagnostic/src/recorder.rs, crates/weaver-diagnostic/tests/manifest.rs, crates/weaver-diagnostic/tests/recorder.rs, crates/weaver-gate/src/channel.rs, crates/weaver-gate/src/hook.rs, crates/weaver-gate/src/lib.rs, crates/weaver-gate/src/main.rs, crates/weaver-gate/src/relay.rs, crates/weaver-gate/src/tools.rs, crates/weaver-gate/tests/boundary.rs, crates/weaver-gate/tests/common/mod.rs, crates/weaver-gate/tests/entry.rs, crates/weaver-gate/tests/manifest.rs, crates/weaver-harness/src/assembly.rs, crates/weaver-harness/src/authorship.rs, crates/weaver-harness/src/bin/pyworker/main.rs, crates/weaver-harness/src/bin/pyworker/py_loop.rs, crates/weaver-harness/src/bin/worker/dev_loop/mod.rs, crates/weaver-harness/src/bin/worker/main.rs, crates/weaver-harness/src/channel.rs, crates/weaver-harness/src/engine.rs, crates/weaver-harness/src/failure.rs, crates/weaver-harness/src/lib.rs, crates/weaver-harness/src/lifecycle.rs, crates/weaver-harness/src/record.rs, crates/weaver-harness/src/replay.rs, crates/weaver-harness/src/spawn.rs, crates/weaver-harness/src/state.rs, crates/weaver-harness/src/tools.rs, crates/weaver-harness/tests/authorship.rs, crates/weaver-harness/tests/channels.rs, crates/weaver-harness/tests/manifest.rs, crates/weaver-harness/tests/service.rs, crates/weaver-internal/src/calculator.rs, crates/weaver-internal/src/lib.rs, crates/weaver-internal/tests/manifest.rs, crates/weaver-spu/build.rs, crates/weaver-spu/src/artifact.rs, crates/weaver-spu/src/bin/classify.rs, crates/weaver-spu/src/channel.rs, crates/weaver-spu/src/decoder/backend.rs, crates/weaver-spu/src/decoder/gguf.rs, crates/weaver-spu/src/decoder/gguf_tap.rs, crates/weaver-spu/src/decoder/mod.rs, crates/weaver-spu/src/decoder/native.rs, crates/weaver-spu/src/decoder/native_pair.rs, crates/weaver-spu/src/decoder/session.rs, crates/weaver-spu/src/family/gemma4.rs, crates/weaver-spu/src/family/gpt_oss.rs, crates/weaver-spu/src/family/llama.rs, crates/weaver-spu/src/family/mistral3.rs, crates/weaver-spu/src/family/mod.rs, crates/weaver-spu/src/family/modernbert.rs, crates/weaver-spu/src/family/phi.rs, crates/weaver-spu/src/family/qwen2.rs, crates/weaver-spu/src/gpu/mod.rs, crates/weaver-spu/src/lib.rs, crates/weaver-spu/src/main.rs, crates/weaver-spu/src/measurement.rs, crates/weaver-spu/src/readout.rs, crates/weaver-spu/src/residency.rs, crates/weaver-spu/src/sampling.rs, crates/weaver-spu/tests/common/mod.rs, crates/weaver-spu/tests/entry.rs, crates/weaver-spu/tests/loaded.rs, crates/weaver-spu/tests/manifest.rs, crates/weaver-spu/tests/markers.rs, crates/weaver-spu/tests/native_loaded.rs, crates/weaver-spu/tests/readout_neutral.rs, crates/weaver-spu/tests/seam.rs, crates/weaver-spu/tests/selection.rs, crates/weaver-spu/tests/two_card.rs, crates/weaver-state/src/engine/mod.rs, crates/weaver-state/src/engine/postgres.rs, crates/weaver-state/src/engine/sqlite.rs, crates/weaver-state/src/lib.rs, crates/weaver-state/src/main.rs, crates/weaver-state/src/store.rs, crates/weaver-trace/src/canonical.rs, crates/weaver-trace/src/event.rs, crates/weaver-trace/src/failure.rs, crates/weaver-trace/src/lib.rs, crates/weaver-trace/src/structure.rs, crates/weaver-trace/src/tee.rs, crates/weaver-trace/src/writer.rs, crates/weaver-trace/tests/kinds.rs, crates/weaver-trace/tests/manifest.rs, crates/weaver-trace/tests/recorder.rs, crates/weaver-traits/src/lib.rs, crates/weaver-traits/src/message.rs, crates/weaver-traits/src/permission.rs, crates/weaver-traits/src/provider.rs, crates/weaver-traits/src/tool.rs, crates/weaver-traits/tests/manifest.rs, crates/weaver-traits/tests/message_model.rs, crates/weaver-types/src/config.rs, crates/weaver-types/src/identity.rs, crates/weaver-types/src/lib.rs, crates/weaver-types/src/wire.rs, crates/weaver-types/tests/config.rs, crates/weaver-types/tests/identity.rs, crates/weaver-types/tests/manifest.rs, crates/weaver-types/tests/wire.rs, crates/weaver-web/src/adapters/gate.rs, crates/weaver-web/src/adapters/mod.rs, crates/weaver-web/src/adapters/upstream.rs, crates/weaver-web/src/bin/weaver-web-connector.rs, crates/weaver-web/src/bin/weaver-web.rs, crates/weaver-web/src/channel.rs, crates/weaver-web/src/config.rs, crates/weaver-web/src/fault.rs, crates/weaver-web/src/lib.rs, crates/weaver-web/src/lifecycle.rs, crates/weaver-web/src/queue.rs, crates/weaver-web/src/registry.rs, crates/weaver-web/src/router.rs, crates/weaver-web/src/store/conversation.rs, crates/weaver-web/src/store/experiment.rs, crates/weaver-web/src/store/key.rs, crates/weaver-web/src/store/mod.rs, crates/weaver-web/src/store/plan.rs, crates/weaver-web/src/store/read.rs, crates/weaver-web/src/surfaces/gate.rs, crates/weaver-web/src/surfaces/mod.rs, crates/weaver-web/src/surfaces/record.rs, crates/weaver-web/src/traceview.rs, crates/weaver-web/src/web/admin.rs, crates/weaver-web/src/web/mod.rs, crates/weaver-web/src/web/user.rs, crates/weaver-web/src/wire.rs; retry ingestion or explicitly allow analysis downgrade",
      "failed": 0,
      "gopls": {
        "edges": 0,
        "failed_files": [],
        "failed_workspaces": [],
        "modules_analyzed": 0,
        "store_errors": 0,
        "store_failed": false,
        "symbols": 0
      },
      "import_edges": 1067,
      "python_call_edges": 153,
      "python_import_edges": 0,
      "relationship_error": null,
      "repointed_inbound_edges": 0,
      "results": [
        {
          "duration_ms": 6334,
          "language": "Rust",
          "num_chunks": 13,
          "num_embeddings": 13,
          "num_symbols": 40,
          "path": "crates/weaver-admin/src/channel.rs",
          "success": true
        },
        {
          "duration_ms": 9709,
          "language": "Rust",
          "num_chunks": 38,
          "num_embeddings": 38,
          "num_symbols": 65,
          "path": "crates/weaver-admin/src/inventory.rs",
          "success": true
        },
        {
          "duration_ms": 333,
          "language": "Rust",
          "num_chunks": 4,
          "num_embeddings": 4,
          "num_symbols": 8,
          "path": "crates/weaver-admin/src/log.rs",
          "success": true
        },
        {
          "duration_ms": 7601,
          "language": "Rust",
          "num_chunks": 36,
          "num_embeddings": 36,
          "num_symbols": 65,
          "path": "crates/weaver-admin/src/main.rs",
          "success": true
        },
        {
          "duration_ms": 1109,
          "language": "Rust",
          "num_chunks": 9,
          "num_embeddings": 9,
          "num_symbols": 18,
          "path": "crates/weaver-admin/src/sink.rs",
          "success": true
        },
        {
          "duration_ms": 654,
          "language": "Rust",
          "num_chunks": 8,
          "num_embeddings": 8,
          "num_symbols": 13,
          "path": "crates/weaver-admin/src/surface.rs",
          "success": true
        },
        {
          "duration_ms": 3000,
          "language": "Rust",
          "num_chunks": 14,
          "num_embeddings": 14,
          "num_symbols": 21,
          "path": "crates/weaver-admin/src/unit.rs",
          "success": true
        },
        {
          "duration_ms": 599,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 9,
          "path": "crates/weaver-admin/src/verbs.rs",
          "success": true
        },
        {
          "duration_ms": 449,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 6,
          "path": "crates/weaver-admin/tests/invocation.rs",
          "success": true
        },
        {
          "duration_ms": 403,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 6,
          "path": "crates/weaver-admin/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 2331,
          "language": "Rust",
          "num_chunks": 15,
          "num_embeddings": 15,
          "num_symbols": 31,
          "path": "crates/weaver-analysis/src/capture.rs",
          "success": true
        },
        {
          "duration_ms": 768,
          "language": "Rust",
          "num_chunks": 7,
          "num_embeddings": 7,
          "num_symbols": 9,
          "path": "crates/weaver-analysis/src/declare.rs",
          "success": true
        },
        {
          "duration_ms": 1031,
          "language": "Rust",
          "num_chunks": 8,
          "num_embeddings": 8,
          "num_symbols": 18,
          "path": "crates/weaver-analysis/src/field.rs",
          "success": true
        },
        {
          "duration_ms": 2643,
          "language": "Rust",
          "num_chunks": 19,
          "num_embeddings": 19,
          "num_symbols": 35,
          "path": "crates/weaver-analysis/src/lens.rs",
          "success": true
        },
        {
          "duration_ms": 287,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 19,
          "path": "crates/weaver-analysis/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 4022,
          "language": "Rust",
          "num_chunks": 17,
          "num_embeddings": 17,
          "num_symbols": 19,
          "path": "crates/weaver-analysis/src/main.rs",
          "success": true
        },
        {
          "duration_ms": 287,
          "language": "Rust",
          "num_chunks": 4,
          "num_embeddings": 4,
          "num_symbols": 7,
          "path": "crates/weaver-analysis/src/preload.rs",
          "success": true
        },
        {
          "duration_ms": 1458,
          "language": "Rust",
          "num_chunks": 12,
          "num_embeddings": 12,
          "num_symbols": 20,
          "path": "crates/weaver-analysis/src/project.rs",
          "success": true
        },
        {
          "duration_ms": 537,
          "language": "Rust",
          "num_chunks": 8,
          "num_embeddings": 8,
          "num_symbols": 8,
          "path": "crates/weaver-analysis/src/reading.rs",
          "success": true
        },
        {
          "duration_ms": 346,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 9,
          "path": "crates/weaver-analysis/src/record.rs",
          "success": true
        },
        {
          "duration_ms": 982,
          "language": "Rust",
          "num_chunks": 8,
          "num_embeddings": 8,
          "num_symbols": 13,
          "path": "crates/weaver-analysis/src/signals.rs",
          "success": true
        },
        {
          "duration_ms": 290,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 7,
          "path": "crates/weaver-analysis/src/stream.rs",
          "success": true
        },
        {
          "duration_ms": 1350,
          "language": "Rust",
          "num_chunks": 12,
          "num_embeddings": 12,
          "num_symbols": 13,
          "path": "crates/weaver-analysis/tests/driver.rs",
          "success": true
        },
        {
          "duration_ms": 2259,
          "language": "Rust",
          "num_chunks": 11,
          "num_embeddings": 11,
          "num_symbols": 13,
          "path": "crates/weaver-analysis/tests/field.rs",
          "success": true
        },
        {
          "duration_ms": 3083,
          "language": "Rust",
          "num_chunks": 19,
          "num_embeddings": 19,
          "num_symbols": 23,
          "path": "crates/weaver-analysis/tests/lens.rs",
          "success": true
        },
        {
          "duration_ms": 191,
          "language": "Rust",
          "num_chunks": 4,
          "num_embeddings": 4,
          "num_symbols": 4,
          "path": "crates/weaver-analysis/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 498,
          "language": "Rust",
          "num_chunks": 7,
          "num_embeddings": 7,
          "num_symbols": 10,
          "path": "crates/weaver-analysis/tests/reading.rs",
          "success": true
        },
        {
          "duration_ms": 2346,
          "language": "Rust",
          "num_chunks": 16,
          "num_embeddings": 16,
          "num_symbols": 17,
          "path": "crates/weaver-analysis/tests/stream.rs",
          "success": true
        },
        {
          "duration_ms": 1042,
          "language": "Rust",
          "num_chunks": 24,
          "num_embeddings": 24,
          "num_symbols": 27,
          "path": "crates/weaver-diagnostic/src/event.rs",
          "success": true
        },
        {
          "duration_ms": 148,
          "language": "Rust",
          "num_chunks": 4,
          "num_embeddings": 4,
          "num_symbols": 4,
          "path": "crates/weaver-diagnostic/src/failure.rs",
          "success": true
        },
        {
          "duration_ms": 263,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 6,
          "path": "crates/weaver-diagnostic/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 696,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 10,
          "path": "crates/weaver-diagnostic/src/recorder.rs",
          "success": true
        },
        {
          "duration_ms": 202,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 5,
          "path": "crates/weaver-diagnostic/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 1062,
          "language": "Rust",
          "num_chunks": 11,
          "num_embeddings": 11,
          "num_symbols": 15,
          "path": "crates/weaver-diagnostic/tests/recorder.rs",
          "success": true
        },
        {
          "duration_ms": 1364,
          "language": "Rust",
          "num_chunks": 9,
          "num_embeddings": 9,
          "num_symbols": 24,
          "path": "crates/weaver-gate/src/channel.rs",
          "success": true
        },
        {
          "duration_ms": 1764,
          "language": "Rust",
          "num_chunks": 16,
          "num_embeddings": 16,
          "num_symbols": 28,
          "path": "crates/weaver-gate/src/hook.rs",
          "success": true
        },
        {
          "duration_ms": 130,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 4,
          "path": "crates/weaver-gate/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 2812,
          "language": "Rust",
          "num_chunks": 15,
          "num_embeddings": 15,
          "num_symbols": 31,
          "path": "crates/weaver-gate/src/main.rs",
          "success": true
        },
        {
          "duration_ms": 2132,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 38,
          "path": "crates/weaver-gate/src/relay.rs",
          "success": true
        },
        {
          "duration_ms": 1822,
          "language": "Rust",
          "num_chunks": 7,
          "num_embeddings": 7,
          "num_symbols": 20,
          "path": "crates/weaver-gate/src/tools.rs",
          "success": true
        },
        {
          "duration_ms": 1262,
          "language": "Rust",
          "num_chunks": 9,
          "num_embeddings": 9,
          "num_symbols": 13,
          "path": "crates/weaver-gate/tests/boundary.rs",
          "success": true
        },
        {
          "duration_ms": 843,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 18,
          "path": "crates/weaver-gate/tests/common/mod.rs",
          "success": true
        },
        {
          "duration_ms": 1629,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 18,
          "path": "crates/weaver-gate/tests/entry.rs",
          "success": true
        },
        {
          "duration_ms": 1386,
          "language": "Rust",
          "num_chunks": 12,
          "num_embeddings": 12,
          "num_symbols": 12,
          "path": "crates/weaver-gate/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 357,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 8,
          "path": "crates/weaver-harness/src/assembly.rs",
          "success": true
        },
        {
          "duration_ms": 1594,
          "language": "Rust",
          "num_chunks": 12,
          "num_embeddings": 12,
          "num_symbols": 25,
          "path": "crates/weaver-harness/src/authorship.rs",
          "success": true
        },
        {
          "duration_ms": 1316,
          "language": "Python",
          "num_chunks": 9,
          "num_embeddings": 9,
          "num_symbols": 17,
          "path": "crates/weaver-harness/src/bin/pyworker/dev_python/alpha_loop.py",
          "success": true
        },
        {
          "duration_ms": 195,
          "language": "Python",
          "num_chunks": 2,
          "num_embeddings": 2,
          "num_symbols": 1,
          "path": "crates/weaver-harness/src/bin/pyworker/dev_python/basic_loop.py",
          "success": true
        },
        {
          "duration_ms": 1314,
          "language": "Python",
          "num_chunks": 9,
          "num_embeddings": 9,
          "num_symbols": 17,
          "path": "crates/weaver-harness/src/bin/pyworker/dev_python/bravo_loop.py",
          "success": true
        },
        {
          "duration_ms": 586,
          "language": "Rust",
          "num_chunks": 2,
          "num_embeddings": 2,
          "num_symbols": 5,
          "path": "crates/weaver-harness/src/bin/pyworker/main.rs",
          "success": true
        },
        {
          "duration_ms": 1381,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 22,
          "path": "crates/weaver-harness/src/bin/pyworker/py_loop.rs",
          "success": true
        },
        {
          "duration_ms": 1956,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 21,
          "path": "crates/weaver-harness/src/bin/worker/dev_loop/mod.rs",
          "success": true
        },
        {
          "duration_ms": 386,
          "language": "Rust",
          "num_chunks": 2,
          "num_embeddings": 2,
          "num_symbols": 4,
          "path": "crates/weaver-harness/src/bin/worker/main.rs",
          "success": true
        },
        {
          "duration_ms": 5142,
          "language": "Rust",
          "num_chunks": 36,
          "num_embeddings": 36,
          "num_symbols": 78,
          "path": "crates/weaver-harness/src/channel.rs",
          "success": true
        },
        {
          "duration_ms": 14359,
          "language": "Rust",
          "num_chunks": 26,
          "num_embeddings": 26,
          "num_symbols": 67,
          "path": "crates/weaver-harness/src/engine.rs",
          "success": true
        },
        {
          "duration_ms": 507,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 13,
          "path": "crates/weaver-harness/src/failure.rs",
          "success": true
        },
        {
          "duration_ms": 449,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 22,
          "path": "crates/weaver-harness/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 18886,
          "language": "Rust",
          "num_chunks": 53,
          "num_embeddings": 53,
          "num_symbols": 84,
          "path": "crates/weaver-harness/src/lifecycle.rs",
          "success": true
        },
        {
          "duration_ms": 834,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 13,
          "path": "crates/weaver-harness/src/record.rs",
          "success": true
        },
        {
          "duration_ms": 3918,
          "language": "Rust",
          "num_chunks": 15,
          "num_embeddings": 15,
          "num_symbols": 45,
          "path": "crates/weaver-harness/src/replay.rs",
          "success": true
        },
        {
          "duration_ms": 2165,
          "language": "Rust",
          "num_chunks": 8,
          "num_embeddings": 8,
          "num_symbols": 20,
          "path": "crates/weaver-harness/src/spawn.rs",
          "success": true
        },
        {
          "duration_ms": 3761,
          "language": "Rust",
          "num_chunks": 18,
          "num_embeddings": 18,
          "num_symbols": 50,
          "path": "crates/weaver-harness/src/state.rs",
          "success": true
        },
        {
          "duration_ms": 501,
          "language": "Rust",
          "num_chunks": 3,
          "num_embeddings": 3,
          "num_symbols": 7,
          "path": "crates/weaver-harness/src/tools.rs",
          "success": true
        },
        {
          "duration_ms": 2401,
          "language": "Rust",
          "num_chunks": 13,
          "num_embeddings": 13,
          "num_symbols": 18,
          "path": "crates/weaver-harness/tests/authorship.rs",
          "success": true
        },
        {
          "duration_ms": 1731,
          "language": "Rust",
          "num_chunks": 12,
          "num_embeddings": 12,
          "num_symbols": 13,
          "path": "crates/weaver-harness/tests/channels.rs",
          "success": true
        },
        {
          "duration_ms": 416,
          "language": "Rust",
          "num_chunks": 7,
          "num_embeddings": 7,
          "num_symbols": 7,
          "path": "crates/weaver-harness/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 2901,
          "language": "Rust",
          "num_chunks": 18,
          "num_embeddings": 18,
          "num_symbols": 29,
          "path": "crates/weaver-harness/tests/service.rs",
          "success": true
        },
        {
          "duration_ms": 1146,
          "language": "Rust",
          "num_chunks": 8,
          "num_embeddings": 8,
          "num_symbols": 12,
          "path": "crates/weaver-internal/src/calculator.rs",
          "success": true
        },
        {
          "duration_ms": 136,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 1,
          "path": "crates/weaver-internal/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 266,
          "language": "Rust",
          "num_chunks": 3,
          "num_embeddings": 3,
          "num_symbols": 3,
          "path": "crates/weaver-internal/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 278,
          "language": "Rust",
          "num_chunks": 2,
          "num_embeddings": 2,
          "num_symbols": 2,
          "path": "crates/weaver-spu/build.rs",
          "success": true
        },
        {
          "duration_ms": 7069,
          "language": "C++",
          "num_chunks": 93,
          "num_embeddings": 93,
          "num_symbols": 48,
          "path": "crates/weaver-spu/kernels/transformer.cu",
          "success": true
        },
        {
          "duration_ms": 4840,
          "language": "Rust",
          "num_chunks": 31,
          "num_embeddings": 31,
          "num_symbols": 52,
          "path": "crates/weaver-spu/src/artifact.rs",
          "success": true
        },
        {
          "duration_ms": 632,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 9,
          "path": "crates/weaver-spu/src/bin/classify.rs",
          "success": true
        },
        {
          "duration_ms": 2888,
          "language": "Rust",
          "num_chunks": 24,
          "num_embeddings": 24,
          "num_symbols": 54,
          "path": "crates/weaver-spu/src/channel.rs",
          "success": true
        },
        {
          "duration_ms": 1024,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 6,
          "path": "crates/weaver-spu/src/decoder/backend.rs",
          "success": true
        },
        {
          "duration_ms": 5524,
          "language": "Rust",
          "num_chunks": 14,
          "num_embeddings": 14,
          "num_symbols": 61,
          "path": "crates/weaver-spu/src/decoder/gguf.rs",
          "success": true
        },
        {
          "duration_ms": 2231,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 24,
          "path": "crates/weaver-spu/src/decoder/gguf_tap.rs",
          "success": true
        },
        {
          "duration_ms": 197,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 6,
          "path": "crates/weaver-spu/src/decoder/mod.rs",
          "success": true
        },
        {
          "duration_ms": 3463,
          "language": "Rust",
          "num_chunks": 16,
          "num_embeddings": 16,
          "num_symbols": 45,
          "path": "crates/weaver-spu/src/decoder/native.rs",
          "success": true
        },
        {
          "duration_ms": 3947,
          "language": "Rust",
          "num_chunks": 13,
          "num_embeddings": 13,
          "num_symbols": 22,
          "path": "crates/weaver-spu/src/decoder/native_pair.rs",
          "success": true
        },
        {
          "duration_ms": 9001,
          "language": "Rust",
          "num_chunks": 22,
          "num_embeddings": 22,
          "num_symbols": 72,
          "path": "crates/weaver-spu/src/decoder/session.rs",
          "success": true
        },
        {
          "duration_ms": 1179,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 29,
          "path": "crates/weaver-spu/src/family/gemma4.rs",
          "success": true
        },
        {
          "duration_ms": 475,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 25,
          "path": "crates/weaver-spu/src/family/gpt_oss.rs",
          "success": true
        },
        {
          "duration_ms": 479,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 24,
          "path": "crates/weaver-spu/src/family/llama.rs",
          "success": true
        },
        {
          "duration_ms": 1211,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 27,
          "path": "crates/weaver-spu/src/family/mistral3.rs",
          "success": true
        },
        {
          "duration_ms": 11955,
          "language": "Rust",
          "num_chunks": 46,
          "num_embeddings": 46,
          "num_symbols": 87,
          "path": "crates/weaver-spu/src/family/mod.rs",
          "success": true
        },
        {
          "duration_ms": 1029,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 18,
          "path": "crates/weaver-spu/src/family/modernbert.rs",
          "success": true
        },
        {
          "duration_ms": 867,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 35,
          "path": "crates/weaver-spu/src/family/phi.rs",
          "success": true
        },
        {
          "duration_ms": 789,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 28,
          "path": "crates/weaver-spu/src/family/qwen2.rs",
          "success": true
        },
        {
          "duration_ms": 1214,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 14,
          "path": "crates/weaver-spu/src/gpu/mod.rs",
          "success": true
        },
        {
          "duration_ms": 197,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 9,
          "path": "crates/weaver-spu/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 8643,
          "language": "Rust",
          "num_chunks": 33,
          "num_embeddings": 33,
          "num_symbols": 60,
          "path": "crates/weaver-spu/src/main.rs",
          "success": true
        },
        {
          "duration_ms": 3957,
          "language": "Rust",
          "num_chunks": 18,
          "num_embeddings": 18,
          "num_symbols": 50,
          "path": "crates/weaver-spu/src/measurement.rs",
          "success": true
        },
        {
          "duration_ms": 2839,
          "language": "Rust",
          "num_chunks": 12,
          "num_embeddings": 12,
          "num_symbols": 36,
          "path": "crates/weaver-spu/src/readout.rs",
          "success": true
        },
        {
          "duration_ms": 4746,
          "language": "Rust",
          "num_chunks": 24,
          "num_embeddings": 24,
          "num_symbols": 60,
          "path": "crates/weaver-spu/src/residency.rs",
          "success": true
        },
        {
          "duration_ms": 2898,
          "language": "Rust",
          "num_chunks": 17,
          "num_embeddings": 17,
          "num_symbols": 33,
          "path": "crates/weaver-spu/src/sampling.rs",
          "success": true
        },
        {
          "duration_ms": 781,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 12,
          "path": "crates/weaver-spu/tests/common/mod.rs",
          "success": true
        },
        {
          "duration_ms": 871,
          "language": "Rust",
          "num_chunks": 8,
          "num_embeddings": 8,
          "num_symbols": 15,
          "path": "crates/weaver-spu/tests/entry.rs",
          "success": true
        },
        {
          "duration_ms": 7647,
          "language": "Rust",
          "num_chunks": 18,
          "num_embeddings": 18,
          "num_symbols": 31,
          "path": "crates/weaver-spu/tests/loaded.rs",
          "success": true
        },
        {
          "duration_ms": 1263,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 10,
          "path": "crates/weaver-spu/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 2354,
          "language": "Rust",
          "num_chunks": 9,
          "num_embeddings": 9,
          "num_symbols": 14,
          "path": "crates/weaver-spu/tests/markers.rs",
          "success": true
        },
        {
          "duration_ms": 3460,
          "language": "Rust",
          "num_chunks": 12,
          "num_embeddings": 12,
          "num_symbols": 19,
          "path": "crates/weaver-spu/tests/native_loaded.rs",
          "success": true
        },
        {
          "duration_ms": 1469,
          "language": "Rust",
          "num_chunks": 8,
          "num_embeddings": 8,
          "num_symbols": 16,
          "path": "crates/weaver-spu/tests/readout_neutral.rs",
          "success": true
        },
        {
          "duration_ms": 743,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 15,
          "path": "crates/weaver-spu/tests/seam.rs",
          "success": true
        },
        {
          "duration_ms": 1039,
          "language": "Rust",
          "num_chunks": 12,
          "num_embeddings": 12,
          "num_symbols": 19,
          "path": "crates/weaver-spu/tests/selection.rs",
          "success": true
        },
        {
          "duration_ms": 1383,
          "language": "Rust",
          "num_chunks": 7,
          "num_embeddings": 7,
          "num_symbols": 14,
          "path": "crates/weaver-spu/tests/two_card.rs",
          "success": true
        },
        {
          "duration_ms": 69,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 2,
          "path": "crates/weaver-state/src/engine/mod.rs",
          "success": true
        },
        {
          "duration_ms": 2964,
          "language": "Rust",
          "num_chunks": 14,
          "num_embeddings": 14,
          "num_symbols": 39,
          "path": "crates/weaver-state/src/engine/postgres.rs",
          "success": true
        },
        {
          "duration_ms": 4562,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 36,
          "path": "crates/weaver-state/src/engine/sqlite.rs",
          "success": true
        },
        {
          "duration_ms": 104,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 3,
          "path": "crates/weaver-state/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 5629,
          "language": "Rust",
          "num_chunks": 30,
          "num_embeddings": 30,
          "num_symbols": 58,
          "path": "crates/weaver-state/src/main.rs",
          "success": true
        },
        {
          "duration_ms": 1387,
          "language": "Rust",
          "num_chunks": 17,
          "num_embeddings": 17,
          "num_symbols": 18,
          "path": "crates/weaver-state/src/store.rs",
          "success": true
        },
        {
          "duration_ms": 298,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 11,
          "path": "crates/weaver-trace/src/canonical.rs",
          "success": true
        },
        {
          "duration_ms": 3097,
          "language": "Rust",
          "num_chunks": 29,
          "num_embeddings": 29,
          "num_symbols": 37,
          "path": "crates/weaver-trace/src/event.rs",
          "success": true
        },
        {
          "duration_ms": 204,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 7,
          "path": "crates/weaver-trace/src/failure.rs",
          "success": true
        },
        {
          "duration_ms": 620,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 12,
          "path": "crates/weaver-trace/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 273,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 14,
          "path": "crates/weaver-trace/src/structure.rs",
          "success": true
        },
        {
          "duration_ms": 1670,
          "language": "Rust",
          "num_chunks": 13,
          "num_embeddings": 13,
          "num_symbols": 30,
          "path": "crates/weaver-trace/src/tee.rs",
          "success": true
        },
        {
          "duration_ms": 2267,
          "language": "Rust",
          "num_chunks": 15,
          "num_embeddings": 15,
          "num_symbols": 41,
          "path": "crates/weaver-trace/src/writer.rs",
          "success": true
        },
        {
          "duration_ms": 514,
          "language": "Rust",
          "num_chunks": 3,
          "num_embeddings": 3,
          "num_symbols": 3,
          "path": "crates/weaver-trace/tests/kinds.rs",
          "success": true
        },
        {
          "duration_ms": 305,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 6,
          "path": "crates/weaver-trace/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 4415,
          "language": "Rust",
          "num_chunks": 33,
          "num_embeddings": 33,
          "num_symbols": 36,
          "path": "crates/weaver-trace/tests/recorder.rs",
          "success": true
        },
        {
          "duration_ms": 403,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 7,
          "path": "crates/weaver-traits/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 375,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 6,
          "path": "crates/weaver-traits/src/message.rs",
          "success": true
        },
        {
          "duration_ms": 146,
          "language": "Rust",
          "num_chunks": 2,
          "num_embeddings": 2,
          "num_symbols": 2,
          "path": "crates/weaver-traits/src/permission.rs",
          "success": true
        },
        {
          "duration_ms": 143,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 0,
          "path": "crates/weaver-traits/src/provider.rs",
          "success": true
        },
        {
          "duration_ms": 353,
          "language": "Rust",
          "num_chunks": 3,
          "num_embeddings": 3,
          "num_symbols": 5,
          "path": "crates/weaver-traits/src/tool.rs",
          "success": true
        },
        {
          "duration_ms": 301,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 6,
          "path": "crates/weaver-traits/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 412,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 6,
          "path": "crates/weaver-traits/tests/message_model.rs",
          "success": true
        },
        {
          "duration_ms": 3080,
          "language": "Rust",
          "num_chunks": 31,
          "num_embeddings": 31,
          "num_symbols": 37,
          "path": "crates/weaver-types/src/config.rs",
          "success": true
        },
        {
          "duration_ms": 349,
          "language": "Rust",
          "num_chunks": 4,
          "num_embeddings": 4,
          "num_symbols": 5,
          "path": "crates/weaver-types/src/identity.rs",
          "success": true
        },
        {
          "duration_ms": 487,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 7,
          "path": "crates/weaver-types/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 5254,
          "language": "Rust",
          "num_chunks": 47,
          "num_embeddings": 47,
          "num_symbols": 54,
          "path": "crates/weaver-types/src/wire.rs",
          "success": true
        },
        {
          "duration_ms": 2835,
          "language": "Rust",
          "num_chunks": 32,
          "num_embeddings": 32,
          "num_symbols": 32,
          "path": "crates/weaver-types/tests/config.rs",
          "success": true
        },
        {
          "duration_ms": 297,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 6,
          "path": "crates/weaver-types/tests/identity.rs",
          "success": true
        },
        {
          "duration_ms": 270,
          "language": "Rust",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 5,
          "path": "crates/weaver-types/tests/manifest.rs",
          "success": true
        },
        {
          "duration_ms": 2160,
          "language": "Rust",
          "num_chunks": 14,
          "num_embeddings": 14,
          "num_symbols": 14,
          "path": "crates/weaver-types/tests/wire.rs",
          "success": true
        },
        {
          "duration_ms": 181,
          "language": "other",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 0,
          "path": "crates/weaver-web/deploy/weaver-admin-verb",
          "success": true
        },
        {
          "duration_ms": 632,
          "language": "Rust",
          "num_chunks": 7,
          "num_embeddings": 7,
          "num_symbols": 16,
          "path": "crates/weaver-web/src/adapters/gate.rs",
          "success": true
        },
        {
          "duration_ms": 57,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 2,
          "path": "crates/weaver-web/src/adapters/mod.rs",
          "success": true
        },
        {
          "duration_ms": 103,
          "language": "Rust",
          "num_chunks": 3,
          "num_embeddings": 3,
          "num_symbols": 5,
          "path": "crates/weaver-web/src/adapters/upstream.rs",
          "success": true
        },
        {
          "duration_ms": 261,
          "language": "Rust",
          "num_chunks": 4,
          "num_embeddings": 4,
          "num_symbols": 8,
          "path": "crates/weaver-web/src/bin/weaver-web-connector.rs",
          "success": true
        },
        {
          "duration_ms": 470,
          "language": "Rust",
          "num_chunks": 4,
          "num_embeddings": 4,
          "num_symbols": 9,
          "path": "crates/weaver-web/src/bin/weaver-web.rs",
          "success": true
        },
        {
          "duration_ms": 533,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 12,
          "path": "crates/weaver-web/src/channel.rs",
          "success": true
        },
        {
          "duration_ms": 501,
          "language": "Rust",
          "num_chunks": 16,
          "num_embeddings": 16,
          "num_symbols": 18,
          "path": "crates/weaver-web/src/config.rs",
          "success": true
        },
        {
          "duration_ms": 146,
          "language": "Rust",
          "num_chunks": 4,
          "num_embeddings": 4,
          "num_symbols": 7,
          "path": "crates/weaver-web/src/fault.rs",
          "success": true
        },
        {
          "duration_ms": 81,
          "language": "Rust",
          "num_chunks": 1,
          "num_embeddings": 1,
          "num_symbols": 13,
          "path": "crates/weaver-web/src/lib.rs",
          "success": true
        },
        {
          "duration_ms": 349,
          "language": "Rust",
          "num_chunks": 3,
          "num_embeddings": 3,
          "num_symbols": 7,
          "path": "crates/weaver-web/src/lifecycle.rs",
          "success": true
        },
        {
          "duration_ms": 1186,
          "language": "Rust",
          "num_chunks": 9,
          "num_embeddings": 9,
          "num_symbols": 19,
          "path": "crates/weaver-web/src/queue.rs",
          "success": true
        },
        {
          "duration_ms": 424,
          "language": "Rust",
          "num_chunks": 11,
          "num_embeddings": 11,
          "num_symbols": 14,
          "path": "crates/weaver-web/src/registry.rs",
          "success": true
        },
        {
          "duration_ms": 802,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 13,
          "path": "crates/weaver-web/src/router.rs",
          "success": true
        },
        {
          "duration_ms": 1227,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 25,
          "path": "crates/weaver-web/src/store/conversation.rs",
          "success": true
        },
        {
          "duration_ms": 716,
          "language": "Rust",
          "num_chunks": 11,
          "num_embeddings": 11,
          "num_symbols": 23,
          "path": "crates/weaver-web/src/store/experiment.rs",
          "success": true
        },
        {
          "duration_ms": 975,
          "language": "Rust",
          "num_chunks": 11,
          "num_embeddings": 11,
          "num_symbols": 15,
          "path": "crates/weaver-web/src/store/key.rs",
          "success": true
        },
        {
          "duration_ms": 319,
          "language": "Rust",
          "num_chunks": 3,
          "num_embeddings": 3,
          "num_symbols": 16,
          "path": "crates/weaver-web/src/store/mod.rs",
          "success": true
        },
        {
          "duration_ms": 3846,
          "language": "Rust",
          "num_chunks": 11,
          "num_embeddings": 11,
          "num_symbols": 26,
          "path": "crates/weaver-web/src/store/plan.rs",
          "success": true
        },
        {
          "duration_ms": 4821,
          "language": "Rust",
          "num_chunks": 14,
          "num_embeddings": 14,
          "num_symbols": 42,
          "path": "crates/weaver-web/src/store/read.rs",
          "success": true
        },
        {
          "duration_ms": 321,
          "language": "Rust",
          "num_chunks": 6,
          "num_embeddings": 6,
          "num_symbols": 9,
          "path": "crates/weaver-web/src/surfaces/gate.rs",
          "success": true
        },
        {
          "duration_ms": 202,
          "language": "Rust",
          "num_chunks": 3,
          "num_embeddings": 3,
          "num_symbols": 9,
          "path": "crates/weaver-web/src/surfaces/mod.rs",
          "success": true
        },
        {
          "duration_ms": 2654,
          "language": "Rust",
          "num_chunks": 17,
          "num_embeddings": 17,
          "num_symbols": 41,
          "path": "crates/weaver-web/src/surfaces/record.rs",
          "success": true
        },
        {
          "duration_ms": 1181,
          "language": "Rust",
          "num_chunks": 10,
          "num_embeddings": 10,
          "num_symbols": 27,
          "path": "crates/weaver-web/src/traceview.rs",
          "success": true
        },
        {
          "duration_ms": 1623,
          "language": "Rust",
          "num_chunks": 23,
          "num_embeddings": 23,
          "num_symbols": 34,
          "path": "crates/weaver-web/src/web/admin.rs",
          "success": true
        },
        {
          "duration_ms": 1003,
          "language": "Rust",
          "num_chunks": 21,
          "num_embeddings": 21,
          "num_symbols": 36,
          "path": "crates/weaver-web/src/web/mod.rs",
          "success": true
        },
        {
          "duration_ms": 1608,
          "language": "Rust",
          "num_chunks": 21,
          "num_embeddings": 21,
          "num_symbols": 36,
          "path": "crates/weaver-web/src/web/user.rs",
          "success": true
        },
        {
          "duration_ms": 3051,
          "language": "Rust",
          "num_chunks": 21,
          "num_embeddings": 21,
          "num_symbols": 48,
          "path": "crates/weaver-web/src/wire.rs",
          "success": true
        },
        {
          "duration_ms": 5035,
          "language": "Python",
          "num_chunks": 20,
          "num_embeddings": 20,
          "num_symbols": 43,
          "path": "process/gates/census.py",
          "success": true
        },
        {
          "duration_ms": 1960,
          "language": "Python",
          "num_chunks": 12,
          "num_embeddings": 12,
          "num_symbols": 22,
          "path": "process/gates/grounds_parity.py",
          "success": true
        },
        {
          "duration_ms": 5158,
          "language": "Python",
          "num_chunks": 13,
          "num_embeddings": 13,
          "num_symbols": 71,
          "path": "process/gates/test_census.py",
          "success": true
        },
        {
          "duration_ms": 795,
          "language": "Python",
          "num_chunks": 5,
          "num_embeddings": 5,
          "num_symbols": 23,
          "path": "process/gates/test_grounds_parity.py",
          "success": true
        },
        {
          "duration_ms": 2614,
          "language": "Python",
          "num_chunks": 17,
          "num_embeddings": 17,
          "num_symbols": 35,
          "path": "process/ingest/chunk_plan.py",
          "success": true
        }
      ],
      "rust_analyzer": {
        "crates_analyzed": 1,
        "edges": 37,
        "failed_files": [
          "/opt/weavertools/WeaverTools/crates/weaver-admin/src/inventory.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-admin/src/log.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-admin/src/main.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-admin/src/sink.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-admin/src/surface.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-admin/src/unit.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-admin/src/verbs.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-admin/tests/invocation.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-admin/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/capture.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/declare.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/field.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/lens.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/main.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/preload.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/project.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/reading.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/record.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/signals.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/src/stream.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/tests/driver.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/tests/field.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/tests/lens.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/tests/reading.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-analysis/tests/stream.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-diagnostic/src/event.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-diagnostic/src/failure.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-diagnostic/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-diagnostic/src/recorder.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-diagnostic/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-diagnostic/tests/recorder.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/src/channel.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/src/hook.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/src/main.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/src/relay.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/src/tools.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/tests/boundary.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/tests/common/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/tests/entry.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-gate/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/assembly.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/authorship.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/bin/pyworker/main.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/bin/pyworker/py_loop.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/bin/worker/dev_loop/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/bin/worker/main.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/channel.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/engine.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/failure.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/lifecycle.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/record.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/replay.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/spawn.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/state.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/src/tools.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/tests/authorship.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/tests/channels.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-harness/tests/service.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-internal/src/calculator.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-internal/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-internal/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/build.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/artifact.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/bin/classify.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/channel.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/decoder/backend.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/decoder/gguf.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/decoder/gguf_tap.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/decoder/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/decoder/native.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/decoder/native_pair.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/decoder/session.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/family/gemma4.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/family/gpt_oss.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/family/llama.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/family/mistral3.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/family/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/family/modernbert.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/family/phi.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/family/qwen2.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/gpu/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/main.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/measurement.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/readout.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/residency.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/src/sampling.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/common/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/entry.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/loaded.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/markers.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/native_loaded.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/readout_neutral.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/seam.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/selection.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-spu/tests/two_card.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-state/src/engine/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-state/src/engine/postgres.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-state/src/engine/sqlite.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-state/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-state/src/main.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-state/src/store.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/src/canonical.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/src/event.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/src/failure.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/src/structure.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/src/tee.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/src/writer.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/tests/kinds.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-trace/tests/recorder.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-traits/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-traits/src/message.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-traits/src/permission.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-traits/src/provider.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-traits/src/tool.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-traits/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-traits/tests/message_model.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-types/src/config.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-types/src/identity.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-types/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-types/src/wire.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-types/tests/config.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-types/tests/identity.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-types/tests/manifest.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-types/tests/wire.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/adapters/gate.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/adapters/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/adapters/upstream.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/bin/weaver-web-connector.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/bin/weaver-web.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/channel.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/config.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/fault.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/lib.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/lifecycle.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/queue.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/registry.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/router.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/store/conversation.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/store/experiment.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/store/key.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/store/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/store/plan.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/store/read.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/surfaces/gate.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/surfaces/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/surfaces/record.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/traceview.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/admin.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/mod.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/user.rs",
          "/opt/weavertools/WeaverTools/crates/weaver-web/src/wire.rs"
        ],
        "failed_workspaces": [],
        "store_errors": 0,
        "store_failed": false,
        "symbols": 37
      },
      "rust_import_edges": 1067,
      "skipped": 0,
      "structural_call_edges": 0,
      "structural_import_edges": 0,
      "total": 171
    },
    "document_phase_error": null,
    "documents": {
      "completed": 46,
      "duration_ms": 328152,
      "failed": 0,
      "results": [
        {
          "duration_ms": 1834,
          "input": "/opt/weavertools/WeaverTools/README.md",
          "num_chunks": 6,
          "success": true
        },
        {
          "duration_ms": 2560,
          "input": "/opt/weavertools/WeaverTools/crates/weaver-spu/kernels/PROVENANCE.md",
          "num_chunks": 7,
          "success": true
        },
        {
          "duration_ms": 958,
          "input": "/opt/weavertools/WeaverTools/crates/weaver-web/README.md",
          "num_chunks": 3,
          "success": true
        },
        {
          "duration_ms": 159,
          "input": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/VENDORED.md",
          "num_chunks": 1,
          "success": true
        },
        {
          "duration_ms": 788,
          "input": "/opt/weavertools/WeaverTools/crates/weaver-web/deploy/agent-setup.md",
          "num_chunks": 2,
          "success": true
        },
        {
          "duration_ms": 6771,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-admin-harness-contract.md",
          "num_chunks": 21,
          "success": true
        },
        {
          "duration_ms": 1944,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-admin-operator-contract.md",
          "num_chunks": 6,
          "success": true
        },
        {
          "duration_ms": 2596,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-admin-systemd-contract.md",
          "num_chunks": 9,
          "success": true
        },
        {
          "duration_ms": 2123,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-analysis-state-contract.md",
          "num_chunks": 7,
          "success": true
        },
        {
          "duration_ms": 5176,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-analysis-web-contract.md",
          "num_chunks": 17,
          "success": true
        },
        {
          "duration_ms": 2130,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-gate-world-contract.md",
          "num_chunks": 7,
          "success": true
        },
        {
          "duration_ms": 1943,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-diagnostic-contract.md",
          "num_chunks": 6,
          "success": true
        },
        {
          "duration_ms": 4474,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-gate-contract.md",
          "num_chunks": 14,
          "success": true
        },
        {
          "duration_ms": 2041,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-spu-classify-contract.md",
          "num_chunks": 6,
          "success": true
        },
        {
          "duration_ms": 4358,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-spu-contract.md",
          "num_chunks": 14,
          "success": true
        },
        {
          "duration_ms": 6261,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-spu-decode-contract.md",
          "num_chunks": 20,
          "success": true
        },
        {
          "duration_ms": 3544,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-state-contract.md",
          "num_chunks": 12,
          "success": true
        },
        {
          "duration_ms": 3317,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-trace-contract.md",
          "num_chunks": 11,
          "success": true
        },
        {
          "duration_ms": 2225,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-organ-channel.md",
          "num_chunks": 8,
          "success": true
        },
        {
          "duration_ms": 13179,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-admin/weaver-admin-PRD.md",
          "num_chunks": 46,
          "success": true
        },
        {
          "duration_ms": 19519,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-admin/weaver-admin-Spec.md",
          "num_chunks": 62,
          "success": true
        },
        {
          "duration_ms": 10666,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-agents-PRD.md",
          "num_chunks": 37,
          "success": true
        },
        {
          "duration_ms": 3816,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-analysis/weaver-analysis-PRD.md",
          "num_chunks": 13,
          "success": true
        },
        {
          "duration_ms": 10105,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-analysis/weaver-analysis-Spec.md",
          "num_chunks": 32,
          "success": true
        },
        {
          "duration_ms": 6101,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-gate/weaver-gate-PRD.md",
          "num_chunks": 20,
          "success": true
        },
        {
          "duration_ms": 11091,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-gate/weaver-gate-Spec.md",
          "num_chunks": 34,
          "success": true
        },
        {
          "duration_ms": 3356,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/Loops/basic-inference-loop.md",
          "num_chunks": 11,
          "success": true
        },
        {
          "duration_ms": 2202,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/Loops/diagnostic-replay-loop.md",
          "num_chunks": 7,
          "success": true
        },
        {
          "duration_ms": 3788,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-diagnostic/weaver-diagnostic-PRD.md",
          "num_chunks": 12,
          "success": true
        },
        {
          "duration_ms": 6141,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-diagnostic/weaver-diagnostic-Spec.md",
          "num_chunks": 19,
          "success": true
        },
        {
          "duration_ms": 7771,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-harness-PRD.md",
          "num_chunks": 26,
          "success": true
        },
        {
          "duration_ms": 26285,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-harness-Spec.md",
          "num_chunks": 81,
          "success": true
        },
        {
          "duration_ms": 4760,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-state/weaver-state-PRD.md",
          "num_chunks": 16,
          "success": true
        },
        {
          "duration_ms": 5551,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-state/weaver-state-Spec.md",
          "num_chunks": 18,
          "success": true
        },
        {
          "duration_ms": 12931,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-trace/weaver-trace-PRD.md",
          "num_chunks": 45,
          "success": true
        },
        {
          "duration_ms": 15715,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-trace/weaver-trace-Spec.md",
          "num_chunks": 45,
          "success": true
        },
        {
          "duration_ms": 870,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-internal/weaver-internal-PRD.md",
          "num_chunks": 3,
          "success": true
        },
        {
          "duration_ms": 1948,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-internal/weaver-internal-Spec.md",
          "num_chunks": 6,
          "success": true
        },
        {
          "duration_ms": 14529,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-spu/weaver-spu-PRD.md",
          "num_chunks": 50,
          "success": true
        },
        {
          "duration_ms": 29460,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-spu/weaver-spu-Spec.md",
          "num_chunks": 89,
          "success": true
        },
        {
          "duration_ms": 2103,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-traits/weaver-traits-PRD.md",
          "num_chunks": 7,
          "success": true
        },
        {
          "duration_ms": 6001,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-traits/weaver-traits-Spec.md",
          "num_chunks": 18,
          "success": true
        },
        {
          "duration_ms": 7538,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-types/weaver-types-PRD.md",
          "num_chunks": 23,
          "success": true
        },
        {
          "duration_ms": 19966,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-types/weaver-types-Spec.md",
          "num_chunks": 60,
          "success": true
        },
        {
          "duration_ms": 7326,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-web/weaver-web-PRD.md",
          "num_chunks": 26,
          "success": true
        },
        {
          "duration_ms": 20205,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-web/weaver-web-Spec.md",
          "num_chunks": 67,
          "success": true
        }
      ],
      "skipped": 0,
      "total": 46
    },
    "duration_ms": 668728,
    "root": "/opt/weavertools/WeaverTools",
    "routed": {
      "code": 171,
      "documents": 46,
      "unrouted": 42
    },
    "unrouted": [
      {
        "path": "/opt/weavertools/WeaverTools/Cargo.lock",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/LICENSE",
        "reason": "no extension and no shebang"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-admin/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-analysis/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-diagnostic/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-gate/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-harness/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-internal/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-spu/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-state/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-trace/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-traits/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-types/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/LICENSE",
        "reason": "no extension and no shebang"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/askama.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/htmx.min.js",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/sse.js",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/style.css",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/surfaces/instrument.css",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/deploy/config.example.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/deploy/connector.example.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/deploy/weaver-web.sudoers",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/surfaces/templates/instrument.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/surfaces/templates/record.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/agent_config.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/base.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/channel.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/channels.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/event.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/lifecycle.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/name.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/sidebar.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/trace.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/trace_event.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/deploy/create-agent.sh",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/deploy/update-stack.sh",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/Loops/basic-inference-loop.png",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/process/gates/census-baseline.json",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/process/gates/lock.sh",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/rust-toolchain.toml",
        "reason": "no handler for extension"
      }
    ]
  },
  "success": false,
  "timestamp": "2026-09-21T20:29:56.287996537+00:00"
}
```

</details>

<details>
<summary>Attempt 2: complete stdout (SHA-256 7d5a4160c56625d6ab9d92dbd356759875c488f0e33c227998bd2672bf7dbb8b)</summary>

```json
{
  "command": "ingest",
  "data": {
    "code": {
      "completed": 171,
      "cpp_call_edges": 0,
      "dangling_inbound_edges": 0,
      "duration_ms": 4812,
      "embedding": {
        "embedding_failure_paths": [],
        "files_embedded": 0,
        "files_with_embedding_failures": 0,
        "service_connected": true,
        "total_embeddings": 0
      },
      "enrichment_error": null,
      "failed": 0,
      "gopls": {
        "edges": 0,
        "failed_files": [],
        "failed_workspaces": [],
        "modules_analyzed": 0,
        "store_errors": 0,
        "store_failed": false,
        "symbols": 0
      },
      "import_edges": 0,
      "python_call_edges": 0,
      "python_import_edges": 0,
      "relationship_error": "failed to atomically store cross-file relationships; earlier file replacements remain committed: request error: relationship endpoint no longer exists; retry ingestion",
      "repointed_inbound_edges": 0,
      "results": [
        {
          "duration_ms": 39,
          "language": "Rust",
          "num_symbols": 40,
          "path": "crates/weaver-admin/src/channel.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 106,
          "language": "Rust",
          "num_symbols": 65,
          "path": "crates/weaver-admin/src/inventory.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 8,
          "path": "crates/weaver-admin/src/log.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 91,
          "language": "Rust",
          "num_symbols": 65,
          "path": "crates/weaver-admin/src/main.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 16,
          "language": "Rust",
          "num_symbols": 18,
          "path": "crates/weaver-admin/src/sink.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 10,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-admin/src/surface.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 29,
          "language": "Rust",
          "num_symbols": 21,
          "path": "crates/weaver-admin/src/unit.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 9,
          "path": "crates/weaver-admin/src/verbs.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-admin/tests/invocation.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-admin/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 34,
          "language": "Rust",
          "num_symbols": 31,
          "path": "crates/weaver-analysis/src/capture.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 12,
          "language": "Rust",
          "num_symbols": 9,
          "path": "crates/weaver-analysis/src/declare.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 17,
          "language": "Rust",
          "num_symbols": 18,
          "path": "crates/weaver-analysis/src/field.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 40,
          "language": "Rust",
          "num_symbols": 35,
          "path": "crates/weaver-analysis/src/lens.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "language": "Rust",
          "num_symbols": 19,
          "path": "crates/weaver-analysis/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 59,
          "language": "Rust",
          "num_symbols": 19,
          "path": "crates/weaver-analysis/src/main.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 7,
          "path": "crates/weaver-analysis/src/preload.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 17,
          "language": "Rust",
          "num_symbols": 20,
          "path": "crates/weaver-analysis/src/project.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 9,
          "language": "Rust",
          "num_symbols": 8,
          "path": "crates/weaver-analysis/src/reading.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "language": "Rust",
          "num_symbols": 9,
          "path": "crates/weaver-analysis/src/record.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 16,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-analysis/src/signals.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 7,
          "path": "crates/weaver-analysis/src/stream.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 16,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-analysis/tests/driver.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 22,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-analysis/tests/field.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 35,
          "language": "Rust",
          "num_symbols": 23,
          "path": "crates/weaver-analysis/tests/lens.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "language": "Rust",
          "num_symbols": 4,
          "path": "crates/weaver-analysis/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 10,
          "path": "crates/weaver-analysis/tests/reading.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 26,
          "language": "Rust",
          "num_symbols": 17,
          "path": "crates/weaver-analysis/tests/stream.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 11,
          "language": "Rust",
          "num_symbols": 27,
          "path": "crates/weaver-diagnostic/src/event.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "language": "Rust",
          "num_symbols": 4,
          "path": "crates/weaver-diagnostic/src/failure.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-diagnostic/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 10,
          "language": "Rust",
          "num_symbols": 10,
          "path": "crates/weaver-diagnostic/src/recorder.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "language": "Rust",
          "num_symbols": 5,
          "path": "crates/weaver-diagnostic/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 13,
          "language": "Rust",
          "num_symbols": 15,
          "path": "crates/weaver-diagnostic/tests/recorder.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 16,
          "language": "Rust",
          "num_symbols": 24,
          "path": "crates/weaver-gate/src/channel.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 18,
          "language": "Rust",
          "num_symbols": 28,
          "path": "crates/weaver-gate/src/hook.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "language": "Rust",
          "num_symbols": 4,
          "path": "crates/weaver-gate/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 36,
          "language": "Rust",
          "num_symbols": 31,
          "path": "crates/weaver-gate/src/main.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 29,
          "language": "Rust",
          "num_symbols": 38,
          "path": "crates/weaver-gate/src/relay.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 23,
          "language": "Rust",
          "num_symbols": 20,
          "path": "crates/weaver-gate/src/tools.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 14,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-gate/tests/boundary.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 13,
          "language": "Rust",
          "num_symbols": 18,
          "path": "crates/weaver-gate/tests/common/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 20,
          "language": "Rust",
          "num_symbols": 18,
          "path": "crates/weaver-gate/tests/entry.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 15,
          "language": "Rust",
          "num_symbols": 12,
          "path": "crates/weaver-gate/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "language": "Rust",
          "num_symbols": 8,
          "path": "crates/weaver-harness/src/assembly.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 22,
          "language": "Rust",
          "num_symbols": 25,
          "path": "crates/weaver-harness/src/authorship.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 13,
          "language": "Python",
          "num_symbols": 17,
          "path": "crates/weaver-harness/src/bin/pyworker/dev_python/alpha_loop.py",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "language": "Python",
          "num_symbols": 1,
          "path": "crates/weaver-harness/src/bin/pyworker/dev_python/basic_loop.py",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 13,
          "language": "Python",
          "num_symbols": 17,
          "path": "crates/weaver-harness/src/bin/pyworker/dev_python/bravo_loop.py",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 9,
          "language": "Rust",
          "num_symbols": 5,
          "path": "crates/weaver-harness/src/bin/pyworker/main.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 20,
          "language": "Rust",
          "num_symbols": 22,
          "path": "crates/weaver-harness/src/bin/pyworker/py_loop.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 23,
          "language": "Rust",
          "num_symbols": 21,
          "path": "crates/weaver-harness/src/bin/worker/dev_loop/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "language": "Rust",
          "num_symbols": 4,
          "path": "crates/weaver-harness/src/bin/worker/main.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 62,
          "language": "Rust",
          "num_symbols": 78,
          "path": "crates/weaver-harness/src/channel.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 182,
          "language": "Rust",
          "num_symbols": 67,
          "path": "crates/weaver-harness/src/engine.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-harness/src/failure.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "language": "Rust",
          "num_symbols": 22,
          "path": "crates/weaver-harness/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 219,
          "language": "Rust",
          "num_symbols": 84,
          "path": "crates/weaver-harness/src/lifecycle.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 13,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-harness/src/record.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 48,
          "language": "Rust",
          "num_symbols": 45,
          "path": "crates/weaver-harness/src/replay.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 25,
          "language": "Rust",
          "num_symbols": 20,
          "path": "crates/weaver-harness/src/spawn.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 46,
          "language": "Rust",
          "num_symbols": 50,
          "path": "crates/weaver-harness/src/state.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 7,
          "path": "crates/weaver-harness/src/tools.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 31,
          "language": "Rust",
          "num_symbols": 18,
          "path": "crates/weaver-harness/tests/authorship.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 20,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-harness/tests/channels.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "language": "Rust",
          "num_symbols": 7,
          "path": "crates/weaver-harness/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 37,
          "language": "Rust",
          "num_symbols": 29,
          "path": "crates/weaver-harness/tests/service.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 18,
          "language": "Rust",
          "num_symbols": 12,
          "path": "crates/weaver-internal/src/calculator.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "language": "Rust",
          "num_symbols": 1,
          "path": "crates/weaver-internal/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "language": "Rust",
          "num_symbols": 3,
          "path": "crates/weaver-internal/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "language": "Rust",
          "num_symbols": 2,
          "path": "crates/weaver-spu/build.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 703,
          "language": "C++",
          "num_symbols": 48,
          "path": "crates/weaver-spu/kernels/transformer.cu",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 61,
          "language": "Rust",
          "num_symbols": 52,
          "path": "crates/weaver-spu/src/artifact.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 11,
          "language": "Rust",
          "num_symbols": 9,
          "path": "crates/weaver-spu/src/bin/classify.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 35,
          "language": "Rust",
          "num_symbols": 54,
          "path": "crates/weaver-spu/src/channel.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 11,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-spu/src/decoder/backend.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 59,
          "language": "Rust",
          "num_symbols": 61,
          "path": "crates/weaver-spu/src/decoder/gguf.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 25,
          "language": "Rust",
          "num_symbols": 24,
          "path": "crates/weaver-spu/src/decoder/gguf_tap.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-spu/src/decoder/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 42,
          "language": "Rust",
          "num_symbols": 45,
          "path": "crates/weaver-spu/src/decoder/native.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 58,
          "language": "Rust",
          "num_symbols": 22,
          "path": "crates/weaver-spu/src/decoder/native_pair.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 102,
          "language": "Rust",
          "num_symbols": 72,
          "path": "crates/weaver-spu/src/decoder/session.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 14,
          "language": "Rust",
          "num_symbols": 29,
          "path": "crates/weaver-spu/src/family/gemma4.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 8,
          "language": "Rust",
          "num_symbols": 25,
          "path": "crates/weaver-spu/src/family/gpt_oss.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 8,
          "language": "Rust",
          "num_symbols": 24,
          "path": "crates/weaver-spu/src/family/llama.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 13,
          "language": "Rust",
          "num_symbols": 27,
          "path": "crates/weaver-spu/src/family/mistral3.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 112,
          "language": "Rust",
          "num_symbols": 87,
          "path": "crates/weaver-spu/src/family/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 16,
          "language": "Rust",
          "num_symbols": 18,
          "path": "crates/weaver-spu/src/family/modernbert.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 11,
          "language": "Rust",
          "num_symbols": 35,
          "path": "crates/weaver-spu/src/family/phi.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 12,
          "language": "Rust",
          "num_symbols": 28,
          "path": "crates/weaver-spu/src/family/qwen2.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 15,
          "language": "Rust",
          "num_symbols": 14,
          "path": "crates/weaver-spu/src/gpu/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "language": "Rust",
          "num_symbols": 9,
          "path": "crates/weaver-spu/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 91,
          "language": "Rust",
          "num_symbols": 60,
          "path": "crates/weaver-spu/src/main.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 39,
          "language": "Rust",
          "num_symbols": 50,
          "path": "crates/weaver-spu/src/measurement.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 27,
          "language": "Rust",
          "num_symbols": 36,
          "path": "crates/weaver-spu/src/readout.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 51,
          "language": "Rust",
          "num_symbols": 60,
          "path": "crates/weaver-spu/src/residency.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 31,
          "language": "Rust",
          "num_symbols": 33,
          "path": "crates/weaver-spu/src/sampling.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 12,
          "language": "Rust",
          "num_symbols": 12,
          "path": "crates/weaver-spu/tests/common/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 11,
          "language": "Rust",
          "num_symbols": 15,
          "path": "crates/weaver-spu/tests/entry.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 90,
          "language": "Rust",
          "num_symbols": 31,
          "path": "crates/weaver-spu/tests/loaded.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 15,
          "language": "Rust",
          "num_symbols": 10,
          "path": "crates/weaver-spu/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 21,
          "language": "Rust",
          "num_symbols": 14,
          "path": "crates/weaver-spu/tests/markers.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 39,
          "language": "Rust",
          "num_symbols": 19,
          "path": "crates/weaver-spu/tests/native_loaded.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 15,
          "language": "Rust",
          "num_symbols": 16,
          "path": "crates/weaver-spu/tests/readout_neutral.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 10,
          "language": "Rust",
          "num_symbols": 15,
          "path": "crates/weaver-spu/tests/seam.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 12,
          "language": "Rust",
          "num_symbols": 19,
          "path": "crates/weaver-spu/tests/selection.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 16,
          "language": "Rust",
          "num_symbols": 14,
          "path": "crates/weaver-spu/tests/two_card.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "language": "Rust",
          "num_symbols": 2,
          "path": "crates/weaver-state/src/engine/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 33,
          "language": "Rust",
          "num_symbols": 39,
          "path": "crates/weaver-state/src/engine/postgres.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 56,
          "language": "Rust",
          "num_symbols": 36,
          "path": "crates/weaver-state/src/engine/sqlite.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "language": "Rust",
          "num_symbols": 3,
          "path": "crates/weaver-state/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 63,
          "language": "Rust",
          "num_symbols": 58,
          "path": "crates/weaver-state/src/main.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 18,
          "language": "Rust",
          "num_symbols": 18,
          "path": "crates/weaver-state/src/store.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 11,
          "path": "crates/weaver-trace/src/canonical.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 28,
          "language": "Rust",
          "num_symbols": 37,
          "path": "crates/weaver-trace/src/event.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "language": "Rust",
          "num_symbols": 7,
          "path": "crates/weaver-trace/src/failure.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 12,
          "path": "crates/weaver-trace/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 14,
          "path": "crates/weaver-trace/src/structure.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 19,
          "language": "Rust",
          "num_symbols": 30,
          "path": "crates/weaver-trace/src/tee.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 26,
          "language": "Rust",
          "num_symbols": 41,
          "path": "crates/weaver-trace/src/writer.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 3,
          "path": "crates/weaver-trace/tests/kinds.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-trace/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 54,
          "language": "Rust",
          "num_symbols": 36,
          "path": "crates/weaver-trace/tests/recorder.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 7,
          "path": "crates/weaver-traits/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-traits/src/message.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "language": "Rust",
          "num_symbols": 2,
          "path": "crates/weaver-traits/src/permission.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "language": "Rust",
          "num_symbols": 0,
          "path": "crates/weaver-traits/src/provider.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "language": "Rust",
          "num_symbols": 5,
          "path": "crates/weaver-traits/src/tool.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-traits/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-traits/tests/message_model.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 31,
          "language": "Rust",
          "num_symbols": 37,
          "path": "crates/weaver-types/src/config.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 5,
          "path": "crates/weaver-types/src/identity.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "language": "Rust",
          "num_symbols": 7,
          "path": "crates/weaver-types/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 49,
          "language": "Rust",
          "num_symbols": 54,
          "path": "crates/weaver-types/src/wire.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 27,
          "language": "Rust",
          "num_symbols": 32,
          "path": "crates/weaver-types/tests/config.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 6,
          "path": "crates/weaver-types/tests/identity.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 5,
          "path": "crates/weaver-types/tests/manifest.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 25,
          "language": "Rust",
          "num_symbols": 14,
          "path": "crates/weaver-types/tests/wire.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "language": "other",
          "num_symbols": 0,
          "path": "crates/weaver-web/deploy/weaver-admin-verb",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 10,
          "language": "Rust",
          "num_symbols": 16,
          "path": "crates/weaver-web/src/adapters/gate.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "language": "Rust",
          "num_symbols": 2,
          "path": "crates/weaver-web/src/adapters/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "language": "Rust",
          "num_symbols": 5,
          "path": "crates/weaver-web/src/adapters/upstream.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "language": "Rust",
          "num_symbols": 8,
          "path": "crates/weaver-web/src/bin/weaver-web-connector.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 8,
          "language": "Rust",
          "num_symbols": 9,
          "path": "crates/weaver-web/src/bin/weaver-web.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 12,
          "path": "crates/weaver-web/src/channel.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 18,
          "path": "crates/weaver-web/src/config.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "language": "Rust",
          "num_symbols": 7,
          "path": "crates/weaver-web/src/fault.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-web/src/lib.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 7,
          "path": "crates/weaver-web/src/lifecycle.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 18,
          "language": "Rust",
          "num_symbols": 19,
          "path": "crates/weaver-web/src/queue.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 7,
          "language": "Rust",
          "num_symbols": 14,
          "path": "crates/weaver-web/src/registry.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 13,
          "language": "Rust",
          "num_symbols": 13,
          "path": "crates/weaver-web/src/router.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 18,
          "language": "Rust",
          "num_symbols": 25,
          "path": "crates/weaver-web/src/store/conversation.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 9,
          "language": "Rust",
          "num_symbols": 23,
          "path": "crates/weaver-web/src/store/experiment.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 10,
          "language": "Rust",
          "num_symbols": 15,
          "path": "crates/weaver-web/src/store/key.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "language": "Rust",
          "num_symbols": 16,
          "path": "crates/weaver-web/src/store/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 38,
          "language": "Rust",
          "num_symbols": 26,
          "path": "crates/weaver-web/src/store/plan.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 49,
          "language": "Rust",
          "num_symbols": 42,
          "path": "crates/weaver-web/src/store/read.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "language": "Rust",
          "num_symbols": 9,
          "path": "crates/weaver-web/src/surfaces/gate.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "language": "Rust",
          "num_symbols": 9,
          "path": "crates/weaver-web/src/surfaces/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 29,
          "language": "Rust",
          "num_symbols": 41,
          "path": "crates/weaver-web/src/surfaces/record.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 19,
          "language": "Rust",
          "num_symbols": 27,
          "path": "crates/weaver-web/src/traceview.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 26,
          "language": "Rust",
          "num_symbols": 34,
          "path": "crates/weaver-web/src/web/admin.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 17,
          "language": "Rust",
          "num_symbols": 36,
          "path": "crates/weaver-web/src/web/mod.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 27,
          "language": "Rust",
          "num_symbols": 36,
          "path": "crates/weaver-web/src/web/user.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 48,
          "language": "Rust",
          "num_symbols": 48,
          "path": "crates/weaver-web/src/wire.rs",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 36,
          "language": "Python",
          "num_symbols": 43,
          "path": "process/gates/census.py",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 27,
          "language": "Python",
          "num_symbols": 22,
          "path": "process/gates/grounds_parity.py",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 49,
          "language": "Python",
          "num_symbols": 71,
          "path": "process/gates/test_census.py",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 9,
          "language": "Python",
          "num_symbols": 23,
          "path": "process/gates/test_grounds_parity.py",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 27,
          "language": "Python",
          "num_symbols": 35,
          "path": "process/ingest/chunk_plan.py",
          "skipped": true,
          "success": true
        }
      ],
      "rust_analyzer": {
        "crates_analyzed": 0,
        "edges": 0,
        "failed_files": [],
        "failed_workspaces": [],
        "store_errors": 0,
        "store_failed": false,
        "symbols": 0
      },
      "rust_import_edges": 0,
      "skipped": 171,
      "structural_call_edges": 0,
      "structural_import_edges": 0,
      "total": 171
    },
    "document_phase_error": null,
    "documents": {
      "completed": 46,
      "duration_ms": 90,
      "failed": 0,
      "results": [
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/README.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/crates/weaver-spu/kernels/PROVENANCE.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/crates/weaver-web/README.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/VENDORED.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/crates/weaver-web/deploy/agent-setup.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-admin-harness-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-admin-operator-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-admin-systemd-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-analysis-state-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-analysis-web-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-gate-world-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-diagnostic-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-gate-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-spu-classify-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-spu-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-spu-decode-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-state-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-harness-trace-contract.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/docs/crates/contracts/weaver-organ-channel.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-admin/weaver-admin-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-admin/weaver-admin-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-agents-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-analysis/weaver-analysis-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-analysis/weaver-analysis-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-gate/weaver-gate-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-gate/weaver-gate-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/Loops/basic-inference-loop.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/Loops/diagnostic-replay-loop.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-diagnostic/weaver-diagnostic-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-diagnostic/weaver-diagnostic-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-harness-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 5,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-harness-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-state/weaver-state-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-state/weaver-state-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 2,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-trace/weaver-trace-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/weaver-trace/weaver-trace-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-internal/weaver-internal-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 0,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-internal/weaver-internal-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 3,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-spu/weaver-spu-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 6,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-spu/weaver-spu-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-traits/weaver-traits-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-traits/weaver-traits-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-types/weaver-types-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-types/weaver-types-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 1,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-web/weaver-web-PRD.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        },
        {
          "duration_ms": 4,
          "input": "/opt/weavertools/WeaverTools/docs/crates/weaver-web/weaver-web-Spec.md",
          "reason": "content hash unchanged",
          "skipped": true,
          "success": true
        }
      ],
      "skipped": 46,
      "total": 46
    },
    "duration_ms": 4921,
    "root": "/opt/weavertools/WeaverTools",
    "routed": {
      "code": 171,
      "documents": 46,
      "unrouted": 42
    },
    "unrouted": [
      {
        "path": "/opt/weavertools/WeaverTools/Cargo.lock",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/LICENSE",
        "reason": "no extension and no shebang"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-admin/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-analysis/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-diagnostic/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-gate/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-harness/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-internal/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-spu/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-state/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-trace/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-traits/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-types/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/Cargo.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/LICENSE",
        "reason": "no extension and no shebang"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/askama.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/htmx.min.js",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/sse.js",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/style.css",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/assets/surfaces/instrument.css",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/deploy/config.example.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/deploy/connector.example.toml",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/deploy/weaver-web.sudoers",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/surfaces/templates/instrument.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/surfaces/templates/record.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/agent_config.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/base.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/channel.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/channels.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/event.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/lifecycle.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/name.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/sidebar.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/trace.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/crates/weaver-web/src/web/templates/trace_event.html",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/deploy/create-agent.sh",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/deploy/update-stack.sh",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/docs/crates/weaver-harness/Loops/basic-inference-loop.png",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/process/gates/census-baseline.json",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/process/gates/lock.sh",
        "reason": "no handler for extension"
      },
      {
        "path": "/opt/weavertools/WeaverTools/rust-toolchain.toml",
        "reason": "no handler for extension"
      }
    ]
  },
  "success": false,
  "timestamp": "2026-09-21T20:30:39.436451474+00:00"
}
```

</details>
