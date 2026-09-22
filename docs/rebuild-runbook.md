# WeaverTools rebuild rehearsal runbook

## Current checkpoint — fresh run, 2026-09-22

**Fresh-from-empty execution and verification passed. Comparison limitations remain;
no production cutover or merge is authorized by this result.** PR #160 remains draft
for review. Todd authorized `scratch_rebuild_wt5_r2` plus read-only live
schema/count/report comparisons after the review-seat reconciliation. No prior
scratch database was cleared or reused for this run; no `--force` was used.

- Refreshed main: `90bab3ae8c46ba930d162bda9a3d8d5f61b802c9`.
- Documentation starting revision: `9cbab044714102c3d89fd75feaed7ac0b1f8c51f`.
- Binary build revision: `8facda9`, identical implementation to refreshed main;
  reused the existing private binary rather than rebuilding unchanged code.
- Binary SHA-256: `c2988d354a395cccb790ee3b4e1bc5ce23bf707c560ae3ef4153c74ad375d463`.
- Source: `/opt/weavertools/WeaverTools`, clean before and after at
  `f03142a85e1c6bf088b6299063455272df93d68d`.
- Evidence captured September 22, 2026, 09:58–10:12 America/Chicago (CDT,
  UTC−05:00). Command logs use UTC; table times below use CDT.
- [Main CI](https://github.com/toddwbucy/HADES/actions/runs/35735357279) and
  [starting documentation CI](https://github.com/toddwbucy/HADES/actions/runs/35742744334)
  passed. Final documentation CI is tracked on PR #160, separately from runtime evidence.

The pre-create lookup returned database-not-found (ArangoDB 404 / 1228), as expected.
Creation succeeded once. After schema application all 35 collections existed;
only `hades_schema` held rows (20), every data collection was empty. The live and
scratch schema query results matched exactly after excluding `_id` and `_rev`.
This includes every metadata field, all 18 relation definitions and the named graph.
The creation command necessarily calls the `_system` database-creation endpoint;
its only created database was the authorized `scratch_rebuild_wt5_r2`.

## Exact commands and phase timings

Executed from the existing isolated checkout `/tmp/hades-epic12`. Preparation:

```bash
git fetch origin
mkdir -p /tmp/hades-rebuild-rehearsal/r2-2026-09-22
cp /tmp/hades-rebuild-rehearsal/2026-09-22/hades /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades
```

Each captured command below additionally inherited
`CARGO_TARGET_DIR=/tmp/hades-rebuild-rehearsal/r2-2026-09-22/analyzer-target` and
`PYTHONDONTWRITEBYTECODE=1`, keeping generated analyzer output private. The wrapper
records stdout, stderr, start epoch, subprocess wall time and exit status separately.
Preparation wall time was not measured. These are completed historical invocations,
**not an instruction to recreate an existing database**.

| Phase | Start (CDT) | Seconds | Exit |
| --- | --- | ---: | ---: |
| scratch-before | 09:58:55 | 0.0098 | 1 |
| create-scratch | 09:59:14 | 0.0131 | 0 |
| apply-schema | 09:59:14 | 0.0363 | 0 |
| live-schema | 09:59:32 | 0.0109 | 0 |
| scratch-schema | 09:59:32 | 0.0100 | 0 |
| live-counts | 09:59:32 | 0.0186 | 0 |
| scratch-empty | 09:59:32 | 0.0189 | 0 |
| live-report | 09:59:32 | 0.0093 | 0 |
| ingest-1 | 09:59:49 | 705.6009 | 0 |
| adapter-dry-run | 10:12:02 | 0.2313 | 0 |
| adapter-write | 10:12:02 | 0.5071 | 0 |
| scratch-collections-after | 10:12:03 | 0.0199 | 0 |
| drift | 10:12:03 | 0.1022 | 0 |
| validate | 10:12:03 | 0.2221 | 0 |
| query-default | 10:12:03 | 1.8346 | 0 |
| query-codebase | 10:12:05 | 2.4788 | 0 |
| metadata-embeddings | 10:12:07 | 0.0100 | 0 |
| metadata-codebase_embeddings | 10:12:07 | 0.0139 | 0 |
| adapter-report | 10:12:08 | 0.0091 | 0 |

`scratch-before` exit 1 is the expected absence check, not an ingestion failure.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db collections
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db create-database scratch_rebuild_wt5_r2
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 schema apply services/adapters/weavertools/schema.yaml
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db WeaverTools_v5 db aql 'FOR d IN hades_schema SORT d._key RETURN UNSET(d, "_id", "_rev")'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR d IN hades_schema SORT d._key RETURN UNSET(d, "_id", "_rev")'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db WeaverTools_v5 db collections
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db collections
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db WeaverTools_v5 db get wt_ingest_report latest
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades ingest /opt/weavertools/WeaverTools --db scratch_rebuild_wt5_r2
HADES_CONFIG=/home/todd/.config/hades/hades.yaml python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5_r2 --repo /opt/weavertools/WeaverTools --dry-run
HADES_CONFIG=/home/todd/.config/hades/hades.yaml python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5_r2 --repo /opt/weavertools/WeaverTools
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db collections
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 codebase drift /opt/weavertools/WeaverTools
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 codebase validate
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db query 'state persistence' -n 5
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db query 'state persistence' -c codebase -n 5
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection LET invalid = !IS_STRING(e.model) OR e.model == "" OR !IS_STRING(e.model_hash) OR e.model_hash == "" OR !IS_NUMBER(e.dimension) OR e.dimension <= 0 OR e.dimension != FLOOR(e.dimension) COLLECT AGGREGATE total = SUM(1), invalid_rows = SUM(invalid ? 1 : 0) RETURN {total, invalid_rows}' --bind '{"@collection":"embeddings"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection LET invalid = !IS_STRING(e.model) OR e.model == "" OR !IS_STRING(e.model_hash) OR e.model_hash == "" OR !IS_NUMBER(e.dimension) OR e.dimension <= 0 OR e.dimension != FLOOR(e.dimension) COLLECT AGGREGATE total = SUM(1), invalid_rows = SUM(invalid ? 1 : 0) RETURN {total, invalid_rows}' --bind '{"@collection":"codebase_embeddings"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db get wt_ingest_report latest
```

## Observed results and limits

- Unified ingestion: **705.6009 seconds**, exit 0, `success:true`; 171 code files
  and 46 documents completed, zero failed and **zero skipped**. Code produced
  1,976 embeddings; no embedding failures, enrichment error or relationship error.
- Rust-analyzer: 161 files; 4,409 symbols and 7,754 edges emitted, one crate,
  zero store errors, `store_failed:false`, no failed files/workspaces. Final
  collection totals below include the other analyzers. The envelope emitted
  1,067 import and 153 Python-call edges; these are distinct from total stored calls.
- Adapter dry-run and write: exit 0; **2,177 rows plus one report acknowledged**.
  Both live and scratch reports have zero dangling document/code citations,
  missing citation-source files and missing declaring-document targets. These
  are report-based checks, not a new exhaustive scan of every adapter edge.
- Drift: clean, 171 source files matched 171 graph nodes; zero changed, stale,
  unverifiable or uningested files. This also confirms the observed route count
  is 171, not the earlier review's 170. No discovery policy was changed.
- Validation: **14 queried invariants passed, zero failed, three skipped** because
  they are database/unit-test guarantees; not 17 runtime checks.
- Queries: `state persistence` returned five results on each of `default` and
  `codebase`, model `/bulk-store/books/models/jinaai--jina-embeddings-v4`, dimension
  2048. This proves query execution, not scored retrieval-quality acceptance.
- Metadata: exhaustive scans of all 1,050 document and 1,976 code embeddings found
  zero rows lacking nonempty string model/hash or a positive integer dimension.
  Hash provenance and vector quality were not independently established.

Non-fatal LSP teardown warnings (`incomplete or oversized LSP header`,
`LSP reader terminated`, `LSP client dropped`) occurred; the enrichment store and
complete ingestion succeeded. No implementation correction was made.

## Live comparison

Live was read before ingestion; scratch was read after the adapter. This is not
an atomic snapshot of the live service. No production rows were written or retired.

| Collection | WeaverTools_v5 | scratch_rebuild_wt5_r2 |
| --- | ---: | ---: |
| `chunks` | 1050 | 1050 |
| `codebase_calls_edges` | 3431 | 3430 |
| `codebase_chunks` | 1993 | 1976 |
| `codebase_defines_edges` | 4868 | 4868 |
| `codebase_embeddings` | 1993 | 1976 |
| `codebase_files` | 188 | 171 |
| `codebase_implements_edges` | 68 | 68 |
| `codebase_imports_edges` | 1067 | 1067 |
| `codebase_symbols` | 4868 | 4868 |
| `documents` | 46 | 46 |
| `embeddings` | 1050 | 1050 |
| `hades_schema` | 20 | 20 |
| `wt_artifacts` | 3 | 3 |
| `wt_assertions` | 418 | 418 |
| `wt_asserts_edges` | 423 | 423 |
| `wt_axioms` | 5 | 5 |
| `wt_cites_edges` | 504 | 500 |
| `wt_crates` | 12 | 12 |
| `wt_declared_in_edges` | 498 | 498 |
| `wt_defines_edges` | 46 | 46 |
| `wt_documents` | 13 | 13 |
| `wt_draws_edges` | 59 | 59 |
| `wt_elects_edges` | 2 | 2 |
| `wt_floor_link_edges` | 8 | 8 |
| `wt_grounds_edges` | 87 | 87 |
| `wt_holds_edges` | 8 | 8 |
| `wt_ingest_report` | 1 | 1 |
| `wt_parent_edges` | 12 | 12 |
| `wt_party_edges` | 23 | 23 |
| `wt_reads_edges` | 1 | 1 |
| `wt_seam_edges` | 10 | 10 |
| `wt_systems` | 1 | 1 |
| `wt_terms` | 5 | 5 |
| `wt_vocabulary` | 41 | 41 |
| `wt_writes_edges` | 2 | 2 |

Five counts differ: files/chunks/code embeddings are 17 fewer each, calls one
fewer, and adapter citations four fewer. Fresh imports are **1,067**, matching
live, compared with 1,263 in retained `scratch_rebuild_wt5`. This removes the prior
mixed-history import excess from the new run; it does not identify every old edge
or prove semantic correctness of all targets. The other count differences remain
unreconciled, not silently accepted as loss or classified as new defects.
Crate-aware import correctness remains a separate known follow-up.

| Comparison | Live | Fresh scratch |
| --- | --- | --- |
| Adapter dangling document / code citations | 0 / 0 | 0 / 0 |
| Missing citation-source / declaring-document targets | 0 / 0 | 0 / 0 |
| Unified-ingest unrouted paths | Unknown: historical live envelope unavailable | 42, listed below |
| Citing files out of adapter scope | 0 in stored live report | 3 in new report |

Live report `_rev` is `_mJ-mKhS--g`; it contains no run timestamp or source revision.
Its zero counts are historical report observations, not proof of current live
coverage. It additionally lists `weaver-analysis (socket)` as an unknown document
tag; the new report does not. Both retain the malformed/unknown `WeaverTools`
notes. The scratch report explicitly covers present extracted rows only and does
not retire absent older rows (there were no prior rows in this fresh database).

The historical live unrouted set cannot be reconstructed from collection counts.
Supplying its original ingest envelope is the only remaining comparison input;
without it, unrouted parity remains unknown. Successful runtime checks do not
establish live parity or authorize production migration.

## Unrouted paths (current unified ingest)

All paths are relative to `/opt/weavertools/WeaverTools`. No handler or discovery
policy was changed to consume these files.

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

## Adapter coverage notes

Three out-of-scope declaration/citation files were reported: `crates/weaver-analysis/Cargo.toml`, `crates/weaver-diagnostic/Cargo.toml`, `crates/weaver-spu/Cargo.toml`. They were not silently ingested. The report also retains these notes:

- document_notes: malformed node id in docs/crates/weaver-agents-PRD.md: 'WeaverTools'
- document_notes: unknown tag in docs/crates/weaver-agents-PRD.md: WeaverTools (ratified)
- document_notes: declaring_files_out_of_scope=0
- code_notes: out of scope and holds a citation, not read: crates/weaver-analysis/Cargo.toml
- code_notes: out of scope and holds a citation, not read: crates/weaver-diagnostic/Cargo.toml
- code_notes: out of scope and holds a citation, not read: crates/weaver-spu/Cargo.toml
- code_notes: citing_files_out_of_scope=3
- code_notes: headers_seen=462
- code_notes: sources_without_a_header=38
- code_notes: sources_owing_a_header=160

## Cutover and rollback — DOCUMENTATION ONLY, NOT EXECUTED

This section follows [Code-file identities and migration](code-file-identities.md).
It does not authorize production commands, service operations, snapshots, or
configuration changes. Neither scratch database is an approved cutover target.

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


## Historical evidence and stop disposition

- [Original September 21 stopped rehearsal](https://github.com/toddwbucy/HADES/blob/35445db92da75bc1131c739ebdff8bb667d765de/docs/rebuild-runbook.md):
  both failed envelopes remain immutable; original captures remain untouched.
- [September 22 retained-scratch recovery and review reconciliation](https://github.com/toddwbucy/HADES/blob/9cbab044714102c3d89fd75feaed7ac0b1f8c51f/docs/rebuild-runbook.md):
  preserves all earlier measurements, the review's contamination explanation and
  the authorization gap subsequently resolved by Todd. Its pending-r2 status is
  superseded by this fresh execution, not retroactively rewritten.
- Current raw stdout/stderr/command metadata:
  `/tmp/hades-rebuild-rehearsal/r2-2026-09-22/`. These local temporary captures
  are not a durable off-server archive; compact results and hashes are committed.

All scratch commands finished; no automatic merge or further rehearsal is queued.
Execution stops with fresh verification complete and the historical live unrouted
comparison unavailable. The five count differences require review before any
production acceptance; they do not authorize another investigation. PR stays draft
for independent review. Final-revision CI status is recorded in the PR body.
No production service, data, model, configuration, source tree or active checkout
was modified. No deployment, snapshot, cutover, rollback, #164 refactor or broader
audit work began. Both scratch databases are retained for owner review.

### Capture SHA-256

Each phase has `.json`, `.stdout` and `.stderr` files. The digest below hashes their
concatenation in that order, binding both output streams and command metadata.

| Phase | SHA-256 (`json || stdout || stderr`) |
| --- | --- |
| scratch-before | `116aa66800bbd3abc64ef3f8f3282fc77e38c0d88914f33db326124b73fce990` |
| create-scratch | `4de6042e80a5c6bc11b6d4417648644d7c49befd7fd5d866227ff90e2d914b39` |
| apply-schema | `79477837b4c1644c088aca5f5236ee1e7e2eb14b8039541c9685c5396dc48687` |
| live-schema | `a7abcde8d36eff051621cd66e977dcb023e0dc89e6db56289300d5b48f4835fc` |
| scratch-schema | `150e9f5e8c697ebfe3df5842c55eb6ba1b5e6befd4902903aee469c43a2988d1` |
| live-counts | `7682fa850fd5af2476fb6a3bb93e40203b7bfa07c7136abb146e19439987ed04` |
| scratch-empty | `082ac4aa9d5007495c8555523c77b05cd91abc69a119be12f4d33eed2ef89056` |
| live-report | `6a9703c0df4ba444161b75835e02a535482cf67711429b8be836ccd730cfbe73` |
| ingest-1 | `6b6bff44fecf845f11abad2517505a791efabb9cf3fb9f6b8ffd26e7d43ff730` |
| adapter-dry-run | `911b479caa58bf0ba5f7437f8b4d0e7bd2673ed08771610dc760f3776a27ab84` |
| adapter-write | `7fd812a92225af993a30533592366902d8543643c029fb45f74014ab801b82ab` |
| scratch-collections-after | `dd15c3ddc0354561ed94797ac199f361b02193e93a72d553b5da21fe59f202cb` |
| drift | `ce2d3237d2af0d929b9ae785feca75ef44be6dbc90681a4531bc88ee1d8676f2` |
| validate | `c04a18c2afb4cd3dd919331fa2c985e4f5c1b3a4a4f37bda2616faa38f4e2e3c` |
| query-default | `f39c01529ee256864165e0c227914f71451d3a690830cb577aa9c0f544319f9f` |
| query-codebase | `0d757912992fa28a2bf9614b02fc41a29d8168d95ee9f5e10d433c8a14b1ad9e` |
| metadata-embeddings | `d9cd6f3695fb3d64906aac672ee6c719f8dfdc603066ccf6aea9695f647a8c40` |
| metadata-codebase_embeddings | `0248fbbb9b2e66e2b2ebbe43dac55a021f66bac7e802e079ce48759ab459e43a` |
| adapter-report | `af95fc47268ed63365493d7304275012328fd5cb669cc4ec1746dbd3b6ef8db9` |
