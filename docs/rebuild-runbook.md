# WeaverTools rebuild rehearsal runbook

## Current checkpoint — 2026-09-22

**Retained-scratch recovery passed; fresh-from-empty rehearsal remains outstanding.
Live comparison is incomplete. Keep PR #160 in draft; no cutover approval.**
Todd authorized resumption after #163 merged. This resumed the existing
`scratch_rebuild_wt5` from the stopped September 21 rehearsal; it did **not** create
another database, truncate collections or force re-ingestion. The old source and
embedding writes were reused. This establishes recovery plus downstream execution,
not a fresh-from-empty run of the current binary or acceptance of count differences.

- Current-main implementation: `90bab3ae8c46ba930d162bda9a3d8d5f61b802c9`.
- Rehearsal branch integration/build revision: `8facda9` (`docs/rebuild-runbook`, PR #160).
- Source: clean `f03142a85e1c6bf088b6299063455272df93d68d`, `/opt/weavertools/WeaverTools`.
- All times below use measured subprocess wall time. Log timestamps are UTC;
  the rehearsal date/time zone is America/Chicago (CDT, UTC−05:00).
- [Main integration CI](https://github.com/toddwbucy/HADES/actions/runs/35735357279)
  passed before this resumption's first scratch write. Documentation PR CI is
  tracked separately; neither is production deployment evidence.

The original schema gate remains supported by the
[review-seat comparison of all 20 schema rows](https://github.com/toddwbucy/HADES/pull/160#pullrequestreview-5271355507).
`services/adapters/weavertools/schema.yaml` is unchanged between the original
`5ee16d3` baseline and current main. No schema reapplication or live AQL was run.
The adapter uses the previously verified loopback endpoint of the same user-level
ArangoDB instance. No credentials, configuration or services were changed.

## Review-seat reconciliation — 2026-09-22

The [review submitted at 14:00:55 UTC (09:00:55 CDT)](https://github.com/toddwbucy/HADES/pull/160#pullrequestreview-5279136550)
independently reports zero dangling edges across every code and adapter edge
collection, embedding metadata present on all 1,050 document and 1,976 code rows,
and 4,868 symbols: rust-analyzer 4,409, syn 217, rustpython 194, libclang 48.
All 161 Rust files carry `ra_analyzed`. These are attributed reviewer observations
on `50120e07` / binary `8facda9`, not additional code-seat database queries.
They support recovery, not a fresh rebuild or semantic correctness of every edge.

The reviewer identifies retained pre-#163 import edges as contamination: changing
a target changes its deterministic key, so replacing current keys does not remove
all historical keys. Current import construction uses source/target-derived keys;
the adapter also explicitly retains absent rows. The review reports a fresh
`scratch_resolver_fix` total of 1,067 imports. Retained local evidence independently
records 1,067 emitted imports on candidate `13f201cc`, not the identical `8facda9`
binary or a new full-rebuild comparison. The exact 196-row attribution and the
calls/citations differences were not independently reconstructed here. Do not
promote these mixed-history count differences to new defects or accepted losses.

One detail remains inconsistent: the review says 170 currently routed code files,
whereas this run's retained ingest and drift captures both establish 171. Preserve
171 as the observed run count; source revision alone does not reconcile discovery
options or the live corpus's 188 rows. No additional source investigation was run.

The review requests a fresh `scratch_rebuild_wt5_r2` run and refers to an r2 goal
allowing read-only live AQL. That goal text was not found among the supplied
attachments. The executed authorization named only `scratch_rebuild_wt5` and
excluded live AQL. No rule violation is alleged by the review. Recommended next
step, **pending Todd's authorization**, is one fresh pass on this branch and PR in
`scratch_rebuild_wt5_r2`, permitting read-only live schema/count/report comparison,
while retaining all other safety and stop conditions. Do not create it, clear the
existing scratch database, or broaden live reads based on this review alone.
Fresh ingestion, adapter execution and the full verification/comparison sequence
remain prerequisites for claiming the intended fresh rebuild demonstrated. The
historical live unrouted list remains unavailable unless its envelope is supplied.

## Preparation actually performed

Used the existing isolated checkout `/tmp/hades-epic12`; the active `/opt/HADES`
checkout was untouched. Fetched main, checked out the existing rehearsal branch,
and merged `origin/main`. The only conflict was adjacent changelog entries; both
the rehearsal entry and incoming fix entries were retained. The branch's only
changes relative to main remain `docs/rebuild-runbook.md` and `CHANGELOG.md`.

```bash
git fetch origin
git checkout docs/rebuild-runbook
git merge origin/main --no-edit
CARGO_TARGET_DIR=/tmp/hades-lsp-fix/target CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo build -p hades-cli --bin hades
cp /tmp/hades-lsp-fix/target/debug/hades /tmp/hades-rebuild-rehearsal/2026-09-22/hades
```

Cargo reported 3.83 seconds for this private build; total preparation wall time
was not captured. Nothing was installed. The capture wrapper set
`CARGO_TARGET_DIR=/tmp/hades-rebuild-rehearsal/2026-09-22/analyzer-target` and
`PYTHONDONTWRITEBYTECODE=1` for every command below, keeping analyzer build output
private and preventing Python bytecode writes to the source tree.

## Commands and phase timings

The existing scratch database was reused: **do not replay database creation from
the historical record**. Commands below are the exact resumed invocations, run
from `/tmp/hades-epic12`; no `--force` or analysis downgrade was used.

| Phase | Start (CDT) | Seconds | Exit |
| --- | --- | ---: | ---: |
| live-collections-before | 08:45:42 | 0.0183 | 0 |
| scratch-collections-before | 08:45:42 | 0.0191 | 0 |
| ingest-1 | 08:52:02 | 48.7118 | 0 |
| adapter-dry-run | 08:53:09 | 0.2311 | 0 |
| adapter-write | 08:53:34 | 0.4952 | 0 |
| drift | 08:54:01 | 0.1042 | 0 |
| validate | 08:54:02 | 0.3061 | 0 |
| query-default | 08:54:03 | 4.0631 | 0 |
| query-codebase | 08:54:02 | 6.1033 | 0 |
| metadata-embeddings | 08:54:44 | 0.0128 | 0 |
| metadata-codebase_embeddings | 08:54:41 | 0.0156 | 0 |
| adapter-report | 08:54:40 | 0.0095 | 0 |
| scratch-collections-after | 08:54:00 | 0.0203 | 0 |

Independent read-only checks overlapped; table order is logical, not strict start-time order.

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db WeaverTools_v5 db collections
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db scratch_rebuild_wt5 db collections
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades ingest /opt/weavertools/WeaverTools --db scratch_rebuild_wt5
HADES_CONFIG=/home/todd/.config/hades/hades.yaml python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5 --repo /opt/weavertools/WeaverTools --dry-run
HADES_CONFIG=/home/todd/.config/hades/hades.yaml python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5 --repo /opt/weavertools/WeaverTools
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db scratch_rebuild_wt5 codebase drift /opt/weavertools/WeaverTools
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db scratch_rebuild_wt5 codebase validate
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db scratch_rebuild_wt5 db query 'state persistence' -n 5
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db scratch_rebuild_wt5 db query 'state persistence' -c codebase -n 5
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db scratch_rebuild_wt5 db aql 'FOR e IN @@collection LET invalid = !IS_STRING(e.model) OR e.model == "" OR !IS_STRING(e.model_hash) OR e.model_hash == "" OR !IS_NUMBER(e.dimension) OR e.dimension <= 0 OR e.dimension != FLOOR(e.dimension) COLLECT AGGREGATE total = SUM(1), invalid_rows = SUM(invalid ? 1 : 0) RETURN {total, invalid_rows}' --bind '{"@collection":"embeddings"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db scratch_rebuild_wt5 db aql 'FOR e IN @@collection LET invalid = !IS_STRING(e.model) OR e.model == "" OR !IS_STRING(e.model_hash) OR e.model_hash == "" OR !IS_NUMBER(e.dimension) OR e.dimension <= 0 OR e.dimension != FLOOR(e.dimension) COLLECT AGGREGATE total = SUM(1), invalid_rows = SUM(invalid ? 1 : 0) RETURN {total, invalid_rows}' --bind '{"@collection":"codebase_embeddings"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db scratch_rebuild_wt5 db get wt_ingest_report latest
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/2026-09-22/hades --db scratch_rebuild_wt5 db collections
```

## Observed results

- Ingest: exit 0, `success:true`; 171 code files and 46 documents processed with
  zero failures, all skipped as unchanged in the structural/document passes.
  Rust-analyzer nevertheless ran for 161 files and stored 4,409 symbols and 7,754
  edges, one crate, zero store errors, `store_failed:false`, no failed files or
  workspaces. Relationship stage passed; 1,067 import and 153 Python-call edges
  were reported in this ingest envelope. Those are run output counts, not the
  retained collection totals below.
- Discovery: 171 code, 46 documents, 42 unrouted paths (listed below).
- Adapter: dry-run and write both exit 0; server acknowledged **2,177 rows plus
  one report**. `dangling_documents=0`, `dangling_code=0`,
  `cites_sources_with_no_file_node=0`, `declared_in_targets_with_no_document_row=0`.
  The adapter report explicitly states that absent older rows are retained and
  stale retirement was not performed.
- Drift: `clean:true`, 171 matched nodes, zero changed/unverifiable/uningested or
  stale nodes for this root. The code-only drift command's unhandled list is
  distinct from unified ingestion's unrouted list; documents have their own route.
- Validation: **14 queried invariants passed, zero failed; three invariants are
  delegated to database/unit-test guarantees and were not queried**. Do not call
  this 17 runtime checks.
- Semantic queries: `state persistence` returned five results from `default` and
  five from `codebase`; model identity reported by both was
  `/bulk-store/books/models/jinaai--jina-embeddings-v4`, dimension 2048. This is
  retrieval execution evidence, not a scored retrieval-quality evaluation.
- Exhaustive metadata scans: all **1,050 document embeddings** and **1,976 code
  embeddings** have nonempty string `model`/`model_hash` and a positive integer
  `dimension`; zero invalid rows. This checks presence/types, not a recomputation
  of model hashes or an independent validation of vector provenance.

Existing non-fatal LSP close warnings (`incomplete or oversized LSP header`,
`LSP reader terminated`, `LSP client dropped`) appeared at teardown; no store or
relationship failure occurred. No implementation fix was made in this rehearsal.

## Collection comparison

Live reads were taken before resumed ingestion, scratch reads after adapter write.
These are sequential observations, not a frozen live snapshot. Differences are
recorded without inventing a cause or treating them as approved data loss.

| Collection | WeaverTools_v5 | scratch_rebuild_wt5 |
| --- | ---: | ---: |
| `chunks` | 1050 | 1050 |
| `codebase_calls_edges` | 3431 | 3430 |
| `codebase_chunks` | 1993 | 1976 |
| `codebase_defines_edges` | 4868 | 4868 |
| `codebase_embeddings` | 1993 | 1976 |
| `codebase_files` | 188 | 171 |
| `codebase_implements_edges` | 68 | 68 |
| `codebase_imports_edges` | 1067 | 1263 |
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

The six differing collection counts are not fully reconciled (see the attributed
review-seat explanation above): calls (one fewer), files,
chunks and code embeddings (17 fewer each), imports (196 more), and adapter
citations (four fewer). The original live source/options and retained-row history
were not reconstructed. Do not truncate, retire or relabel rows to make counts
match. In particular, the existing scratch database contains history from the
older writer, so this is not proof of a clean rebuild from empty on current main.

| Comparison item | Live | Scratch |
| --- | --- | --- |
| Adapter dangling citations | Unknown: original live command allowlist excludes report reads | 0 documents; 0 code |
| Unified-ingest unrouted set | Unknown: original live ingestion envelope not provided | 42 paths below |

Remaining owner/reviewer evidence: the live `wt_ingest_report/latest` report and
the historical live ingest envelope. A read-only `db get wt_ingest_report latest`
against `WeaverTools_v5` would supply the former, but is outside the original
live allowlist; it was **not executed**. The latter cannot be recovered from
collection counts. If a current-main fresh-from-empty rehearsal is required,
that needs a separately approved scratch target; the existing database is retained.

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
configuration changes. The resumed scratch database is **not** an approved cutover target.

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


## Evidence and historical checkpoint

The September 21 failures, commands, timings, original comparison and complete
stdout envelopes remain immutable in
[the prior checkpoint](https://github.com/toddwbucy/HADES/blob/35445db92da75bc1131c739ebdff8bb667d765de/docs/rebuild-runbook.md).
Those runs failed under `5ee16d3`: 668.7475 s with incomplete LSP enrichment and
4.9374 s with a relationship-endpoint failure. They are not rewritten as passing.
The raw local captures remain under `/tmp/hades-rebuild-rehearsal/`; the new dated
captures are separate under `/tmp/hades-rebuild-rehearsal/2026-09-22/`.
Large stdout envelopes were removed from the active runbook following review;
the original Git revision and local files preserve them.

Current key captures (SHA-256):

| Capture | SHA-256 |
| --- | --- |
| `ingest-1.stdout` | `33d2c38df895d40c25d8e9d726477f93f8dbcecc7ab66368c42bfea6ddeb86d1` |
| `ingest-1.stderr` | `b154de501f177f2bed55b460f9b3fc7b95200a61e886bfde0d8fc382f1e967ff` |
| `adapter-dry-run.stdout` | `5d018dc9bb72eff55ae4a01127413aec7e9571407b876fc537edbd3b48895df5` |
| `adapter-write.stdout` | `0782b4890096f8cd3cb3e6453e3007835e38f1fa77dd73705547438c7b462888` |
| `adapter-report.stdout` | `5c1605df9b624b0ef1788155ab4a28d3c126d6a7b2bb1334eaba8eb123475bda` |
| `drift.stdout` | `5e9b4e1ae8dbbf29c93f03a2b438d51489ad7e43bb68bb882e92f6bc61a22557` |
| `validate.stdout` | `1997abdd1c9f890326f88815b4198b3d4e9af462e562177447f2c9df89ce3a11` |
| `query-default.stdout` | `ee3a311c5dfcc5187231d907efad563eda6fdda2054432ed72b92b22220a1ff6` |
| `query-codebase.stdout` | `14ed02e06e05c3e37163fb200c4e31e9aa8dd0c656cf0cdbfd9fc4690e8dac3d` |
| `metadata-embeddings.stdout` | `29ac321ab03d70904e4cc993e132fab6e604e1a775187a37df45d8d4d7ac5bc6` |
| `metadata-codebase_embeddings.stdout` | `d80a88014e7c1dd4724ba813a809ed7dd14600676873ef0ec9226d9f385c2895` |

No production write, database creation/drop, force-reingest, deployment, service
restart, configuration change, snapshot, cutover or rollback occurred during this
resumption. All scratch operations have exited. No #164 refactor or broader audit
was started. The PR stays draft pending review and resolution of the comparison
limitations above; CI cannot substitute for those missing records.
