# WeaverTools rebuild rehearsal runbook

## Current checkpoint — 2026-09-22, TOML scope aligned

**All requested counts match live except the documented rust-analyzer ±1 call-edge
variance; every edge collection has zero dangling endpoints.**
No production cutover was performed or authorized.
PR #160 is open for review; no automatic merge is queued.

Todd authorized the incremental `--unparsed-ext toml` pass on existing
`scratch_rebuild_wt5_r2` after the
[15:18:59 UTC review](https://github.com/toddwbucy/HADES/pull/160#pullrequestreview-5280077711).
The flag is required to reproduce live's ingestion scope: its 17 TOML rows use
`analyzer: raw-text`, `tier: text`, one chunk each, as verified by the review seat.
Four `conforms:` citations originate in those files. This was a runbook parameter
omission, not an ingestion regression.

- Main implementation: `90bab3ae8c46ba930d162bda9a3d8d5f61b802c9`.
- Starting documentation revision: `26eae4786c409cb376403ce5764cc3a54613ba38`.
- Private binary built at `8facda9`, same implementation as main; SHA-256
  `c2988d354a395cccb790ee3b4e1bc5ce23bf707c560ae3ef4153c74ad375d463`.
- Source: `/opt/weavertools/WeaverTools` at `f03142a85e1c6bf088b6299063455272df93d68d`.
- Incremental evidence: September 22, 2026, 11:32–11:34 America/Chicago
  (CDT, UTC−05:00); raw log timestamps use UTC.
- Starting-head [CI passed](https://github.com/toddwbucy/HADES/actions/runs/35745942301).
  Final-head CI is recorded separately in the PR body.

## Rebuild sequence and observed execution

The fresh database was created once in the preceding phase, after a 404/1228
absence check. Applied `services/adapters/weavertools/schema.yaml`: 35 collections,
all data collections empty, 20 schema rows exactly equal to live excluding
`_id`/`_rev`, including relation metadata and the named graph. The exact creation,
schema and initial-ingest commands and captures remain in the
[immutable fresh-run record](https://github.com/toddwbucy/HADES/blob/26eae4786c409cb376403ce5764cc3a54613ba38/docs/rebuild-runbook.md).
Do not recreate this retained database.

For a future separately authorized fresh rebuild, follow the creation and schema
steps in the immutable fresh-run record above using a newly approved empty database.
Use that new target for ingestion with `--unparsed-ext toml`, then adapter dry-run
and write. The later command block targets retained `scratch_rebuild_wt5_r2` and
is **incremental-only**, not a fresh-database creation or rebuild procedure.
The sequence actually observed was fresh ingestion without that flag, then the
incremental correction. A single fresh invocation with the corrected flag was
not separately repeated. No `--force`, truncation or code correction was used.

Incremental ingestion took **55.9063 seconds**: 188 code files completed, 171
skipped unchanged, 17 newly embedded TOML files/chunks; all 46 documents skipped
unchanged; zero failures. Relationships and rust-analyzer enrichment still run
on an incremental pass: 4,409 symbols / 7,754 edges, zero store errors, no failed
files/workspaces or relationship error. Adapter dry-run and write passed; the
write acknowledged 2,181 rows plus one report. Code metadata scan checked all
1,993 rows, with zero invalid model/hash/dimension fields.

Prior fresh-phase evidence remains applicable to unchanged behavior: ingestion
705.6009 seconds with no skips, clean drift on the original 171 routed files,
14 queried validation checks passing (three database/unit guarantees not queried),
five results from each search profile, and valid metadata on all 1,050 document
embeddings. These checks were not represented as rerun after the TOML addition.
Query execution is not retrieval-quality acceptance; metadata presence/types are
not proof of vector provenance. Non-fatal LSP teardown warnings remain recorded.

### Exact incremental commands and timings

Run from `/tmp/hades-epic12`, with explicit config in each captured invocation.
The capture wrapper also sets
`CARGO_TARGET_DIR=/tmp/hades-rebuild-rehearsal/r2-toml-2026-09-22/analyzer-target`
and `PYTHONDONTWRITEBYTECODE=1`; no source-tree build output was requested.

| Phase | Start (CDT) | Seconds | Exit |
| --- | --- | ---: | ---: |
| ingest-1 | 11:32:32 | 55.9063 | 0 |
| adapter-dry-run | 11:34:20 | 0.2333 | 0 |
| adapter-write | 11:34:20 | 0.5095 | 0 |
| counts | 11:34:21 | 0.0197 | 0 |
| dangling-codebase_calls_edges | 11:34:21 | 0.0553 | 0 |
| dangling-codebase_defines_edges | 11:34:21 | 0.0911 | 0 |
| dangling-codebase_implements_edges | 11:34:21 | 0.0105 | 0 |
| dangling-codebase_imports_edges | 11:34:21 | 0.0274 | 0 |
| dangling-wt_asserts_edges | 11:34:21 | 0.0162 | 0 |
| dangling-wt_cites_edges | 11:34:21 | 0.0172 | 0 |
| dangling-wt_declared_in_edges | 11:34:21 | 0.0189 | 0 |
| dangling-wt_defines_edges | 11:34:21 | 0.0098 | 0 |
| dangling-wt_draws_edges | 11:34:21 | 0.0102 | 0 |
| dangling-wt_elects_edges | 11:34:21 | 0.0094 | 0 |
| dangling-wt_floor_link_edges | 11:34:21 | 0.0096 | 0 |
| dangling-wt_grounds_edges | 11:34:21 | 0.0102 | 0 |
| dangling-wt_holds_edges | 11:34:21 | 0.0090 | 0 |
| dangling-wt_parent_edges | 11:34:21 | 0.0085 | 0 |
| dangling-wt_party_edges | 11:34:21 | 0.0093 | 0 |
| dangling-wt_reads_edges | 11:34:21 | 0.0084 | 0 |
| dangling-wt_seam_edges | 11:34:21 | 0.0100 | 0 |
| dangling-wt_writes_edges | 11:34:21 | 0.0095 | 0 |
| adapter-report | 11:34:21 | 0.0084 | 0 |
| metadata-codebase | 11:34:21 | 0.0210 | 0 |

```bash
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades ingest /opt/weavertools/WeaverTools --db scratch_rebuild_wt5_r2 --unparsed-ext toml
HADES_CONFIG=/home/todd/.config/hades/hades.yaml python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5_r2 --repo /opt/weavertools/WeaverTools --dry-run
HADES_CONFIG=/home/todd/.config/hades/hades.yaml python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5_r2 --repo /opt/weavertools/WeaverTools
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db collections
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "codebase_calls_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "codebase_defines_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "codebase_implements_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "codebase_imports_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_asserts_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_cites_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_declared_in_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_defines_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_draws_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_elects_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_floor_link_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_grounds_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_holds_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_parent_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_party_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_reads_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_seam_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN @@collection FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null COLLECT WITH COUNT INTO dangling RETURN {dangling}' --bind '{"@collection": "wt_writes_edges"}'
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db get wt_ingest_report latest
HADES_CONFIG=/home/todd/.config/hades/hades.yaml /tmp/hades-rebuild-rehearsal/r2-2026-09-22/hades --db scratch_rebuild_wt5_r2 db aql 'FOR e IN codebase_embeddings LET invalid = !IS_STRING(e.model) OR e.model == "" OR !IS_STRING(e.model_hash) OR e.model_hash == "" OR !IS_NUMBER(e.dimension) OR e.dimension <= 0 OR e.dimension != FLOOR(e.dimension) COLLECT AGGREGATE total = SUM(1), invalid_rows = SUM(invalid ? 1 : 0) RETURN {total, invalid_rows}'
```

## Live comparison and named exception

Live counts below are the preserved 09:59 CDT read; current scratch counts were
captured after the incremental adapter write at 11:34 CDT. This is not an atomic
snapshot of the active live service. All 18 current edge collections were enumerated
from `db collections` and individually queried for missing `_from` or `_to`
documents: **zero dangling edges in every collection**. This supersedes the prior
report-only endpoint evidence; the review's reference to 13 collections is not
used as the scan inventory.

| Collection | WeaverTools_v5 | scratch_rebuild_wt5_r2 |
| --- | ---: | ---: |
| `chunks` | 1050 | 1050 |
| `codebase_calls_edges` | 3431 | 3430 |
| `codebase_chunks` | 1993 | 1993 |
| `codebase_defines_edges` | 4868 | 4868 |
| `codebase_embeddings` | 1993 | 1993 |
| `codebase_files` | 188 | 188 |
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
| `wt_cites_edges` | 504 | 504 |
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

Only calls differ: 3,431 live / 3,430 scratch. The review seat compared semantic
endpoints and isolated the test-cfg edge
`spawn.rs::last_word_tests::the_last_word_survives_a_real_death → spawn.rs::LastWord::take`.
Its three #163 runs reported rust-analyzer totals 7,754 / 7,754 / 7,755; this is
rust-analyzer's own ±1 variance, not a HADES defect. This attribution is reviewer
provided, not a newly repeated variance experiment. The current pass emitted 7,754.

Both stored adapter reports show zero dangling document/code citations and missing
source/declaring-document targets. Scratch now reports `citing_files_out_of_scope=0`,
matching the live report. Live report `_rev` `_mJ-mKhS--g` lacks a source revision
and run timestamp, so report parity alone is historical evidence. The new exhaustive
endpoint scans above apply to scratch only. No live writes were performed.

## Assertion identity and cutover compatibility

PR #110 changed `wt_assertions` keys from slugs such as
`admin-no-library-surface` to **`v2-n-<sha256>`**, retaining the readable slug in
`ident` and recording `adapter_identity_version: 2`. Current adapter code hashes
the framed JSON node identity, rather than sanitizing the slug. The review seat
verified the live/scratch identity difference and citation equivalence on
`(path → ident)`; equal row counts do not imply equal document IDs.

**External readers that address assertions by the old `_key` break at cutover.**
Inventory and migrate those readers or their references to the new identity scheme
before accepting a production switch. This rehearsal does not implement an alias,
compatibility shim or external-reader migration. Crate-aware semantic import
correctness remains its existing separate follow-up.

## Current unrouted paths and adapter notes

The incremental envelope has **25 unrouted paths**, down from 42 because all 17
TOML files are now routed. The original live unified-ingest envelope remains
unavailable, so historical unrouted-set parity is still unknown. Collection counts
cannot reconstruct that list. Paths below are relative to the source root.

| Path | Reason |
| --- | --- |
| `Cargo.lock` | no handler for extension |
| `LICENSE` | no extension and no shebang |
| `crates/weaver-web/LICENSE` | no extension and no shebang |
| `crates/weaver-web/assets/htmx.min.js` | no handler for extension |
| `crates/weaver-web/assets/sse.js` | no handler for extension |
| `crates/weaver-web/assets/style.css` | no handler for extension |
| `crates/weaver-web/assets/surfaces/instrument.css` | no handler for extension |
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

Current adapter coverage notes (retained as limitations, not silently repaired):

- document_notes: malformed node id in docs/crates/weaver-agents-PRD.md: 'WeaverTools'
- document_notes: unknown tag in docs/crates/weaver-agents-PRD.md: WeaverTools (ratified)
- document_notes: declaring_files_out_of_scope=0
- code_notes: citing_files_out_of_scope=0
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
4. During a separately authorized cutover, update all three database selections:
   `ExecStart` in `~/.config/systemd/user/hades-daemon.service` currently carries
   `--database WeaverTools_v5` and `--mcp-dbs WeaverTools_v5,bident_v4`; replace
   the WeaverTools target in both, retaining the intended bident selection.
   `~/.config/hades/daemon.env` carries `HADES_DATABASE=WeaverTools_v3` (stale);
   set it to the same approved replacement. These locations/values were inspected
   read-only for this report. Then run `systemctl --user daemon-reload` and
   `systemctl --user restart hades-daemon`. Neither edits nor commands were executed
   here. Verify CLI/MCP database identity, both retrieval profiles, integrity,
   external assertion readers and writer behavior before restoring normal operation.
5. If verification fails, pause writers and restore both the previous binary and
   previous database selections in daemon and MCP configuration. Reconcile writes
   made after cutover before rollback to avoid silent loss; restore compatible
   checkpoints as needed. Reverify service behavior. Keep old databases and
   snapshots until Todd completes retention and acceptance review.


## Evidence preservation and stop

The [original failed rehearsal](https://github.com/toddwbucy/HADES/blob/35445db92da75bc1131c739ebdff8bb667d765de/docs/rebuild-runbook.md),
[retained-scratch recovery](https://github.com/toddwbucy/HADES/blob/9cbab044714102c3d89fd75feaed7ac0b1f8c51f/docs/rebuild-runbook.md)
and [fresh run before TOML alignment](https://github.com/toddwbucy/HADES/blob/26eae4786c409cb376403ce5764cc3a54613ba38/docs/rebuild-runbook.md)
remain immutable. Their captures and hashes were not modified. Their earlier
unexplained-difference status is superseded by the named explanations and current
measurements here, not retroactively rewritten.

New captures: `/tmp/hades-rebuild-rehearsal/r2-toml-2026-09-22/`; local temporary
storage is not a durable off-server archive. This authorized continuation stops
when the PR body records these counts and final-head CI is green. All scratch
commands are finished. Final CI/review disposition is recorded in the PR body;
a skipped draft review is not approval. No merge, production changes, cutover,
snapshot, service restart, #164 refactor or broader audit work was performed.
Both scratch databases remain for owner review. Historical live unrouted evidence
is still unavailable; it is not an additional execution gate for this continuation.

### Capture SHA-256

Digests bind `.json || .stdout || .stderr` in that order for each phase.

| Phase | SHA-256 |
| --- | --- |
| ingest-1 | `6feb8f628eb9452a9f27781a30a3bd5c1d9e5364eb18f323d138fd7f3247ea99` |
| adapter-dry-run | `9af8f9524f55282646468a8b9022f352955c4783b9100b2f929b83f5656cd9e2` |
| adapter-write | `c43890a29212adad6b17a1c28cd774b3686308cc3a9fb50f405cfa137c9f7fee` |
| counts | `b7abf636c21b3bf8508379d7296737fc11e88ce68ff8c64cce1b6a732b70da2e` |
| dangling-codebase_calls_edges | `2ac1c41dc51010cebe09616ba04d38c06502bfea0c0eb316b2625c772ea2e545` |
| dangling-codebase_defines_edges | `d37551cf249165dec9513a13581a7e0c4adac029e7f0b75a9ef5f5fa353a24b1` |
| dangling-codebase_implements_edges | `366eabe9110038714e883e0d5d0a68dcc50a1e588cd819236395096172e4a579` |
| dangling-codebase_imports_edges | `a80ddfadb4f7706172e7622ef860a4f2436be0a485386a4f555f8ebd4700c798` |
| dangling-wt_asserts_edges | `d5ab750f4be2b5cda4fc8ffb9038fb4f4799e1eb4f50ac17563a1fdd638450f5` |
| dangling-wt_cites_edges | `74dc5a71b00092addc12d27746f4ce9c6cf5bbb286852948270397bee911455c` |
| dangling-wt_declared_in_edges | `0efa7eff8e7580737a752cdb4bc96d9d325f312d7fab8fb5a1c7dbd791e259d1` |
| dangling-wt_defines_edges | `933cc67db59d32b33a0485aef8a689adacd1a155a48840147bca11485fa0a27f` |
| dangling-wt_draws_edges | `6c8bbc4c9edec42e902dd7f04895bf94b634639a7264fc83dd9171621818f63c` |
| dangling-wt_elects_edges | `2ad765733f16ff7826561d20dc8e0845db0ccbd9ff18878a3a932a920a126b14` |
| dangling-wt_floor_link_edges | `ee7af82d2747b671ae2951ac12f4852c7c6ab04f17312ce1228275cde7fc5a02` |
| dangling-wt_grounds_edges | `dcfb26a91807c2a0398205ac487133c76902edcb83d66f4f13074d70824830e7` |
| dangling-wt_holds_edges | `392db474c96d71905e4e8754cac6f92c6e4c969a56f5393807f40e3c83de5013` |
| dangling-wt_parent_edges | `f9b900e90ca6d010c10adf9c509c0d40ef4e90d4592c4f2d4ad98d13dda25482` |
| dangling-wt_party_edges | `9ef381c8ed34cffdc15dcb7759681f317af25cc48857d99a749602738a1afde2` |
| dangling-wt_reads_edges | `513088bb0ad6ae0647d8f74e4ceca9058230401d39d451c5d3ca24b0c84bc96d` |
| dangling-wt_seam_edges | `758f59924758a65610c164fe0a72932a4c40d0000867476dfb89d19a0ea8e0be` |
| dangling-wt_writes_edges | `4221bdf53cf34b58d0ff9456923e677118c71b335bc9deac330ab1ca1a419bdc` |
| adapter-report | `8f9d32dad2f2b2fb5cd78f2c66f5814bb57d7b7040a1d07eb71fe0da22ae3c35` |
| metadata-codebase | `b290726326431699776c5eb33b16b1cba3ed693ad097754c2b4f0c81abbc0d90` |
