# WeaverTools rebuild rehearsal runbook

**BLOCKED at the schema-equivalence gate (stop condition E). This is a partial
rehearsal, not a validated rebuild or cutover procedure.**

Recorded 2026-09-21, America/Chicago (CDT, UTC−05:00). Work began at 15:05:42;
the schema gate stopped execution before ingestion. Repository/build revision:
`5ee16d3ce10eb6af4727594babb47efeaf8b49a4`. Existing clone:
`/tmp/hades-epic12`; branch: `docs/rebuild-runbook`. Only the new database
`scratch_rebuild_wt5` was written. The live `WeaverTools_v5` was read solely
through allowed schema-list, graph-list and collection-list commands.
`bident_v4` and `/opt/weavertools` were not accessed or changed.

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

## Schema comparison and stopping decision

The complete `data` objects from `db schema list` are identical between live and
scratch, as are the `db graph list` objects. Both have 18 edge definitions and
one `codebase_graph`; the graph endpoint collection sets match. Both databases
have the same 35 user collection names.

**No schema difference has been established, and full equality has not been
established either.** `db schema list` projects definitions into names, counts,
and selected attributes; it omits the `meta` document. `db graph list` describes
Gharial topology, not the complete stored `hades_schema`. In particular,
`relation_order`, `model_type`, `feature_dim`, schema version/checksum, and all
stored definition fields cannot be fully compared using the allowed command
outputs. Matching counts or summaries are insufficient to clear this gate.

The live-read allowlist does not include `db get`, `db schema show`, or raw
collection export. No raw AQL or alternate HTTP read was used to bypass it.
**Owner decision required:** authorize read-only `db get hades_schema <key>` on
`WeaverTools_v5`, limited to the 20 expected schema rows: `meta`,
`named_graph__codebase_graph`, and each schema-defined
`edge__<edge-name>__<source-field-key>`. Read the corresponding scratch rows,
compare all semantic fields (exclude only Arango `_id`/`_rev`), and verify their
count against the collection count. Recommend that narrow read extension;
do not weaken equivalence to matching summaries. If it reveals a difference,
record the exact diff and stop again for Todd's choice; do not reconcile it.

## Collection comparison at the schema-only checkpoint

These are counts from sequential reads, not a transactionally frozen live
snapshot and not an end-of-rebuild comparison. Zero means the scratch collection
was created but has not been ingested. Counts cover every collection returned
by the CLI (system collections are outside this inventory).

| Collection | WeaverTools_v5 | scratch_rebuild_wt5 |
| --- | ---: | ---: |
| `chunks` | 1050 | 0 |
| `codebase_calls_edges` | 3431 | 0 |
| `codebase_chunks` | 1993 | 0 |
| `codebase_defines_edges` | 4868 | 0 |
| `codebase_embeddings` | 1993 | 0 |
| `codebase_files` | 188 | 0 |
| `codebase_implements_edges` | 68 | 0 |
| `codebase_imports_edges` | 1067 | 0 |
| `codebase_symbols` | 4868 | 0 |
| `documents` | 46 | 0 |
| `embeddings` | 1050 | 0 |
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

| Additional requirement | Live | Scratch |
| --- | --- | --- |
| Adapter dangling citations | Unknown: allowed reads do not expose report values | Not measured: adapter not run |
| Unrouted list | Unknown: no preserved live ingest envelope read | Not produced: ingestion not run |
| Drift / validation | Not run (outside live allowlist) | Not run: schema gate blocked |
| Default/codebase semantic queries | Not run | Not run: schema gate blocked |
| Every embedding has model/model_hash/dimension | Not checked | No embeddings yet; not a passing verification |

## Remaining rehearsal sequence — NOT EXECUTED

After owner authorization and a complete schema comparison, resume using this
same scratch database only if the gate passes. The following are the requested
next commands, **not commands proven to work by this runbook**:

```bash
/fastpool/venvs/cargo-target/debug/hades ingest /opt/weavertools/WeaverTools --db scratch_rebuild_wt5
python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5 --repo /opt/weavertools/WeaverTools --dry-run
python3 services/adapters/weavertools/write_graph.py --db scratch_rebuild_wt5 --repo /opt/weavertools/WeaverTools
```

Keep the original canonical root. Record each command, selected configuration,
source revision, elapsed time, stdout/stderr and full ingestion envelope,
including unrouted files. The adapter uses TCP and its own `ARANGO_*` environment,
not the CLI YAML: verify that these identify the same authorized server before
running it, without changing server configuration or exposing credentials.
Then verify clean drift, successful validation, results for both semantic search
profiles, and complete embedding metadata in both embedding collections. Capture
adapter dangling citations and repeat the collection comparison. Do not substitute
empty collections or skipped tests for completed verification. Any failure gets
at most two attempts; record it and stop without fixing main or forcing ingestion.

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

No cutover, rollback, snapshot, restart, profile switch, forced ingestion, corpus
rebuild, or production write occurred. The scratch database is retained for Todd
to remove after review. No ingestion or adapter process is running.
