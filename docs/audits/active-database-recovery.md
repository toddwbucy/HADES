# Active database recovery coverage (#63)

## Finding

P1 operational recovery gap in epic #12: the inspected legacy backup mechanism targets a different dataset from the active ArangoDB data. Recovery of the current HADES graphs is unverified. This does not establish that every possible external backup system is absent.

## Read-only evidence (2026-09-20)

- Active user `arangod.service`: PID 3014, zero restarts at inspection; unit selects the per-user configuration.
- Its database directory resolves to the per-user ArangoDB data directory on ZFS dataset `dbpool/home/todd`.
- `zfs list -t snapshot -r dbpool/home/todd` returns no snapshots (successful query).
- The legacy `backup_dbpool.sh` explicitly selects only `olympus`, `postgresql`, and `arangodb` datasets. It omits the active data's dataset.
- Existing replicated ArangoDB snapshots under `bulk-store/backups/dbpool/arangodb` are dated October 22–25, 2025. These belong to the legacy dataset and are not proof of recoverability of the current per-user database.
- Inspected system timers include configuration Git sync and pool scrubs; neither supplies evidence of a current database backup. No user timers were listed.
- No maintained ArangoDB dump/restore procedure was found in the HADES deployment scripts inspected. The owner subsequently identified `/bulk-store`; see the artifact verification below.

## Impact

A current graph/data loss could require rebuilding from sources or recovering from an unknown/stale backup. The backup path must follow the effective live data location; a named ArangoDB backup dataset alone is insufficient evidence. No outage or data loss has been caused or observed by this audit.

## Acceptance

1. Identify the owner and actual backup mechanism, source coverage, retention, destination, last successful artifact and monitoring; explicitly resolve any alternate-backup evidence supplied by the owner.
2. Establish a supported backup strategy for the active database and needed configuration/artifacts, with consistency and compatibility boundaries documented.
3. Restore a representative synthetic database into a separately isolated server and verify documents, edges, indexes/schema and representative reads; document any gaps versus production recovery.
4. Define measurable recovery-point/recovery-time objectives and demonstrate monitoring of stale/failed backups.
5. Any production backup schedule/configuration deployment has a separately reviewed plan and authorization. Do not run the legacy script as an audit probe: it creates, receives and destroys snapshots.

No snapshots, packages, services or production database data were changed during discovery.

## Isolated restore rehearsal

The retained [synthetic result](restore-fixture-result.json) records a successful
ArangoDB 3.12.11 logical dump/restore between two sequential disposable servers.
Three synthetic nodes (including numeric vectors), two edges, named-graph
metadata, strict collection schema and a unique persistent index were restored.
Document/edge values, index definitions, schema and a two-hop traversal matched.
The restored server rejected a schema-invalid document (400) and a duplicate
indexed value (409). Both private server groups exited through the runner's
cleanup. No production endpoint, credentials or corpus were used.

Reproduce from an isolated checkout with existing matching binaries:

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 240 python3 docs/audits/repros/restore_fixture.py \
  --bin-dir /path/to/existing/arangodb/bin
```

The probe imports the maintained isolation helpers, uses new private directories
and Unix sockets, disables configuration discovery, limits each process to one
CPU with nice 10 and an 8 GiB address-space ceiling, and limits dump/restore to
one thread. It records binary/probe/helper and dump-file hashes. Artifacts remain
in a new `/tmp/hades-restore-*` directory. The result's source interval includes
fixture creation and dump; target interval covers restore only. Neither interval
is a production RTO estimate.

This small quiescent logical-restore rehearsal does not verify current backup
coverage, concurrent-write consistency, users/permissions, production scale,
vector indexes, application model/checkpoint compatibility, or loss of the host.
P1 #63 remains open. A full recovery plan needs actual backup evidence, an owner,
RPO/RTO and a separately authorized deployment/verification plan.

## Owner-identified backup destination

The owner confirmed that ZFS backups are stored under `/bulk-store`. Read-only
enumeration of that pool found the replicated ArangoDB snapshots listed above
(October 22–25, 2025), but no replica of the active `dbpool/home/todd` dataset.

A separate logical dump exists at
`/bulk-store/backups/dbpool/arangodump-20260808`. Six `dump.json` manifests record
creation on August 9, 2026 UTC (August 8 local time). One database directory has
a September modification timestamp, but its manifest still records the August
dump; directory mtime alone does not establish a newer backup. Additional older
logical dumps exist under `/bulk-store/arangodb_dumps`. Only metadata and manifest
fields were inspected, not raw database documents or credentials.

This inspection establishes that backup artifacts and manifests are present.
It does not establish their validity or completeness, nor the current
backup schedule or coverage of recent graphs/changes. The subsequent restore
below verifies one selected artifact; the other artifacts remain untested. The synthetic rehearsal above must not be represented as a
restore test of these production backups.


## Actual historical backup restore (2026-09-20)

The owner's backup destination contained a small `WeaverTools_v3` logical dump
inside `arangodump-20260808`: 42 files, 39,755 stored bytes, manifest creation
`2026-08-09T00:11:57Z`. The complete parent backup set occupies approximately
14 GiB; only this small database artifact was selected and restored.

The [retained result](weavertools-backup-restore-result.json) and
[replay probe](repros/restore_weavertools_backup.py) record:

- 20 collections and 733 total rows restored, including 426 edges in 12 edge
  collections. Every document field except the server-generated `_rev` matches
  the dump, including original keys, endpoints and application fields.
- Collection types and schemas match. There are zero secondary indexes in this
  artifact; this run does not demonstrate secondary-index recovery. The separate
  synthetic rehearsal above covers a unique index and enforced schema.
- All restored edge endpoints resolve: zero dangling edges.
- Source file hashes are unchanged after the run. Result hashes bind every dump
  file, both binaries, runtime ICU/timezone data, probe and isolation helper. No document text is published.
- ArangoDB 3.12.11 performed the restore in approximately 0.22 seconds, excluding
  server startup, validation and recovery of the rest of the application. This
  is not a production recovery-time estimate.

Repeat from an isolated checkout with matching existing binaries and their adjacent
`icudtl.dat`, `icudtl_legacy.dat` and `tzdata/` runtime files:

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 210 python3 docs/audits/repros/restore_weavertools_backup.py \
  --dump-dir /bulk-store/backups/dbpool/arangodump-20260808/WeaverTools_v3 \
  --bin-dir /path/to/existing/arangodb/bin
```

The probe accepts only the selected adapter collection layout (20 collections,
`wt_*` plus `hades_schema`), with bounded file counts, file sizes, expanded gzip
bytes and row counts. It reads backup inputs, copies them into a new private
`/tmp/hades-backup-restore-*` directory, and restores into a newly created database
on its own Unix-socket-only server. Parent and children use one CPU with lowered
priority; children have 8 GiB address-space limits, one restore thread and bounded
timeouts. Logs, copied contents and restored data remain private under mode 0700.
Both owned process groups stopped before the successful result was written.
The first exploratory preflight stopped before launching a server because the
allowlist initially omitted `hades_schema`; correcting that verified metadata
assumption allowed both exploratory and retained runs to complete.

**Remaining limits:** this proves this historical artifact can restore its
recorded contents. It does not establish current source coverage, scheduled
backup success, retention/monitoring, recent-data recovery, live consistency,
users/ACLs, named graphs, model checkpoints, all-database recovery or recovery
after loss of this host. Issue #63 stays open. Owner RPO/RTO targets have been
requested; deploying a production backup job still requires its own reviewed plan.


### Replay evidence hardening

The retained run uses explicit validation failures, which remain active under
`python -O` / `PYTHONOPTIMIZE`; subprocess waits are separate from validation
expressions. It copies and hashes the probe, isolation helper, binaries and
required ICU/timezone data before execution, re-executes the private probe copy,
and imports/runs only those copies. Execution files have no write permission and
their hashes are checked again after verification. This isolates the run from
later changes to the original paths. It is not a hermetic operating-system or
shared-library snapshot and does not defend against a hostile process with the
same account's authority.

The final retained run passed with Python optimization enabled. A separate private
copy with a duplicated dump key was rejected under `-O` during preflight, before
starting a server; it produced no passing result. Earlier staging attempts exposed
missing adjacent ICU and timezone resources and exited during server startup;
owned-process cleanup ran and neither attempt produced a passing result. The
recorded successful result belongs to the complete staged implementation.
