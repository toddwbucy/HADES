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
- No maintained ArangoDB dump/restore procedure was found in the HADES deployment scripts inspected. Other backup mechanisms are being requested from the owner.

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
