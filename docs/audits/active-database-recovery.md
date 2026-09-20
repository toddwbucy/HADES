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
