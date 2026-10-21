# Profile-bound background ingestion

Status: document jobs for [#225](https://github.com/toddwbucy/HADES/issues/225).
`ingest.start`, through the shared daemon/MCP handler, can ingest named document
files and the enabled document portion of directories into ready managed databases.
It uses the same [file pipeline](profiled-file-ingestion.md) as foreground ingestion.
Conventional and explicitly configured [late document policies](verified-late-passages.md)
retain the same pinned worker and atomic storage guarantees.
Public database setup and frontend activation remain separate work.

## Admission and worker handoff

The parent verifies the stored profile through the installation's registered
provider catalog before creating a job record. The job persists a compact
`hades-ingest-profile-pin-v1` receipt containing database name/incarnation,
profile revision, generation, artifact MD5 and workload space identities.
Digests bind the writer endpoint and complete validated metadata snapshot,
including every Arango record revision. Rewriting identical metadata therefore
invalidates an admitted job. Pins contain neither host paths nor model manifests.

The sealed configuration envelope advances to version 3, retaining its 64 KiB
limit. A pin is bounded to 16 KiB. It is excluded from ordinary YAML/env config;
only the explicit sealed handoff restores the worker's admission. Old envelopes
remain legacy-only. The inherited CLI refuses database/GPU overrides and ignores
ambient configuration, credentials and provider selection.

Before routing a tree, including an empty tree, the worker reloads and compares
the pin. Document connection repeats the comparison before provider verification
or extraction. Its existing transaction guard rejects profile changes during
preparation or commit. A provider reload is freshly verified by the worker;
identical artifact/behavior remains compatible, while changed identity cannot
substitute another model. Inference still checks its own serving observation.

## Results, cancellation and retries

Start/status responses expose `profile_pin`; the final job row retains the same
pin alongside the bounded CLI result and diagnostics. Explicit `force` uses the
existing atomic file replacement and metadata-preservation rules. Partial batches
fail honestly; prior committed documents remain committed.

The existing process owner still controls concurrency, overlapping source paths,
shutdown, child reaping and terminal record persistence. A changed profile causes
failure, never automatic refresh. Start a new job after inspecting the result;
there is no automatic retry of an uncertain commit or automatic resume of an
unowned historical job. Legacy `--batch`/`--resume` checkpoints remain unavailable
for managed databases.

Named code files and degraded code-enrichment overrides are refused at admission.
A documents-only profile skips the code portion of a mixed directory before any
analyzer starts. The final result lists those paths in `disabled_by_profile.code`,
preserves `unrouted` files and carries the same `profile_pin` as the parent job.
An enabled code workload still fails before corpus writes until verified code
ingestion is implemented; its failed job history remains available.

Managed job history uses a synchronous document collection with its primary
index. If concurrent first jobs race to create it, a failed creator revalidates
the existing collection without issuing another create; incompatible properties
or indexes still refuse admission. Completed setup retries permit and validate
that operational collection, preserving its rows. Initializing workspaces cannot
acquire job history through the managed job path.

## Validation

`profiled_jobs` tests strict compact pins, sealed handoff, database isolation and
changed metadata revisions. The matching CLI target launches real daemons and
workers against disposable databases/private services, checking file/directory
success, disabled-code reporting in mixed directories, explicit replacement,
stale/unpinned workers, provider/profile drift during extraction and shutdown cleanup.
Existing lifecycle and setup contracts cover cleanup and identical setup retry.

```bash
cargo fetch --locked
python3 scripts/test_isolated_database.py --docker --contract profiled_jobs
```

The documented native `--arangod` alternative is supported. Both targets also run
in the default strict CI matrix. These fixtures qualify coordination and storage;
they do not introduce new real-model quality measurements.
