# Model-bound workspace recovery

This supplements the [ThinkPad backup procedure](thinkpad-backup-restore.md) for
databases created through the internal [setup coordinator](../specs/recoverable-workspace-setup.md).
It is an administrative backup/diagnostic contract, not an automated migration
or a public frontend recovery feature. Existing databases are never adopted by
assigning them today's model checksum.

## Preserve the whole setup record

Before taking a consistent backup, stop setup, ingestion and other writers using
the existing workstation procedure. Keep these together in private storage:

- The logical database dump, including `hades_setup`, `hades_workspace`, document
  sources/chunks/vectors, job history and any authored/reading records.
- The complete installation setup ledger: `scope.json`, operation records,
  database-name claims and its empty `.lock` file. Preserve mode 0700 on its root
  and 0600 on files, owned by the service account. A copied lock file does not
  preserve a running process's lock; never restore into an active ledger.
- Protected provider registrations, their exact artifact manifests and either
  the approved model assets or an independently verified way to recover them.
  A model name or cache directory alone cannot establish artifact identity.
- Matching HADES binaries/revision, installation ID, provider routing and user
  grants. Preserve credentials privately and separately from portable reports.
- External ingestion originals and source references. Stored extracted text and
  reading uploads do not necessarily preserve every original input format.

Inventory and hash database records and ledger files before and after the dump
and copy. Reject a backup if writers changed either side. Include the manifest
and SHA-256 checksums in the private backup artifact. The existing ThinkPad
rehearsal predates model-bound setup; its success is not evidence that the new
ledger and registration files were backed up.

## Diagnose before retrying

Trusted installation code can call:

```rust
let report = setup.diagnose("original-operation-id").await?;
```

`setup` must already carry the authorized owner, installation scope, explicit
database endpoint and provider catalog. Diagnostics require an initialized,
protected ledger. They take its process lease, inspect only the named owner's
operation, and make no provider calls or changes to operation/database records.
Reports contain no endpoint, registry paths, credentials or model weights.

| Action | Meaning |
|---|---|
| `none` | The saved completion, current ownership, profile and minimal setup schema agree. This does not verify current provider availability or corpus integrity. |
| `retry_identical_request` | The acknowledged operation can enter its normal retry checks using the original exact request. This report is advisory; retry rechecks everything. |
| `inspect_required` | Ownership, metadata, schema or an earlier administrative hold is unresolved. Preserve the evidence for administrative review. |
| `start_new_operation` | This initialized ledger has no record of that operation ID. This does not grant permission to adopt an existing database. |

Reports distinguish a missing/changed name reservation, unacknowledged create, missing database, changed database
incarnation, unverifiable marker, incomplete initialization, pending provider
verification, incomplete final publication and ready setup. Diagnostic failure
or an unverified check must not be rendered as ready. A report never clears a
`needs_reconciliation` hold or publishes readiness itself.

## Restore and migration boundary

A ledger copy is valid only for its recorded installation and endpoint. Even at
the same name and endpoint, a recreated/restored database has a different database
incarnation. The old setup receipt cannot authorize it. Editing ledger scope,
ownership markers, profile generation or artifact identity to suppress that
failure is not a supported recovery procedure.

Restore a backup into isolated storage for comparison first. Verify every source,
authored record/reference, schema, profile/manifest and registration before planning
activation. Historical job PIDs, admission pins and serving receipts are observations;
do not resume old workers from a backup. Identical model bytes can be loaded again,
but a new provider load requires fresh verification.

Automated cross-host ownership rebinding, legacy provenance adoption and model
migration are still unavailable. Keep restored data and original evidence intact.
The [profile design](../specs/database-embedding-profiles.md#existing-databases-and-compatibility-changes)
requires an explicit offline rebuild or a separately implemented migration that
preserves source/authored/reading references; `ingest --force` across a corpus is
not such a migration.

## Validation scope

Service-free ledger tests prove diagnostics do not create missing scope/lock
files or alter operation records. Disposable ArangoDB tests cover interrupted
checkpoints, lost create acknowledgements, ready and damaged schema, owner
isolation, a copied ledger, missing databases and recreation under the same name.
They assert that operation bytes, metadata revisions and independent authored
content remain unchanged. No live corpus or installed service is used.

```bash
cargo test -p hades-core --lib workspace_setup
python3 scripts/test_isolated_database.py --docker --contract workspace_setup
```
