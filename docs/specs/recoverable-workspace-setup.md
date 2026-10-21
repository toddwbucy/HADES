# Recoverable workspace setup

This internal [#225](https://github.com/toddwbucy/HADES/issues/225) slice composes
registered provider verification, database creation, ownership records, profile
initialization, and readiness. `workspace_setup::WorkspaceSetup` has no public
CLI/MCP/job/frontend caller. It supports the conventional cosine document path;
code, late chunking, graph presets, legacy adoption, and model migration remain
outside this setup schema.

## Authority and confirmed intent

Installation code constructs the coordinator with:

- An existing absolute private `0700` ledger directory owned by the service account.
- A stable installation ID and the already-authorized owner's ID.
- An explicit ArangoDB `_system` client, with database-creation authority.
- An installation-owned [provider catalog](registered-provider-discovery.md).

These settings are not browser input. A `SetupRequest` contains an operation ID,
database name, initializing revision-one profile, and artifact manifest. Model
selection can use `VerifiedProvider::document_profile(...)` to construct the
profile before confirmation. Setup validates the complete intent and verifies
the registered provider again; a serialized receipt is never proof of authority.

Database names are 1–64 ASCII letters, digits, underscores or hyphens, beginning
with a letter. Operation/authority IDs are 1–128 ASCII letters, digits,
underscores or hyphens. The request hash is SHA-256 over typed intent with
canonical manifest ordering. Changing the database, profile, policy, generation,
or model under an existing operation ID is refused. Manifest MD5 and embedding
space identity retain their existing meanings.

## Local durability and serialization

The ledger works before any workspace exists. It stores `scope.json`, one
`op_<id>.json` per accepted operation, and a `db_<name>.json` reservation. Scope
binds the installation and a hash of the explicit endpoint; each operation and
reservation also binds its authorized owner. Credentials are not written to the
ledger or portable receipts. Endpoint aliases are not inferred to be equivalent.

All metadata is single-link `0600` regular files. Descriptor-relative opens refuse
symlinks, including ledger ancestors; records are strictly typed and bounded to
2 MiB. Publication writes a new file, syncs it, atomically renames it, then syncs
the directory before the associated database effect. Incomplete `pending_*`
files are never read as operation records. They are not automatically cleaned up.

One nonblocking `.lock` lease serializes all setup in this ledger, including
different database names. Contention returns a busy error for explicit retry.
Filesystem work uses the blocking pool and retains the lease through cancellation.
The lease spans model and database work. All provisioning for this installation
must use this ledger; a second independent ledger is not a coordination mechanism.
Raw administrative DDL and software controlling the service account remain
outside this ownership boundary.

## Checkpoints and recovery

Receipts expose `validating`, `creating`, `initializing`, `ready`, `failed`, or
`needs_reconciliation`, plus the last durable step and a bounded diagnostic code.
`inspect(id)` reads the authorized owner's local receipt without contacting a
database or model. Diagnostics do not copy server response bodies or secrets.

| Checkpoint | Retry behavior |
| --- | --- |
| Validate | Refuse an existing database; verify the saved model against registration and its actual provider. |
| Create, without an acknowledged database ID | Stop for reconciliation, even if the name is absent. An in-flight create may still complete; existence alone proves nothing. |
| Marker, with a durably acknowledged ID | Recheck that database incarnation. Create/reconcile only an empty marker collection or the exact existing ownership marker. |
| Profile | Reconcile an empty metadata collection or exactly the four initial records in one transaction; never overwrite unknown or mismatched records. |
| Schema | Resume missing collections only in the owned database; validate existing collections and refuse content in an initializing workspace. |
| Verify / publish | Recheck actual provider compatibility, ownership, schema and exact profile before publishing ready revision two. |
| Publish, after a lost readiness response | Accept only the exact ready successor. Preserve immutable metadata instead of replaying publication. |
| Complete | Recheck ownership, schema and saved profile; return the same receipt without rewriting database records. Existing document data is allowed. |

After a positively acknowledged create, the ledger records the database ID before
writing `hades_setup/operation`. That marker binds installation, endpoint hash,
owner, operation, request hash, database name, and database ID. A changed database
incarnation, mismatched marker, or incompatible schema/profile is refused.

No error path drops a database, deletes source data, resets a profile, releases a
name reservation, or adopts a legacy corpus. `needs_reconciliation` is held for
administrative investigation; this slice provides no force/adopt/release API.
Repeated requests with recoverable failures must be identical and explicit.
An interrupted request leaves its preceding checkpoint rather than claiming
that its remote effect was rolled back.

## Schema and readiness

The minimal schema contains `hades_setup`, `hades_workspace`, `documents`,
`chunks`, and `embeddings`: document collections with `waitForSync` enabled and
only their primary unique `_key` indexes. It does not create graph schema seeds
or an unqualified vector index. Retrieval uses the internal exact-scan path.
Unknown collections, indexes, metadata rows, or preexisting vectors are refused.

Final publication locks ownership, profile metadata, and all data collections in
one stream transaction. It rechecks ownership, empty data collections, schema,
and the complete profile snapshot, then preserves revision one and publishes
ready revision two. Metadata reads reconcile an acknowledged or uncertain commit.
The existing standalone `ProfileStore::initialize` still refuses any existing
user collection; only the internal coordinator can use the owned recovery path.

A completed receipt records successful setup, not permanent provider health.
Identical completed retries revalidate database state but do not load a model.
Every document admission still verifies the current provider. A ready database
can outlive a failed final ledger write; the caller receives an error and must
retry to reconcile the receipt before treating setup as completed.

## Backup, validation, and remaining integration

Back up the complete ledger with its installation configuration and corresponding
workspace databases while setup is quiescent. Preserve ownership/permissions,
operation files, scope and reservations together. A database-only restore is not
permission to invent a replacement ledger or marker. Endpoint relocation,
inconsistent restores, unacknowledged creation, and orphaned reservations require
explicit administrative reconciliation. The existing backup scripts do not yet
capture this opt-in ledger automatically; public setup must integrate that policy.

`workspace_setup` is in the strict isolated runner and CI matrix. The tests create
only the exact names owned by `with_temp_db`, and inject failures before/after real
DDL, marker writes, metadata commits, readiness commits, and final ledger writes.
They cover cancellation, competing requests/owners, changed database incarnations,
schema/profile/provider drift, retained data, and idempotent retries. Service-free
unit tests cover ledger locking, corruption, permissions, unsafe links, bounded
records, request changes, and incomplete temporary files.

```bash
cargo test --locked -p hades-core --lib workspace_setup
cargo fetch --locked
python3 scripts/test_isolated_database.py --docker --contract workspace_setup
```

The documented native `--arangod /path/to/arangod` alternative starts a private
Unix-only test server. [Real-Jina qualification](verified-jina-qualification.md)
now creates its disposable database through this coordinator and proves an
identical retry preserves the profile before its fixed ingestion/retrieval gates.
Public entry-point enforcement, broader workload/schema support, scoped setup
authorization, frontend creation, and recovery tooling remain subsequent work.
