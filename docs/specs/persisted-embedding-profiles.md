# Persisted embedding profiles

Status: internal Rust storage primitives for [#225](https://github.com/toddwbucy/HADES/issues/225).
No production caller uses this store yet. It does not enable database creation,
adopt existing vectors, or change the running installation.

## Records and scope

`embedding_profile::store::ProfileStore` uses a dedicated `hades_workspace`
document collection, independent of `hades_schema`. Each record has a strict
`hades-workspace-record-v1` envelope and typed `payload`. Unknown formats/fields,
missing records, invalid ranges, oversized payloads, and inconsistent references
are errors. Portable profiles remain bounded to 64 KiB and manifests to 1 MiB;
HTTP responses have a bounded allowance for ArangoDB metadata.

| Key | Purpose |
| --- | --- |
| `embedding_profile` | Active profile revision, generation label, and manifest MD5 |
| `profile_<revision>` | Immutable complete profile, including capability and workload policy |
| `manifest_<md5>` | Immutable canonical artifact manifest |
| `generation_<sha256-of-label>` | Immutable generation label, artifact MD5, and each enabled workload's space ID |

Generation labels are hashed only for safe document keys; their original value is
retained and checked. This is not a new model digest. Manifest MD5 and embedding
space SHA-256 keep their existing meanings.

Reads use the **writer endpoint** and one exclusive stream transaction to obtain
a coherent snapshot. The opaque `ProfileSnapshot` includes the selected endpoint,
database name and ArangoDB database ID, all four record contents, and their `_rev`
values. Callers can borrow the profile and manifest but cannot change the pin.
Snapshots are in-process values, not serialized background-job credentials.
An endpoint alias change requires reloading; no cache or alias equivalence is
inferred.

## Initial publication

The provisioning owner must already own a newly created database and serialize
its setup. `initialize` validates an **initializing, revision-one** profile and
its manifest before writing. It refuses any existing user collection, then creates
`hades_workspace` without create-if-absent behavior. One metadata transaction
inserts the manifest, generation, profile, and active pointer.

This empty-database check is an additional refusal rule, **not proof of ownership**
or protection against unrelated administrative collection creation. The
[durable setup coordinator](recoverable-workspace-setup.md) now establishes
ownership and uses a separate internal recovery method, which checks its marker
under the same metadata transaction. The standalone initializer retains its
strict empty-database behavior.
Existing corpora are not automatically assigned the currently configured model.

After independently verifying the required schema and trusted provider, the
provisioning owner may call `mark_ready(snapshot)`. It rechecks the complete pin,
inserts immutable ready revision two, and replaces only the active pointer with
an `_rev` precondition. Revision one remains intact. This follows ArangoDB's
[`ignoreRevs=false` document replacement contract](https://docs.arango.ai/arangodb/3.12/develop/http-api/documents/).
Only that readiness transition is implemented. Changing the model, policy,
generation, or an already-ready profile requires future explicit migration work.

Storage does not itself verify a provider or domain schema. Consequently there
is no CLI/MCP/frontend readiness action in this slice.

## Prepared-work guard

`guarded_write(snapshot, collections, callback)` admits only ready snapshots.
It locks workspace metadata and all declared data collections in the same stream
transaction, reloads and compares the full pin, then calls trusted core code with
the transaction-scoped client. A stale pin, cross-database snapshot, missing
identity, or changed record prevents callback admission.

The callback must use that client exclusively, declare every target collection,
and perform inference/extraction beforehand. A second metadata check detects
accidental policy writes inside the callback. Callback errors or cancellation
before commit abort the transaction through the existing transaction owner.
Even a change followed by restoration of the same metadata invalidates an old
pin because its server revision changed.

This is a database freshness guard, not a vector validator. The internal
[document integration](profile-bound-documents.md) adds provider receipt,
dimension, workload/role, and space/generation validation around prepared rows,
and checks stored provenance before query scoring. Existing production
write/query paths do not acquire this guard yet.

## Failure and recovery

Collection creation is outside the metadata transaction. A failure can leave an
empty metadata collection; initialization will refuse to take it over on retry.
Failed readiness publication rolls back its inserted revision and pointer update.
Repeated/stale readiness requests fail rather than silently succeeding.

A lost commit acknowledgement is an unknown outcome. The owning provisioning
operation must reload and reconcile durable metadata against its operation ledger;
it must not reset a database or equate a matching name with ownership. The
[setup coordinator](recoverable-workspace-setup.md) now implements the local
ledger and safe resume for conventional document databases. The public setup
workflow, recovery tooling, and portable job pins remain subsequent work.
Raw administrative writes can still corrupt records: validation detects them,
but this collection is not an authorization boundary.

## Validation

`embedding_profile_store` is included in the strict isolated database runner and
CI matrix. It uses `with_temp_db` and synthetic identities, with no model, GPU,
installed service, or research corpus:

```bash
cargo fetch --locked
python3 scripts/test_isolated_database.py --docker --contract embedding_profile_store
# Or use an existing binary to start a private Unix-only test server:
python3 scripts/test_isolated_database.py --arangod /path/to/arangod \
  --contract embedding_profile_store
```

Contracts cover round-trip identity and binding, retained revisions, competing
initializers/readiness transitions, failed publication, legacy refusal, corrupt
or oversized metadata, database isolation, stale pins, rollback, cancellation,
and atomic rejection of accidental metadata changes during prepared writes.
