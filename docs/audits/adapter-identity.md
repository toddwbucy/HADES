# WeaverTools adapter identity review (#109)

Source: 7a735c8aa357e71c44a5b08471542c7b24db403e. The
[retained probe](repros/adapter_identity.py) runs the real extractor and writer on
private synthetic Markdown with every database API call stubbed. The
[result](adapter-identity-baseline.json) retains source/probe hashes.

Three cases emit two distinct edge rows with one identical key and exit zero:
a valid 250-character source whose distinguishing target is truncated away;
ambiguous double-hyphen concatenation of otherwise valid identifiers; and two
tags on the same source/relation/target/via tuple. The latter contradicts the
Edge record's stated identity semantics. The writer submits onDuplicate=replace.
The probe simulates replacement retaining one row; real database persistence has
not yet been replayed for this finding. No live corpus or production loss is claimed.

These are graph-identity defects rather than failed-write acknowledgement errors.
Successful import counters can include replaced rows and therefore do not prove
that distinct logical relationships survive. Existing writer failure tests do not
cover these collisions.

Other source observations: node key_for replaces unsupported characters without
a digest; malformed node IDs are noted but retained, and duplicate identifiers
can enter the extraction list. The writer's kind_of dictionary picks a kind by
identifier, so conflicting duplicates need a defined rejection policy. No whole-run
transaction or stale-record retirement exists in this writer. A rerun imports
present rows and replaces the latest report; absence from the new extraction does
not delete an older adapter row. This review did not run a live retirement.

Remediation requires full-tuple deterministic identity, key-size validation,
duplicate/conflicting-declaration handling, consistent endpoint derivation and
an explicit legacy compatibility/migration policy. Changing key generation alone
would risk leaving old and new rows coexisting. Test distinct persistence and
idempotent replay in a disposable database before considering a migration.
Production changes remain separately controlled by the owner's snapshot/downtime
policy. This report is baseline evidence, not a completed remediation or complete
real-WeaverTools corpus conformance audit.


## Actual database verification

The subsequent [disposable ArangoDB result](adapter-identity-database-baseline.json)
confirms persistence loss in all three cases on ArangoDB 3.12.11: two declared
seam edges produce one persisted row while the writer exits zero, both initially
and on a repeat run. Actual extraction, collection creation, imports, replacement
acknowledgements and verification queries ran through an owned loopback bridge
to a fresh Unix-only server. No production endpoint was used.

The [database probe](repros/adapter_identity_database.py) is retained byte-for-byte;
copy it to scripts/verify_adapter_identity_database.py in an isolated baseline
checkout before running so its repository-relative imports resolve. It accepts an
existing --arangod binary, never a database endpoint. The replay used one CPU,
nice 10, an 8 GiB child address-space limit, small caches and a 90-second external
watchdog. Both private server group and bridge thread stopped before success was
recorded. Synthetic data/artifacts remain under a private temporary directory.

This supersedes the earlier simulated-persistence limitation for these three
fixtures only. It does not measure prevalence in the real WeaverTools corpus.
No remediation is yet implemented. A corrected writer must explicitly reject
or safely migrate legacy identity rows before a new encoding is admitted; merely
adding a hash can leave duplicate legacy/new representations. Conflicting node
declarations and stale-row policy also need explicit handling. Any migration
against live data remains outside discovery and subject to the snapshot policy.

## Local remediation and validation

Version 2 derives 69-byte ASCII node/edge keys from SHA-256 over JSON-framed
full identities. Edge identity includes persisted endpoints, relation, basis,
via and tag; endpoint construction uses the same node-key function and leaves
HADES's persisted file/document keys intact. Readable node identifiers remain
attributes. Exact duplicate rows are deduplicated, while conflicting duplicate
node declarations fail before mutation.

A read-only preflight checks all existing adapter node/relation/report collections.
Legacy or mixed-version rows and changed stored node kinds are refused before
collection creation/imports. There is no automatic migration/bypass: rebuild
legacy adapters into a fresh, independently verified graph and review consumer
cutover/rollback separately. The guard is not a lock against simultaneous legacy
or arbitrary writers; those must be stopped during a version cutover.

The [real-database remediation result](adapter-identity-remediation-result.json)
passes all three collision cases with two persisted edges on initial and repeat
runs, stable keys/values and valid endpoints. Legacy rows, conflicting declarations
and stored-kind changes are rejected while preserving exact old rows/revisions.
The maintained scripts/verify_adapter_identity_database.py runs these positive
contracts against a fresh server. Both private server and bridge stopped before
success. The historical baseline probe/results remain unchanged.

All 75 adapter identity/failure/redirect/scope CPU tests pass (0.31 seconds).
The original disposable-database writer suite also checks first/repeat success,
schema rejection and wrong collection type. The [result of that replay](adapter-identity-writer-regression-result.json)
records those outcomes. This is no claim of full real-WeaverTools corpus conformance.

Stale retirement remains explicitly unsupported: absent rows are retained, and
the latest report now says so. A successful incremental write is not a current
whole-graph snapshot. Existing partial-import/error accounting remains; this
change adds no whole-run transaction. Repository changes have not been deployed.
Final integration/CI/review remain required before #109 can close.
