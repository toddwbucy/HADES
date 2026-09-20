# Batch insert outcome baseline

At runtime source `e22ad53`, the actual CLI against a disposable ArangoDB 3.12.11
reports a partially failed document-array insert as success. The test creates
`documents/existing`, then inserts `new` and a conflicting `existing` together.
ArangoDB commits `new` and returns a per-item unique-constraint error (1210) for
`existing`; read-back confirms the original existing document/revision is intact.
The CLI nevertheless exits zero, with an outer `success:true` and the error
inside its data array.

[Retained evidence](cli-insert-outcomes-baseline.json) records the actual output,
read-back assertions and seven source/probe hashes. The reproduction is
`codebase_lifecycle::batch_insert_partial_failure_is_not_reported_as_success`,
run using `scripts/test_isolated_database.py --arangod <private-binary>
--cli-lifecycle`. Its final nonzero-exit assertion intentionally fails at the
baseline. The runner creates its own Unix-only server and sanitized configuration;
no live database is used.

The native CLI forwards `DbInsert` to shared dispatch, which calls
`crud::insert_document` and returns the raw successful HTTP response. The transport
checks HTTP status, not each element of a bulk response. Thus per-item errors do
not reach the command error path. The shared daemon route is affected by the same
source path; actual daemon-envelope reproduction remains to be added.

This P2 outcome-reporting defect can cause callers to treat an incomplete batch
as complete or retry already committed documents. It does not imply the batch
was atomic, that the failed item overwrote existing data, or that a production
incident occurred. Remediation must detect partial/all-item failures, preserve
useful failure and partial-completion diagnostics, and make CLI/daemon failure
semantics explicit. Cover single/all-success, mixed/all-failed batches, real
read-back and valid response-shape checks. Do not silently retry or promise
rollback of successful items.


## Remediation behavior

The shared insert handler validates input before sending a request and requires
one typed outcome per submitted document. Existing successful single-object and
array response shapes are preserved. Per-item failures produce `InsertFailed`
with confirmed success/failure counts, submitted-item indexes, successful keys,
and server error details. The daemon maps this to `INSERT_FAILED` with
`success:false`; the CLI returns nonzero and prints the diagnostic on stderr.
It does not automatically retry or roll back committed items.

Malformed shapes, cardinality or success metadata produce a separate uncertain
outcome error instructing callers to inspect stored state before retrying. A
submitted document's own `error` field is not an operation result and does not
cause rejection. This classification is specific to `DbInsert`, not a blanket
rule applied to arbitrary database documents or every CRUD transport call.

Private shared-service tests cover mixed/all-failed envelopes, valid single/array
results, malformed responses and invalid inputs rejected before any request.
The real CLI fixture also checks an all-failed repeat preserves exact rows and
revisions and that successful single/batch inserts commit the expected keys.
Final passing evidence is retained separately from the original baseline.

[Remediation evidence](cli-insert-outcomes-remediation.json) records 15 passing
private database lifecycle tests, four passing shared-service contracts, actual
partial-insert diagnostics, eight source/probe hashes and clean server shutdown.
