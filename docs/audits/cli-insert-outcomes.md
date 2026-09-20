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
