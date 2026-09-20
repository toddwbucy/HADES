# Materialization failure-status audit

At source `71ddf59fa2cb9862e08bac479be968cc60b8a20e`, actual CLI executions against
private Unix HTTP peers returned exit zero and outer success:true despite a
source scan failing. Both normal and --dry-run modes reproduced the problem.
The injected cursor reply had result:null; query validation rejected it, but
materialization accumulated the error in totals.errors and returned Ok.

Valid empty source scans passed controls in both modes. Each failed case made
four read requests (collection list, schema cursor, collection list, source
cursor); neither attempted an import or graph registration. This establishes
failure-status handling, not persisted partial writes or a production incident.
[Baseline evidence](materialize-outcomes-baseline.json) records normalized
observations and five source/probe hashes. Reproduce in an isolated checkout:

```sh
cargo test -p hades-cli --test db_response_shapes materialize_scan_failure_is_not_success -- --nocapture
```

The source also accumulates collection-creation, import, unknown-strategy and
registration errors before returning success. Remediation must classify an
incomplete run as failure while retaining per-definition errors, counts, dry-run
state and completed registrations. Preserve successful output compatibility and
explicitly state that earlier writes may remain; do not claim rollback or retry
automatically. Verify daemon envelopes, successful controls and failure branches,
including a disposable-database partial-write case before claiming persistence
coverage. Missing collections and deliberately skipped references need separate
semantics from execution errors.

## Remediation

Shared materialization now returns MaterializationFailed when accumulated
execution errors are nonempty, retaining the complete per-definition/totals
report in the diagnostic. The daemon maps this to MATERIALIZATION_FAILED with
success:false; the CLI exits nonzero and writes diagnostics to stderr. The
report retains dry-run state and completed registrations. Successful output is
unchanged. Missing collections and skipped references remain reported counters,
not execution errors. No automatic retry or rollback is introduced; earlier
writes, if any, remain committed.

Private peer contracts exercise scan, unknown-strategy, collection creation,
import and registration errors, successful registration, dry-run suppression,
and missing/skipped inputs. A separate disposable-database fixture exercises
partial edge import with persisted-state read-back. [Remediation evidence](materialize-outcomes-remediation.json) records seven
source/probe hashes, four passing shared-service tests, five passing CLI response
tests and all 16 passing disposable-database lifecycle tests (62.53 seconds).
The private ArangoDB 3.12.11 server stopped with exit zero.

The persistence fixture imports two derived edges: one valid key and one overlong
key derived from valid source-document keys. Exactly one edge remains committed;
read-back verifies its endpoints, total edge count one and source-document count
three. CLI exit is one with empty stdout and retained report: one edge created,
one of two import items failed. This proves reporting and preservation for this
partial-import case, not atomic materialization or every backend failure mode.

After integration of PR #124, the recorded pre-integration hashes remain pinned
to a1e2bee; the materialization handler is unchanged. The service and CLI response
suites are rerun on the integrated source to cover both fixes together.

[Integration evidence](materialize-outcomes-integration.json) pins the combined
source: 32 service tests and six CLI response tests pass.
