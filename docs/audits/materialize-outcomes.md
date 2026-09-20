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
