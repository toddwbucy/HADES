# Orientation metadata failure audit

At source `739ad99b5a39ffcbf5bebccdc82db6c69a508c0a`, actual CLI `orient --collection
docs` runs against private Unix HTTP peers returned exit zero and success:true
for three separately injected failures: malformed sample cursor, malformed recent
cursor, and malformed index-list reply. Each affected field became empty without
a stderr diagnostic, while the count still reported two documents. A valid
sample/recent/index control passed first.

[Baseline evidence](orientation-outcomes-baseline.json) records normalized
observations and five source/probe hashes. The fixture invokes the real CLI with
an empty environment and explicit fixture database, no production endpoint. Run:

```sh
cargo test -p hades-cli --test db_response_shapes orientation_does_not_hide_metadata_read_failures -- --nocapture
```

Source handling uses `.ok()` for the sample and unconditional empty defaults for
recent documents and indexes. The recent helper also serves profile overview;
that path shares the source defect but is not exercised by this baseline.

Remediation must distinguish legitimate not-found/empty results from query,
transport, authorization and malformed-response failures. Preserve valid output,
propagate stage-specific diagnostics and nonzero CLI/daemon failure status, and
cover both single-collection and profile-overview paths. This is a private
failure-status reproduction, not a production incident or write-integrity claim.

## Remediation

Collection sampling, recent-document reads and index listing now tolerate only
not-found as empty. Other errors propagate with collection/stage context through
shared dispatch. The recent helper returns Result, so profile overview also
fails when a recent-document read fails. CLI errors exit nonzero with no success
output; daemon replies preserve request IDs and report QUERY_FAILED with no data.
Successful output and existing missing-collection semantics remain unchanged.

Private daemon tests cover malformed replies and HTTP 403/503 at each detail
stage, the same faults in overview recent reads, and valid empty/404 controls
for detail and every registered profile. Actual CLI tests cover all three
malformed detail stages, populated controls, stage diagnostics and stopping after
the failed request. This is response handling evidence, not deployed readiness,
full metadata-shape validation or a production incident. Snapshot-consistent
orientation across concurrent database changes remains outside this fix.

[Remediation evidence](orientation-outcomes-remediation.json) records five
source/probe hashes, seven passing CLI tests and three passing service tests.
