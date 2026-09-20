# Structural export failure contract

P2 #83 was reproduced at `cdcf58fe05041e6b266c25f7a8758b6d9c3e7527` using
private Unix-socket mocks. Three baseline regressions fail: a rejected batch
returns success with zero updates, a one-row acknowledgment for two updates
returns partial success, and a zero batch size panics. Both CLI callers only
warn about partial counts before constructing a success response.

Export now fails on the first rejected, short or malformed acknowledgment.
The error records prior fully acknowledged updates and the attempted failing
batch where available. Existing CLI error propagation prevents their success
response after such an error. Earlier batches are not rolled back; a failed
transport can leave the current batch's outcome unknown. No whole-export
transaction or automatic retry is introduced. Hash-map collection traversal
means inter-collection execution order is unspecified.

Preflight rejects zero batch size, zero/overflowing dimensions, size mismatch
and nonfinite vectors before sending updates. Both full and compact subset
exports use this validation. Collection names are supplied as AQL bind values,
and ignoreErrors is removed. Successful acknowledgments must contain one numeric
1 per update; a present writesIgnored statistic must be numeric zero. Counts
for malformed/short replies are not promoted to verified progress.

Eight private transport tests cover rejected/short/malformed replies, invalid
inputs without traffic, a failure following acknowledged progress, and compact
subset row/value order plus collection binding. Invalid compact subsets are
rejected without requests, including duplicate and out-of-range indices.
Acknowledgment errors distinguish row counts, values and ignored-write stats. The existing graph-export unit
contracts remain applicable. A separate `structural_export_db` test is wired
into the strict disposable-server CI runner: it verifies full and subset writes,
then a missing target after an acknowledged batch, checking earlier persistence
and that a later node is untouched. Real-server execution is pending final CI;
mock success alone does not certify ArangoDB transaction behavior.

No production export, repair, model or data was changed. This review does not
establish that a trained model is useful, that target documents still match the
training snapshot, or that structural vectors remain current after ingestion.
Those are separate full-audit alignment/provenance and retrieval requirements.
