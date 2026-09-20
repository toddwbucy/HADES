# Detached ingestion ownership review

Tracks #51 under epic #12. Baseline: `7571036`, with the original static finding
recorded in #51. No production ingestion job has been launched for this review.

## Local admission increment

The current isolated branch reserves one of two process-wide slots before the
first database await. All served databases share the registry. Canonical paths
conflict with equal, ancestor or descendant paths; similar string prefixes alone
do not conflict. Lock poisoning and admission-query failures fail closed.
Cancelling a handler before spawn drops its reservation. After spawn, the handler
transfers both child and reservation into the detached task without an intervening
await. Direct-child kill-on-drop is enabled as a fallback.

Four private registry tests exercise simultaneous admission, overlap, cancellation
and poisoned state. A real-handler/private-socket contract holds two starts in
different database configurations before spawn, proves subsequent starts perform
no database work, then tests cancellation, admission-query failures and retries.
These are admission contracts, not measurements of actual live child count.

## Remaining requirements before publication or closure

- Replace the legacy database/PID-existence heuristic with explicit owner identity
  and restart reconciliation; process-wide admission does not coordinate separate
  daemons or establish ownership of historical children.
- Own the process group through normal completion, shutdown, panic and task
  cancellation. Retain admission until descendants are stopped and the direct
  child is reaped; direct-child kill-on-drop alone does not prove that invariant.
- Make starting/running/failed/orphaned states truthful across insertion, spawn,
  PID-update and completion-recording failures. Bound persistence/retry work.
- Capture stdout/stderr within byte budgets before disk growth or full-file
  allocation, create private artifacts exclusively, and define cleanup/retention.
- Preserve the effective selected configuration as well as the database in the
  spawned command; verify provisioning and root boundaries remain intact.
- Add actual synthetic-child/process-group, noisy-output, failure-injection and
  real disposable-database fixtures, including simultaneous starts across databases
  and actual peak live-child counts. Complete review and final CI.

The current local increment is deliberately not presented as completion of #51.
No production binary, service configuration, database or GPU workload is changed.
