# Detached ingestion ownership review

Tracks #51 under epic #12. Baseline: `7571036`, with the original static finding
recorded in #51. No production ingestion job has been launched for this review.

## Local admission increment

The current isolated branch reserves one of two process-wide slots before the
first database await. All served databases share the registry. Canonical paths
conflict with equal, ancestor or descendant paths; similar string prefixes alone
do not conflict. Lock poisoning and admission-query failures fail closed.
Cancelling before job insertion releases the handler's reservation. Once the
startup owner is created, it retains the reservation through record cleanup,
including cancellation during insertion. Direct-child kill-on-drop remains a
fallback behind explicit process-group cleanup.

Four private registry tests exercise simultaneous admission, overlap, cancellation
and poisoned state. A real-handler/private-socket contract holds two starts in
different database configurations before spawn, proves subsequent starts perform
no database work, then tests cancellation, admission-query failures and retries.
These are admission contracts, not measurements of actual live child count.

## Local output and process-group increment

The child now owns a fresh process group. Output is drained asynchronously through
pipes: stdout is capped at 8 MiB before buffer growth; stderr retains only its last
64 KiB while counting consumed bytes. The stored diagnostic text is also capped
after UTF-8 replacement. Overflow or a provisional six-hour runtime limit fails
the job. A successful process returning malformed JSON is reported as failed.

There are no new output files: `log_path` and `stderr_path` are null and
`output_storage` is `bounded_job_record`. Completed job records retain the bounded
result and, on failure, bounded diagnostics. Truncation and byte counts are explicit;
capture failures record unknown counts as null. This removes the shared-temp-file
creation and unbounded disk-growth path rather than retaining full logs privately.
Historical output files are not touched by this audit or automatically cleaned up.

Exit observation uses Linux `waitid(WNOWAIT)` so the leader cannot be reused before
group signalling. Normal exit signals lingering descendants before waiting for
pipe EOF; overflow and runtime expiry signal the group and reap the direct child.
Three finite synthetic-child contracts cover a 1 MiB single-line diagnostic,
stdout overflow, deadline expiry and a leader exiting while a descendant holds
its pipes. They verify direct-child disappearance and descendant termination.
They do not yet establish service shutdown, persisted outcomes or worst-case RSS.

## Local shutdown wiring

The daemon registers SIGINT/SIGTERM before opening either listener. Shutdown
closes ingestion admission under the same lock used to reserve paths and broadcasts
cancellation to owners. The daemon's outer cleanup path waits for reservations to
drain on ordinary shutdown and startup/accept errors. Owners observe cancellation
during PID persistence and process supervision; supervision signals the group and
reaps its leader before releasing the reservation on this normal cleanup path.

Private tests verify admission stays closed, drain waits for an outstanding
reservation, and an already-cancelled or newly-cancelled owner reaps its synthetic
child before returning its reservation.

The actual `daemon_shutdown` CLI fixture verifies SIGINT and SIGTERM both idle
and with a pending ingestion reservation. A gated private database response keeps
the reservation occupied after the signal; the daemon must remain alive until
the gate releases and admission fails closed. It then exits successfully and
removes its socket. The mock asserts exactly collection/admission requests and no
job insertion or child spawn. The fixture runs in service-free CI with a cleared
environment and explicit existing private database sockets to prevent discovery
of ambient endpoints. Actual daemon shutdown with a running ingestion child and
persisted job outcomes are covered by the additional fixture below.

The maintained `codebase_lifecycle` target now starts the actual daemon and an
actual ingestion child against the disposable ArangoDB server. A private mock
embedder holds a Python source-file ingestion during preparation. The initiating
connection has closed, but the job remains owned and running; a duplicate start
is refused. SIGINT and SIGTERM each cause direct-child reaping before daemon exit,
a persisted failed job with a shutdown reason, removal of the daemon socket and
preservation of the prior graph. This uses synthetic CPU embedding responses,
not installed services or a model. It does not yet measure simultaneous child
counts across multiple served databases. For each signal, a new daemon instance
also retries the same source tree, completes ingestion with a new owner identity,
reaps the child and passes graph validation against the disposable database.
Before retry, the fixture inserts an unfinished record belonging to the previous
owner, with the live test process's PID. The restarted daemon must report
`recovery_required`, refuse admission and leave that record (including its
revision) and graph unchanged. Only explicit fixture-side reconciliation to a
terminal state permits the successful retry. This checks conservative recovery;
it does not provide automatic recovery or establish the outcome of a lost job.

The added lifecycle check exposed a completion/status race: a database read could
return a running snapshot after the owner persisted completion and released its
reservation. Status now samples ownership once and, for an unfinished row from
the current instance whose owner is gone, refreshes the record once before
classifying it. A private HTTP regression supplies a stale running response then
a completed response, while the persistence-failure case supplies two unfinished
responses and still requires recovery. Reads remain bounded and do not mutate
the job record.

Validation: the maintained private-database matrix passed after this fix
(`strict-prerequisite`, `database-contracts`, `codebase`, `cli-lifecycle`), as did
all 13 ingestion ownership unit tests. The initial lifecycle failure and the
deterministic stale-snapshot regression establish why the refresh is needed.
These results do not certify the remaining cross-database concurrency criteria.

Process supervision now catches panics around capture/exit observation while
retaining the process outside the unwind boundary, then signals its group and
reaps the direct child. A private injected-panic contract verifies the child is
absent while its reservation is still held, followed by readmission after release.
This does not promise cleanup after an uncatchable process abort or SIGKILL.
Canonical paths that cannot be represented as UTF-8 are rejected before admission
or database access; a private symlink fixture prevents a serialization panic from
replacing an ordinary invalid-parameter response.

## Remaining requirements before publication or closure

- Verify concurrent starts against a real disposable database; process-wide
  admission does not coordinate separate daemons.
- Own the process group through normal completion, shutdown, panic and task
  cancellation. Retain admission until descendants are stopped and the direct
  child is reaped; direct-child kill-on-drop alone does not prove that invariant.
- Verify persisted phases and output bounds against a real disposable database,
  including ambiguous commits and response-size rejection. Existing private HTTP
  fixtures below do not substitute for real ArangoDB semantics.
- Verify the sealed effective-configuration handoff in the complete disposable
  database/daemon ingestion lifecycle, retaining provisioning and root boundaries.
- Add actual synthetic-child/process-group, noisy-output, failure-injection and
  real disposable-database fixtures, including simultaneous starts across databases
  and actual peak live-child counts. Complete review and final CI.

The current local increment is deliberately not presented as completion of #51.
No production binary, service configuration, database or GPU workload is changed.

## Local persisted-state increment

`ingest_jobs::records` now owns startup and completion, extracted from dispatch.
A random daemon-instance identity and in-memory `(database, job)` identity replace
PID-existence admission checks. The admission query returns at most three compact
rows with explicit cursor/response/server-memory limits. An unowned unfinished row
blocks new jobs conservatively; status reports `recovery_required` and preserves
the recorded phase. It does not infer whether another daemon's process is alive,
signal a saved PID, or silently change historical records.

The detached owner starts before the first job insertion await. Records begin as
`starting`; a confirmed PID update changes them to `running`. If PID persistence
fails, the child is stopped and reaped before failure persistence. Insertion
uncertainty, pre-spawn client cancellation and spawn failures attempt a terminal
failure record. Completion has three five-second write attempts with short fixed
backoff; exhaustion leaves an explicitly unowned unfinished record requiring
reconciliation. Startup writes have 64 KiB response limits and status reads a
16 MiB ceiling. Successful transport exit with malformed JSON is a failed job.

Private HTTP/actual synthetic-child contracts cover successful phases, uncertain
insertion, cancelled insertion without spawn, missing executables, failed PID
persistence with verified reaping, malformed JSON, non-UTF-8 diagnostic tails,
three exhausted completion attempts, unowned-row admission refusal and invalid
job IDs with no database traffic. A stale row naming the fixture process's live
PID still reports unknown ownership. These are local, not deployment evidence.

## Local effective-configuration handoff

The child receives the daemon's resolved configuration in a Linux memfd capped at
64 KiB during serialization, with mode 0600 and write/grow/shrink/seal seals. Only
the selected child's pre-exec hook clears CLOEXEC. The inherited descriptor is
above the stdio range, and the loader restores CLOEXEC before later analyzer execs.
Positional reads avoid a shared file-offset race. Memory-file ownership is retained
by the command/child and ends with those owners; no snapshot path is created.

The internal `--resolved-config-fd` option is accepted only for ingestion. It
loads the snapshot without reapplying ambient configuration or environment
overrides, followed by explicit CLI overrides. Password and CUDA visibility are
preserved separately from fields skipped in ordinary configuration serialization;
CUDA visibility is also aligned in the child's environment. Ordinary CLI loading
and password exclusion from YAML remain unchanged. No credential bytes enter argv.

Private contracts check full configuration round trips, skipped fields, seal
enforcement, size/descriptor rejection, a child with conflicting environment, and
closed-stdin allocation. Actual CLI tests verify the missing ambient config is
bypassed for empty ingestion (which stops before service access), while unrelated
commands reject the internal option. A real startup command fixture reads the
inherited snapshot and returns only preservation booleans, never credentials.
