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
persisted job outcomes remains required.

## Remaining requirements before publication or closure

- Verify conservative restart reconciliation and concurrent starts against a real
  disposable database; process-wide admission does not coordinate separate daemons.
- Own the process group through normal completion, shutdown, panic and task
  cancellation. Retain admission until descendants are stopped and the direct
  child is reaped; direct-child kill-on-drop alone does not prove that invariant.
- Verify persisted phases and output bounds against a real disposable database,
  including ambiguous commits and response-size rejection. Existing private HTTP
  fixtures below do not substitute for real ArangoDB semantics.
- Preserve the effective selected configuration as well as the database in the
  spawned command; verify provisioning and root boundaries remain intact.
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
