# Analyzer preflight execution boundaries

Reviewed at `b0d14f0`. `session::preflight_binary` uses synchronous
`std::process::Command::output()` from the selected workspace. It has no runtime
or stdout/stderr capture ceiling. Analyzer resolution and version argument
selection are shared by ingestion and tools status/install. The asynchronous
ingestion path calls the synchronous probe directly before per-file processing.

A bounded private reproduction invokes the real helper with Python that sleeps
300 ms, writes exactly 2 MiB, then exits. The helper accepts all 2,097,152 bytes
as its version string. On a current-thread Tokio runtime, a concurrent 20 ms
timer finishes after 313 ms. The child terminates normally; no real analyzer,
workspace scan, database, installation or production service was involved.
This demonstrates blocking and permissive capture, not an executed infinite
hang, OOM, or production outage. Selected hashes and observations are retained
in `analyzer-preflight-result.json`; the historical fixture is in
`repros/analyzer_preflight.rs`.

Remediation must bound runtime and captured output on both streams, keep the
workspace and version-argument semantics, preserve useful bounded diagnostics,
and move blocking work off asynchronous ingestion's executor. Process ownership
must survive cancellation and close inherited pipes without leaving descendants
running. Tests must cover normal/error responses, sleep, both noisy streams,
retained descendant pipes and caller cancellation using private synthetic peers.
Linux process-group ownership cannot contain children that intentionally escape
the group; stronger sandboxing is a separate boundary.

The ordinary LSP stream's frame/notification memory and descendant boundaries,
and the `go install` build/download subprocess, are separate remaining scope.
No implementation or production change is represented by this baseline.
