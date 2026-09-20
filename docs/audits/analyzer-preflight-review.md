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
The baseline above is preserved separately from the remediation below.


## Remediation implementation

Version probes now capture at most 64 KiB per stream using nonblocking pipes,
with a ten-second runtime ceiling and explicit overflow errors. Both streams
are serviced in bounded chunks. A fresh process group is owned until signalling
and direct-child reap, including on errors and unwinding. Natural leader exit
is observed with `waitid(WNOWAIT)` so descendants can be signalled before the
leader's PID is released; buffered version output is then drained.

Synchronous tools callers retain the public probe API. Async ingestion uses
`resolve_and_probe_async`, whose blocking worker owns cleanup after its caller
is cancelled. A drop guard signals cancellation, checked before spawning and
during bounded capture. The same resolution helper supplies configured/managed/
PATH precedence to both APIs, retaining workspace and version-argument semantics.

The deadline does not promise preemption of kernel spawn/reap or blocking-worker
queue admission. The worker keeps ownership through cleanup; cancellation may
return to the caller before the worker finishes. Descendants that deliberately
leave the process group are not contained, and descendants killed after leader
exit can await reaping by the system's adopter. No production deployment.
