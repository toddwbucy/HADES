# Viewer boundary review (epic #12)

Source snapshot: `e27e909`. This is a bounded source review of
`crates/hades-frontend`, not a production deployment or browser certification.

## Architecture and authority

The crate has no HADES crate dependencies. `main.rs` constructs a `Backend` and
`assemble.rs` invokes the configured `hades` executable with argument vectors,
consuming JSON envelopes. `server.rs` exposes graph discovery, snapshots,
neighborhood expansion, node records and source content. The browser renders
those responses through Graphology/Sigma and bundled JavaScript assets.

`serve` permits loopback/private binds and rejects a non-loopback bind without a
password. Every registered route passes through exact Host-header validation and
optional Basic authentication. A missing explicit database list triggers discovery
of all databases accessible to the backend; request database selection is limited
to that list. Thus the introductory source comment describing one fixed database
is stale. This is operator-selected broad exposure, not an observed allowlist bypass.

The backend uses direct argv, rejects leading-dash graph/node values, separates
untrusted positional values with `--`, and whitelists traversal direction.
Graph-data requests may narrow the configured export cap, not widen it. An unset
operator cap still permits complete export. These source controls are distinct
from tested browser/network behavior.

## Rendering observations

Inspector titles, identifiers and properties pass through `esc`; source content
uses `textContent`. Copied shell commands use `shq` to quote database, graph and
record identifiers. Numeric snapshot counts are deserialized into typed Rust
fields before serialization. This review does not establish complete DOM-XSS
coverage, vendor-library vulnerability status or behavior of every style/color path.

## Open verification and resource questions

`Backend::run` and initial database discovery use `Command::output().await`.
They capture both output streams without a byte limit, subprocess deadline or
explicit cancellation/process-group owner. The HTTP routes show no shared
backend-process admission control. A private synthetic-child probe now confirms that `Backend::list_graphs` retains
at least 2 MiB of child stderr in its error even with `limit: Some(1)`. The child
only emits synthetic text and exits; no HADES process or database is used.
`error_response` subsequently logs and serializes the error without a further cap.
Stalled-child, concurrent-request and cleanup fixtures remain outstanding;
no such load was sent to the active service.

Existing assembler tests cover parsing, argument rejection and partial snapshots.
They do not establish end-to-end Host/auth enforcement, subprocess cleanup or
aggregate resource bounds. Follow-up should exercise the actual router with a
synthetic backend, verify default/disallowed database selection and request limits,
and separately test browser rendering with adversarial fixture strings. Password
transport/deployment configuration and bundled dependency provenance remain to be
reviewed. No claim of complete frontend acceptance is made here.

## Subprocess remediation in progress (#52)

The isolated branch replaces both backend invocation paths with an owned process
runner. Initial policy limits are four simultaneous children, 8 MiB stdout,
16 KiB stderr and 30 seconds of child runtime. These are explicit policy defaults,
not measured workload capacity. Excess output fails before retaining it, and
nonzero exit diagnostics are not reflected into HTTP errors or logs.

Each child starts a private process group with null stdin. An independent owner
retains its semaphore permit, captures both streams concurrently, and observes
leader exit with Linux `waitid(WNOWAIT)` before group cleanup. Keeping the leader
unreaped prevents PID reuse during group signaling. Cancellation, overflow and
deadline expiry signal the owned group; the direct child is reaped before its
permit is released. Cleanup retains admission until the OS completes reaping;
this does not promise a hard deadline for an uninterruptible kernel task.

Nineteen frontend tests pass, including corrected oversized-diagnostic rejection,
normal exit, stdout overflow, spawn failure, cancellation and timeout fixtures.
The latter two start a synthetic child and grandchild and verify that neither is
running after cleanup and the slot is available again. All-target frontend
Clippy passes. The probe never invokes a deployed HADES binary or database.

This is not complete #52 acceptance: graph assembly can retain output from many
collections after individual child permits are released. HTTP connection/request
admission, serialization/response retention, end-to-end router contracts, shutdown
behavior and measured aggregate budgets still require work. No deployment or
complete frontend/security certification is implied.

The next local increment adds a cumulative graph budget (32 collections, 50,000
documents, 8 MiB serialized document content), a 16 MiB serialization writer cap,
16 connection slots retained through socket writes, a 120-second connection
lifetime and a 60-second request deadline. All 22 current frontend tests and
Clippy pass. These limits are provisional policy, not a peak-RSS claim. Tests
currently exercise exact serialization boundaries and stalled I/O ownership;
actual-router assembly/overload/disconnect tests and measured budgets remain.
Explicit signal-driven server shutdown and private descendant cleanup during
shutdown must be added before #52 can be considered complete.

## Shutdown and router verification

Signal handlers are now registered before startup discovery. SIGINT/SIGTERM
close backend admission and broadcast cancellation to all owned children; the
server begins graceful HTTP shutdown and main keeps the runtime alive until
child permits have returned after reaping. An actual-binary fixture verifies both
signals during discovery, dump and an active loopback HTTP request. It also
verifies that disconnecting the HTTP client stops its synthetic child/grandchild
before viewer shutdown. These fixtures never invoke an installed HADES binary,
database or real document corpus.

The actual router now has contracts for rejected Host/auth/database inputs
(with no child launch), default/explicit allowed database requests, malformed
backend JSON and sanitized nonzero-exit responses. A separate shutdown-broadcast
contract verifies new child requests are refused after shutdown. Twenty-four
unit tests, the process lifecycle target and all-target Clippy pass. CI explicitly
runs the new service-free `process_lifecycle` target.

The signal test must run in an environment that delivers signals to its private
children. Under the restrictive local execution wrapper the registered task did
not receive SIGINT/SIGTERM; the same actual-binary fixture passed outside that
wrapper. Concurrent temporary-script creation also exposed `ETXTBSY`; script
fixture lifetimes are serialized, while their internal concurrent-child tests
retain their concurrency. Fifteen repeated unit-suite runs passed after that
fixture correction. No production spawn failures are ignored or retried.

Remaining #52 evidence includes end-to-end cumulative graph-budget cases, measured
concurrent child/response retention and peak memory, plus final review/CI. Signal
shutdown and router boundary checks above supersede the earlier pending items;
this report still does not claim complete frontend acceptance.
