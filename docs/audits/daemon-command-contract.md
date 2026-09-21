# Daemon command contract

Epic #12 source review at `95e8629a192112ecce92e8c13169cef73c0441f1`.
The `daemon` leaf routes from `main.rs` to `commands::daemon::run` and its
Unix/MCP transports, shared service authorization and ingestion-owner cleanup.
This maps command behavior, not every MCP tool, SDK implementation or live host.

## Startup and authority

The Unix listener defaults to `/run/hades/hades.sock`, with `--socket` override.
It serves one configured database. Pool construction validates configuration and
constructs clients; it does not prove backend readiness. MCP is optional: bind
and token-file flags (or their environment equivalents) must appear together.
Extra-database, prefix and ingest-root options alone do not enable MCP.

MCP accepts explicit loopback, RFC1918 or IPv6 unique-local bind addresses and
plain HTTP bearer tokens. Token files are read once at startup, with blank/comment
lines ignored; missing/empty files fail. File ownership/mode, token entropy and
hot rotation are not enforced by this loader. The host allowlist includes local
forms and the bind IP. `/healthz` is unauthenticated and returns a static `ok`;
it is not a database, model or ingestion readiness check.

MCP uses an agent authority ceiling. Provisioning requires both a nonblank
name-prefix list and ingest-root list; one alone leaves provisioning disabled.
The shared service checks command tier, create-database prefixes and canonical
ingest paths. Prefixes also widen the database read allowlist. Otherwise only
the configured default plus explicitly listed databases are served. Names are
limited to 64 ASCII letters/digits/underscore/hyphen. Lazy pools are cached per
name; there is no cache eviction in the reviewed implementation, relevant when
provisioning prefixes admit many names. All accepted bearer tokens share policy.
Filesystem canonicalization is a check, not proof against later filesystem races.

Unix peers receive a local-admin ceiling by default, including provisioning;
requests can self-restrict to agent. There is no peer-credential authentication
or explicit chmod/chown in this module: socket/parent permissions, process umask
and deployment configuration form that local access boundary. This does not
establish the live server's access permissions or resolve the separate #74 review.
The stale-socket path follows metadata, refuses non-sockets and successful live
connects, then removes a socket after any connection failure. Permission-denied
versus connection-refused distinction and path races need private fixtures.
Parent-directory creation errors are ignored until bind reports failure.

## Framing, admission and request outcomes

Unix frames have a four-byte big-endian length followed by JSON, capped at
16 MiB before allocating the body. Header and body each have a 30-second read
budget; requests on a connection are sequential. Dispatch has a 60-second
budget. Responses are serialized fully, checked only against the u32 framing
range, then written/flushed within 15 seconds; the inbound 16 MiB cap does not
cap response allocation. Oversized requests get an error then disconnection.
Shared service parsing, session/tier checks and provisioning checks precede the
dispatch timeout. Timeout returns an error, not rollback proof for prior effects.

Unix and MCP sockets share 64 connection permits, retained through response I/O.
Excess sockets close before request allocation. MCP adds a 300-second absolute
socket lifetime, 16 MiB/15-second body collection with 64 body permits, and
bounded session/request/stream owners (32/64/64 respectively). Its session module
also declares a 2 MiB message bound and 15-second forwarding budget. These are
source policy values, not measured production capacity or a full SDK review.
MCP wraps shared service envelopes as successful/error tool results; the tool
list and CLI parity still require their own complete inventory.

## Shutdown and failure-path candidates

Signal handlers register before listeners and request ingestion admission closure
on SIGINT/SIGTERM. An outer cleanup calls `shutdown_and_wait` even after startup
or accept errors, retaining the runtime for owned ingestion cleanup. Cleanup
failure takes precedence over the original `run_inner` result. See the existing
[ingestion ownership audit](ingest-job-ownership.md) for its separate evidence.

MCP binds/spawns before the Unix socket lifecycle checks. Early errors after that
point bypass normal MCP cancellation/join cleanup. The Unix accept loop does not
monitor MCP task completion, so an MCP task failure is only observed when normal
shutdown joins it. Accepted Unix connection tasks are detached rather than
explicitly drained. Normal shutdown removes the Unix path (ignoring removal
errors), cancels MCP and waits five seconds; timeout logs a warning and drops its
join handle without an explicit abort. CLI runtime teardown eventually ends
remaining async tasks, but is not equivalent to graceful request completion.
These are source-level lifecycle candidates, not reproduced production incidents.

## Evidence and remaining work

The prior [readiness audit](daemon-readiness.md) covers framed health checks and
private daemon/backend peers with explicit limits. No daemon was launched or
service contacted for this map. Pinning sources and tracing the entry point adds
one leaf: combined coverage is 73/80, with seven codebase leaves remaining.
Private startup-failure/shutdown fixtures, socket access verification, full MCP
parity, response-memory behavior, deployed configuration, recovery and independent
retrieval-quality reviews remain required by the full epic.
