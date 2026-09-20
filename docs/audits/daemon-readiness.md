# Daemon installation readiness (#64)

The installation harness previously checked a daemon PID and then invoked
`hades db stats` directly. That CLI command bypasses the daemon, so the success
banner did not establish daemon socket readiness.

The harness now runs `scripts/check_daemon_ready.py` as the `hades` user against
its actual socket and checks the expected configured database. The Unix protocol
has no per-request database selector; the probe checks the returned identity.
It sends a framed `db.health` request, requires a matching request ID, successful
error-free envelope, and healthy reader/writer connection flags. The health command is Internal-tier, so the probe uses the local socket’s
admin session under the service identity. It does not write to the database.

The overall startup deadline defaults to 30 seconds, each attempt gets at most
two seconds, and every read uses the remaining deadline rather than restarting
a timeout. Responses are limited to 64 KiB before allocation. Invalid framing,
JSON, command errors, wrong database identity and degraded health cannot pass.
The container installs Python 3 for this standard-library-only probe.

Six private Unix-peer tests cover successful framing, missing and stalled
listeners, truncated/oversized/invalid frames, failed or mismatched responses,
degraded connections and invalid timeout values. They run through existing CI
script discovery. A freshly built private daemon was also tested against a synthetic Unix HTTP
database peer: healthy passed, wrong database failed, and degraded backend failed.
That integration caught and corrected the initial agent-session authorization
error. A complete container installation remains a separate unperformed check.

`db.health` checks the configured clients' version endpoints. Passing proves
transport/service dispatch and those connection checks, not write authorization,
collection-level access, ML readiness, inference latency, or production health.
The probe does not replace end-to-end ingestion/query validation or the separate
real-systemd installation rehearsal. No active service is contacted by tests.
