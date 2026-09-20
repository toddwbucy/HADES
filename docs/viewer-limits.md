# Viewer resource limits

`hades-viewer` uses the HADES CLI JSON contract. It starts its own bounded CLI
children; it does not link the database/core crates. The following limits apply
to the viewer process, including startup database discovery.

| Boundary | Default |
|---|---|
| Concurrent backend children | 4; excess requests receive HTTP 503 |
| One child's stdout / stderr | 8 MiB / 16 KiB; overflow fails explicitly |
| One child's runtime | 30 seconds, followed by owned-group cleanup/reaping |
| Whole HTTP request | 60 seconds across all backend calls |
| Accepted HTTP connections | 16; excess sockets are closed before parsing |
| Connection lifetime | 120 seconds, including stalled response writes |
| Serialized JSON response | 16 MiB, enforced while writing JSON |
| One assembled graph | 32 collections, 50,000 documents, 8 MiB cumulative serialized document content |

`--limit` and the graph request's `limit` can reduce collection exports. A request
cannot widen the operator's cap. Global-budget overflow returns an error rather
than pretending a complete graph was returned; existing successful capped
snapshots retain their `partial` metadata. Reduce the selected graph/export size
when a budget is exceeded.

Children have null stdin and separate process groups. Timeout, overflow, client
cancellation and SIGINT/SIGTERM trigger cleanup. Admission remains held until the
direct child is reaped; uninterruptible kernel work is not promised a hard cleanup
deadline. Raw child diagnostics are not copied into HTTP errors or server logs.
Signal handlers are installed before discovery begins.

The current workspace enables Axum HTTP/1, not HTTP/2. Connection admission stays
with the socket through response writes, rather than ending when a handler
returns. These are resource-policy limits, not a universal RSS guarantee.

## Isolated measurement

[The recorded pilot](benchmarks/viewer-slow-readers.json) held 16 unread responses
with synthetic names of 8,387,584 bytes each. Viewer peak RSS/high-water mark was
196,488 KiB (191.9 MiB), versus a 9,376 KiB baseline. The seventeenth connection
was refused; releasing one client allowed another request. The private viewer
exited zero after cleanup. Backend sampled peak RSS is reported separately;
separate maxima must not be added as a simultaneous process-tree peak.

The viewer and synthetic children ran on one CPU with a 2 GiB address-space limit.
Measurements exclude kernel socket buffers and use neither production corpora nor
a real HADES backend. Source, lockfile, executable and synthetic-backend hashes
are preserved. This is a slow-reader retention pilot, not a worst-case graph
transformation benchmark or production capacity certification.

To repeat against an explicit isolated build:

```bash
cargo build -p hades-frontend --locked
python3 scripts/benchmark_viewer.py \
  --viewer /absolute/path/to/isolated/target/debug/hades-viewer \
  --output /tmp/viewer-slow-readers.json
```

The runner supplies a private synthetic backend and loopback port, confines the
viewer, and cleans up on success, failure or interruption. Signal/lifecycle tests
must run where signals to the test's own processes are delivered normally.
