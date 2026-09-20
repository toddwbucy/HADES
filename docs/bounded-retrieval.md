# Bounded exact retrieval

`db.query` ranks stored vectors with a streaming exact top-K heap. It retains
keys/scores for at most the requested K and hydrates only those results. Equal
scores use stable chunk/parent-key ordering. Index search is measured separately
in [the benchmark](benchmarks/retrieval.md); the handler does not silently switch
to approximate ranking when an index appears.

## Resource controls

| Control | Current bound |
|---|---:|
| Query text | 64 KiB UTF-8 |
| Results requested | 1–1,000 |
| Query vector width | 1–8,192 |
| Exact scan work | 100,000 returned embedding rows |
| Vector/structural cursor page | 4 rows, at most 256 KiB HTTP body |
| Detail cursor page | 1 row, at most 1 MiB + 64 KiB cursor envelope |
| Single-query embedding response | 256 KiB HTTP body |
| Hydrated result data | 1 MiB total, checked again after reranking |
| Structural vector width | 2,048 |
| Server AQL memory request | 32 MiB per search cursor |
| Cursor lifetime / cleanup attempt | 60 seconds / 2 seconds |
| Process-wide admission accounting | 512 MiB, 128 MiB reserved per search |

The admission reservation currently permits four simultaneous searches across
all databases/profiles in one process. Further requests receive
`SEARCH_OVERLOADED`; callers can retry with bounded backoff. A timed-out caller
retains its cursor reservation until the independent owner finishes cleanup.
The existing service timeout envelope remains `INTERNAL` with `request timed out`.
Standalone CLI processes have separate admission accounting.

The Unix daemon and MCP listener share a cap of 64 open connections, retaining
slots through response writes. Excess connections close before request parsing;
clients must reconnect with bounded backoff. Unix writes have a 15-second deadline,
including framing and flushing. MCP TCP connections have a five-minute absolute
lifetime, including stalled reads/writes and SSE streams. Clients reconnect using
their existing session and Last-Event-ID; a socket expiry does not delete the
session. MCP additionally admits 32 sessions, 64 retained request owners, and
64 concurrent GET/resume streams per process. A request reservation survives HTTP
cancellation and remains held through explicit SDK cache cleanup and any queued
response stream. Session slots return only after the worker has exited and both
manager maps are cleaned. These operational count limits are tested at saturation
and constrain retained work; they are not process RSS guarantees.

MCP request bodies are collected through an explicit 16 MiB byte limit before
SDK parsing, including chunked bodies with no Content-Length. Body reads have a
15-second deadline and a process-wide cap of 64 simultaneous admitted requests.
Bearer authorization runs first. This explicit middleware is required because
the SDK does not use Axum's body extractors.

Messages entering MCP SDK queues are limited to 2 MiB of serialized JSON, measured
without allocating an extra serialized copy. Oversized responses become terminal
`MCP_RESPONSE_TOO_LARGE` errors with their original request IDs; oversized
non-responses terminate the transport. Handler-side allocations still need their
own limits. SDK channel/cache capacity remains 16 messages.

An isolated test verifies reconnecting an active request using Last-Event-ID.
The installed rmcp 2.2 implementation removes normal terminal-response caches
immediately; completed-response replay therefore fails, despite its exposed
completed-cache TTL setting. Do not promise late replay after completion. Cancelled
requests use a conservative 60-second retention window followed by explicit cache
removal. Normal terminal responses release their owner after cleanup acknowledgment;
any response stream still queued to HTTP retains its own reservation reference.
Clients should back off on admission errors and reinitialize after session expiry.

Wire limits apply before JSON parsing, including chunked responses. Row count is
only a work limit, not an estimate of memory use. The reservations are conservative
accounting, not an OS RSS limit: transport buffers, JSON allocation, runtime
state, unrelated commands, and database caches have separate costs.

Keep the measured handler policy at **128 MiB per search and 512 MiB total**.
Full-handler pilots with 1,000 results, 64 KiB queries, and maximum-width structural
reranking peaked at 30.8–31.7 MiB for one request and 68.4–72.1 MiB for four.
Growing the corpus from 1,024 to 4,096 vectors increased latency without increasing
retained memory. The reservation is over four times the observed single-request
process peak, retaining headroom for allocator, input, and model-width variation.
This is a conservative admission decision, not a measured worst-case guarantee.

MCP response retention is separate: 64 unread near-2 MiB responses peaked at
273–288 MiB in loopback fixtures including clients. Its count and message-size limits
must not be described as part of the 512 MiB handler pool. At the configured caps,
32 sessions can retain up to 512 common-channel messages (1 GiB of serialized
payload at 2 MiB each); active request channels and parsing have additional costs.
Maximum-size request bodies alone can occupy 1 GiB across 64 admitted readers.
Those simultaneous extremes were not measured by the response pilot. Budget
host/cgroup capacity for the full command mix and SDK queues before deployment;
never set a daemon memory ceiling to 512 MiB on the strength of search accounting.
[Reproduction and captured measurements](benchmarks/retrieval.md) state their scope. The server's AQL accounting likewise does not bound
its process RSS. Keep OS-level capacity planning separate.

Known cursors are deleted on success, failure, and cancellation. If an oversized
or failed creation response prevents learning the cursor ID, server runtime/TTL
bounds remain the fallback; see [cursor ownership](database-cursors.md). Cleanup
cannot execute after process termination. Errors return no partial result list.

## Existing corpora and rollout

Search now rejects missing/mismatched stored `model` or `dimension`, malformed
vector values, and zero vectors. Previously some malformed rows were silently
skipped or compared without provenance checks. Do not repair metadata by guessing
or relabeling a vector: rebuild the affected corpus using the intended model and
profile, then verify representative query results in an isolated database.

Do not replace the active server's binary or modify its corpus as part of audit
validation. Follow the isolated rebuild/cutover/rollback process in
[code-file identities](code-file-identities.md), retain the prior binary and
corpus, and establish a maintenance window for an eventual deployment. An older
binary may silently skip invalid vectors; restoring it does not fix corrupt data.
If response limits reject large chunks or high result counts, reduce K or review
chunking in an isolated rebuild. Limits never silently truncate text or rankings.
