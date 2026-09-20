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
| Cursor page | 4 rows, at most 256 KiB HTTP body |
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
manager maps are cleaned. These counts are provisional capacity controls pending
full-handler measurements, not RSS guarantees.

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
removal. Normal responses currently retain their reservation for that same window.
Clients should back off on admission errors and reinitialize after session expiry.

Wire limits apply before JSON parsing, including chunked responses. Row count is
only a work limit, not an estimate of memory use. The reservations are conservative
accounting, not an OS RSS limit: transport buffers, JSON allocation, runtime
state, unrelated commands, and database caches have separate costs. The current
numbers remain subject to full-handler workload measurements before rollout.
Engine-only pilots stayed below 16 MiB client RSS and cannot establish an entire
daemon's memory requirement. The server's AQL accounting likewise does not bound
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
