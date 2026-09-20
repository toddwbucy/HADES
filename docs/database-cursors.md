# AQL cursor lifetime and cancellation

`db::query` owns each cursor in a Tokio task separate from the calling future.
The task uses one client for creation, continuation, and deletion. With different
reader/writer endpoints it uses the writer throughout because the read-only proxy
cannot service cursor continuation or deletion.

Creation retains response ownership even if the caller disappears. Once an ID
arrives, the task records it before parsing results. Pagination observes caller
cancellation and stops fetching pages; known IDs are then deleted. A changed ID
is treated as malformed, with both known IDs retained for cleanup. Invalid IDs
are never interpolated into a request path.

## Bounds and limitations

- Query execution, including response bodies and all pages: 60 seconds.
- Cleanup after success, failure, timeout, or cancellation: two seconds total.
  Delivery waits for this bounded attempt so a one-shot CLI cannot normally
  shut down its runtime before deletion. The tradeoff is up to two seconds of
  additional latency if cleanup stalls.
- Requests set `options.maxRuntime=60` and cursor `ttl=60` as server fallbacks.
  TTL applies to idle cursor retention and garbage collection is server-driven;
  it is not a precise wall-clock deletion guarantee. See the
  [ArangoDB cursor API](https://docs.arango.ai/arangodb/stable/develop/http-api/queries/aql-queries/).
- If creation never returns a readable ID, cleanup cannot address that cursor.
  Transport failure, process termination, or Tokio runtime shutdown can also
  prevent deletion; runtime/TTL limits remain the fallback. Non-404 cleanup
  errors and cleanup timeouts are logged.
- Cancellation does not roll back an AQL mutation already submitted to the server.
- A completed query is returned even if best-effort cleanup fails. A missing
  cursor during deletion is normal after the final batch.
- Result rows are still accumulated in memory. Aggregate search admission and
  bounded retrieval are separate work under #22; the lifecycle budget is not a
  memory guarantee.

## Verification

`cargo test -p hades-core --lib db::query::tests::` covers normal completion,
malformed first/continuation responses, server errors, cancellation during
creation and pagination, query timeout, and a stalled deletion response using
private Unix socket mocks. Split-endpoint tests give the reader an absent socket
so an accidental reader request fails.

`cargo test -p hades-core --test cursor_lifecycle` runs the separate
disposable-database contract. A private proxy delays
creation or continuation, the caller is cancelled, and direct requests to the
separate ArangoDB instance must report the captured cursor ID missing after
cleanup. Use only a resource-limited test server, `ARANGO_SOCKET` pointing at its
private socket, fixture credentials, and `ARANGO_TESTS=1`. Missing prerequisites
then fail instead of silently skipping. Never run it against a production server.
