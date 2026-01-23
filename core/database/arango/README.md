# ArangoDB Optimized HTTP/2 Client

## Benchmarks

| Operation                                   | PHP Subprocess | HTTP/2 (direct) | HTTP/2 via proxies |
|---------------------------------------------|----------------|-----------------|--------------------|
| GET single doc (hot cache, p50)             | ~100 ms        | ~0.4 ms         | ~0.6 ms            |
| GET single doc (hot cache, p95 target)      | n/a            | 1.0 ms          | 1.0 ms             |
| Insert 1000 docs (waitForSync=false, p50)   | ~400–500 ms    | ~6 ms           | ~7 ms              |
| Query (LIMIT 1000, batch size 1000, p50)    | ~200 ms        | ~0.7 ms         | ~0.8 ms            |

## Usage

### Client

```python
from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config

config = ArangoHttp2Config(
    database="arxiv_repository",
    socket_path="/run/hades/readonly/arangod.sock",
    username="arxiv_reader",
    password="...",
)
with ArangoHttp2Client(config) as client:
    doc = client.get_document("arxiv_metadata", "0704_0003")
    print(doc)
```

### Workflow Integration

```python
from core.database.database_factory import DatabaseFactory

memory_client = DatabaseFactory.get_arango_memory_service()
try:
    documents = memory_client.execute_query(
        "FOR doc IN @@collection LIMIT 5 RETURN doc",
        {"@collection": "arxiv_metadata"},
    )
finally:
    memory_client.close()
```

### Proxy Binaries

1. Build: `cd core/database/arango/proxies && go build ./...`
2. Run RO proxy: `go run ./cmd/roproxy`
3. Run RW proxy: `go run ./cmd/rwproxy`

Sockets default to `/run/hades/readonly/arangod.sock` and `/run/hades/readwrite/arangod.sock` (systemd-managed). Ensure permissions (0640/0600) and adjust via env vars `LISTEN_SOCKET`, `UPSTREAM_SOCKET`.

### Benchmark CLI (Phase 4)

`tests/benchmarks/arango_connection_test.py` now supports:

- TTFB and E2E timing (full body consumption).
- Cache-busting via multiple `--key` values or varying bind variables.
- Adjustable payload size (`--doc-bytes`), `waitForSync`, and concurrency (`--concurrency`).
- JSON report emission (`--report-json`) for regression tracking.

Example:

```
poetry run python tests/benchmarks/arango_connection_test.py \
    --socket /run/hades/readonly/arangod.sock \
    --database arxiv_repository \
    --collection arxiv_metadata \
    --key 0704_0001 --key 0704_0002 \
    --iterations 20 --concurrency 4 \
    --report-json reports/get_hot.json
```

### Testing Infrastructure

- The HTTP/2 memory client is now the default access path for automated tests.
- Run `poetry run pytest tests/core/database/test_memory_client_config.py` for a quick sanity check.
- Future regression suites should share the proxy-aware fixtures so workflows exercise the same transport stack.

### Production Hardening Notes

- Treat the RO (`/run/hades/readonly/arangod.sock`) and RW (`/run/hades/readwrite/arangod.sock`) proxies as the security boundary. Plan to ship them via systemd socket units with explicit `SocketUser`/`SocketGroup` assignments and 0640/0600 modes.
- Arango HTTP responses are enforced to negotiate HTTP/2; mismatches raise immediately.
- Reference benchmark summary: see `docs/benchmarks/arango_phase4_summary.md` for the latest latency table.
- Systemd templates for the proxies live in `docs/deploy/arango_proxy_systemd.md`.
