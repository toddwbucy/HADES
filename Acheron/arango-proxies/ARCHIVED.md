# Archived: ArangoDB Unix Socket Proxies

This Go code was embedded in HADES but has been archived in favor of the standalone repository:

**[arango-unix-proxy](https://github.com/toddwbucy/arango-unix-proxy)**

## Why Archived

1. **Duplication**: Maintaining the same code in two places leads to drift
2. **Separation of concerns**: The proxy is an infrastructure component, not application code
3. **The HADES Python client now auto-detects socket availability and falls back to network transport**

## Configuration

The HADES database client (`core/database/arango/memory_client.py`) now supports:

- `use_proxies=True`: Explicitly use proxy sockets
- `use_proxies=False`: Use direct socket or network
- `use_proxies=None` (default): Auto-detect - check if sockets exist, fall back to network

Environment variables:
- `ARANGO_RO_SOCKET`: Read-only proxy socket path
- `ARANGO_RW_SOCKET`: Read-write proxy socket path
- `ARANGO_SOCKET`: Direct ArangoDB socket path
- `ARANGO_HTTP_BASE_URL`: Network URL (default: `http://localhost:8529`)

## Running the Proxies

Install and run from the standalone repository:

```bash
git clone https://github.com/toddwbucy/arango-unix-proxy
cd arango-unix-proxy
go build -o roproxy ./cmd/roproxy
go build -o rwproxy ./cmd/rwproxy
./roproxy &
./rwproxy &
```
