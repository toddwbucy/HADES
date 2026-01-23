"""Advanced benchmark helper for the ArangoDB HTTP/2 client.

Phase 4 expansion for Issue #51. Collects both time-to-first-byte (TTFB)
and end-to-end (E2E) latencies, supports cache-busting, configurable
payload sizes, waitForSync toggles, and simple concurrency sweeps.

Usage examples:

```
poetry run python tests/benchmarks/arango_connection_test.py \
    --socket /tmp/arango_ro_proxy.sock \
    --database arxiv_repository \
    --collection arxiv_metadata \
    --key 0704_0001 --key 0704_0002 --iterations 20 \
    --report-json results/get.json

poetry run python tests/benchmarks/arango_connection_test.py \
    --socket /tmp/arango_rw_proxy.sock \
    --database arxiv_repository \
    --collection arxiv_metadata \
    --insert-docs 1000 --doc-bytes 1024 --iterations 10 \
    --wait-for-sync \
    --report-json results/insert.json

poetry run python tests/benchmarks/arango_connection_test.py \
    --socket /tmp/arango_ro_proxy.sock \
    --database arxiv_repository \
    --collection arxiv_metadata \
    --query 'FOR doc IN arxiv_metadata LIMIT 1000 RETURN doc._key' \
    --iterations 10 --batch-size 1000 \
    --concurrency 8 --report-json results/query.json
```

Notes:
- These benchmarks target a single host using Unix domain sockets with
  HTTP/2 enabled. They assume hot caches by default but include toggles
  to vary keys/documents for cache-busting scenarios.
- The tool intentionally reads the entire response body so that E2E
  timings reflect full payload transfer and parsing overhead.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import string
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from core.database.arango.optimized_client import ArangoHttp2Config

DEFAULT_BASE_URL = "http://localhost"


@dataclass(slots=True)
class Samples:
    """Container for TTFB/E2E samples."""

    ttfb: list[float]
    e2e: list[float]

    def summary(self) -> dict[str, dict[str, float]]:
        def quantile(values: list[float], q: float) -> float:
            if not values:
                return float("nan")
            k = (len(values) - 1) * q
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return values[int(k)]
            return values[f] * (c - k) + values[c] * (k - f)

        def build(values: list[float]) -> dict[str, float]:
            if not values:
                return {}
            ordered = sorted(values)
            return {
                "min": ordered[0],
                "max": ordered[-1],
                "avg": statistics.mean(ordered),
                "p50": statistics.median(ordered),
                "p95": quantile(ordered, 0.95),
                "p99": quantile(ordered, 0.99),
            }

        return {"ttfb": build(self.ttfb), "e2e": build(self.e2e)}


def ensure_http2(response: httpx.Response) -> None:
    if response.http_version not in {"HTTP/2", "HTTP/1.1"}:
        raise RuntimeError(
            f"Unexpected HTTP version {response.http_version!r} for "
            f"{response.request.method} {response.request.url}"
        )


def build_httpx_client(config: ArangoHttp2Config) -> httpx.Client:
    auth = None
    if config.username and config.password:
        auth = (config.username, config.password)

    transport = httpx.HTTPTransport(uds=config.socket_path, retries=0)
    timeout = httpx.Timeout(
        connect=config.connect_timeout,
        read=config.read_timeout,
        write=config.write_timeout,
        pool=config.connect_timeout,
    )

    return httpx.Client(
        http2=True,
        base_url=config.base_url or DEFAULT_BASE_URL,
        transport=transport,
        timeout=timeout,
        auth=auth,
        limits=config.pool_limits,
    )


def format_stats(samples: Samples, label: str) -> str:
    summary = samples.summary()

    def fmt(section: dict[str, float]) -> str:
        return ", ".join(f"{k}={v:.2f}ms" for k, v in section.items())

    return (
        f"{label}: "
        f"TTFB[{fmt(summary['ttfb']) if summary.get('ttfb') else 'n/a'}], "
        f"E2E[{fmt(summary['e2e']) if summary.get('e2e') else 'n/a'}]"
    )


def random_payload(size: int, base: dict[str, Any] | None = None) -> dict[str, Any]:
    base_doc = dict(base or {})
    filler_bytes = max(size - len(json.dumps(base_doc).encode("utf-8")), 0)
    if filler_bytes <= 0:
        return base_doc

    alphabet = string.ascii_letters + string.digits
    chunk = ''.join(alphabet[i % len(alphabet)] for i in range(64))
    repeats = filler_bytes // len(chunk) + 1
    filler = (chunk * repeats)[:filler_bytes]
    base_doc["__payload__"] = filler
    return base_doc


def timed_stream(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json_payload: Any | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> tuple[float, float, bytes]:
    """Perform a request, capturing both TTFB and E2E durations."""

    start = time.perf_counter()
    request = client.build_request(
        method,
        url,
        json=json_payload,
        params=params,
        headers=headers,
        content=data,
    )

    response = client.send(request, stream=True)
    try:
        ensure_http2(response)
        response.raise_for_status()
        ttfb_ms = (time.perf_counter() - start) * 1000.0

        body_chunks = bytearray()
        for chunk in response.iter_bytes():
            body_chunks.extend(chunk)

        e2e_ms = (time.perf_counter() - start) * 1000.0
        return ttfb_ms, e2e_ms, bytes(body_chunks)
    finally:
        response.close()


def run_iterations(
    operation: Callable[[], tuple[float, float]],
    iterations: int,
    concurrency: int,
) -> Samples:
    if iterations <= 0:
        return Samples([], [])

    ttfb_samples: list[float] = []
    e2e_samples: list[float] = []
    lock = threading.Lock()

    def wrapped() -> None:
        ttfb_ms, e2e_ms = operation()
        with lock:
            ttfb_samples.append(ttfb_ms)
            e2e_samples.append(e2e_ms)

    if concurrency <= 1:
        for _ in range(iterations):
            wrapped()
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(wrapped) for _ in range(iterations)]
            for future in as_completed(futures):
                future.result()

    return Samples(ttfb_samples, e2e_samples)


def benchmark_get(
    client: httpx.Client,
    config: ArangoHttp2Config,
    collection: str,
    keys: list[str],
    *,
    iterations: int,
    concurrency: int,
    randomize: bool,
) -> Samples:
    counter = Counter()
    key_lock = threading.Lock()

    def choose_key() -> str:
        with key_lock:
            index = counter['i']
            counter['i'] += 1
        if not keys:
            raise ValueError("At least one --key must be provided for GET benchmark")
        if randomize:
            return random.choice(keys)
        return keys[index % len(keys)]

    def operation() -> tuple[float, float]:
        key = choose_key()
        path = f"/_db/{config.database}/_api/document/{collection}/{key}"
        ttfb_ms, e2e_ms, _ = timed_stream(client, "GET", path)
        return ttfb_ms, e2e_ms

    samples = run_iterations(operation, iterations, concurrency)
    print(format_stats(samples, f"GET {collection}/{keys[0] if keys else '?'}"))
    return samples


def benchmark_insert(
    client: httpx.Client,
    config: ArangoHttp2Config,
    collection: str,
    *,
    iterations: int,
    concurrency: int,
    documents_per_batch: int,
    doc_bytes: int,
    wait_for_sync: bool,
) -> Samples:
    base_doc = {"created_at": time.time()}

    def build_payload(batch_index: int) -> bytes:
        docs = []
        for idx in range(documents_per_batch):
            doc = random_payload(doc_bytes, base_doc | {"value": f"bench-{batch_index}-{idx}"})
            docs.append(json.dumps(doc))
        ndjson = "\n".join(docs)
        return ndjson.encode("utf-8")

    def operation_factory(batch_index: int) -> Callable[[], tuple[float, float]]:
        payload = build_payload(batch_index)
        params = {
            "collection": collection,
            "type": "documents",
            "complete": "true",
            "overwrite": "false",
            "onDuplicate": "error",
        }
        if wait_for_sync:
            params["waitForSync"] = "true"

        headers = {
            "Content-Type": "application/x-ndjson",
            "Content-Length": str(len(payload)),
        }
        path = f"/_db/{config.database}/_api/import"

        def op() -> tuple[float, float]:
            ttfb_ms, e2e_ms, body = timed_stream(
                client,
                "POST",
                path,
                params=params,
                headers=headers,
                data=payload,
            )
            response_json = json.loads(body.decode("utf-8"))
            if response_json.get("errors"):
                raise RuntimeError(f"Bulk insert reported errors: {response_json}")
            return ttfb_ms, e2e_ms

        return op

    counter = Counter()
    counter_lock = threading.Lock()

    def operation() -> tuple[float, float]:
        with counter_lock:
            batch_index = counter['batch']
            counter['batch'] += 1
        op = operation_factory(batch_index)
        return op()

    samples = run_iterations(operation, iterations, concurrency)
    print(
        format_stats(
            samples,
            f"INSERT {documents_per_batch} docs (size≈{doc_bytes}B, waitForSync={wait_for_sync})",
        )
    )
    return samples


def benchmark_query(
    client: httpx.Client,
    config: ArangoHttp2Config,
    query: str,
    *,
    iterations: int,
    concurrency: int,
    batch_size: int,
    bind_vars: dict[str, Any],
    vary_bind_key: str | None,
) -> Samples:
    counter = Counter()
    bind_lock = threading.Lock()

    def initial_payload(bind_index: int) -> dict[str, Any]:
        payload = {
            "query": query,
            "batchSize": batch_size,
            "bindVars": bind_vars.copy(),
            "options": {"fullCount": False},
        }
        if vary_bind_key:
            payload["bindVars"][vary_bind_key] = bind_index
        return payload

    def operation() -> tuple[float, float]:
        with bind_lock:
            bind_index = counter['bind']
            counter['bind'] += 1

        payload = initial_payload(bind_index)
        path = f"/_db/{config.database}/_api/cursor"
        ttfb_ms, e2e_ms, body = timed_stream(client, "POST", path, json_payload=payload)
        data = json.loads(body)
        results = data.get("result", [])

        while data.get("hasMore"):
            cursor_id = data.get("id")
            if not cursor_id:
                raise RuntimeError("Cursor indicated hasMore but no id returned")
            follow_path = f"/_db/{config.database}/_api/cursor/{cursor_id}"
            ft, fe, fbody = timed_stream(client, "PUT", follow_path)
            e2e_ms += fe
            data = json.loads(fbody)
            results.extend(data.get("result", []))

        # Consume the results to ensure parsing cost is included.
        _ = len(results)
        return ttfb_ms, e2e_ms

    samples = run_iterations(operation, iterations, concurrency)
    print(format_stats(samples, f"QUERY batch={batch_size}"))
    return samples


def dump_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Report written to {path}")


def parse_bind_vars(raw: Iterable[str]) -> dict[str, Any]:
    bind_vars: dict[str, Any] = {}
    for item in raw:
        key, _, value = item.partition(":")
        if not key:
            continue
        key = key.strip()
        value = value.strip()
        try:
            bind_vars[key] = json.loads(value)
        except json.JSONDecodeError:
            bind_vars[key] = value
    return bind_vars


def main() -> None:
    parser = argparse.ArgumentParser(description="ArangoDB HTTP/2 benchmark helper")
    parser.add_argument(
        "--socket",
        default="/run/hades/readonly/arangod.sock",
        help="Unix socket path (use the read-write proxy socket for insert tests)",
    )
    parser.add_argument("--database", default="_system", help="Database name")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="HTTP base URL (default: http://localhost)")
    parser.add_argument("--collection", required=True, help="Collection name for operations")
    parser.add_argument("--key", action="append", help="Document key for GET benchmark (repeatable)")
    parser.add_argument("--randomize-get", action="store_true", help="Append unique suffix to each GET key to bust caches")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per benchmark")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent workers")
    parser.add_argument("--query", help="AQL query to benchmark")
    parser.add_argument(
        "--query-bind",
        action="append",
        default=[],
        help="Bind variable as key:json (e.g. offset:0). Repeat for multiple variables.",
    )
    parser.add_argument("--vary-bind-key", help="Bind variable key to vary per iteration (integer suffix)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Cursor batch size")
    parser.add_argument("--insert-docs", type=int, default=0, help="Number of documents per insert batch")
    parser.add_argument("--doc-bytes", type=int, default=1024, help="Approximate size of each document in bytes")
    parser.add_argument("--wait-for-sync", action="store_true", help="Enable waitForSync on bulk import")
    parser.add_argument("--report-json", type=Path, help="Optional path to save benchmark results as JSON")
    args = parser.parse_args()

    config = ArangoHttp2Config(
        database=args.database,
        socket_path=args.socket,
        base_url=args.base_url,
        username=args.username,
        password=args.password,
    )

    report: dict[str, Any] = {
        "database": args.database,
        "collection": args.collection,
        "socket": args.socket,
        "base_url": args.base_url,
        "iterations": args.iterations,
        "concurrency": args.concurrency,
    }

    with build_httpx_client(config) as client:
        if args.key:
            samples = benchmark_get(
                client,
                config,
                args.collection,
                args.key,
                iterations=args.iterations,
                concurrency=args.concurrency,
                randomize=args.randomize_get,
            )
            report["get"] = {
                "keys": args.key,
                "randomize": args.randomize_get,
                "stats": samples.summary(),
            }

        if args.insert_docs:
            samples = benchmark_insert(
                client,
                config,
                args.collection,
                iterations=args.iterations,
                concurrency=args.concurrency,
                documents_per_batch=args.insert_docs,
                doc_bytes=args.doc_bytes,
                wait_for_sync=args.wait_for_sync,
            )
            report["insert"] = {
                "docs_per_batch": args.insert_docs,
                "doc_bytes": args.doc_bytes,
                "wait_for_sync": args.wait_for_sync,
                "stats": samples.summary(),
            }

        if args.query:
            bind_vars = parse_bind_vars(args.query_bind)
            samples = benchmark_query(
                client,
                config,
                args.query,
                iterations=args.iterations,
                concurrency=args.concurrency,
                batch_size=args.batch_size,
                bind_vars=bind_vars,
                vary_bind_key=args.vary_bind_key,
            )
            report["query"] = {
                "query": args.query,
                "batch_size": args.batch_size,
                "bind_vars": bind_vars,
                "vary_bind_key": args.vary_bind_key,
                "stats": samples.summary(),
            }

    if args.report_json:
        dump_report(args.report_json, report)


if __name__ == "__main__":
    main()
