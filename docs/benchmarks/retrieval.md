# Isolated retrieval benchmarks

Run only against the runner's disposable server:

```bash
cargo fetch --locked
python3 scripts/test_isolated_database.py --arangod /path/to/test/arangod \
  --benchmark --benchmark-rows 1024 --benchmark-dimension 64 --benchmark-trials 16
```

The opt-in Rust integration benchmark seeds deterministic clustered vectors into
its own temporary database, records exact top-10 reference results, builds a
16-centroid cosine index, and checks that EXPLAIN selects `use-vector-index`.
It compares bounded streaming with index search at 4 and 16 probes, at concurrency
1, 4, and 8. Parameters permit 1,024–100,000 rows, 16–2,048 dimensions, and
16–512 trials per case. The runner enforces one CPU affinity, lower priority,
an 8 GiB address-space ceiling, and a command deadline; it stops its own server.

Each `BENCH` JSON line reports p50/p95/p99 latency, client RSS sampled every 2 ms,
starting RSS, local-server RSS (when using `--arangod`), recall@10 against exact ranking, index-build time, and wall time.
Logs and results remain in the printed private artifact directory.

## Interpretation and limitations

The checked-in 1,024 × 64 pilot validates the harness. Sixteen samples are too few
for a stable tail-latency estimate. Every indexed pilot query matched exact top-10
results, but clustered synthetic vectors are easy for the index. This is neither
a representative language-model quality evaluation nor a production capacity claim.

The benchmark directly measures retrieval engines; it does not exercise handler
admission, embedding inference, result hydration, or graph reranking. GPU memory is excluded. Docker server RSS is not sampled. Cases share a process and allocator state, so compare
reported baseline RSS alongside sampled peaks. Sampling can miss short peaks.
Further corpus sizes, production-width vectors, end-to-end admission tests, and
versioned query/relevance evaluation are required before setting final budgets
or recommending an indexed production path.

The 1,024 × 2,048 pilot records 32 trials per case. Single-query median latency
was 1,160 ms for streaming and 2.9 ms for four-probe indexed search. Both indexed
settings matched exact top-10 results on this synthetic fixture. Sampled client
RSS peaked below 16 MiB across these engine-only cases. These observations do
not size the complete daemon or establish production-model relevance quality.

## Full-handler allocation pilot

Run `python3 scripts/test_isolated_database.py --arangod /path/to/arangod
--handler-benchmark --benchmark-rows 1024 --benchmark-trials 16` as one command.
The runner creates a private database and fixed-vector mock embedder. This opt-in
fixture exercises service admission, embedding-response parsing, exact scan,
1,000-result hydration, hybrid and 2,048-dimensional structural reranking, a
64 KiB query, and retained response JSON plus serialization buffers. It excludes
model inference and Unix/MCP transport queues.

[Captured pilot](search-handler-pilot-1024x2048.json): peak client RSS was 30.8 MiB
at concurrency one and 72.1 MiB at concurrency four. Responses serialized to
1,059,729 bytes including the query envelope. Median latency was 2.70 s and
10.35 s respectively on the runner's single CPU affinity. Both cases passed and
the private server exited cleanly. Sixteen trials and a shared allocator make
these allocation pilots, not stable tail-latency estimates or an RSS guarantee.
Corpus scaling and transport retention still require separate validation before
finalizing the provisional search and MCP budgets.

The [4,096-row repeat](search-handler-pilot-4096x2048.json) retained similar client
memory: 31.3 MiB at concurrency one and 70.0 MiB at four, with medians 4.02 s and
15.32 s. It used the same maximum result count, query size, and structural width;
only the scanned corpus grew. Both private database runs completed and shut down.

## MCP slow-reader allocation pilot

Run `python3 scripts/benchmark_transport_memory.py` with the same isolated Cargo
target/settings. This separate runner applies one CPU, nice 10, an 8 GiB
address-space cap, and a 180-second timeout; it owns and cleans up its subprocess
group. An ephemeral loopback server exposes only a synthetic fixture tool.

[Captured near-limit pilot](mcp-slow-readers-pilot.json): 64 responses containing
2,096,128 text bytes each were held unread for two seconds. Peak process RSS was
273.0 MiB, including both HTTP clients and server. Production session, connection,
stream, serialized-message, and request-body controls were exercised. It excludes
search/inference work and does not saturate all common notification caches or
maximum input bodies. Kernel socket buffers are outside the RSS measurement.
These separate pilots cannot be added together as a measured combined-process
ceiling; allocator behavior, input parsing, and other handlers remain separate.

After normal terminal responses were changed to release request admission as soon
as cache cleanup is acknowledged, the [verification repeat](mcp-slow-readers-final.json)
passed at 288.2 MiB peak RSS. This retains the same 64 near-limit unread responses;
the variation reinforces that these are measured workloads, not exact RSS caps.
