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
starting RSS, recall@10 against exact ranking, index-build time, and wall time.
Logs and results remain in the printed private artifact directory.

## Interpretation and limitations

The checked-in 1,024 × 64 pilot validates the harness. Sixteen samples are too few
for a stable tail-latency estimate. Every indexed pilot query matched exact top-10
results, but clustered synthetic vectors are easy for the index. This is neither
a representative language-model quality evaluation nor a production capacity claim.

The benchmark directly measures retrieval engines; it does not exercise handler
admission, embedding inference, result hydration, or graph reranking. It excludes
server RSS and GPU memory. Cases share a process and allocator state, so compare
reported baseline RSS alongside sampled peaks. Sampling can miss short peaks.
Further corpus sizes, production-width vectors, end-to-end admission tests, and
versioned query/relevance evaluation are required before setting final budgets
or recommending an indexed production path.
