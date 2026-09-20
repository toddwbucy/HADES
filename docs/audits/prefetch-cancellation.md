# Prefetch cancellation contract

Part of #77 and epic #12. The earlier audit demonstrated graph captures surviving
producer abort. The prefetcher now owns cancellation through both blocking jobs.

`stop()` and `Drop` request cancellation and close the receiver without waiting.
The producer aborts queued blocking jobs, signals running samplers and joins both
handles. `shutdown().await` additionally waits for that producer to finish and
reports a producer join failure. Dropping the shutdown future still requests
cancellation, but forfeits its completion acknowledgement. Cleanup requires the
runtime to remain available; runtime teardown is not a hard termination bound.

Sampling checks cancellation before allocation, every 1,024 indexed edges,
between candidate batches of at most 1,024 attempts, and before returning a
result. Cancelled partial samples are discarded. Allocator calls and scheduler
queue delays are not preemptible, so no wall-clock shutdown guarantee is made.
The ordinary seeded sampling function keeps its existing deterministic behavior.

The normal/early-stop training path now joins prefetch before restoring the best
checkpoint. Error returns and dropped training futures request cooperative cleanup
through Drop; they do not claim synchronous completion. One-time graph preparation
and final test-negative generation are separate blocking operations and are not
made cancellable by this change. Python model-step cancellation remains open in
#77; this change alone does not close that issue.

Validation covers queued jobs behind a finite worker gate, shutdown with a full
output buffer, dropped-prefetch graph release, cancellation at each exercised
sampler checkpoint, unchanged seeded output, cross-thread cancellation after
a real sampler has entered its indexing work, and cancellation after one child
result has already been consumed while the other child remains running. The
combined join future is retained across cancellation to avoid polling a completed
child handle twice. All fixtures use synthetic graphs
and no database, GPU or production provider.

```sh
CARGO_BUILD_JOBS=1 cargo test --offline -p hades-prefetch --lib
cargo fmt --all -- --check
```

No production deployment has occurred. Any later maintenance risking data loss
requires the owner's verified active-data snapshot and arranged downtime policy.
