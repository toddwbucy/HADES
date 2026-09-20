# HADES Audit: Isolated Execution and Deeper Code Review

Date: 2026-09-19 (America/Chicago; test output uses UTC on September 20)  
Parent epic: https://github.com/toddwbucy/HADES/issues/12  
Audited revision: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`

## Outcome and scope

This pass reproduced silent source-file overwriting, malformed embedding-response acceptance, and five training-contract failures. Existing tests largely pass, which demonstrates that the missing contracts matter more than the existing pass count.

The full audit remains open. This is deeper selected code review and isolated execution, not a statement that every source line, supported analyzer, GPU backend, live ACL, backup, or retrieval-quality claim has been verified.

No production implementation was edited. All persistent additions are audit documentation and reproduction programs under `docs/audits/`. Existing `AGENTS.md` was not changed. No production database writes, service restarts, GPU inference/training, dependency installations, or binary replacements were performed.

## Execution environment and checks

Rust 1.98.1 and protoc 36.1 were already installed. Cargo used its existing dependency cache offline, one build job, reduced scheduling priority, disabled incremental/debug artifacts, and `CARGO_TARGET_DIR=/tmp/hades-audit-target`. Python used the existing services environment with bytecode writing disabled, CUDA hidden, and numerical thread pools limited to one thread.

A separate ArangoDB process used a fresh mode-0700 temporary directory, its own data store, Unix-only socket, authentication disabled only within that private disposable instance, one CPU at reduced priority, and 64 MiB block/write-buffer budgets. HADES configuration and socket overrides pointed exclusively to it; embedding/extraction endpoints pointed to nonexistent sockets inside the same temporary directory. Both successful audit database processes exited cleanly after their fixtures.

| Check | Result and qualification |
| --- | --- |
| CLI unit target | 100 reported passing after rerunning two socket-pair tests outside the socket-restricting sandbox |
| Core unit target | 456 reported passing after rerunning its temporary-socket client tests outside the sandbox |
| Prefetch unit target | 34 passed |
| Frontend unit target | 14 passed |
| Protobuf unit target | Zero unit tests |
| Service-free integration targets | 11 passed: config 2, pipeline 5, protobuf 4 |
| ArangoDB cache integration | 5 passed in strict mode against the isolated database |
| Codebase CLI subset | 60 passed in strict mode against the isolated database, including ingest refresh, edge remapping, retirement, and pruning |
| Existing Python training tests | 16 passed; 1 CUDA-specific test deliberately skipped |
| Existing WeaverTools scope tests | 6 passed using the existing corpus read-only |
| New Python contract probes | 5 failed as intended to reproduce defects; 2 unawaited-coroutine warnings |
| New Rust contract probes | 3 failed as intended to reproduce defects |
| End-to-end collision fixture | CLI reported two successful inputs; database retained one file record |

The 60 codebase tests are a subset of the 100 CLI tests, not 60 additional distinct tests. The initial no-database unit runs can count resource-skipping tests as successful; the strict isolated run explicitly exercised the selected database-dependent codebase tests. Four initial socket failures were sandbox limitations and passed with the required socket access; they are not application defects.

Main evidence directories:
- `/tmp/hades-audit-db-fnrk3e0f/`: initial isolated cache and collision run.
- `/tmp/hades-audit-db-sepxu_28/`: strict codebase run, cache tests, and collision output.
- `/tmp/hades-audit-unit.log`, `/tmp/hades-audit-other-units.log`, `/tmp/hades-audit-remaining-units.log`, `/tmp/hades-audit-service-free.log`: existing Rust suite output. Initial unit logs retain the sandbox failures; successful targeted reruns are recorded above.

## Verified findings

### B01 — Distinct code paths overwrite each other (P1)

`keys::file_key` in `crates/hades-core/src/db/keys.rs:71` maps both `a/b.py` and `a_b.py` to `a_b_py`. Ingest uses that identity for file replacement, chunk keys, and symbol cleanup.

The isolated fixture contained two Python functions in those distinct files. HADES returned exit 0, `total: 2`, and `success: true` for both. The database returned only:
```json
{"key":"a_b_py","path":"a_b.py","symbol_count":1}
```
This is confirmed data loss within the disposable graph, not merely a theoretical hash collision. Fix identity generation and pre-write conflict detection before applying migrations to existing graphs.

### B02 — Held-out edges participate in training message passing (P1)

`services/training/server.py:155-263` loads all edges, encodes all edges, then uses `train_edge_indices` only to select positive loss targets. A CPU spy showed encoder inputs `[(0,1),(1,2),(2,3)]` when training indices contained only edge 0.

Held-out link-prediction relationships are therefore visible to the encoder. Validation/test metrics cannot establish unseen-link quality under the intended split. Separate training adjacency from target edges and define the evaluation protocol explicitly.

### B03 — Incompatible graph load discards restored weights (P1)

A CPU fixture saved and restored a six-feature checkpoint, then loaded a graph with seven-feature vectors. `LoadGraph` succeeded and rebuilt the model with fresh parameters. The real update path calls checkpoint loading before graph loading in this same order.

Fresh-training initialization may legitimately choose feature width; checkpoint-backed inference must validate compatibility and preserve weights on rejection. Also review semantic compatibility: checkpoints store relation/collection counts without their identity mappings, while the loader sorts the currently present collection names.

### B04 — Malformed embedding responses are accepted (P2)

Local Unix-socket mock servers returned:
1. Two vector rows both claiming `index: 0` for two inputs.
2. Vectors `[1,"bad"]` and `[0,"bad"]`.

Both were accepted by `crates/hades-core/src/persephone/embedding.rs:614-653`. Indices are sorted without checking a one-to-one mapping; nonnumeric elements are dropped, yielding accepted one-dimensional vectors in the second case.

Require exact index coverage, numeric finite elements, nonempty vectors, expected dimensions and model compatibility before associating results with chunks.

### B05 — Training metrics mishandle ties and empty splits (P2)

`_auc(torch.zeros(1), torch.zeros(1))` returns 0.0 instead of 0.5. Double argsort assigns distinct ranks to ties rather than average ranks.

`_bce_link_loss(torch.empty(0), torch.empty(0))` returns `loss=nan, accuracy=0.0`. Split sizes are floored in Rust, so small graphs can reach empty validation/test splits. Dense graphs can also exhaust negative sampling attempts.

Define valid split sizes and unavailable metrics explicitly; add tie-correct AUC and deterministic evaluation fixtures. These are two separate failing probes grouped into one metric workstream.

### B06 — Training precondition aborts are not awaited (P2)

Calling `TrainStep` before initialization with an asynchronous gRPC context produces unawaited-coroutine warnings and an AttributeError on `self.model.train()`.

`_require_loaded`, `LoadGraph`, and `Checkpoint` contain unawaited `context.abort` calls although the service uses `grpc.aio`. Make rejection asynchronous and verify status codes and state preservation.

## Additional review observations

These are narrower static observations or follow-up hypotheses, not newly reproduced production failures:
- Ordinary ingestion deletes and rewrites related collections without a transaction; some cleanup errors are logged as nonfatal. Exercise failure between steps, including forced re-ingest of unchanged content, before accepting self-healing claims.
- The Python package include list contains extraction, embedding and generated bindings but excludes training and adapters. Package discovery confirmed those omissions; checkout-based service execution currently masks them. Validate the intended wheel/install contract.
- The ordinary embedding TCP client constructs a plain HTTP connector although endpoint parsing accepts HTTPS. Verify advertised TLS support with an isolated TLS fixture.
- Plain gzip extraction reads all expanded text without the tar path's expansion bound. Review decompression and archive-member budgets with small synthetic fixtures.
- Ingest admission reads running jobs before writing the new job and treats read failure as an empty list. Concurrent requests and failure injection need an isolated test of the intended cap.
- Checkpoint filenames and inference files are shared within a chosen directory, while the training service holds mutable global model/graph state. Concurrent-client isolation still needs explicit review.
- Graph loading derives nodes from edges and indexes only currently observed collections. Review isolated-node behavior and stable collection-index identity across schema/graph evolution.
- Vector model hashes derive from model identifiers; replacing weights under the same identifier requires a defined invalidation policy.
- Frontend subprocess lifetime and response-size bounds, LSP cancellation cleanup, authentication/ACL coverage, and backup/restore behavior remain open.

## Reproduction artifacts

- `repros/test_training_contracts.py`: five CPU contract probes.
- `repros/rust_contracts.rs`: file-key and mock-embedding probes.
- `repros/run_isolated_database.py`: disposable database runner, with optional `--codebase-tests`.
- `repros/README.md`: commands, expected failures, and isolation requirements.

These probes assert the intended contracts and intentionally fail on the audited revision. They are kept outside the normal service test directory until corresponding fixes are implemented.

## Tracking and publication

Ten focused issue drafts combine the six newly verified groups with boot-profile selection, isolated CI, cursor cancellation, and search memory work from the initial audit. Exact titles and bodies are in `2026-09-19-remediation-issues.md`.

After automatic approval review requested explicit destination approval, the user approved publishing all ten drafts. They are now published as issues #13–#22 in `toddwbucy/HADES`; the exact issue mapping and approved bodies are recorded in `2026-09-19-remediation-issues.md`.

## Live-service verification and remaining scope

After all isolated execution, HADES daemon, production ArangoDB, extractor and GPU2 embedder remained active with the same PIDs (4240, 3014, 3018, 3016) and restart counts (1, 0, 0, 0) as the baseline. This confirms no observed service restart; it is not a measurement of workload latency.

Next priorities:
1. Resolve B01 and checkpoint/training correctness through focused changes and isolated regressions.
2. Complete atomicity, authorization, packaging and multi-client lifecycle review with fault-injection/mock fixtures.
3. Establish representative retrieval relevance labels and CPU/service-contract baselines.
4. Plan GPU correctness, concurrency/load, analyzer installation, and restore drills in an isolated resource window.
5. Keep epic #12 open until each workstream has evidence and remaining risks have an explicit disposition.

