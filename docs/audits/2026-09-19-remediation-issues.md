# Published Audit Remediation Issues

Published to `toddwbucy/HADES` after explicit user approval, under [epic #12](https://github.com/toddwbucy/HADES/issues/12). The ten approved titles and bodies are preserved below.

- [Issue #13](https://github.com/toddwbucy/HADES/issues/13) — [Audit P1] Prevent code-file key collisions from silently overwriting distinct source files
- [Issue #14](https://github.com/toddwbucy/HADES/issues/14) — [Audit P1] Keep held-out link-prediction edges out of training message passing
- [Issue #15](https://github.com/toddwbucy/HADES/issues/15) — [Audit P1] Reject incompatible graph loads after checkpoint restore instead of randomizing the model
- [Issue #16](https://github.com/toddwbucy/HADES/issues/16) — [Audit P2] Reject malformed embedding indices and vector elements before storing results
- [Issue #17](https://github.com/toddwbucy/HADES/issues/17) — [Audit P2] Correct ROC-AUC ties and explicitly handle empty training/evaluation splits
- [Issue #18](https://github.com/toddwbucy/HADES/issues/18) — [Audit P2] Await gRPC training precondition aborts and stop invalid RPC execution
- [Issue #19](https://github.com/toddwbucy/HADES/issues/19) — [Audit P2] Persist exactly one enabled embedder profile across server boots
- [Issue #20](https://github.com/toddwbucy/HADES/issues/20) — [Audit P2] Gate CI on isolated database and Python service contracts
- [Issue #21](https://github.com/toddwbucy/HADES/issues/21) — [Audit P2] Clean up database cursors when requests time out or are cancelled
- [Issue #22](https://github.com/toddwbucy/HADES/issues/22) — [Audit P2] Bound aggregate search memory and measure an indexed retrieval path

## 1. [Audit P1] Prevent code-file key collisions from silently overwriting distinct source files

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Confirmed reproduction
In a disposable ArangoDB instance, ingest a directory containing:
```
a/b.py   # def second(): return 2
a_b.py   # def first(): return 1
```
Both files map to `a_b_py`. The CLI exits 0 and reports `success: true` for both files, but querying `codebase_files` returns only `a_b.py`. The second ingest purges/replaces the first file's graph data.
Source: [file_key](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/crates/hades-core/src/db/keys.rs#L71), [ingest replacement](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/crates/hades-cli/src/commands/codebase_ingest.rs#L1690).

### Required outcome
- Use collision-resistant, ArangoDB-legal file identities and define the namespace across ingest roots.
- Detect identity conflicts before purging existing data.
- Plan compatibility/migration for existing file, chunk, symbol and edge keys; no live automatic migration.
- Regression fixture must retain both files and their symbols/chunks across initial ingest, re-ingest, modification and retirement. Include dotted/underscored, Unicode, long-path and multi-root cases.

---

## 2. [Audit P1] Keep held-out link-prediction edges out of training message passing

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Confirmed reproduction
A CPU fixture loads edges `0→1, 1→2, 2→3`, calls `TrainStep(train_edge_indices=[0])`, and spies on the encoder input. All three edges are passed to `model.encode`.
[LoadGraph and _encode](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/services/training/server.py#L155) retain the complete graph; [TrainStep](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/services/training/server.py#L236) uses the split only for the loss. Rust serializes train/validation/test indices but the Python loader does not use them to mask message-passing adjacency.

### Impact
Held-out relationships are available to the encoder during training and evaluation, so reported held-out link-prediction quality does not establish performance on unseen relationships.

### Required outcome
Separate the message-passing training graph from evaluation target edges. Define the intended transductive evaluation protocol, handle inverse/duplicate edges consistently, and verify held-out topology is absent with a CPU spy fixture. Recompute affected model/evaluation baselines in an isolated environment.

---

## 3. [Audit P1] Reject incompatible graph loads after checkpoint restore instead of randomizing the model

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Confirmed reproduction
On CPU: initialize a six-feature SAGE model, save and restore its checkpoint, then call `LoadGraph` with seven-feature node vectors. The call succeeds and invokes `_build_model(7)`, discarding the restored parameters. The `graph-embed update` path loads a checkpoint and then a graph in this order.
Sources: [LoadGraph](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/services/training/server.py#L155), [LoadCheckpoint](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/services/training/server.py#L320), [update](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/crates/hades-cli/src/commands/graph_embed_update.rs#L184).

### Required outcome
- Distinguish fresh-training dimension setup from checkpoint-backed inference.
- Reject incompatible feature dimensions before modifying the loaded model.
- Persist and validate relation order, collection-name/index mapping, feature/model provenance and architecture. Current checkpoint configuration contains counts but no semantic mapping, while graph loading sorts the currently present collection names.
- Add CPU checkpoint round-trip and incompatible-schema tests; verify a rejected load preserves model weights and graph state.
- Do not overwrite deployed structural embeddings while investigating compatibility.

---

## 4. [Audit P2] Reject malformed embedding indices and vector elements before storing results

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Confirmed mock-server reproductions
The ordinary embedding client accepts both responses for a two-input request:
```json
{"data":[{"index":0,"embedding":[1,0]},{"index":0,"embedding":[0,1]}]}
```
```json
{"data":[{"index":0,"embedding":[1,"bad"]},{"index":1,"embedding":[0,"bad"]}]}
```
The first silently associates duplicate-index data with separate inputs. The second silently drops nonnumeric values, yielding accepted one-dimensional vectors.
[Parsing](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/crates/hades-core/src/persephone/embedding.rs#L614) defaults missing indices to zero, sorts them without validating the sequence, and uses `filter_map` for vector elements.

### Required outcome
Require exactly one index per input in range; reject missing, duplicate and out-of-range indices. Require finite numeric values, nonempty vectors, consistent expected dimensions and compatible model identity. Test with local mock responses, including valid reordered responses. No live embedder is needed.

---

## 5. [Audit P2] Correct ROC-AUC ties and explicitly handle empty training/evaluation splits

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Confirmed CPU reproductions
- `_auc(torch.zeros(1), torch.zeros(1))` returns `0.0`; one tied positive/negative pair requires `0.5`.
- `_bce_link_loss(torch.empty(0), torch.empty(0))` does not reject an empty split; its mean reductions produce an undefined loss.
[Metrics](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/services/training/server.py#L57) use double argsort without average tie ranks. [Splitting](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/crates/hades-prefetch/src/tensor.rs#L148) floors split sizes, allowing empty validation/test splits for small graphs.

### Required outcome
Implement tie-correct AUC and define explicit behavior for empty splits and insufficient negative samples. Reject unsupported runs or return an explicit unavailable metric rather than NaN or misleading zero. Test tied scores, empty classes, tiny and dense graphs, and early-stopping/checkpoint behavior. Add seedable split/evaluation sampling for repeatable comparisons.

---

## 6. [Audit P2] Await gRPC training precondition aborts and stop invalid RPC execution

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Confirmed CPU reproduction
Calling `TrainStep` before initialization with an async context produces unawaited-coroutine warnings and `AttributeError: 'NoneType' object has no attribute 'train'`, rather than the intended `FAILED_PRECONDITION`.
[_require_loaded](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/services/training/server.py#L131), `LoadGraph` and `Checkpoint` call `context.abort` without awaiting it, although the server uses `grpc.aio`.

### Required outcome
Use awaited aborts consistently and prevent all work after a failed precondition. Cover missing model/graph, invalid indices, malformed tensors and invalid checkpoint lifecycle with CPU RPC tests. Verify actual status codes and no state mutation on rejection.

---

## 7. [Audit P2] Persist exactly one enabled embedder profile across server boots

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Confirmed live observation (read-only)
Both `hades-embedder@gpu1` and `hades-embedder@gpu2` were enabled. GPU2 was running; GPU1 had `start-limit-hit` after address-in-use failures. Both profiles share one endpoint.
[Profile switcher](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/deploy/systemd/hades-embedder-profile) stops active instances and starts the selected one without reconciling boot enablement.

### Required outcome
Define one persistent selected profile, reconcile enabled instances safely, and verify repeated switches/reboot behavior in isolation. Preserve GPU resource preflight and clear failure reporting. A metadata response alone must not be described as proof that inference works.
The active server was left unchanged. Deployment of this fix needs an explicit maintenance/rollback plan; do not stop either live profile during audit discovery.

---

## 8. [Audit P2] Gate CI on isolated database and Python service contracts

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Evidence
Current CI executes Rust unit tests and three service-free integration targets; Python tests and full database workflows are not execution gates. Named-database/root fixtures remain in `arango_crud`, `arango_index`, `arango_query` and `arango_transport`.
The audit ran five cache tests and 60 codebase-related tests successfully against a separate disposable ArangoDB instance in strict mode. Existing CPU training tests pass while targeted audit contracts expose defects.

### Required outcome
- Migrate write tests to disposable databases and a separate server; never point the suite at production.
- Make skips/prerequisites explicit; strict mode must fail when setup is absent.
- Gate Python CPU tests, service-free contracts and isolated graph lifecycle tests.
- Add regressions for #12 findings and a small ingest/query/modify/move/delete fixture with partial failures.
- Align `docs/specs/workstation-specific-tests.md`, README and contributor commands with the actual test matrix.
- Resource-limit the integration runner and clean up on failure/cancellation.

---

## 9. [Audit P2] Clean up database cursors when requests time out or are cancelled

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Confirmed control-flow finding
[query](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/crates/hades-core/src/db/query.rs#L116) deletes the cursor after awaiting pagination. [service timeout](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/crates/hades-core/src/service.rs#L362) drops the dispatch future on expiry. Cancellation during pagination bypasses deletion; normal pagination errors do attempt cleanup. This is also acknowledged in the search handler's comments.
No timeout/failure injection was run on the live database.

### Required outcome
Provide cancellation-safe cursor ownership with bounded cleanup and query lifetime. Isolated tests must cover normal completion, server error, malformed first/continuation response, timeout and caller cancellation, and confirm server cursor release. Preserve reader/writer endpoint consistency.

---

## 10. [Audit P2] Bound aggregate search memory and measure an indexed retrieval path

Parent epic: #12. Audited source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`. Discovery did not modify the running deployment. Fixes and tests must use isolated fixtures.

### Confirmed design limitation
[db.query](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/crates/hades-core/src/dispatch.rs#L5053) checks a 100,000-embedding count, then materializes/scans vectors and sorts results. Count and fetch are separate operations. [Daemon acceptance](https://github.com/toddwbucy/HADES/blob/a71d73bfe988e17d487a2db1d35dadf3a18f0664/crates/hades-cli/src/commands/daemon.rs#L179) spawns per-connection tasks without an evident shared search budget; the sampled deployment has no systemd memory ceiling.

### Required outcome
Establish isolated corpus-size/concurrency benchmarks with p50/p95/p99 latency and peak memory. Compare indexed retrieval with streaming top-K, add shared admission/byte/result limits, and enforce cancellation behavior. Define budgets from measurements rather than treating the row cap as a memory guarantee. Verify retrieval-quality parity and meaningful overload errors.
Do not load-test the running HADES service or switch GPU profiles as part of discovery.

