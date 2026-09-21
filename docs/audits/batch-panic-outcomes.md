# Batch panic outcomes audit

Tracking: epic #12; finding #141. Severity: P2 (false success and incomplete recovery state).

## Scope and baseline

Source `2bb0e6a13cdc705d6708010047d487cd37dbc911` includes the pending #138 failed-item resume fix. Baseline commit `eff54cf` adds an isolated two-item test with concurrency one. Each item position is made to panic in turn; its successful neighbor returns normally. No database, model, GPU, or production service is accessed.

The first-item panic is consumed by the opportunistic `try_join_next` loop: total=2, completed=1, failed=0, only one result, and the checkpoint is removed. A last-item panic reaches the final drain: failed=1 and two results, but no failed checkpoint entry and progress reports failed=0. Both violate the same accounting contract. Normalized observations and the executed test source hash are preserved in `2026-09-19/batch-panic-outcomes-baseline.json`.

## Remediation

Both drain paths now collect join outcomes with Tokio task IDs and resolve them through one helper. Successful and failed joins release their identity mapping. A task failure becomes an identified failed `ItemResult` and follows the existing common checkpoint/progress path. Successful neighbors remain recorded. Explicit resume uses the #138 behavior: skip completed items, retry failures; a successful retry clears the checkpoint.

The regression exercises both collection paths, checks failure identity and stage, reads the checkpoint, and resumes with a callback that rejects any successful neighbor being retried. All 28 batch tests pass. Run:

```sh
cargo test --offline -p hades-core --lib batch:: -- --nocapture
cargo clippy --offline -p hades-core -p hades-cli --all-targets -- -D warnings
```

## Limits

This demonstrates a library accounting defect, not an observed production panic. A panic may happen after a caller has already committed side effects; this change does not roll them back or make retries idempotent. Abort-on-panic builds and process termination cannot be recovered by Tokio joining. Checkpoint namespace isolation, concurrent writers, and crash durability remain separate audit questions. No running HADES component was deployed or changed.
