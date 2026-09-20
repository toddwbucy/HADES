# HADES Audit: Initial Code Review and Live Deployment Baseline

Date: 2026-09-19  
Epic: https://github.com/toddwbucy/HADES/issues/12  
Source revision: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`

## Scope and status

This is the initial audit pass, not a completed full-code audit or production certification. It combines selected source review with bounded read-only inspection of the active deployment. The epic tracks the remaining component reviews and isolated execution phases.

Reviewed areas include workspace structure, CI, database test fixtures and cursor handling, dispatch/search, request authorization, daemon and MCP transports, viewer boundaries, embedding metadata, service units, profile switching, and selected training/sampling code. Ingestion, analyzers, adapters, training, and recovery require deeper review and execution.

No services were stopped, restarted, reloaded, or reconfigured. No binaries were replaced; no dependencies were installed; no database writes, corpus queries, inference, training, load tests, or recovery tests were run. Environment values and raw journal contents were not published. Existing `AGENTS.md` was left unchanged.

## Runtime baseline

| Component | Observation |
| --- | --- |
| HADES daemon | Active user service; PID 4240; one recorded restart; approximately 7.5 MiB systemd-accounted memory at sampling |
| ArangoDB | Active user service; PID 3014; approximately 929 MiB memory; loopback TCP listener and user-owned Unix socket |
| Extractor | Active user service; PID 3018; working directory `/opt/HADES/services`; user-owned Unix socket |
| GPU2 embedder | Active user service; PID 3016; loopback HTTP; approximately 1.2 GiB systemd-accounted memory (not a VRAM measurement) |
| GPU1 embedder | Failed, `start-limit-hit`, three recorded restarts; journal contains address-in-use failures |
| Trainer | No active trainer unit appeared in the user-service inventory; this does not establish whether training is supported through another launch mechanism |

Three requests, each bounded to three seconds, returned:
- Embedder `GET /v1/models`: profile `gpu2`, physical device `cuda:2`, dimension 2048, maximum sequence length 11,900.
- Daemon MCP listener `GET /healthz`: `{"status":"ok"}`.
- Unauthenticated `GET /mcp`: HTTP 401.

These establish metadata availability, HTTP liveness, and rejection of one unauthenticated request. They do not establish database connectivity, successful inference, authorization correctness for every operation, or retrieval quality.

The running daemon executable and installed `~/.local/bin/hades` have the same SHA-256:
`993712814319be5f663a3c98fdb51b13c27a144def1adcb81c1178109af986ac`.
There was no `target/release/hades` artifact available for comparison. The running binary's exact source commit remains unverified. Python services execute from this checkout, but already imported modules may differ from files subsequently edited on disk.

## Findings

Priorities indicate remediation order: P1 before database test execution; P2 planned reliability/correctness work; P3 hardening/documentation.

### A01 — Named-database fixtures prevent safe blanket testing (P1, confirmed source)

`crates/hades-core/tests/arango_crud.rs:52` connects as root to `bident_burn`; its setup drops/recreates collections. Related query, index, and transport tests retain named-database assumptions. Their socket guards do not provide database isolation.

Impact: a full test command can fail against missing historical fixtures or modify a shared database if those prerequisites exist. No such tests were run here.

Remediation: migrate every mutating fixture to `test_support::with_temp_db`, require explicit test credentials and strict prerequisite checks, and run against a separate ArangoDB instance. Acceptance: enumerate all write-capable tests and prove that they operate only on disposable databases.

### A02 — Competing embedder profiles are enabled at boot (P2, confirmed deployment)

Both `hades-embedder@gpu1.service` and `hades-embedder@gpu2.service` report `UnitFileState=enabled`. GPU2 is active; GPU1 has address-in-use failures and exhausted its restart limit. The profile template intentionally uses one shared endpoint.

`deploy/systemd/hades-embedder-profile` stops active instances and starts the requested profile but does not reconcile enablement.

Impact: boot attempts competing profiles; the active profile can depend on startup order. Current GPU2 metadata is healthy.

Remediation: define one persistent selected profile and reconcile enablement through the profile-switch mechanism. Verify reboot behavior in isolation or a maintenance window; do not change the active server during discovery.

### A03 — CI does not verify the complete deployed workflow (P2, confirmed configuration)

`.github/workflows/ci.yml:96-122` runs Rust library/binary tests and three service-free integration targets. Database-backed workflows and Python service tests are not execution gates in this workflow.

Impact: passing CI cannot establish extraction/embedding/ingestion/search interoperability or graph integrity after updates.

Remediation: add isolated database-backed regression coverage and Python/service contract tests, then a small end-to-end fixture covering ingest, query, modification, movement, deletion, and partial failure. Report skipped prerequisites explicitly.

### A04 — Search memory and concurrency remain inadequately bounded (P2, confirmed design; capacity unmeasured)

`crates/hades-core/src/dispatch.rs:5053-5210` sizes the embeddings collection, rejects counts above 100,000, materializes vectors as JSON, scores them in Rust, and sorts candidates. `db/query.rs` accumulates cursor pages. The daemon spawns a task per accepted connection; inspected paths do not establish a shared search admission budget. The running daemon has no configured systemd memory ceiling.

Impact: per-request count limits are not a combined memory limit. Concurrent searches can multiply allocations. Collection sizing and retrieval are separate operations, so the count check is not a snapshot-enforced row budget.

Remediation: benchmark only against isolated fixtures; compare indexed retrieval with streaming top-K; introduce explicit concurrency and byte/result budgets. Record latency percentiles and peak memory across corpus sizes before choosing thresholds.

### A05 — Request cancellation bypasses cursor cleanup (P2, confirmed control flow; not reproduced live)

`crates/hades-core/src/service.rs:362` wraps dispatch in a timeout. In `db/query.rs`, cursor deletion follows `paginate(...).await`. Dropping that future on timeout bypasses the subsequent cleanup code. The search handler's own documentation acknowledges this behavior.

Impact: server-side cursor state can persist until expiry following cancellation. Normal returned pagination errors do attempt cleanup.

Remediation: implement cancellation-aware cursor ownership and bounded query lifetime. Acceptance: an isolated delayed-pagination test verifies cursor cleanup on timeout, client cancellation, malformed responses, and normal completion.

### A06 — Deployment provenance and readiness are incomplete (P2, confirmed evidence gap)

Executable hashes match the installed artifact, but no commit-to-binary mapping was established. `/healthz` returns a constant success object; `/v1/models` returns configuration without loading the model.

Impact: those checks can remain green while a dependency or inference path is unavailable. The profile switcher uses metadata response as its readiness check.

Remediation: expose build revision and distinguish process liveness, dependency readiness, and inference readiness. Keep potentially expensive GPU probes separate and explicit.

### A07 — Documentation and credential-file hardening need alignment (P3, confirmed)

`docs/specs/workstation-specific-tests.md` still describes CI as library-only and groups service-dependent tests with self-contained tests, whereas current CI and `CLAUDE.md` describe a different matrix. The repository's `config/hades.yaml` is not automatically discovered from that location.

The daemon environment file contains an `ARANGO_PASSWORD` assignment and has mode 0644. Its ancestor home directory is mode 0700, so the observed mode does **not** prove exposure to other local users. The MCP token file is 0600.

Remediation: reconcile test/setup documentation and make credential-file permissions restrictive in installation guidance. Review actual ACLs before making exposure claims. No permissions were changed.

## Further investigations, not confirmed defects

- Training `_auc` uses double argsort without average tie ranks; add a CPU-only equal-score fixture before treating its metric as reliable.
- Edge splits use an unseeded RNG; determine the required reproducibility contract and verify no held-out edges leak into message passing.
- Verify stored vector/model/task provenance, malformed-vector reporting, and dimension compatibility across ingestion and search.
- Trace full incremental ingest and edge remapping under crashes, retries, concurrent writers, and partial embedder failures.
- Review all provisioning path uses, symlink handling during traversal, subprocess arguments, body limits, checkpoint loading, and parser resource bounds.
- Establish the database ACL matrix, backup freshness, restore procedure, GPU headroom, and actual query workload without publishing private data.
- Locate or construct a versioned retrieval evaluation set; the initial file scan did not find a dedicated suite, but that is not proof none exists elsewhere.
- Review Python packaging, dependency locking, generated bindings, and the deployed environment in more detail.

## Next audit phase

First create a resource-limited isolated execution environment after reviewing every fixture's external effects. Keep builds and tests separate from the service's working artifacts. Run strict database fixtures, Python CPU tests and service contract tests; then build the end-to-end graph fixture. GPU, performance, failure injection, and restore work require a separately agreed resource window.

Close workstreams only after their reviewed files, executed checks, results, and remaining limitations are recorded. Create focused remediation issues from confirmed findings; do not mark the epic complete on the strength of this baseline.

