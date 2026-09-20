# Epic #12 acceptance reconciliation

Evidence cut: September 20, 2026, repository `1c1adf8962723c83535d317951a13b102c8afe59`.
This is a current reconciliation of earlier reports, not a claim of full audit
completion. Historical inventories and baseline reproductions retain their own
revision boundaries. Repository fixes have not been deployed.

## Deployment identity and limits

The [sanitized metadata capture](deployment-metadata-refresh.json) records hashes
of the running daemon, database and Python interpreter, scoped listener ownership,
installed Python distribution metadata and the embedder's reported configuration.
The daemon executable and installed `~/.local/bin/hades` have identical SHA-256
`993712814319be5f663a3c98fdb51b13c27a144def1adcb81c1178109af986ac`.
This identifies bytes, not their exact source commit. The earlier
[provenance investigation](2026-09-20-deployment-provenance.md) remains applicable.

| Process | PID | Observed application listeners |
|---|---:|---|
| HADES daemon | 4240 | `192.168.0.203:10443`; per-user `hades.sock` |
| ArangoDB | 3014 | `127.0.0.1:8529`; per-user `arangodb.sock` |
| Extraction | 3018 | Per-user `extractor.sock` |
| Embedding GPU2 | 3016 | `127.0.0.1:8087` |

The GPU2 process also owns an abstract CUDA runtime socket, not an application
RPC endpoint. Binding alone establishes neither TLS/authentication nor remote
reachability. Unit inspection found these four services active/running, with
restart counts 1/0/0/0 respectively; GPU1 was failed with three restarts. Boot
enablement was not refreshed in this capture.

Both Python launchers select `/opt/HADES/services/.venv/bin/python`, resolving to
CPython 3.12.13. Distribution versions in the JSON are on-disk metadata; they do
not prove which modules are already imported. No ML package was imported for
inspection. No vulnerability conclusion follows from these version numbers alone.

The bounded metadata request to `/v1/models` on port 8087 returned HTTP 200 and
reported Jina embeddings V4, dimension 2048, maximum sequence length 11900,
profile `gpu2`, logical `cuda:0` and physical `cuda:2`. An initial request to
guessed port 8000 was refused; the actual listener was then mapped to PID 3016.
That refusal is not evidence the active embedder was down. The successful metadata
response does not establish loaded weight-byte identity or inference readiness.
No inference, model loading, dependency installation or service change was requested.

## Workstream evidence and remaining work

Each row names established scope and the evidence still needed. A passing fixture
does not certify its entire component. Older statements that worker ownership or
document replacement remain wholly untested are superseded by the linked reports.

| Workstream | Established evidence | Remaining scope |
|---|---|---|
| Deployment | Metadata above; [recovery](active-database-recovery.md) includes an actual historical dump restore | Exact daemon source mapping remains unproven; loaded Python/model identity, deployed schema and complete configuration precedence need reconciliation |
| Architecture/API | Five-crate/Python [component inventory](component-coverage.md), shared dispatch and selected transport contracts | Complete command/envelope/status/provisioning matrix and remaining frontend/protocol compatibility review |
| Security/isolation | [ACL matrix](database-access-boundary.md), filesystem authority, archive limits, installer signature and subprocess findings | #74 remains open; remaining parser/input/dependency/install subprocess review; no production authentication change |
| Ingestion/integrity | [File lifecycle/transactions](ingestion-atomicity.md) and [document transactions](document-replacement.md), including rollback/cancellation/concurrency | Adapter identity/truncation/stale retirement and versioned real-corpus conformance; not a whole-ingest transaction |
| Retrieval/embedding | Response/vector validation, bounded ranking and [embedding ownership](embedding-worker-lifetime.md) | Model/task freshness, independently judged relevance and learned-graph comparison |
| Training | Leakage/metrics/checkpoints; [prefetch cancellation](prefetch-cancellation.md), [worker ownership](training-worker-ownership.md), [atomic artifacts](training-artifact-publication.md) | Feature/node/export alignment and [structural generation/freshness](structural-vector-provenance.md); CPU contracts do not prove GPU preemption or quality |
| Performance/reliability | Declared isolated retrieval/transport pilots; viewer, ingest, extraction and embedding ownership; LSP deadlines, memory, preflight and descendant reports | Remaining queue/admission/readiness/install paths; no production load or hard kernel-time guarantees |
| Tests/CI | Rust, Python CPU and disposable database gates; explicit workstation prerequisite limits | Refresh [fixture inventory](test-fixture-inventory.md) for later targets; distinguish synthetic providers from real analyzers/models |
| Retrieval evaluation | Versioned seed results; private Bastion passages, frozen CPU vectors and blinded owner/third-party packet | Code run completion, both reviewers' judgments for both workloads, learned comparison and measured regression thresholds; no representative quality scores yet |
| Packaging/operations | Installed wheel/sdist contracts, installer signatures, [daemon readiness](daemon-readiness.md), synthetic and historical restore evidence | Fresh-host end-to-end installation, upgrade/rollback, full application recovery and current backup monitoring/coverage |
| Findings/remediation | Original #13–#22 merged; later findings have baseline evidence and individual remediation reports | Reconcile unresolved high-severity disposition and remaining confirmed findings; issue closure alone is not technical verification |

## Acceptance criteria assessment

1. **Workstream evidence: partial.** All eleven areas have evidence and explicit
   limitations above; remaining review is substantive, not waived by this table.
2. **Deployment provenance: partial.** Running artifact hashes and selected
   configuration are recorded; exact source/model/schema mapping remains incomplete.
3. **Finding traceability: ongoing.** Preserve each baseline and its linked issue,
   impact, severity and verification; do not convert source limitations into
   observed production failures.
4. **Isolated lifecycle: established for selected code/document contracts.**
   Maintained CLI `codebase_lifecycle::ingest_query_modify_move_delete_and_partial_failure_recover`
   and file identity contracts exercise the
   ingest/query/modify/move/delete path; transaction reports retain partial-failure
   evidence. This does not establish every language or adapter workflow.
5. **Repeatable quality/performance: partial.** Isolated performance and seed
   evaluation artifacts exist. Representative independent judgments and learned
   comparisons are still required. Bastion drafts describe the intended system;
   their claims are not evidence of implemented behavior.
6. **High-severity disposition: not fully demonstrated.** Repository P1 fixes
   have verification, but recovery closure supplies no new coverage evidence.
7. **Production preservation: maintained.** Audit discovery has not deployed
   repository changes; later maintenance needs its own verification/rollback plan.

## Recovery and latest remediation disposition

The owner closed #63 on September 20. Preserve that disposition. Its historical
active-dataset coverage gap, limited restore scope and unverified monitoring
remain evidence limitations; closure without a rationale does not prove explicit
acceptance of each risk. The owner's policy permits arranging downtime and
requires a verified snapshot of effective active data before a risky operation,
with database consistency and rollback steps. It does not set numeric incident
RPO/RTO or prove routine backup freshness. No snapshot was taken during discovery.

PR #97 merged as `9826de8` and PR #101 as `1c1adf8`, closing #96 and #100.
Their final CI runs `35532803352` and `35533568380` passed all three gates.
PR #97 records author self-review because external review was limited. PR #101's
runtime/test changes had external review at `67a9488`; its final integration was
author-reviewed and tested, not independently reviewed anew.
PR #103 at `e2ac534` addresses #102 and is pending final gates at this evidence
cut; its green external status is a rate-limit notice, not completed review.

The epic remains open. Further P2 fixes cannot substitute for representative
retrieval judgments, complete boundary review or deployment/recovery evidence.
