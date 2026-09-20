# Epic #12: Audit Evidence and Remaining Acceptance Work

This register distinguishes repository remediation from deployment and full audit
completion. Baseline source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`.
Repository status sampled on 2026-09-20. Epic #12 remains open.

## Evidence locations

- [Initial review and deployment baseline](2026-09-19-initial-audit.md):
  read-only service inventory, binary hash, selected static findings and gaps.
- [Isolated audit](2026-09-19-isolated-audit.md): reproducible defects,
  fixture results and isolation boundaries.
- [Approved remediation issues](2026-09-19-remediation-issues.md): exact ten
  published drafts, linked to #13–#22.
- [Historical reproductions](repros/README.md): baseline-only investigation
  artifacts. Use the current `scripts/test_isolated_database.py` for maintained
  regression execution; historical probes are not the current CI suite.
- PR #31 / CI run `35494691052`: Rust, Python CPU and disposable database gates
  passed before merge. No production endpoint was used.
- PR #32: bounded retrieval and transport changes, indexed/streaming experiments,
  handler and MCP memory pilots, frozen retrieval judgments and model vectors.
  Merged as `16c4f7d`; use its final artifacts for measured limits.

## Confirmed findings and disposition

| Finding | Evidence / repository outcome | Deployment status |
|---|---|---|
| File identity collisions | #13, PR #28; lifecycle fixtures preserve distinct roots/paths | Not deployed; no live key migration |
| Held-out topology leakage | #14, PR #25; encoder adjacency contracts | Not deployed; production quality not re-evaluated |
| Checkpoint schema mismatch | #15, PR #27; reject incompatible loads without replacing state | Not deployed |
| Malformed embedding responses | #16, PR #24; mock response validation | Not deployed |
| Tied AUC and empty splits | #17, PR #26; CPU metric/sampling contracts | Not deployed |
| Unawaited gRPC aborts | #18, PR #23; invalid RPC contracts | Not deployed |
| Competing boot profiles | #19, PR #29; persistent selection tests | Both profiles still enabled at last inspection |
| Unsafe/missing CI database coverage | #20, PR #31; strict disposable database and Python gates | CI active; no live database tests |
| Cancellation cursor leaks | #21, PR #30; private delayed-cursor contracts | Not deployed |
| Search/transport retention | #22, PR #32, merge `16c4f7d`; final CI gates passed | Not deployed |
| bfloat16 NumPy conversion | #33, PR #34, merge `d464c9d`; ten CPU contracts | Not deployed |
| LaTeX/archive expansion | #35, PR #36, merge `74ff141`; 13 byte/header/count regressions | Not deployed |
| Incomplete Python distributions | #38, PR #39; installed wheel/sdist and isolated build contracts | Review pending; live environment unchanged |
| Failed file replacement loses committed graph | P1 #40; schema-rejection fixture changes chunk/symbol counts from 1/1 to 0/0 | Remediation pending; isolated discovery only |

## Workstream coverage and explicit gaps

| Epic workstream | Evidence established | Still required before closure |
|---|---|---|
| Deployment inventory/provenance | Units, endpoints, executable hash and restart baseline | Exact running binary commit mapping or documented inability; backup inventory and restore evidence |
| Architecture/API | Shared dispatch, authorization and service contracts reviewed in affected paths | Complete component inventory and review of frontend, analyzers, adapters and remaining API boundaries |
| Security/isolation | Authentication rejection baseline; isolated tests; body, cursor, archive limits under review | Database ACL matrix, provisioning/path handling, subprocess/parser limits and dependency review |
| Ingestion/graph integrity | Collision and modify/move/delete regression coverage | Concurrent writers, partial failures, transaction/retry semantics and document/adaptor pipelines |
| Retrieval/embedding | Model/dimension validation; versioned exact-ranking parity | Stale model/task provenance, broader independently reviewed quality set |
| Training | Leakage, split metrics, aborts and checkpoint schema fixes | Concurrent sessions, cancellation, graph alignment and trained structural baseline |
| Performance/reliability | Isolated engine, handler and transport pilots | Remaining non-search queue/time-out paths |
| Tests/CI | Rust, Python CPU and disposable ArangoDB gates | Full write-fixture inventory; broader failure injection beyond the selected end-to-end contract |
| Retrieval evaluation | 24 author-judged queries, frozen CPU Jina vectors, file-membership comparator | Representative independent judgments and learned graph comparison; seed results cannot certify production relevance |
| Packaging/operations | Manifest and launch scripts inspected | Fresh installation/protobuf packaging, health/readiness, upgrade/rollback and isolated restore rehearsal |
| Findings/remediation | Ten approved issues plus four additional confirmed findings | Complete severity-ranked dispositions, linked evidence and owner acceptance for unresolved high-severity findings |

## Deployment boundary

Repository merges have not changed the active service. The last read-only unit
inspection retained the baseline PIDs and restart counts: daemon 4240/1,
ArangoDB 3014/0, extractor 3018/0, GPU2 embedder 3016/0. GPU1 remained failed and
enabled with three restarts. The unchanged PIDs and counters show no additional restart between the
2026-09-19 baseline and the 2026-09-20 read-only inspection,
not inference readiness or workload latency. Systemd memory accounting is not
process RSS or GPU VRAM.

Deployment requires an explicit artifact-to-revision mapping, configuration and
schema compatibility checks, backup/rollback plan, and separately authorized
maintenance execution. Discovery has not altered production data, dependencies,
permissions, services or model files. Do not check off full workstreams merely
because the ten initial repository defects have fixes.

## Packaging reproduction

A tracked-file copy of merged revision `d464c9d` was built with the existing
isolated CPU interpreter and setuptools, without installing dependencies. The
clean wheel contained five extraction and five embedding files but zero generated,
training or adapter files. After `make proto-gen`, a second wheel contained 15
generated files; training, adapters and `schema.yaml` were still absent. #38 tracks
build prerequisites, distribution scope, generated/runtime version compatibility,
and installed-wheel verification outside the source directory. Source/editable
launches are not evidence that a wheel can run the declared services.

## Ingestion failure boundaries

The maintained CLI lifecycle fixture covers ingest/query, text/vector modification,
definition-line movement, rename with explicit retirement, graph validation,
document-phase failure after successful code persistence, idempotent retry and
final deletion. This is evidence for the selected end-to-end acceptance fixture;
it does not establish atomicity of each file replacement.

A separate fault fixture against `16c4f7d` configured only its disposable database
to reject new chunk documents by schema. After an existing file was changed,
replacement failed with the expected CLI error, but both previously committed
chunk and symbol counts fell from one to zero. The private server exited cleanly.
P1 #40 tracks transactional replacement, purge-error propagation, concurrency,
cancellation and equivalent guarantees for fallback/enrichment paths. Production
was not used to reproduce this failure.
