# Epic #12: Audit Evidence and Remaining Acceptance Work

This register distinguishes repository remediation from deployment and full audit
completion. Baseline source: `a71d73bfe988e17d487a2db1d35dadf3a18f0664`.
Repository status sampled on 2026-09-20. Epic #12 remains open.

## Evidence locations

- [Deployment provenance follow-up](2026-09-20-deployment-provenance.md): live
  executable hash, ELF build ID/compiler evidence and limits of source mapping.

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
| Incomplete Python distributions | #38, PR #39 merged `5c82e95`; installed wheel/sdist and isolated build contracts | Not deployed |
| Failed file replacement loses committed graph | #40, PR #41 merged `26e71ea`; all five requirements mapped to rollback, revision, timeout/cancellation and retry fixtures | Not deployed |
| Shared trainer state crosses client lifecycles | P1 #42, PR #43 merged `4fdad18`; two-channel CPU reproduction, 16 CPU session and 10 Rust client contracts; final CI `35506989556` passed | Not deployed; lease expiry is not compute preemption |
| Unsafe installer credential/response handling | #44, PR #45 merged `979fe97`; malformed-password reproduction and six private Unix HTTP contracts | Not deployed |
| Ambient database endpoints in unit fixtures | #47, PR #48; accepted-AQL unit tests used normal endpoint discovery and accepted backend errors | Private-mock remediation and fixture inventory awaiting integrated-head CI; no live reproduction |
| Agent filesystem access through smell reports | #49; private service fixture returned a matching synthetic line without permitted roots | Admin scan / database-only report separation in isolated development; not deployed |

## Workstream coverage and explicit gaps

| Epic workstream | Evidence established | Still required before closure |
|---|---|---|
| Deployment inventory/provenance | Units, endpoints, executable hash and restart baseline | Exact running binary commit mapping or documented inability; backup inventory and restore evidence |
| Architecture/API | Shared dispatch, authorization and service contracts reviewed in affected paths | Complete component inventory and review of frontend, analyzers, adapters and remaining API boundaries |
| Security/isolation | Authentication rejection baseline; isolated tests; body, cursor, archive limits under review | Database ACL matrix, provisioning/path handling, subprocess/parser limits and dependency review |
| Ingestion/graph integrity | Collision/lifecycle fixtures; PR #41 file/stage transactions, races, timeout/cancellation and retry contracts | Remaining document/adapter pipeline review and failure boundaries |
| Retrieval/embedding | Model/dimension validation; versioned exact-ranking parity | Stale model/task provenance, broader independently reviewed quality set |
| Training | Leakage, split metrics, aborts and checkpoint schema fixes | Concurrent sessions, cancellation, graph alignment and trained structural baseline |
| Performance/reliability | Isolated engine, handler and transport pilots | Remaining non-search queue/time-out paths |
| Tests/CI | Rust, Python CPU and disposable ArangoDB gates | Full write-fixture inventory; broader failure injection beyond the selected end-to-end contract |
| Retrieval evaluation | 24 author-judged queries, frozen CPU Jina vectors, file-membership comparator | Representative independent judgments and learned graph comparison; seed results cannot certify production relevance |
| Packaging/operations | Manifest and launch scripts inspected | Fresh installation beyond verified wheel/sdist contracts, health/readiness, upgrade/rollback and isolated restore rehearsal |
| Findings/remediation | Ten approved issues plus eight additional confirmed findings | Complete severity-ranked dispositions, linked evidence and owner acceptance for unresolved high-severity findings |

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


## Subsequent ingestion and installer evidence

PR #41 merged as `26e71ea` after all three gates passed at final head `fcefac7`
in run 35505635214. [Its requirement review](ingestion-atomicity.md#issue-40-requirement-review)
maps every #40 bullet to maintained tests and explicit stage boundaries. The
issue is closed; no code or data migration was deployed.

The installer probe used synthetic credentials, a private Unix socket and mock
curl. An ordinary password generated valid JSON and completed three requests;
a quote/backslash password generated invalid JSON and failed on the first request.
Source also passed the root password in curl argv and used a fixed response file.
#44 is closed through PR #45, merged as `979fe97` after final CI and review. It
replaces this with in-memory standard-library HTTP/JSON handling.
Six private contracts cover special passwords, existing grants, sanitized HTTP
and transport failures, argv/interruption behavior and the real request deadline.
This does not prove fresh-host installation or production ACL correctness.

## Representative retrieval workload decision

The user selected **both code search and document research**. Build balanced
coverage and report quality separately for each workload as well as in aggregate.
Code queries should include implementation lookup, behavior, debugging and change
impact; document queries should include factual, conceptual and cross-document
evidence retrieval. Freeze representative relevance judgments before scoring.
Compare vector-only and graph-assisted retrieval with recall@k, MRR/nDCG, latency
and memory, distinguishing learned structural embeddings from file membership.
The existing 24-query seed remains preliminary. Representative judgments, measured
results and baseline-derived regression thresholds are still outstanding.

The research corpus consists of evolving public-facing paper drafts describing
the intended completed project. The [workload plan](../../evaluation/retrieval/workload-plan.md)
records candidate questions, a version-aware judgment protocol and a planned
claim-to-code check. Draft claims alone do not establish implemented capabilities. The exact
collection snapshot and independent judgments remain outstanding.
