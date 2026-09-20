# Component and API audit coverage

This map pins tracked source to `307892f94ae37b4b4e33e4bede7b8edec0e91717`.
The [file inventory](component-inventory.json) lists the five Rust crates, four
Python packages and protocol sources. File counts are navigation aids, not coverage
percentages. The [evidence register](2026-09-20-evidence-register.md) remains the
workstream acceptance register; this map does not certify epic #12 complete.

## Component boundaries and evidence

| Component | Tracked source / test-tree files | Boundary reviewed and existing evidence | Remaining scope |
|---|---:|---|---|
| `hades-cli` | 36 Rust / 3 Rust | `main.rs` routes CLI commands; daemon and MCP supply transport authority. [Detached ingest](ingest-job-ownership.md) covers admission, child ownership, output and lifecycle. [Filesystem authority](smell-filesystem-authority.md) separates agent reports from administrative scans. | Complete command/envelope/error-code matrix across CLI, daemon and MCP; provisioning and remaining analyzer subprocess paths. |
| `hades-core` | 68 / 27 | Shared dispatch, configuration, database transport and ingestion pipeline. [Atomicity](ingestion-atomicity.md), cursor contracts and bounded-retrieval evidence cover selected transactions, failures, cancellation and response/memory limits. | Remaining document lifecycle, model/task staleness, parser boundaries and configuration precedence beyond covered paths. |
| `hades-frontend` | 7 / 1 | Binary `hades-viewer`, not a library; communicates through CLI JSON. [Viewer boundary review](viewer-boundary-review.md) covers owned children, transport deadlines and retained response budgets. | Complete UI/assets and contract-version review; no claim that bounded subprocess tests cover every frontend behavior. |
| `hades-prefetch` | 4 / 0 | `tensor`, `prefetcher`, `orchestrator`: graph serialization, splits/sampling and training orchestration. #14/#17 and [training ownership](training-session-ownership.md) provide selected topology, metric and client-lifecycle contracts. | Active sampling cancellation, feature/node alignment through export/update, and a measured learned-graph retrieval baseline. Inline tests exist; zero separate test-tree files does not mean zero tests. |
| `hades-proto` | 1 / 1 | Build script compiles four proto sources into client/server code; library exports common, embedding, extraction and training modules. Generated-type tests and service/client contracts exercise selected compatibility. | Full error/status, size-limit and client/provider version compatibility matrix. Generated stubs alone do not establish a running provider. |
| Python extraction | 5 Python | gRPC `extraction/server.py`; dispatches PDF/unknown to Docling, LaTeX to its parser, and text/Markdown/code to text extraction using an executor. #35 covers archive/header expansion limits. | Full path/format/input limits, executor cancellation and concurrent model initialization; an awaited executor future is not proof its thread stops on cancellation. |
| Python embedding | 5 | `embedding/http_server.py` exposes OpenAI-compatible HTTP; `jina_v4.py` owns model encoding, `tensors.py` handles export. #16/#33 and retrieval preflight cover selected response and bfloat16/token contracts. | Complete HTTP failure/admission/model-readiness review, production task/model provenance and independently judged retrieval quality. |
| Python training | 7 | `training/server.py` behind session ownership in `session.py`. #14–#18 and #42 cover held-out topology, metric/precondition/checkpoint and ownership errors. | Active-compute cancellation/responsiveness, remaining alignment/update paths and learned structural evaluation. |
| Python adapters | 6 | Optional WeaverTools extractor, records, standalone SCIP exploration and HTTP writer. #66/#68/#70 cover redirect policy, failed writes and mixed-record metadata. [Writer limitations](adapter-failure-handling.md) state partial persistence. | Identity collisions/truncation, stale-record retirement and complete extraction-to-write conformance against a versioned real WeaverTools corpus. Local file traversal is not an untrusted-input sandbox. |

Rust test-tree counts include support modules, exclude inline tests, and do not
count integration targets. Python tests are shared under `services/tests`.
The [fixture inventory](test-fixture-inventory.md) has its own earlier revision;
subsequent focused reports add evidence rather than silently updating its scope.

## Protocol and authority distinctions

The four proto files declare two extraction RPCs, two embedding RPCs, ten training
RPCs and shared message types. The training service has three session-lifecycle
RPCs and seven state-bearing RPCs. The declared embedding gRPC protocol must not
be confused with the current Python HTTP embedding entry point.

Shared command dispatch is not an authorization boundary by itself. CLI, Unix
daemon, MCP, viewer subprocesses and direct database access have different entry
conditions. Deployment finding #74 records disabled ArangoDB authentication;
PR #75 supplies a private authenticated ACL matrix and clarifies shared
reader/writer credentials. Those database permissions do not replace HADES's
transport tiers or establish production grants. The active service is still the
baseline deployment, not this source revision.

## Cancellation scope still requiring execution

Static review of `Prefetcher::producer` finds two `spawn_blocking` sampling tasks
per epoch behind a bounded channel. `stop()` and `Drop` abort the producer handle;
they supply no cooperative cancellation token or explicit join for already-running
sampling. This does not demonstrate a measured shutdown delay, but it prevents
using producer abort as proof that all sampling work has stopped.

The valid Python `TrainStep` path calls synchronous encoding, scoring, backward
and optimizer-step operations without yielding to the event loop. Session ownership
prevents another client taking over during that work. Its lease tests do not prove
compute preemption, responsive cancellation or rollback after a response is lost.
A bounded CPU-only cancellation experiment is still required; no live training or
GPU failure injection was performed for this map.

## Practical seams and completion limits

The large dispatch module combines command schemas, tier mapping and handler
orchestration. Those are practical extraction seams if maintainability work is
undertaken, provided shared authorization and response contracts remain tested.
Ingestion already has documented per-file and cross-file stage boundaries; a
future decomposition should preserve their explicit transaction/failure outcomes.
Line count alone is not a finding and this map performs no behavioral refactor.

Cross-cutting work remains: analyzer/LSP subprocess and parser limits, dependency
reproducibility, full envelope/status parity, active computation cancellation,
representative independent judgments, learned-graph comparison, and operations
beyond the selected readiness/restore fixtures. Confirmed findings must retain
severity, evidence and a remediation issue or owner disposition. Inventory entries
must not be checked off as fully reviewed merely because tests compile or one
neighboring path has a passing fixture.
