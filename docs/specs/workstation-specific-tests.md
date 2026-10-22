# Test environments and CI gates

Updated for audit [#20](https://github.com/toddwbucy/HADES/issues/20).

## Test matrix

| Gate | Coverage | External resources |
|---|---|---|
| Rust build/test/lint | Workspace library and binary tests, rustfmt, all-target Clippy | Pinned Rust toolchain, protoc |
| Service-free contracts | `proto_types`, `pipeline`, `config_integration`, embedding contract/client, extraction client, training client | Private mock sockets; no installed ML services |
| Python CPU contracts | Training topology, metrics, checkpoint/schema/RPC validation, models, subset export, adapter scope | Python 3.12.14 and hashed CPU dependency lock; generated protobufs |
| Isolated database contracts | CRUD/index/query/transport/cache, graph loader/contract, cursor cancellation, codebase invariants, file identity, full CLI lifecycle | Disposable ArangoDB with vector indexes enabled |
| Profile selection | Persistent enablement, switching, simulated reboot, failure/rollback | Mock systemctl/curl only |
| Optional workstation probes | `clang_cuda_probe`, `gopls_semantic`, `ra_span_agreement`, CUDA-specific Python case | Explicit analyzer/CUDA prerequisites; not evidence supplied by CPU CI |

CI installs the checksum-pinned Linux x86_64 protoc 23.4 archive through
`scripts/install_ci_protoc.sh`, without querying GitHub's release listing.
When updating the pin, verify its checksum and compile all canonical `proto/` files.

The opt-in [Jina code qualification](verified-code-provider.md) uses approved
offline model assets and a private HTTP server to check actual document/code
adapters through Rust. `verified_jina_workloads` is ignored in ordinary CI;
CPU provider/qualification regressions run in `services/tests`. This check creates
no database and does not activate managed code ingestion.

The CLI lifecycle uses a private deterministic embedding HTTP fixture. It covers
real storage, late-chunk response mapping, query retrieval, content and symbol-line
changes, file moves, drift/retirement, graph validation, deletion, and a failed
document phase alongside durable code ingestion/retry. These vectors establish
pipeline correctness, not the quality of a production embedding model.

## Run database contracts

Use an isolated checkout on an active server. Never export a live socket or corpus
name into these tests. The runner creates a private 0700 directory and Unix-only
server, replaces inherited HADES/Arango configuration, sets `ARANGO_TESTS=1`, and
terminates its own server/process groups on success, error, timeout, or caught
interruption. SIGKILL and machine failure cannot run cleanup.

```bash
cargo fetch --locked
python3 scripts/test_isolated_database.py --arangod /path/to/test/arangod
# Or use the pinned official container image (Docker required):
python3 scripts/test_isolated_database.py --docker
```

Each write test uses `hades_core::test_support::with_temp_db`; no named corpus or
preexisting seed data is required. The harness requires explicit `ARANGO_SOCKET`
and `ARANGO_PASSWORD` and never defaults to a system socket. Missing prerequisites
are visible skips outside strict mode and failures under `ARANGO_TESTS=1`. The
runner also proves the missing-socket failure path. Vector-index unavailability
is a strict failure, not a passing skip.

Server caches and thread counts are capped. Local server/test commands have one
CPU affinity, reduced priority, an 8 GiB address-space ceiling, and timeouts. The
CI server container additionally has a 1 GiB memory ceiling, PID limit, read-only
root, private writable fixture mount, and no network. Logs and command results
remain in the printed `/tmp/hades-tests-*` directory; CI uploads only logs/results.

## Run CPU service contracts

Install into a new test environment, never the running services' environment:

```bash
python3.12 -m venv /tmp/hades-test-python
/tmp/hades-test-python/bin/python -m pip install --require-hashes --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cpu -r services/requirements-ci.txt
make -C services proto-gen PROTO_DIR=../proto PYTHON=/tmp/hades-test-python/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/hades-test-python/bin/python -m pytest services/tests -v
```

The lock pins direct and transitive dependencies with distribution hashes. Its
input records the regeneration command. CUDA-specific coverage remains an explicit
skip on CPU runners. Adapter tests generate their own corpus; no WeaverTools
checkout, model download, production database, GPU, or installed service is needed.

## Adding tests

Prefer service-free fixtures. Put database workflows in integration binaries,
with deterministic seed data and teardown through `with_temp_db`. Do not mix
credential-dependent database tests with unit tests that mutate process-wide
environment variables. Add the target to the isolated runner, ensure strict mode
fails when prerequisites are missing, and assert meaningful stored state after
failure/retry. CI green is evidence only for the gates listed above, not live
inference, production reboot behavior, backup restoration, or retrieval quality.

## Embedding profile persistence

The `embedding_profile_store` database contract covers the internal
[profile persistence and prepared-write guard](persisted-embedding-profiles.md):
immutable identity records, atomic readiness publication, stale/cross-database
refusal, metadata validation, and transaction rollback/cancellation. Select it
with `--contract embedding_profile_store`; it is also in the default CI matrix.

The `profiled_documents` contract covers the internal
[document ingestion/search integration](profile-bound-documents.md), using
private verified HTTP fixtures and disposable two- and three-dimensional
databases. It checks provider/profile drift, stored provenance, atomic replacement,
scan budgets, and database isolation. Select it with `--contract profiled_documents`;
the default CI matrix includes it. No real model or installed service is used.

The `profiled_code` contract checks internal [code graph persistence](profile-bound-code.md):
workload selection, bounded preparation, complete graph publication, replacement
conflicts/rollback, profile and provider drift, edited-record preservation and
cross-database refusal. Use `--contract profiled_code`; the default matrix includes it.
Public code ingestion remains gated pending its integration.

The `workspace_setup` contract covers [recoverable setup](recoverable-workspace-setup.md):
durable reservations, ownership markers, interrupted/uncertain writes, exact
retry, concurrent attempts, and refusal of existing or changed databases.
Select it with `--contract workspace_setup`; the default CI matrix includes it.
Ledger filesystem contracts run in the service-free `workspace_setup` unit tests.

The `profiled_search` contract checks the public search handler and runs the
`profiled_admission` CLI target against disposable databases. It covers model
isolation, source text, profile/provider drift and refusals before legacy vector
work; use `--contract profiled_search`. Both targets run in the default CI matrix.
See [public search admission](profiled-search-admission.md).

The `profiled_files` contract checks [foreground file ingestion](profiled-file-ingestion.md)
and runs the `profiled_ingestion` CLI target. It covers bounded immutable source
bytes, atomic metadata/replacement, cancellation, provider/profile/source drift,
policy refusal and partial batch failure. Select it with `--contract profiled_files`;
both targets also run in the default isolated CI matrix.

The `profiled_jobs` contract checks [background job pins](profiled-ingestion-jobs.md)
and runs the matching CLI target. It covers database/profile isolation, sealed
handoff, real daemon workers, explicit replacement and profile drift during
extraction. Both targets also run in the default isolated CI matrix.

The opt-in [real Jina qualification](verified-jina-qualification.md) uses an
approved local registration, fixed smoke cases, private verified HTTP, and the
ignored `verified_jina_documents` test. Its owner selects `--verified-jina` plus
a new `--verified-jina-report` on the same strict runner. It is separate from the
default matrix and cannot turn absent real-model/GPU prerequisites into passing
qualification evidence.

## Codex reading contracts

The isolated database matrix includes `reading_chat`: durable request deduplication,
immutable passage/reference context, ownership exclusion, approvals, interruption,
receipt reconciliation, and transcript pagination. It uses a deterministic App
Server fixture and requires no Codex account. CI also runs
`node crates/hades-frontend/web/tests/reading-chat.test.cjs` for browser recovery
and stale response handling.

The opt-in [browser and real Codex checks](../codex-reading.md#limits-and-validation)
use the same private server runner and `with_temp_db`. They exercise the actual
viewer/daemon/MCP path, human graph acceptance, and worker reconnect. Real inference
requires explicit authentication input; it is not a CI prerequisite.
