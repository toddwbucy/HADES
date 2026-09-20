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
