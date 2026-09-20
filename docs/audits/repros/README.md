# Audit Reproductions

These are investigation artifacts for epic #12, not production fixes. The contract probes assert intended behavior and **fail on revision a71d73b**. Do not interpret their failures as a failed setup when they match the audit report.

Use the existing dependency environments. Do not install packages into the running services' environment or run the repository's blanket database/smoke tests against a live instance.

## Python CPU contracts

From the repository root:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 120s nice -n 10 services/.venv/bin/python -m pytest \
  -p no:cacheprovider docs/audits/repros/test_training_contracts.py -q --tb=short
```

Expected on the audited revision: five failures (AUC ties, held-out adjacency, async abort, checkpoint dimension compatibility, empty split). Tests use tiny tensors, temporary files and a mock context; they do not contact a server.

## Rust file-key and mock-embedding contracts

Create a temporary Cargo package, keeping build output outside the working service tree:

```bash
mkdir -p /tmp/hades-audit-rust
cat > /tmp/hades-audit-rust/Cargo.toml <<'EOF'
[package]
name = "hades-audit-contracts"
version = "0.0.0"
edition = "2024"

[workspace]

[lib]
path = "/opt/HADES/docs/audits/repros/rust_contracts.rs"

[dependencies]
hades-core = { path = "/opt/HADES/crates/hades-core" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
tempfile = "3"
EOF
cp Cargo.lock /tmp/hades-audit-rust/Cargo.lock
env CARGO_TARGET_DIR=/tmp/hades-audit-target CARGO_BUILD_JOBS=1 \
  CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  timeout 300s nice -n 10 cargo test --offline \
  --manifest-path /tmp/hades-audit-rust/Cargo.toml -- --test-threads=1
```

Adjust the two absolute source paths for another checkout. The copied lockfile preserves the repository's existing versions while Cargo adds the audit package locally. Expected: three contract failures. Socket-restricting sandboxes require approval for temporary Unix socket creation; the mocks never contact the actual embedder.

## Disposable database and ingest fixture

After checking resource headroom, use an existing ArangoDB executable:

```bash
python3 docs/audits/repros/run_isolated_database.py \
  --arangod /home/todd/git/arangodb/build/bin/arangod --codebase-tests
```

The runner creates a fresh mode-0700 directory under `/tmp/hades-audit-db-*`, a separate data store and a Unix-only listener. It clears inherited HADES/Arango settings, disables CUDA visibility, fixes both database sockets to that instance, and directs ML endpoints to nonexistent sockets. ArangoDB and child commands use one CPU at reduced priority. The server uses small RocksDB caches; this is not a complete cgroup memory cap.

It runs cache tests, builds a temporary debug CLI, ingests the two colliding Python paths, prints the resulting rows, and optionally runs the 60 codebase tests in strict mode. Expected on the audited revision: ordinary tests pass, both ingests report success, but only one file record survives. The collision observation is printed rather than used as the runner's exit status. Command failures raise an error.

The runner terminates only the server process it created, even after an exception. It retains logs/data for inspection; do not commit those temporary artifacts. No production sockets, databases, service units, binaries or model files are modified.

