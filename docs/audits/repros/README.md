# Historical Audit Reproductions

These probes assert intended behavior and **fail on audited revision
`a71d73bfe988e17d487a2db1d35dadf3a18f0664`**. They are historical evidence,
not the current regression suite. Use `scripts/test_isolated_database.py` for
maintained database contracts. The historical runner now shares its memory and
process cleanup controls with that maintained runner.

## Required source checkout

Keep the probe files in this current checkout and create a separate, clean
worktree for the code under test. Run the following commands from the current
repository root. Set `AUDIT_PYTHON` to an existing dedicated test interpreter
with CPU test dependencies; never install into a running service environment.

```bash
git worktree add --detach /tmp/hades-historical-source a71d73bfe988e17d487a2db1d35dadf3a18f0664
export HADES_AUDIT_SOURCE=/tmp/hades-historical-source
export AUDIT_PYTHON=/path/to/test/python
python3 docs/audits/repros/historical_source.py
```

The validator rejects another commit, tracked modifications or nonignored untracked files. Python/database
probes invoke it before executing the historical source. Generated bindings and
build outputs are ignored; keep this worktree dedicated to historical probes.

## Python CPU contracts

```bash
make -C "$HADES_AUDIT_SOURCE/services" proto-gen PROTO_DIR=../proto PYTHON="$AUDIT_PYTHON"
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout 120s nice -n 10 "$AUDIT_PYTHON" -m pytest \
  -p no:cacheprovider docs/audits/repros/test_training_contracts.py -q --tb=short
```

Expected: five failures (AUC ties, held-out adjacency, async abort, checkpoint
dimension compatibility, empty split). Tiny tensors, temporary files and mock
contexts are used; no server is contacted.

## Rust key and mock-embedding contracts

Generate an isolated manifest only after validating the historical source:

```bash
python3 - <<'PY'
import json, os, pathlib, shutil, subprocess
repo = pathlib.Path.cwd()
source = pathlib.Path(subprocess.check_output(
    ['python3', 'docs/audits/repros/historical_source.py'], text=True).strip())
root = pathlib.Path('/tmp/hades-audit-rust')
root.mkdir(exist_ok=True)
(root / 'Cargo.toml').write_text('''[package]
name = "hades-audit-contracts"
version = "0.0.0"
edition = "2024"
[workspace]
[lib]
path = %s
[dependencies]
hades-core = { path = %s }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
tempfile = "3"
''' % (json.dumps(str(repo / 'docs/audits/repros/rust_contracts.rs')),
       json.dumps(str(source / 'crates/hades-core'))))
shutil.copyfile(source / 'Cargo.lock', root / 'Cargo.lock')
PY
env CARGO_TARGET_DIR=/tmp/hades-audit-target CARGO_BUILD_JOBS=1 \
  CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
  timeout 300s nice -n 10 cargo test --offline \
  --manifest-path /tmp/hades-audit-rust/Cargo.toml -- --test-threads=1
```

Expected: three contract failures. Temporary Unix mocks do not contact the
actual embedder. Use a disposable build environment with sufficient capacity.

## Disposable database and ingest fixture

```bash
python3 docs/audits/repros/run_isolated_database.py \
  --source-root "$HADES_AUDIT_SOURCE" --arangod /path/to/test/arangod --codebase-tests
```

The runner validates the historical commit before starting anything. It creates
its own private directory, store and Unix-only endpoint, clears inherited HADES
and Arango settings, hides CUDA and points ML endpoints at nonexistent sockets.
Children use one CPU, reduced priority and an 8 GiB address-space ceiling. Each
command and server owns a private process group; cleanup includes descendants
on timeout, error and normal completion. This is not a cgroup-wide memory cap.

It runs cache tests and optionally codebase contracts, builds a temporary CLI,
and ingests two colliding Python paths. Expected: both ingests report success,
but only one file remains. This historical observation is printed, not encoded
in the runner exit status. Logs/data remain under its printed temporary path.
No live endpoint, database, service unit, installed binary or model is modified.

## Synthetic restore rehearsal (#63)

`restore_fixture.py` uses existing matching ArangoDB binaries and two sequential
private servers. See [scope and reproduction](../active-database-recovery.md) and
[recorded result](../restore-fixture-result.json). This is an opt-in audit probe,
not the maintained database CI suite or a production backup procedure.
