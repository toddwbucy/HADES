# Real Jina qualification

This opt-in protocol qualifies the internal [verified provider](verified-jina-provider.md)
and [profile-bound document flow](profile-bound-documents.md) for #225 with an
already approved local Jina v4 registration. Ordinary CI uses synthetic weights.
The real run needs an explicit ML environment, device, registration, and disposable
database. It does not activate database pairing or change installed services.

## Fixed protocol and evidence

`services/tests/fixtures/jina_qualification.json` defines eight short documents,
six independently specified retrieval answers, and Unicode/whitespace samples.
The protocol and criteria are committed before real inference. Retain failed
runs; do not weaken thresholds or choose different queries after seeing results.
This is a multilingual retrieval smoke test, not the #167 evaluation baseline
or evidence that this model is preferable to another model.

`scripts/qualify_verified_jina.py` requires all of these gates:

- Every registered adapter checkpoint tensor matches the loaded model tensor.
  Missing-adapter warnings fail the run.
- Vectors have 2,048 finite components and L2 norm error at most 0.001. Every
  expected document ranks first using independently computed cosine similarity.
- A repeated query has cosine at least 0.99999; after unload/reload it is at least
  0.9999. Artifact and space identities stay stable while load identity changes.
- Exact Unicode and whitespace survive tokenizer encode/decode; IDs remain
  identical after reload. No tokenizer regex correction is applied.
- Both task prefixes accept an exactly fitting input. A forward hook observes
  the actual IDs reaching the model, including prefix/special-token policy.
  One additional token rejects the entire mixed batch before any forward pass.
  Wrong artifact, space, and load receipts also fail before inference.
- A private verified HTTP server supplies real vectors to the Rust bound client.
  `with_temp_db` initializes a fresh profile, stores all eight documents and their
  bindings, and retrieves all six expected answers from stored vectors.

`report.json` records success/failure, the failing stage and partial observations,
exact suite SHA-256, implementation commit/dirty status, manifest, runtime versions,
receipts, retrieval results, norm error, synchronized timings, and peak PyTorch
CUDA allocated/reserved bytes. These allocator peaks exclude non-PyTorch GPU
memory. Timings include receipt checks and are smoke measurements, not throughput
benchmarks. `database.json` and `database.log` record the Rust round trip.
Success requires both provider and database gates, followed by provider cleanup.

## Run on an approved workstation

Use a dedicated checkout and the approved installed model environment; do not
upgrade a running service's dependencies. Create a private registration as
described in [registered loading](registered-model-loading.md). Registration
copies artifacts, and each load makes another independent checked snapshot.
Choose a filesystem with room for these copies and runtime caches; avoid a
small `/tmp` for model data. Temporary sockets/context contain no weight files.

Fetch Rust dependencies before starting the isolated runner:

```bash
cargo fetch --locked
```

Example using an existing local test server binary:

```bash
QUALIFICATION_DATA="$PWD/target/jina-qualification"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HOME="$QUALIFICATION_DATA/hf-cache" \
TORCHINDUCTOR_CACHE_DIR="$QUALIFICATION_DATA/torch-cache" \
TRITON_CACHE_DIR="$QUALIFICATION_DATA/triton-cache" \
PYTHONPATH="$PWD/services" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /path/to/approved/ml-venv/bin/python scripts/qualify_verified_jina.py \
  --registry "$QUALIFICATION_DATA/registry" --provider qualification-jina \
  --device cuda:0 --max-input-tokens 2048 \
  --output-dir "$QUALIFICATION_DATA/run-001" \
  --arangod /path/to/test/arangod
```

The evidence directory must be new; each run gets its own directory. CPU fallback
is refused for an unavailable requested CUDA device. `--device cpu` explicitly
selects CPU qualification and does not establish CUDA evidence. Replace
`--arangod ...` with `--docker` to use the pinned private container from the
[isolated test runner](workstation-specific-tests.md).

The provider listens only on a private Unix socket. Its qualification owner
retains the observed load through a possible cold Rust build, drains requests,
and unloads at exit. The existing strict runner owns the disposable database,
clears inherited HADES/Arango configuration, and cleans up its process groups.
The command never discovers a live corpus or starts/restarts installed services.
Caught interruption writes a failed report and unwinds cleanup; SIGKILL or
machine failure cannot guarantee cleanup.

## Scope of the result

Qualification applies to the recorded artifacts, runtime, numerical contract,
and serving ceiling. A different library version, precision, tokenizer behavior,
or capability needs its own evidence and compatible identity checks. A warning
from a different Transformers installation does not establish a defect in this
runtime; retain and investigate warnings rather than silently editing assets.

The database test uses one explicit passage per small document. Token-aware
planning, all public CLI/MCP/job paths, code/late-chunk capabilities, trusted
provider discovery, recoverable provisioning, legacy policy, and frontend
creation remain separate work. Public pairing remains gated.
