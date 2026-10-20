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
runs; do not weaken numerical/retrieval thresholds or choose different queries after seeing results.
This is a multilingual retrieval smoke test, not the #167 evaluation baseline
or evidence that this model is preferable to another model.

`scripts/qualify_verified_jina.py` requires all of these gates:

- Every registered adapter checkpoint tensor matches the loaded model tensor.
  Missing-adapter warnings fail the run.
- Vectors have 2,048 finite components and L2 norm error at most 0.001. Every
  expected document ranks first using independently computed cosine similarity.
- A repeated query has cosine at least 0.99999; after unload/reload it is at least
  0.9999. Artifact and space identities stay stable while load identity changes.
- Tokenizer encode/decode preserves whitespace and follows only its declared
  identity or NFC normalization, checked against Python's independent Unicode
  implementation. IDs remain identical after reload. Original database source
  text remains byte-for-byte intact. No tokenizer regex correction is applied.
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

## ThinkPad result: 2026-10-20

The [complete report](../evidence/jina-qualification-thinkpad-2026-10-20.json)
passed on clean implementation commit `d9474e5200027b83b2dc58fe89688c9308bb0bb3`.
The [failed tokenizer preflight](../evidence/jina-qualification-thinkpad-2026-10-20-preflight.json)
is retained for the protocol correction described below.

The run used the NVIDIA RTX PRO 5000 Blackwell Generation Laptop GPU, CUDA runtime
13.0, PyTorch 2.14.1, Transformers 4.52.4, tokenizers 0.21.4, PEFT 0.15.2,
and Accelerate 1.15.0. No FlashAttention package was installed. The private native
ArangoDB reported 3.12.12-1. Jina revision
`853c867b65b749f3c3c72a06868140d842e04f06` supplied 22 assets totaling
7,885,849,280 bytes, with aggregate manifest MD5
`446160668af00248832ed8cf5d8d4432`.

| Observation | Result |
|---|---|
| Loaded adapter tensors checked against registered bytes | 1,518 |
| Provider retrieval / stored-vector retrieval | 6/6 expected first in each |
| Maximum vector norm error | 7.47e-8 |
| Repeated / reloaded query cosine | 1.0 / 1.0 |
| Both task boundaries | 2,048 actual forward tokens accepted; 2,049 refused with zero forwards |
| Private load / reload | 18.06 s / 16.44 s |
| Eight passages / six queries, direct provider | 3.11 s / 0.89 s |
| Eight document commits / individual stored-vector query | 4.40 s / 0.51–0.53 s |
| Peak CUDA allocation / reservation during provider gates | 8.71 GiB / 9.16 GiB |

The final space ID is
`3adfde754e36707cac76908558b20477bacf1a5d19391490550e6c987d4f00a3`.
The database round trip used the new post-reload receipt; the original receipt
was refused. Serving snapshots were removed and GPU memory returned to its
pre-run level. Installed HADES services were not restarted.

This runtime emitted a video-processor configuration deprecation warning and a
Triton C-header macro redefinition warning; neither altered registered assets.
It emitted no tokenizer regex warning. These results qualify this recorded
runtime, not the separate Transformers 4.57.6 environment or legacy vector space.

## Scope of the result

Qualification applies to the recorded artifacts, runtime, numerical contract,
and serving ceiling. A different library version, precision, tokenizer behavior,
or capability needs its own evidence and compatible identity checks. A warning
from a different Transformers installation does not establish a defect in this
runtime; retain and investigate warnings rather than silently editing assets.

The first v1 attempt stopped before embedding inference because it incorrectly
required byte-equivalent decoded Unicode. The registered tokenizer explicitly
declares NFC and turns combining `é` into composed `é`. Protocol v2 corrects that
gate to verify declared NFC independently, rejects other normalization policies,
and still requires exact whitespace preservation. The cases and numerical/ranking
criteria did not change; the failed v1 report is retained with the run evidence.

The database test uses one explicit passage per small document. Token-aware
planning, all public CLI/MCP/job paths, code/late-chunk capabilities, trusted
provider discovery, recoverable provisioning, legacy policy, and frontend
creation remain separate work. Public pairing remains gated.
