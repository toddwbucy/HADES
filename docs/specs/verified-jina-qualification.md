# Real Jina qualification

This opt-in protocol qualifies the internal [verified provider](verified-jina-provider.md)
and [profile-bound document flow](profile-bound-documents.md) for #225 with an
already approved local Jina v4 registration. Ordinary CI uses synthetic weights.
The real run needs an explicit ML environment, device, registration, and disposable
database. It does not activate database pairing or change installed services.

## Public-search evidence on the ThinkPad

The [2026-10-21 UTC public-search run](../evidence/jina-public-search-thinkpad-2026-10-21.json)
passed on clean implementation `60616ac77b17cab07ef02b53b6c832db94420a4e`,
using the existing approved Jina registration on the RTX PRO 5000 laptop GPU.
The shared `db.query` handler resolved explicit provider routes with the legacy
endpoint deliberately unavailable. All six expected answers ranked first, every
returned passage matched its fixture text, and each response carried the ready
profile's verified query binding. No corpus or installed service was changed.

The same run retained setup/retry checks, eight stored retrieval documents,
and the fixed 12,384-byte Unicode source: 2,402 whole-input tokens, six passages,
and six successful independent token recounts. The long-source ingest/search
check took 2.956 seconds. The report records provider/runtime identities and
per-query timings; concurrent workstation tests and these small fixtures do not
establish production throughput or a model-quality comparison.

The [post-review run](../evidence/jina-public-search-reviewed-thinkpad-2026-10-21.json)
also passed on clean `1791df30d017c934891729846e7d30a63be17109` after grouping
detail reads by parent while retaining only one complete source at a time.
All six public retrieval/source-text checks and the long-source tokenizer gates
still passed. The original measurement is retained above.

## ThinkPad file-ingestion result: 2026-10-21 UTC

The [foreground file-ingestion run](../evidence/jina-file-ingestion-thinkpad-2026-10-21.json)
passed on clean implementation `9929668ae0b31dc48f12f80f0da136d6df6edb8a` on
2026-10-21 UTC. All eight fixed sources entered through bounded file snapshots
and private gRPC extraction before actual verified Jina planning/inference.
All six expected answers ranked first through public search, and source text and
namespaced file metadata were preserved. The long Unicode source retained all
12,384 bytes and six passages with independent tokenizer recounts; its full
file-ingest/search check took 2.418 seconds. Model artifacts, runtime and space
matched the previous qualification. The extractor was a text-passthrough test
fixture; this is embedding/storage integration evidence, not Docling quality evidence.

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
- Rust reads the protected registration itself and obtains the live description
  from a private verified HTTP server. This independent observation must match
  the qualified manifest/receipt before constructing the initializing profile.
  Document admission resolves the same registration and observation again.
  The internal setup coordinator creates the disposable database with its durable
  ledger and ownership marker, checks the provider before readiness, and proves
  an identical setup retry preserves the profile snapshot.
  `with_temp_db` initializes a fresh profile, stores all eight documents and their
  bindings through foreground file ingestion, and retrieves all six expected
  answers from stored vectors through
  the shared public `db.query` service handler using explicit provider routes.
  The legacy endpoint is deliberately unavailable. Returned source text must
  equal the expected fixture, and the response must carry a ready profile binding.
  A private text-passthrough gRPC extractor receives immutable fixture file bytes.
  File provenance and namespaced metadata must be committed with the document.
  This controlled extractor isolates transport/storage behavior; the run does
  not qualify Docling quality or nontext extraction formats.
- After those retrieval checks, the fixed Unicode text in
  `services/tests/fixtures/jina_passage_planning.txt` must produce multiple
  conventional passages with overlap. The source bytes survive storage; every
  stored body/input/overlap count is independently checked with the same live
  tokenizer. Search validates all persisted chunks. The report retains the plan,
  source SHA-256, whole-input token count, and recount result.

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

## ThinkPad token-planning result: 2026-10-20

The [token-planning report](../evidence/jina-token-planning-thinkpad-2026-10-20.json)
passed on clean implementation commit `8dc3de2b6aadc52d52d70bbac26c0272c0bc80d1`,
using the same registered artifacts, runtime, and embedding space as above.
The original six retrieval answers ranked first in both provider and stored-vector
checks. The additional fixed Unicode source produced:

| Observation | Result |
| --- | --- |
| Original source | 12,384 UTF-8 bytes; 2,402 prompt-inclusive tokens |
| Saved policy / effective serving ceiling | 500 body tokens, at most 50 overlap tokens / 2,048 input tokens |
| Persisted passages | 6; five with 500 body / 502 input tokens, one with 150 / 152 |
| Actual overlaps | 0 for the first passage, then 50 tokens each |
| Independent recount | All 6 persisted body/input/overlap counts match |
| Long-document planning, embedding, commit, and full search scan | 2.52 s |

The full plan and original source SHA-256 are retained in the report. The source
bytes, including combining accents, survived unchanged. These measurements
qualify this conventional planning path; they do not establish semantic chunk
quality or late-chunk behavior. Private database/provider cleanup completed,
GPU usage returned to its initial 495 MiB, and installed services were unchanged.

## ThinkPad registered-discovery result: 2026-10-20

The [registered-discovery report](../evidence/jina-provider-discovery-thinkpad-2026-10-20.json)
passed on clean implementation commit `885c4dd4d6fc7a9c0d4c25eb8dfcaed645a1b344`,
using the same registered artifacts, runtime, and embedding space. Rust resolved
`qualification-jina` through the explicit installation catalog, read the protected
registration, and obtained the actual HTTP description independently. That
observation agreed with Python's qualification before profile initialization;
document admission repeated the registration and live-description checks.

| Observation | Result |
| --- | --- |
| Independent registration / live-description agreement | Passed before initialization and admission |
| Provider retrieval / stored-vector retrieval | 6/6 expected first in each |
| Loaded adapter tensors / task input boundaries | 1,518 checked; 2,048 accepted and 2,049 refused before inference |
| Long Unicode document | Same 12,384 source bytes and 6 persisted passages |
| Independent body/input/overlap recount | All 6 match |
| Long-document planning, embedding, commit, and full search scan | 2.00 s |

The complete receipt, manifest, runtime, plan, and timings are retained in the
report. The private database stopped cleanly, serving snapshots were removed,
and GPU usage returned to 495 MiB. Installed services were unchanged. This is
evidence for the internal registered-provider flow; public setup remains gated.

## ThinkPad recoverable-setup result: 2026-10-21 UTC

The [recoverable-setup report](../evidence/jina-recoverable-setup-thinkpad-2026-10-21.json)
passed on clean implementation commit `d8ce9c668101b5411ccd8b1c9c23724cb1253324`,
using the same registered artifacts, runtime, and embedding space. The setup
coordinator created the harness-owned disposable database, persisted its ownership
marker and profile, and verified the provider before readiness. Repeating the
identical request returned the same complete receipt and preserved the exact
database profile snapshot.

| Observation | Result |
| --- | --- |
| Setup / identical retry | Ready revision two; same receipt and unchanged profile snapshot |
| Provider retrieval / stored-vector retrieval | 6/6 expected first in each |
| Long Unicode document | Same 12,384 source bytes and 6 persisted passages |
| Independent body/input/overlap recount | All 6 match |
| Long-document planning, embedding, commit, and full search scan | 2.02 s |

The report retains the setup receipt, request digest, model receipt, manifest,
runtime and complete passage plan. The private database stopped cleanly, serving
snapshots were removed, and GPU usage returned to 495 MiB. Installed services
were unchanged. Failure/retry behavior is separately exercised by the disposable
fault-injection contracts; this run qualifies successful setup with the real model.

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

The original result above used one explicit passage per small document. The
current protocol adds [token-aware planning](profile-token-planning.md) and a
long Unicode fixture; its evidence is recorded separately. Public `db.query`
search and foreground file ingestion are now qualified through their shared paths.
Background job pins, code/late-chunk capabilities, legacy policy and frontend
creation remain separate work. Registered local provider discovery and internal recoverable
document setup are now part of the qualification flow; public setup authorization
and administrative reconciliation/backup tooling remain pending.
Public pairing remains gated.
