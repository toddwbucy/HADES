# Verified conventional Jina code provider

Part of [#225](https://github.com/toddwbucy/HADES/issues/225), following
[workload discovery](registered-provider-discovery.md#workload-observations).

## Contract and scope

`services/embedding/verified_code.py::VerifiedJinaCodeProvider` is an opt-in
operator-created provider. It serves two conventional workloads from one
registered model snapshot and returns `hades-provider-description-v2`:

| Workload | Passage / query tasks | Prompts | Adapter |
| --- | --- | --- | --- |
| Documents | `retrieval.passage` / `retrieval.query` | `Passage: ` / `Query: ` | `retrieval` |
| Code | `code` / `code` | `Passage: ` for both roles | `code` |

The symmetric code prompt matches HADES's existing explicit code-task choice;
it is part of this adapter's declared space, not an implicit model default.
The immutable `workload_contracts.py` table supplies these task/prompt/adapter
records to serving and qualification. HTTP task literals are built from the
operator-created provider's supported tasks, preserving strict request admission.
Both workloads use the standard Qwen2 fast tokenizer without truncation,
attention-masked mean pooling in float32, and L2-normalized 2048-dimensional
float32 vectors. Model parameter precision, tokenizer behavior and numerical
runtime are recorded in the space identity. The original conventional document
space is preserved; code gets a different space ID even at the same dimension.

Receipts share the registered artifact MD5 and one serving generation. Every
request must supply the receipt for its selected task. A document receipt cannot
authorize code or vice versa. Parameter, tokenizer, configuration or artifact
drift invalidates the complete load. Unload/reload refreshes both generations;
ordinary requests never silently reload or replace their binding.

The constructor requires `code_max_input_tokens` independently of the document
`max_input_tokens`. Both must fit the loaded model/tokenizer context. The provider
counts the exact prompt and special-token overhead for every batch member before
any forward pass. A later oversized input refuses the whole batch. Conventional
planning clamps to the smaller saved-profile and selected live-workload limit,
preserving source bytes, complete coverage and independently counted overlaps.

`create_verified_app()` exposes code embedding/planning only for the code adapter.
Existing `VerifiedJinaProvider` and `VerifiedJinaLateProvider` retain their original
single-space HTTP descriptions and task admission. The new adapter advertises no
images, contextual code, or general tokenizer-offset support. There is no new
production CLI/environment activation, database setup flow or code graph writer.
Managed code ingestion remains refused until its admission and persistence path
is implemented. These vectors must not be assigned to legacy records by relabeling.

## Qualification

The fixed code suite contains eight snippets across Python, Rust, SQL and
JavaScript and six predeclared natural-language searches. The existing document
suite runs separately. This is a contract/retrieval smoke check, not a general
retrieval-quality benchmark.

`scripts/qualify_verified_code.py` checks every registered adapter checkpoint
tensor against the loaded weights, exact tokenizer normalization/whitespace,
finite normalized output, each expected top-ranked result, repetition and reload.
Fixed cosine thresholds are `0.99999` for repetition and `0.9999` across reload;
maximum norm error is `0.001`. Real forward hooks check the selected adapter and
untruncated IDs at exact-fit boundaries. One-over and swapped receipts must fail
before inference. Workload ceilings must differ during qualification so accidental
sharing can be detected. Fixtures and thresholds must be committed before runs;
retain failed reports without tuning cases to observed results.

The command owns a private Unix HTTP server and runs the ignored Rust
`verified_jina_workloads::registered_conventional_workloads` contract. Rust reads
the protected registration independently, compares the complete V2 observation,
constructs an explicitly mixed profile in memory, and binds both passage/query
roles. It checks retrieval and long-source plans through the actual transport;
Python independently recounts accepted byte slices before shutdown. The recount
pins the task and receipt from the expected workload's actual observation before
inspecting counts; Rust output cannot select its own verification workload.
Rust also asserts the planned task
and receipt and sizes each initial chunk policy against that workload's ceiling,
including unequal cases such as 1024/64 tokens. No database or installed service
is used. The server, child process and model load are owned through failure,
timeout and cancellation cleanup.

Both workloads run on the same initial load and then share one reload. The full
adapter checkpoint is compared once per generation, shared across workloads;
validation is never reused across a reload. `initial_observation_seconds` and
`initial_observation_loaded_model` distinguish a cold load from description
revalidation. The original, contextual and code CLIs share preflight, exclusive
evidence creation, implementation identity, interruption handling and reporting
through `QualificationRun`. Their private HTTP transports also share cleanup.
Secondary shutdown failures are retained in `cleanup_errors` alongside the
original error, and a cleanup-only failure cannot produce successful evidence.

Use the approved ML environment and a previously approved registration. Do not
upgrade the installed runtime or download model assets as part of this command:

```bash
cargo fetch --locked
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /path/to/approved/python scripts/qualify_verified_code.py \
  --registry /path/to/protected/registry --provider approved-jina \
  --device cuda:0 --max-input-tokens 1024 --code-max-input-tokens 512 \
  --output-dir /path/to/new/code-qualification
```

Use disk-backed registry/session and runtime cache directories with adequate free
space. A requested CUDA device must exist; CPU fallback is refused. Evidence
records the implementation commit/dirty status, suite and planning-source hashes,
artifact manifest, runtime, receipts, numerical observations and Rust result.
The output directory must be new. Failure or cleanup failure keeps `passed: false`
and exits unsuccessfully; do not discard the failure report.

CPU regressions use tiny deterministic model fixtures and real registration,
tokenization and HTTP logic. Run them with the dependency/protobuf setup in
[the test matrix](workstation-specific-tests.md); ordinary CI never loads the
approved large model. Real-model evidence is recorded separately under
`docs/evidence/` after the implementation and protocol are committed.

## Reviewed protocol v2 qualification, 2026-10-21

[Committed v2 evidence](../evidence/jina-code-v2-thinkpad-2026-10-21.json) passed
on clean implementation `fe0c6dc34bbc4a7a0882000c7b9b99c0a24aeeb0` after the
review corrections. It used the same approved artifact, runtime, GPU, fixed
cases and thresholds as the initial run below. Both workload space IDs stayed
unchanged. The run occurred October 22 UTC / October 21 in America/Chicago.

| Check | Documents | Code |
| --- | --- | --- |
| Predeclared top-one retrieval, Python and Rust | 6/6 | 6/6 |
| Exact-fit / one-over token boundary | 1024 / 1025 | 512 / 513 |
| Repeat and reload cosine | approximately 1.0 | approximately 1.0 |
| Long-source rendered tokens | 2402 | 1619 |
| Planned, embedded and workload-bound recounted passages | 10 | 7 |
| Initial observation loaded the model | yes | no |

All 1518 adapter tensors matched on each of two serving generations; both
workloads shared each verified load. Oversized input and all three cross-workload
receipt substitutions caused zero forward calls. Maximum norm error stayed
below `7.5e-8`. The complete run took 78 seconds, including 19 seconds for Rust
HTTP and recount checks. Peak allocated CUDA memory was 8.88 GB. Cleanup left
no serving copies and restored GPU memory to its pre-run baseline. This evidence
qualifies the declared conventional provider behavior; code graph persistence,
contextual code and production activation still require subsequent work.

## Initial ThinkPad qualification, 2026-10-21

[Committed evidence](../evidence/jina-code-thinkpad-2026-10-21.json) passed on clean
implementation `2e2a7cd307b4480b6f89f6c3cad14313037831a2`, using the approved local
artifact MD5 `446160668af00248832ed8cf5d8d4432`. The runtime was Torch 2.14.1,
Transformers 4.52.4, CUDA 13.0 and the RTX PRO 5000 Blackwell laptop GPU.
The run occurred October 22 UTC / October 21 in America/Chicago.

This v1 report is retained as historical evidence. Review identified that its
independent recount did not pin the expected workload. Protocol v2 adds that
check and the shared-load, policy and lifecycle changes described above. The
fresh committed v2 run above supersedes this report for qualification.

| Check | Documents | Code |
| --- | --- | --- |
| Predeclared top-one retrieval, Python and Rust | 6/6 | 6/6 |
| Exact-fit / one-over token boundary | 1024 / 1025 | 512 / 513 |
| Repeat and reload cosine | approximately 1.0 | approximately 1.0 |
| Long-source rendered tokens | 2402 | 1619 |
| Planned, embedded and independently recounted passages | 10 | 7 |

All 1518 registered adapter tensors matched loaded values. Oversized batches and
all three cross-workload receipt substitutions caused zero forward calls.
Maximum norm error stayed below `7.5e-8`. The complete run took about 106 seconds;
the Rust HTTP/recount stage took 24 seconds. Peak allocated CUDA memory during
the provider checks was 8.88 GB. Cleanup completed successfully. This qualifies
the declared conventional provider behavior, not code graph persistence,
contextual code, production activation or general retrieval quality.
