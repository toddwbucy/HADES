# Verified Jina text provider

This slice of [#225](https://github.com/toddwbucy/HADES/issues/225) produces the
provider receipts consumed by the [bound Rust client](bound-embedding-client.md).
It combines [registered loading](registered-model-loading.md) with an explicit
conventional text-retrieval implementation. Production startup does not select
this provider; there is no new environment flag, CLI command, database setup,
or frontend switch.

## Explicit behavior contract

`embedding.verified_jina.VerifiedJinaProvider` takes a trusted `RegisteredModel`,
device/precision settings, and a serving token budget. `describe()` explicitly
loads the private checked snapshot and derives a portable description from the
loaded model and tokenizer. Registration alone never issues a serving receipt.

The `hades-jina-raw-text-v1` implementation:

- Requires the standard Qwen2 fast tokenizer, 2,048 hidden dimensions, and the
  retrieval adapter. Custom tokenizer classes and quantized/trainable parameters
  are refused.
- Supplies exactly `Passage: ` or `Query: `, adds special tokens, preserves
  special-token interpretation, and disables padding and truncation. Every input
  is tokenized and checked, including its prompt, before any forward pass.
- Executes one input per raw forward pass in evaluation mode with gradients
  and autocasting disabled. The request's batch-size hint does not change this
  execution policy.
- Pools all attention-masked token hidden states in float32, including the
  prefix/special tokens, then applies L2 normalization and emits dense float32
  vectors. Invalid widths, nonfinite values, or zero norms fail the operation.

The space document records the artifact MD5, the declared tokenizer asset subset
and its MD5, a fingerprint of the loaded tokenizer backend, exact task/prompt
settings, pooling policy, actual parameter dtypes/shapes, loaded model
configuration, attention implementation, numerical library versions, and relevant
math settings. Acquisition paths and wheel device suffixes are excluded.
The v1 implementation label must change when its numerical semantics change.
Python space canonicalization is tested against the existing Rust fixture.

Tokenizer dependencies are explicitly selected from the registered root:
`config.json`, `tokenizer_config.json`, `tokenizer.json`, and any present
`special_tokens_map.json`, `added_tokens.json`, `vocab.json`, or `merges.txt`.
The adapter uses this standard tokenizer directly, not the model's processor.
The full artifact MD5 still covers all registered model/code/adapter dependencies.

The provider advertises conventional text retrieval only. Images, specialized
code, late chunking, and general token-offset mappings are unavailable. Conventional
passage planning returns source-byte spans with exact token counts; it does not
advertise the `tokenizer_offsets` capability needed for late chunking. This is a new
declared embedding space: the legacy high-level encoder uses implicit defaults,
including bfloat16 autocasting, and historical vectors must not be adopted as
compatible. Late/code behavior needs its own implementation and qualification
before capabilities are expanded.

## Admission and serving lifecycle

`describe()` returns model identity, capabilities, the full space document, and
`hades-provider-receipt-v1`. Returned descriptions are detached from internal
state. A reload gets a fresh serving generation even when artifact and space
identities match. Device relocation and a different serving token budget do not
change the space when the effective numerical contract is unchanged.

Every embedding or planning operation requires an exact expected receipt.
Admission never silently loads or rebinds after unload. One lifecycle lock covers admission,
tokenization, all forward passes, the final artifact/behavior checks, and unload.
Drift discards the complete operation and invalidates the observed load.
The response contains the provider's actual checked receipt, not an unchecked
echo of request identity.

Per-request checks inspect serving-file metadata, runtime configuration,
tokenizer state, parameter identity/version counters and dtype/shape metadata,
and evaluation state. They do not hash weight bytes again. Tokenizer/runtime
metadata hashing has a 32 MiB bound and adds CPU overhead that requires measurement
with a real model. These checks detect ordinary drift; they are not protection
against hostile approved code or a process that controls the service account.
Arbitrary memory changes bypassing framework version tracking are outside this
trusted-runtime boundary.

## Internal HTTP integration

`embedding.verified_http.create_verified_app(provider)` creates an ASGI app for
integration and qualification. It reuses the existing worker ownership, idle
unload, and shutdown-drain lifecycle. Production `embedding.http_server` startup
continues to construct its current backend.

- `POST /v1/verify` explicitly observes the operator-selected provider. HTTP
  callers cannot submit registry paths, artifact digests to adopt, or profiles.
- `POST /v1/embeddings` requires `model`, `task`, `input`, and `expected_receipt`;
  only float encoding and an optional positive batch-size hint are accepted.
  Unknown fields, duplicate JSON keys, coercions, and unsupported capabilities
  fail validation. A successful response includes `receipt` in the shape
  required by the bound client.
- `POST /v1/passages` accepts one source string and the saved conventional token
  policy. It returns a receipt-bound complete plan with measured body/input/overlap
  counts and original UTF-8 offsets. [Planning limits and semantics](profile-token-planning.md)
  apply; it performs no model inference.
- Changed/unavailable identity returns HTTP 409 (`PE_RECEIPT_MISMATCH`). Oversized
  inputs return structured token counts with HTTP 400. Requests are capped at
  8 MiB, 128 inputs, and 262,144 total tokens; the per-input serving ceiling also
  applies. Backend failures never return partial vectors.

Cancellation drains admitted inference before releasing ownership. Shutdown and
idle cleanup cannot unload between inputs or before the final receipt check.
Authentication/credential discovery and installation activation are subsequent
integration work. This factory is not a deployment guide.

## Validation and remaining work

CPU tests use synthetic artifact files, a real local Qwen2 fast tokenizer, and a
tiny deterministic model substituted only at weight acquisition. They exercise
registered loading, exact prompt-inclusive token counting, raw forward/pooling,
strict HTTP parsing, receipt admission, artifact/runtime drift, reloads,
independent copies, request limits, cancellation, and shutdown.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python -m pytest services/tests/test_verified_jina_provider.py \
  services/tests/test_embedding_identity.py -v
```

The installed tokenizer was loaded read-only with offline, standard-library
tokenizer loading; its runtime metadata fits the bound. Transformers 4.57.6
emitted a regex warning during that check. Its local detection also considers
configuration metadata, so this warning alone does not establish a Jina tokenizer
defect. Record and resolve it with tokenizer/quality fixtures during real-model
qualification; no regex correction was applied in that initial slice. No installed
weights were registered, copied, loaded, or used for inference in that slice.

The opt-in [real-model qualification protocol](verified-jina-qualification.md)
now covers fixed retrieval/tokenizer checks, performance observations, reloads,
and the actual HTTP/Rust/disposable-database flow. Internal profile persistence and
guarded document ingest/query commits and conventional token-aware planning are
implemented. General quality evaluation, late/code capabilities, trusted provider
discovery, public entry-point enforcement, legacy policy, and frontend creation
remain pending. Database pairing stays disabled until the applicable public entry
points enforce it together.
