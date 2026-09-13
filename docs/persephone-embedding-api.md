# Persephone Embedding API Specification

**Version:** 1.1
**Status:** Implemented
**Path prefix:** `/v1`

> **Amended in 1.1.** Version 1.0 specified late chunking as the only
> behaviour, with the backend choosing chunk boundaries from a token size and
> overlap, and returned a nested `chunk` object per vector. The implementation
> diverged on all three points and the spec is amended to match it rather than
> the reverse. What changed and why is in "Amendments in 1.1" at the end.


## Overview

The Persephone Embedding API is HADES's contract for embedding service backends. It follows OpenAI's `/v1/embeddings` request shape and extends it. By default a request returns one vector per input, which is what an OpenAI client expects. A request carrying a `late_chunk` object instead returns *N* pooled vectors per input, one per chunk, each conditioned on the whole surrounding document.

HADES is **engine-agnostic** at this contract level — any backend that implements PE-API is a valid backend (the FastAPI service in this repo, a future Rust-native loader, a `hades-weaver-bridge` translator, an external service). HADES is **model-bound** at the data layer: implementations must serve a model with Jina V4's capability profile (2048-dim, 32k context, multimodal, task-conditional via LoRA, late-chunking-capable). Wrong model class → silently incompatible vector geometry.

### Why diverge from OpenAI

OpenAI's `/v1/embeddings` returns one vector per input. HADES depends on **late chunking**, meaning full-document encoding followed by chunk-aware pooling, for retrieval quality on documents above roughly 500 tokens. That returns *N* vectors per input rather than one, so the response shape cannot match OpenAI's whenever late chunking is in use.

The divergence is confined to requests that ask for it. A request with no `late_chunk` field returns a response an OpenAI client reads correctly, which keeps short queries, probes, and third-party tooling working against the same endpoint. A request with `late_chunk` returns the extended shape, and a client that does not understand it will misread the response, so the field is opt-in on the client's side rather than a mode the server chooses.

## Endpoints

### `GET /v1/models`

Discover the model the backend has loaded. OpenAI-compatible response shape.

**Request:** none.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "jinaai/jina-embeddings-v4",
      "object": "model",
      "created": 0,
      "owned_by": "hades",

      "dimension": 2048,
      "max_seq_length": 32768,
      "supported_tasks": [
        "retrieval.passage",
        "retrieval.query",
        "text-matching",
        "code"
      ]
    }
  ]
}
```

`dimension`, `max_seq_length`, and `supported_tasks` are **PE-API extensions** to OpenAI's model object.

- **`supported_tasks` is REQUIRED.** Conforming backends MUST include it, accurately reflecting the tasks they can serve (see Implementation requirements). Clients use this field to validate `task` values before sending requests, avoiding unnecessary round-trips that would only fail with `PE_INVALID_TASK`.
- **`dimension` and `max_seq_length` are OPTIONAL.** Backends MAY omit them; clients that need either value can determine it via an embedding round-trip (`/v1/embeddings` with a probe input).

### `POST /v1/embeddings`

Primary embedding endpoint. Returns one vector per input by default, and *N* pooled vectors per input when `late_chunk` is supplied.

**Request:**

```jsonc
{
  "model": "jinaai/jina-embeddings-v4",       // OpenAI standard
  "input": "long document text...",           // OpenAI standard: string OR array of strings
  "encoding_format": "float",                 // OpenAI standard; only "float" supported

  // PE-API extensions:
  "task": "retrieval.passage",                // LoRA adapter selector (see "Tasks" below)
  "images": ["base64-encoded image", ...],    // multimodal inputs paired with `input` strings
  "batch_size": 8,                            // server-side batch size override

  // Opt in to late chunking. Omit for one vector per input.
  "late_chunk": {
    "boundaries": [[0, 487], [430, 1022]],    // character ranges to pool at
    "chunk_size_tokens": 500,                 // used only when `boundaries` is omitted
    "overlap_tokens": 200                     // used only when `boundaries` is omitted
  }
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | none | Model identifier. **Not validated by the reference implementation**, see below |
| `input` | string \| string[] | yes | none | Text(s) to embed |
| `encoding_format` | string | no | `"float"` | Reserved for future encoding variants |
| `task` | string | no | `"retrieval.passage"` | Jina V4 LoRA adapter, see Tasks |
| `images` | string[] | no | `[]` | Per-input base64-encoded images for multimodal embedding. When provided, `len(images)` must equal the number of inputs, so `1` if `input` is a string and otherwise `len(input)` |
| `batch_size` | int | no | backend-configured | Server-side batch size override |
| `late_chunk` | object | no | absent | Enable late chunking. See below |

**`late_chunk` object:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `boundaries` | [int, int][] | no | `null` | Character ranges over the input to pool at, such as AST definition spans or document sections. One vector is returned per range, in the order given |
| `chunk_size_tokens` | int | no | `500` | Uniform window size, used only when `boundaries` is omitted |
| `overlap_tokens` | int | no | `200` | Uniform window overlap, used only when `boundaries` is omitted |

**`boundaries` are character offsets, not byte offsets.** They are matched against the tokenizer's offset mapping, which is character-indexed. A client working in a language whose string offsets are byte-based (Rust among them) MUST convert before sending. Passing byte offsets through produces no error and no failed request: the pooling is correct over whatever range it is handed, so every chunk in a file containing a multibyte character is silently pooled from the wrong span, with the gap widening through the file. This cost a corpus and is the single sharpest edge in this contract.

**`boundaries` describes one input.** A request supplying `boundaries` with more than one element in `input` MUST be rejected with 400. Ranges valid for input 0 are meaningless against input 1, and applying them anyway would pool from the wrong document without erroring whenever the later inputs are long enough to contain the offsets.

**Boundaries that map to no tokens MUST be rejected**, with 400, rather than skipped. A boundary maps to nothing when the input was truncated at the context window, so the caller sent more text than the model can hold. Dropping those quietly returns fewer vectors than boundaries were sent, and the caller has no way to notice, so chunks are stored with no embedding and are unsearchable in a corpus that otherwise looks whole.

**Response:**

```jsonc
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,                             // index into request `input` array
      "embedding": [0.0123, -0.0456, ...],    // length = model dimension (2048 for Jina V4)

      // Present only when the request carried `late_chunk`.
      "chunk_index": 0,                       // chunk position within input[index]
      "token_start": 0,                       // token range, relative to the document
      "token_end": 195,
      "char_start": 0,                        // character range over input[index]
      "char_end": 487
    },
    {
      "object": "embedding",
      "index": 0,                             // same input, next chunk
      "embedding": [...],
      "chunk_index": 1,
      "token_start": 172,
      "token_end": 404,
      "char_start": 430,
      "char_end": 1022
    }
    // ...
  ],
  "model": "jinaai/jina-embeddings-v4",
  "usage": {
    "prompt_tokens": 0,
    "total_tokens": 0
  }
}
```

| Field | Type | Present | Description |
|-------|------|---------|-------------|
| `index` | int | always | Position in the request's `input` array |
| `embedding` | float[] | always | The vector, `dimension` long |
| `chunk_index` | int | late chunking only | Zero-based position of this chunk within `input[index]`, contiguous from zero |
| `token_start` / `token_end` | int | late chunking only | Token range, relative to the document rather than to any prompt prefix the backend added |
| `char_start` / `char_end` | int | late chunking only | Character range over `input[index]`, token-aligned |

**Chunk metadata fields are required when late chunking was requested.** A backend that does not implement `late_chunk` MUST reject the request rather than ignore the field and return plain embeddings. A client cannot distinguish "one vector because the server pooled to one chunk" from "one vector because the server ignored the request", and defaulting the missing `chunk_index` to 0 collapses every vector of an input onto its first chunk.

**The returned range is token-aligned and so is wider than the one requested, at both ends.** `char_start` falls at the start of the first token overlapping the boundary, so it is at or before the requested start. `char_end` falls at the end of the last overlapping token, so it is at or after the requested end.

Consequently **returned ranges from adjacent boundaries can overlap**, even when the boundaries sent did not. Requesting `[[0,25],[27,52],[54,80]]` over an 80-character input returns `[0,27]`, `[27,54]`, `[54,80]`. A client intersecting these with symbol spans, which is the stated reason character ranges are returned at all, will see two chunks claiming the characters between them. Intersect against the boundaries you sent, and treat the returned range as the span the vector was pooled from.

Clients SHOULD compare the returned `char_start` against the boundary they sent. A `char_start` LATER than the requested start cannot come from alignment and means the vector carries a different chunk's range.

**Critical shape difference from OpenAI:** under late chunking, `data` contains one entry per chunk rather than one per input. Each entry's `index` references its position in the request's `input` array, so several entries share an `index` when they came from the same input. Use `chunk_index` for position within the input.

**Counts are a contract.** When `boundaries` is supplied, the number of returned entries for an input MUST equal the number of boundaries sent, and `chunk_index` MUST run contiguously from zero. Clients SHOULD verify both. Three separate defects in this pipeline have taken the shape of returning fewer vectors than chunks and reporting success, and a count check catches all three.

**Usage counts.** `usage.prompt_tokens` and `usage.total_tokens` are reported as `0`. The field is present for OpenAI shape compatibility and is not yet populated.

## Tasks (Jina V4 LoRA adapter routing)

The `task` field selects which LoRA adapter Jina V4 uses to produce embeddings. Different adapters produce different geometries; **mismatched task between query and corpus drastically degrades retrieval quality.**

| Task | Adapter | Use case |
|------|---------|----------|
| `retrieval.passage` | `retrieval` | Embedding documents for retrieval (default) |
| `retrieval.query` | `retrieval` | Embedding search queries (paired with `retrieval.passage` corpus) |
| `text-matching` | `text-matching` | Symmetric text similarity (paired-document scoring) |
| `code` | `code` | Code embedding |

Backends that don't honor `task` (e.g., a backend serving Jina V4 with a single pre-baked adapter) **must reject requests with unsupported `task` values** with HTTP 400, rather than silently routing to the wrong adapter and producing geometrically incomparable vectors.

## Multimodal (text + images)

When `images` is provided, it must be the same length as `input`. Each `input[i]` is paired with `images[i]` as a multimodal input. `images[i]` is a base64-encoded image (PNG, JPEG; format detected from bytes).

A backend that does not support multimodal **must** reject requests with non-empty `images` with HTTP 400, rather than silently ignoring images and producing text-only embeddings (which would be silently lower-quality and cohort-mismatched against multimodal-corpus data).

## Error handling

Errors follow OpenAI's error envelope:

```json
{
  "error": {
    "message": "human-readable description",
    "type": "invalid_request_error" | "server_error",
    "param": "task" | "input" | null,
    "code": "PE_INVALID_TASK" | "PE_DIMENSION_MISMATCH" | ...
  }
}
```

By convention, **4xx** HTTP responses (caller's fault) use `type: "invalid_request_error"`, and **5xx** responses (server's fault) use `type: "server_error"`. Unclassified or library-specific errors default to `"server_error"`. The `code` field is PE-API-specific and provides a more granular machine-readable identifier than `type`.

PE-API-specific error codes:

| Code | HTTP | Meaning |
|------|------|---------|
| `PE_INVALID_TASK` | 400 | `task` value not in backend's `supported_tasks` |
| `PE_UNSUPPORTED_ENCODING_FORMAT` | 400 | `encoding_format` value other than `"float"` (only `"float"` is supported in v1.0) |
| `PE_MULTIMODAL_UNSUPPORTED` | 400 | `images` provided but backend serves text-only model |
| `PE_INPUT_IMAGES_LENGTH_MISMATCH` | 400 | `len(images) != input_count`, where `input_count = 1` if `input` is a string, else `len(input)` |
| `PE_INPUT_TOO_LARGE` | 400 | An input exceeds backend's max-context fallback handling |
| `PE_MODEL_NOT_LOADED` | 503 | Model is unloaded (e.g., idle-timeout); retry after warm-up |
| `PE_BACKEND_OOM` | 503 | GPU OOM on this batch; retry with smaller batch or wait |

## Cohort identity (sketched, deferred to v1.1)

A `cohort` field on the response is reserved for v1.1+. Its purpose is to enable forensic and cross-corpus queries to verify that two vector sets are geometrically comparable before running similarity operations.

```jsonc
// v1.1 response (sketched, not implemented in v1.0):
{
  // ... v1.0 fields ...
  "cohort": {
    "model": "jinaai/jina-embeddings-v4",
    "model_revision": "<HF model revision SHA>",
    "lora_adapter": "retrieval",
    "tokenizer_hash": "<hash>",
    "fa2_kernel_version": "<candle-flash-attn build identifier>",
    "precision": "bf16"
  }
}
```

Concrete shape lands once HADES's forensic / multi-cohort query work begins (see `project_weaver_embedder_cohort_pin.md`). v1.0 backends should record the equivalent metadata internally; surfacing it on the wire is v1.1.

## Implementation requirements

A conforming PE-API v1.1 backend MUST:

1. Implement `GET /v1/models` returning at least one model with the Jina V4 capability profile (2048-dim, 32k context, late-chunking-capable). The model's `supported_tasks` field is REQUIRED and MUST accurately list the tasks the backend can serve — neither over- nor under-claiming.
2. Implement `POST /v1/embeddings` returning one vector per input when `late_chunk` is absent, and one vector per chunk when it is present.
2a. Reject a request carrying `late_chunk` if the backend does not implement late chunking, rather than ignoring the field and returning plain embeddings. A client cannot tell the two apart from the response.
2b. Reject a request carrying `late_chunk.boundaries` with more than one element in `input`, with 400.
2c. Reject a request whose boundaries map to no tokens, with 400, rather than returning fewer vectors than boundaries.
3. Honor `task` for adapter selection when the value is listed in the model's `supported_tasks`. Reject `task` values not in `supported_tasks` with `PE_INVALID_TASK`. Silently routing to a different adapter is forbidden — the cost of "wrong adapter" silently degraded retrieval quality is much higher than a hard error.
4. Reject multimodal requests it cannot serve (i.e., `images` provided to a text-only backend) with `PE_MULTIMODAL_UNSUPPORTED` rather than silently producing text-only embeddings.
5. Reject `encoding_format` values other than `"float"` with `PE_UNSUPPORTED_ENCODING_FORMAT`. v1.0 supports only `"float"`; the field is reserved for future variants.
6. Return HTTP 503 with `PE_MODEL_NOT_LOADED` when the model is unloaded; do not block clients on model warm-up.

A conforming PE-API v1.1 backend SHOULD:

- Support all four standard tasks (`retrieval.passage`, `retrieval.query`, `text-matching`, `code`) where the loaded model is capable. Backends limited to a single pre-baked adapter are valid (they list a single task in `supported_tasks` and reject the others) but reduce HADES's retrieval quality and are NOT the recommended deployment shape.

A conforming backend MAY:

- Surface `device`, `dimension`, and `max_seq_length` in `/v1/models` response (these fields are optional; `supported_tasks` is required per item 1 above).
- Pre-allocate or batch internally on top of the per-request `late_chunk.chunk_size_tokens` and `late_chunk.overlap_tokens` overrides.
- Idle-unload the model after configurable timeout, returning `PE_MODEL_NOT_LOADED` until reload.

## Compatibility with OpenAI ecosystem

The OpenAI Python and TypeScript client libraries can construct PE-API requests because the request shape is a superset of OpenAI's.

**Without `late_chunk`** the response is OpenAI-shaped: one `data` entry per input, in order, with the extra fields absent rather than null. An OpenAI client reads it correctly. This is the path for queries, probes, and third-party tooling.

**With `late_chunk`** an OpenAI client will misread the response, because `data` has one entry per chunk rather than one per input and several entries share an `index`. There is no error, only silent chunk confusion, so a client that does not understand the extended shape must not send the field.

Clients consuming the late-chunked shape are expected to use a PE-API-aware client, such as the Rust client at `crates/hades-core/src/persephone/embedding.rs`.

## Reference implementation

The FastAPI service at `services/embedding/http_server.py` is the v1.1 reference implementation. The Rust client at `crates/hades-core/src/persephone/embedding.rs` is the v1.1 reference consumer. Both are normative for behavior questions not resolved by this spec.

## Future direction

- **v1.1:** add `cohort` field to response (concrete shape).
- **v1.2:** streaming responses for very-long-document embedding (reduces TTFB on >100k-token inputs).
- **v2:** consider whether the token budget should move to the server. Clients currently pre-window long documents against a hardcoded character estimate of the token ceiling, because only the server knows its own `MAX_TOKENS` and owns the tokenizer. Exposing the budget, or doing the windowing server-side, removes a constant that has already been wrong once.

The v1.0 roadmap asked whether the chunk-array shape should split onto a separate path so `/v1/embeddings` could stay strictly OpenAI-compatible. v1.1 settles it a different way: the two shapes share the path and the request selects between them, so an OpenAI client keeps working as long as it does not ask for late chunking.

## Known gaps in the reference implementation

Recorded rather than quietly tolerated, because this document names
`services/embedding/http_server.py` as normative and a backend author will
read the requirements above as descriptions of it.

**`model` is not validated.** The requirement table says it must match a model
from `GET /v1/models`. It is not checked. The Rust client sends a configured
alias (`jinaai/jina-embeddings-v4`) while the backend reports the local path it
loaded the weights from, so enforcing the match would refuse every request
HADES makes. Fixing it properly means the client reads the served name at
connect, which is a change on the other side of the wire.

Note the related hazard that this one masks: `model` is also what a client
stamps into its stored rows. If one code path records the configured alias and
another records the served name, a corpus ends up holding two identifiers for
one model, and anything keyed on that identifier, incremental skip included,
sees half the corpus. Both paths in the Rust client now take the name from the
response for that reason.

## Amendments in 1.1

v1.0 was written before the implementation. Three of its decisions did not survive contact, and this revision follows the implementation rather than the other way round.

**Late chunking is opt-in, not the only mode.** v1.0 stated there is no single-vector-per-input path. There is one, it is the default, and it is what queries and probes use. Making late chunking unconditional would have meant a query returning a chunk array of one, and would have broken OpenAI-shaped clients on every request rather than only on requests that ask for the extended shape.

**The caller supplies boundaries.** v1.0 had the backend cut uniform windows from `chunk_size_tokens` and `chunk_overlap_tokens`. The primary caller is code ingest, which already has AST definition spans and wants chunks aligned to them, and a uniform window cutting across a function boundary is the thing late chunking exists to avoid. `boundaries` is now the main input and the uniform window is the fallback for callers that have no structure to offer.

**Response fields are flat.** v1.0 nested them under a `chunk` object with `start_char` and `start_token` naming. They are flat, and named `char_start` and `token_start`. This is the least defensible of the three changes, being a rename with no behavioural argument behind it, and it is recorded here rather than quietly reverted because the implementation, its client, and one ingested corpus already use it.

**`total_chunks` and `text` were dropped.** v1.0 returned both per chunk. `text` is redundant, since the caller sent the text and has the character range back. `total_chunks` is not redundant and was worth keeping: it is what lets a client check it received every chunk it asked for. With caller-supplied boundaries the client already knows how many it sent, so the count check is available without it, and the conformance rules above now require that check. A backend using the uniform-window fallback gives its caller no such number, which is a real gap in this revision.
