# Provider contracts and enforced limits

Source cut: `66ebc4d6c6c88c41d5e5c5ba3371d3521b88aa10`, September 20, 2026.
This complements [transport contracts](api-boundaries.md). Source hashes and the
focused CPU replay are retained in [provider results](provider-contract-result.json).
No production provider was called for this review and no model/GPU was loaded.

## Protocol versus implementation

| Surface | Implemented contract | Status behavior and limits |
|---|---|---|
| Extraction gRPC | `Extract`, `Capabilities`; Python `extraction/server.py`, Rust `persephone/extraction.rs` | Closing rejects `Extract` with `UNAVAILABLE`. A returned extraction error sets `INTERNAL`; unexpected exceptions are not normalized by that branch. Rust channel defaults: 600-second request timeout, ten-second connection timeout. |
| Embedding HTTP | `GET /v1/models`, `POST /v1/embeddings`; Python `embedding/http_server.py`, Rust `persephone/embedding.rs` | Explicit validation/worker `ValueError`: 400; closing admission: 503; other caught worker exceptions: 500. Request-model validation is handled by FastAPI/Pydantic. Rust default request budget is 300 seconds, connection configuration ten seconds; one request deadline spans response acquisition and body collection. |
| Embedding gRPC declaration | `Embed`, `Info` in `proto/persephone/embedding/embedding.proto` | Generated bindings declare a different interface. They do not prove the current Python HTTP entry point serves these RPCs. Do not infer a listening embedding Unix RPC service from proto comments. |
| Training gRPC | Three session-lifecycle and seven state-bearing RPCs; Python `training/session.py` wraps `training/server.py` | Closing: `UNAVAILABLE`; non-owner/missing/expired session: `FAILED_PRECONDITION`; competing acquisition: `RESOURCE_EXHAUSTED`. Handler validation adds `INVALID_ARGUMENT`, `NOT_FOUND`, `FAILED_PRECONDITION`. Rust requests carry ownership metadata and per-call deadlines (60 seconds ordinary, 600 seconds slow, ten seconds connection). |

These are distinct error vocabularies. A client-side deadline does not establish
worker preemption or rollback. Worker reports document draining and the possibility
that cancelled admitted work has already changed state or published an artifact.

## Request and model identity

Embedding accepts a string or list of strings and requires a `model` string, but
explicitly does not compare it with the served model: configured aliases differ
from the local model path. The response names the actual backend's configured
model. The accepted string must not be treated as verified weight identity.
`/v1/models` reports configuration and capabilities, not an inference health test.
`usage` currently returns zero counters, not measured token consumption.

The handler rejects an empty input list, non-float encoding, nonempty images,
unknown tasks, negative batch size and shared explicit late-chunk boundaries
across multiple inputs. Batch size zero means use the configured default.
An empty string is distinct from an empty list. Pydantic enforces positive chunk
size and nonnegative overlap, but the request schema has no maximum text length,
input count or batch-size upper bound. Backend token limits are not a bound on
HTTP JSON buffering, queued requests or aggregate process memory.

Extraction accepts uploaded bytes or a local file path; uploaded bytes take
precedence and are staged in an owned temporary file. Source type is explicit or
extension-derived. Code/Markdown/text use plain-text extraction, not the
Tree-sitter implementation suggested by the proto's introductory comments.
The text path calls `read_text`; no application byte ceiling is imposed there.
Local file reads rely on the provider process's filesystem authority. MCP ingest
root policy does not automatically apply to direct extraction RPC callers.

## Transport, memory and concurrency boundaries

Both Python gRPC entry points bind insecure Unix sockets and set socket modes for
local access; this is filesystem access control, not TLS or per-RPC user identity.
Training ownership tokens separate lifecycles but do not establish operator roles.
Training explicitly configures 512 MiB send and receive message ceilings.
Extraction constructs `grpc_aio.server()` without explicit message-size options;
the private Python default-message probe below measures selected cases, but not
the deployed library or Rust client defaults. Neither
setting bounds files referenced by path or all decoded/working memory.

Embedding starts one Uvicorn worker. Its entry point does not configure a request
body ceiling or concurrency limit; owned executor operations protect resource
lifetimes, not a bounded admission queue. Extraction similarly counts/owns admitted
operations without a configured admission maximum. Training serializes ownership
and state-bearing work; queued callers are not a fixed global memory budget.

The Rust embedding client supports `with_response_limit(bytes)` but initializes
that option to `None`; collection falls back to `usize::MAX`. Selected bounded
search paths opt in. Therefore those search tests do not establish a universal
embedding-response cap for every caller. The Rust extraction/training connection
paths inspected do not explicitly override generated client message-size limits;
server settings alone do not establish client/server capacity parity.

These are source-derived resource and compatibility limitations, not an injected
OOM, demonstrated unauthorized read or observed production outage. Remaining
verification needs bounded private oversized-body/message fixtures, call-site
response-limit mapping, filesystem trust review, and explicit admission budgets.

## Executed evidence

A focused replay generated protocol stubs only in the isolated worktree and ran:

```sh
python -m pytest -p no:cacheprovider \
  services/tests/test_embedding_ownership.py \
  services/tests/test_extraction_ownership.py \
  services/tests/test_training_rpc_validation.py \
  services/tests/test_training_sessions.py -q
```

All **78 tests passed in 7.52 seconds**, with the private CPU interpreter,
bytecode/cache disabled, one OpenMP/MKL thread, no visible CUDA devices and an
external 120-second watchdog. Embedding uses the real HTTP framework with a fake
model backend; extraction uses synthetic text and private RPC fixtures; training
uses CPU fixtures. This validates the named status/ownership paths, not every
malformed request, default gRPC size, real model or GPU behavior. Source limits
above were read; the small additional probes below are not exhaustion stress tests. See the individual worker, session and artifact
reports for cancellation/publication boundaries and historical reproductions.


## Bounded request/message probes

Two retained scripts run against the same application source with fake backends.
Both completed successfully under 30-second watchdogs; the extraction script
starts and stops only its own temporary Unix-socket server. Results retain source
and probe hashes. No model/GPU or live endpoint is involved.

- [Embedding ASGI probe](repros/embedding_request_boundaries.py) and
  [result](embedding-request-boundary-result.json): missing model and numeric
  input returned 422, empty input list returned 400, all without backend calls.
  A different model alias returned 200 with the fixture backend's own model name.
  One 4 MiB synthetic string and a one-million batch-size hint each reached the
  fake backend and returned 200. These are accepted samples, not measured maxima;
  no inference or large-batch allocation was performed.
- [Extraction gRPC probe](repros/extraction_message_boundaries.py) and
  [result](extraction-message-boundary-result.json): with private `grpcio 1.84.0`
  server/channel defaults, a 1 MiB upload succeeded. An upload of 4 MiB + 1024
  bytes returned `RESOURCE_EXHAUSTED` without calling the extraction route.
  A synthetic response of the same size returned `RESOURCE_EXHAUSTED` after the
  route ran. Thus inbound rejection and client-side response rejection protect
  different stages; message caps do not prevent all work/allocation beforehand.

The first extraction attempt used the wrong keyword for the fake `ExtractionResult`
(`full_text` instead of `text`) and returned `UNKNOWN`. The fixture was corrected;
that failed setup is not a provider defect. Final successful results above are
from the corrected probe. These three samples do not locate exact protobuf-byte
thresholds or verify Tonic/Rust defaults. The deployed Python metadata records a
different grpcio version, so the private result is not asserted as its effective
limit. Remaining admission/body/response-cap and trust-boundary review stays open.
