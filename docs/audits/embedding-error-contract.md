# Embedding API error contract

Part of epic #12; fixes #153. Requirements authority is the owner's `/opt/HADES/docs/`, specifically `persephone-embedding-api.md` v1.1 at `a71d73b`. Bastion is excluded. This report covers the error-response requirement group, not complete embedding conformance.

## Baseline

At source `67a8fb2`, actual in-process ASGI requests to the FastAPI application returned framework `detail` responses for all six tested failures: invalid task (400), unsupported encoding (400), unsupported images (400), empty input (400), invalid request shape (422), and injected inference failure (500). None had the required `error` object with `message`, `type`, `param` and `code`. Invalid task lacked `PE_INVALID_TASK`; encoding/image codes were embedded only in prose.

Baseline commit `ede0f5e` preserves the failing test and `embedding-error-contract-baseline.json`, including specification and source hashes. The first restricted-environment run was deliberately interrupted while waiting in the event loop and is not a validation result. The same private fixture run outside that restriction terminated with the expected assertion failure in 0.41 seconds. No production incident is asserted.

## Correction

Intentional PE errors carry a status, stable code, public-safe message and parameter name. HTTP/framework and request-validation handlers produce the same envelope. Validation responses omit Pydantic input values. An unexpected-error handler supplies a generic 500 rather than exposing arbitrary exceptions.

Invalid task, encoding and image requests use the specified codes. Typed input-length failures retain the input index, token count and ceiling; the existing late-chunk saturation guard is classified without reflecting its full message. PyTorch's `OutOfMemoryError` is exported by the backend for type-based classification as 503/`PE_BACKEND_OOM`. Other input failures remain 400; other backend failures remain 500. Shutdown admission remains 503 with `PE_SERVICE_CLOSING`. Additional codes describe cases not named by the current spec. Missing model fields remain HTTP422 with the specified 4xx error type; the spec does not mandate a different status for schema errors.

Inference ownership/cancellation logic is unchanged. Plain and late success shapes retain their existing behavior. The Rust consumer already treats non-success HTTP statuses as errors and retains the response body; this change does not require its error parser to recognize a new success shape.

## Verification and limits

All 29 embedding ownership/ASGI tests pass (2.71 seconds). Coverage includes successful plain/late requests; six original failures; both embedding paths under typed oversized input, legacy saturation, OOM, value and runtime exceptions; sensitive-detail non-reflection; invalid-schema and missing-route envelopes; unexpected response-construction failure; shutdown rejection; existing cancellation, drain and unload behavior.

Only model inference is stubbed. The app, request models, exception handlers, worker ownership and ASGI serialization are real. OOM injection proves classification, not actual GPU exhaustion or recovery. No real model, GPU, production endpoint or database was used. Request-timeout/model warm-up policy, full late-boundary semantics, model-alias policy and the spec's deferred cohort/version ambiguity remain separate conformance work. Existing model-alias behavior is deliberately preserved because the specification explicitly records that exception.

Server-side diagnostic logging remains; sanitization here concerns HTTP responses, not a certification of log redaction. Production deployment, upgrade and rollback require separate coordination. Full epic acceptance remains open.
