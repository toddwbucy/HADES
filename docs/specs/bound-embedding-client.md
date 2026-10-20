# Profile-bound embedding client

This slice of [#225](https://github.com/toddwbucy/HADES/issues/225) connects the
[portable profile/receipt contracts](embedding-identity-contract.md) to the Rust
HTTP client. `persephone::bound_embedding::BoundEmbeddingClient` is an explicit,
opt-in library boundary. No CLI, MCP, background job, or frontend operation uses
it yet. The installed Jina HTTP service does not issue full provider receipts;
its responses will be rejected by this client.

## Construct a fixed binding

`BoundEmbeddingClient::connect` accepts a profile, artifact manifest, workload,
role, `ProviderConnection`, and observed `ProviderReceipt`. It validates the ready
profile and enabled workload, compares artifact and behavior-space identities,
and requires the resolved provider ID to match the profile. Construction is lazy:
it neither probes the endpoint nor loads a model.

The caller must obtain the connection and observed receipt from a trusted
provider resolver. Supplying matching strings does not establish artifact
provenance or authenticate an endpoint. Registry lookup, credential resolution,
and provider discovery are not implemented by this factory.

The model, vector width, task, profile revision, corpus generation, and expected
receipt come from the binding. Operational input supplies only provider ID,
endpoint, positive request timeout, and a positive response-byte limit. No Jina
defaults, alternate endpoint, arbitrary task override, or unbound inner client
are exposed. Cloning retains the original pin. Changing the caller's profile
object does not retarget existing clients.

## Receipt wire contract

Every `POST /v1/embeddings` request made by the bound client includes
`expected_receipt`, with exactly this shape:

```json
{
  "format": "hades-provider-receipt-v1",
  "artifact_md5": "36ad00d2e76767b48b3a55053472dcf5",
  "space_id": "ec03fc217ec93d932741ef76c3bc39b91fbf35176c3e563177749447950db6c4",
  "serving_generation": "load-fixture-1"
}
```

These values are from the shared synthetic fixture, not an installed model.
A successful response must contain an identical `receipt` object and the exact
profile model ID. The bound path does not accept installation-path aliases.
Missing/null receipts, unknown formats or fields, malformed digests, duplicate
receipt/model identity keys, or changed artifact, space, or serving generation
fail before vectors are returned. Existing vector shape, finite-value, index,
row-count, and late-boundary response checks still apply. HTTP responses are
bounded before parsing, including error responses.

The same pin is sent for every batch, OOM split/retry, and late-chunked input.
Receipt failure discards the entire operation's accumulated result. It does not
refresh identity, retry against another provider, or accept a newer generation.
An HTTP 409 provider refusal is returned to the caller. A new load requires a
fresh trusted observation and explicit client construction.

The future provider implementation must compare `expected_receipt` against its
actual loaded artifact/behavior/generation before inference, hold that identity
through the complete operation, and recheck before returning its own `receipt`.
Echoing the request is insufficient. This PR implements the consuming client;
provider admission enforcement remains pending.

## Request policy and persistence boundary

Queries use ordinary embeddings with the profile's query task, including queries
against late-chunked corpora. Conventional passage workloads use `embed`; late
passage workloads use `embed_late_chunked`. Unsupported mode/role combinations
and empty work are rejected before dispatch. Late requests require nonempty
character ranges within every input; validation covers the entire input list
before sending the first request.

Results carry the captured `BoundEmbeddingRequest` alongside vectors. This is
not permission to write to a database. The orchestrator must retain database
identity and recheck profile revision, corpus generation, and job ownership
before committing. This client does not tokenize, enforce token budgets, choose
chunk boundaries, or manage caches. Those policies, Jina behavior-space derivation,
provider receipt production, profile persistence, and enforcement at every public
entry point remain prerequisites for enabling database pairing. Existing clients
retain their previous wire behavior.

## Validation

The synthetic provider uses private temporary Unix sockets and the shared
384-dimensional profile fixture; no model, GPU, or database is required:

```sh
cargo test --locked -p hades-core --test bound_embedding_contract \
  --test embedding_contract --test embedding_client
```

CI includes the new target in its service-free contract gate. Tests cover explicit
profile resolution, asymmetric tasks, same-width incompatible spaces, malformed
and duplicate identity fields, batch/reload rejection, OOM retry pins, late-mode
policy, independent bindings/clones, response limits, and provider refusal.
