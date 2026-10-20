# Embedding identity contract implementation

This is the first implementation slice of [#225](https://github.com/toddwbucy/HADES/issues/225),
following the [approved audit/design](database-embedding-profiles.md). It supplies
portable Rust profile/space/receipt types and Rust/Python artifact primitives.
No CLI, MCP, worker, HTTP route, or database operation activates these contracts
yet. Existing databases retain their current behavior; no historical vectors
are labeled verified. Review this foundation before adding the serving and
persistence paths from the first planned implementation stage.

## Artifact manifests

`hades_core::embedding_profile::ArtifactManifest` and
`embedding.identity.ArtifactManifest` implement `hades-model-manifest-v1`.
Both reproduce the [shared vectors](database-embedding-manifest-vectors.json)
byte for byte. `canonical_bytes()` sorts relative POSIX paths by UTF-8 bytes
and object keys lexically, preserves Unicode scalar sequences, and emits compact
UTF-8 JSON. `digest()` computes the aggregate MD5 of those bytes. Use these
methods, rather than ordinary struct serialization, to persist canonical data.

Each file has exactly `path`, `bytes` (an unsigned 64-bit integer), and lowercase
`md5`. Empty asset sets, duplicates, traversal, file/directory conflicts, invalid
UTF-8, unknown fields/versions, and malformed digests are rejected. Limits are
4,096 assets, 4,096 UTF-8 bytes per path, and 1 MiB per serialized manifest.
Literal backslashes are POSIX filename characters, not directory separators.
Readers also bound incoming JSON before parsing it.

`ArtifactFile::from_reader` / `ArtifactFile.from_stream` hash an already opened
asset with a 64 KiB buffer. They neither discover required assets nor protect a
file from concurrent mutation. The future registrar must select every required
weight, configuration, tokenizer, adapter, and model-local code dependency and
prepare an immutable snapshot. A checksum comparison establishes consistency
with a trusted registration, not source authenticity or loaded-model proof.

## Profiles and embedding spaces

The Rust `EmbeddingProfile` is bounded to 64 KiB and rejects unknown fields and
versions. Its model references a complete manifest and an explicit tokenizer
asset subset. Tokenizer identity uses the same manifest format over that subset.
`null` workloads are disabled independently of advertised capabilities.
Enabled document/code workloads require text support and declared tasks. Late
chunking requires both late-chunk and tokenizer-offset capabilities. Specialized
code is an explicit claim; ordinary text code processing does not imply it.

Each enabled workload contains a complete `EmbeddingSpace`: artifact/tokenizer
digests, tokenizer contract, representation, and resolved passage/query behavior.
`space.id()` computes SHA-256 over its canonical JSON, using the manifest's JSON
encoding rules. This behavior digest complements the model artifact MD5. Exact
prompts, adapters, special-token rules, pooling, normalization, precision, and
implementation version affect it. An asymmetric query/passage pair shares one
declared space. Endpoint, provider ID, device name, and corpus generation do not
enter this hash. The supported dense vector width is currently 1–8,192, matching
the retrieval ceiling.

The [synthetic profile/receipt vector](../../crates/hades-core/tests/fixtures/embedding_profile.json)
fixes the wire shapes and expected behavior digest for a 384-dimensional text
fixture. It is not a registered or downloadable model. Representation and
tokenizer contract labels must eventually be supplied by adapter implementations;
syntax validation alone cannot certify their meaning. Token budgets are validated
for positive sizes, context ceilings, and overlap; actual prompt-inclusive token
counting remains the adapter's responsibility.

## Binding and provider receipts

`profile.bind(manifest, workload, role, observed_receipt)` requires a structurally
valid ready profile, an enabled workload, and matching artifact/space identities.
It returns the explicit model, dimension, task, profile revision, corpus generation,
provider ID, and expected receipt. There are no Jina defaults in this resolution.
The result is a planning value, not permission to store data. The opt-in
[bound client](bound-embedding-client.md) now consumes it for receipt-enforced
HTTP requests; production call sites do not use that client yet.

`hades-provider-receipt-v1` has exactly `format`, `artifact_md5`, `space_id`, and
`serving_generation`. Rust and Python implement the same shape. A future verified
loader must issue a fresh generation for every load/reload, including reloads of
identical bytes. `check_matches` rejects changed artifacts, behavior, or generation;
call it at provider admission and again before accepting response vectors. It
cannot detect a dishonest provider repeating an old receipt.

## Remaining integration boundary

The next slice implements [local registration and checked Jina loading](registered-model-loading.md).
It supplies artifact provenance internally; behavior-space receipts and database
enforcement remain separate integration work.

The [bound client](bound-embedding-client.md) implements the explicit client
factory and consuming side of receipt transport. The next dependent changes
must connect the checked loader to resolved behavior spaces and provider-side
receipt enforcement, isolated
database/profile persistence, and a synthetic ingest/query path. Before
activating database pairing, all applicable public entry points must enforce it,
including background jobs and generation checks before commit. Profile CAS,
provisioning recovery, cache isolation, legacy handling, and frontend creation
remain as specified in the audit. No feature switch is exposed by this slice.

## Validation

Run without a model, GPU, or database:

```sh
cargo test --locked -p hades-core --lib embedding_profile::
python3 -m unittest discover -s services/tests -p test_embedding_identity.py -v
```

The Python suite is also collected by the existing service pytest job; the Rust
suite runs in the existing workspace library test gate. Coverage includes shared
canonical vectors, relocation, every required asset changing identity, strict
parsing, capability/budget rejection, asymmetric tasks, same-width incompatible
spaces, and provider reload rejection. Filesystem fixtures use temporary directories.
