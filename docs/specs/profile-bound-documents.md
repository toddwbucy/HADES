# Profile-bound document ingestion and search

Status: internal Rust integration for [#225](https://github.com/toddwbucy/HADES/issues/225).
`pipeline::profiled_documents::ProfiledDocuments` connects the persisted profile,
verified embedding client, atomic document storage, and exact cosine retrieval.
It has no CLI, MCP, job, or frontend callers. Production ingestion and the running
installation continue to use their existing paths.

## Admission and preparation

A trusted caller supplies the explicit database pool, registered provider
connection, and observed provider receipt. `connect` loads the authoritative
profile from the writer endpoint and derives separate passage/query clients.
Documents must be enabled, chunking must be conventional, and the distance must
be cosine. The three document collections must already exist. This operation
does not provision a database or establish provider trust.

`prepare(DocumentInput, replace)` accepts already-extracted text, a stable source
identifier, and ordered UTF-8 byte spans with exclusive ends. Spans must cover the
whole source without gaps; overlap is allowed, but duplicate/contained spans are
rejected. Limits are 2 MiB of source text, 128 passages, 4 MiB of aggregate passage
text, and 16 MiB of serialized prepared records. Source identifiers are hashed
into safe keys; their original spelling is retained without normalization.

The stored profile is checked before inference. For replacement, the old source's
metadata revision is pinned first. Every provider batch must return the expected
artifact MD5, embedding-space ID, and serving-load receipt. Model, width, finite
nonzero values, and declared L2 normalization are validated. Any batch failure
discards the whole preparation. Preparation performs no database writes.

The supplied spans are recorded as `hades-supplied-byte-spans-v1`. This slice does
**not** establish that those spans satisfy the profile's token-size/overlap policy.
The trusted planner must provide that evidence before activation. Tokenizer-exact
planning, provider-budget discovery, extraction integration, and long-document
window planning remain follow-up work. There is no truncation, rechunking, or
fallback when the provider rejects a passage.

## Records and publication

All three collections use the internal `hades-bound-document-v1` record format:

| Collection | Stored information |
| --- | --- |
| `documents` | Original source identifier, full extracted text, source SHA-256, passage plan, chunk count, complete passage binding |
| `chunks` | Source and chunk keys, ordinal, source digest, passage text and byte span, complete passage binding |
| `embeddings` | The same source/chunk provenance, model, dimension, vector, complete passage binding |

The binding includes profile revision, corpus generation, provider ID, passage
task, model, dimension, artifact MD5, embedding-space ID, and observed serving
generation. Source digests identify text, not model artifacts. Chunk keys include
the corpus generation and source key; this is not an implementation of model
migration or historical anchor retention.

`PreparedDocument` has private fields. `commit` accepts it only for its original
database/profile/provider admission. One guarded transaction locks workspace
metadata and all three data collections, rechecks the profile and source revision,
validates any existing managed records, and publishes the complete replacement.
Both bulk and individual write acknowledgements are checked.

Replacement requires explicit opt-in. Changed sources, orphan/foreign children,
unknown fields that replacement would discard, and legacy records are refused.
Unrelated sources are preserved. Write errors and cancellation before commit
abort the transaction; a lost commit response remains an unknown outcome that
must be reconciled by reading durable state. Automatic retries, adoption, source
removal, and migrations are unavailable.

## Search and compatibility

`search(text, top_k, max_rows)` embeds the query with the profile's query task,
then acquires the same guarded transaction for a coherent metadata/data view.
Eight-row pages validate each vector's binding, source/chunk references, model,
width, and normalization **before scoring it**. Provenance is not an AQL filter:
an incompatible or legacy row fails the whole search.

Exact cosine ranking retains at most K keys/scores. K is limited to 1–1000;
the explicit scan budget is 1–100,000 rows. Exceeding it discards all results.
Responses are bounded to 4 MiB per search page, with a five-second AQL runtime,
32 MiB server query-memory budget, and the transaction owner's 60-second deadline.
The initial exclusive scan favors correctness; it is not a concurrent or indexed
production search design.

A freshly verified provider load may query vectors from an older load only when
the artifact, space, profile revision, corpus generation, provider, model, task,
and width still agree. Existing clients remain pinned to their original load and
reject a changed receipt. The returned binding describes the query observation;
persisted records retain their original passage observation.

Database administrators can still corrupt data directly. These checks detect
incompatible identity and broken row references; they are not cryptographic
attestation of stored vector values or a repair tool.

## Validation and activation gate

The `profiled_documents` contract runs in the strict isolated database matrix:

```bash
cargo fetch --locked
python3 scripts/test_isolated_database.py --docker --contract profiled_documents
# Existing binary, private Unix-only server:
python3 scripts/test_isolated_database.py --arangod /path/to/arangod \
  --contract profiled_documents
```

Tests use temporary databases and private HTTP peers with synthetic two- and
three-dimensional vectors. They cover complete ingestion/search, asymmetric
tasks, replacement conflicts and rollback, provider failures across batches,
mid-inference profile drift, compatible reloads, cross-database isolation,
incompatible stored provenance, scan limits, disabled/unsupported policy,
unknown-record preservation, and inference cancellation.

This closes the synthetic document round trip in stage 1 of the
[implementation plan](database-embedding-profiles.md). Real-Jina qualification,
token-aware planning, all-entry-point enforcement, portable job pins, recoverable
provisioning, and frontend setup remain required. No public pairing feature
should be enabled until those applicable paths share the same admission rules.
