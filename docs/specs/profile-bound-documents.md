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

`prepare(DocumentInput, replace)` accepts already-extracted text and a stable
source identifier. The bound provider plans conventional passages with its actual
serving tokenizer, including prompt/special-token overhead and measured overlap.
Rust validates the complete plan before slicing or inference. See
[token-aware planning](profile-token-planning.md) for policy semantics, limits,
strict wire fields, and original UTF-8 byte coordinates.

Limits are 2 MiB of source text, 128 passages, 4 MiB of aggregate passage text,
and 16 MiB of serialized prepared records. Source identifiers are hashed into
safe keys; their original spelling is retained without normalization.

The stored profile is checked before planning and again before inference. For
replacement, the old source's metadata revision is pinned first. Planning and
all embedding batches retain the expected artifact MD5, embedding-space ID, and
serving-load receipt. Model, width, finite nonzero values, and declared L2
normalization are validated. Any failure discards the whole preparation, which
performs no database writes. Extraction integration and late-chunked context
window planning remain follow-up work.

## Records and publication

All three collections use the internal `hades-bound-document-v2` record format:

| Collection | Stored information |
| --- | --- |
| `documents` | Original source identifier, full extracted text, source SHA-256, passage plan, chunk count, complete passage binding |
| `chunks` | Source and chunk keys, ordinal, source digest, passage text and counted byte span, complete passage binding |
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
The previous internal v1 format had unproven caller-supplied spans; it is not
automatically adopted as token-checked v2 data.
Unrelated sources are preserved. Write errors and cancellation before commit
abort the transaction; a lost commit response remains an unknown outcome that
must be reconciled by reading durable state. Automatic retries, adoption, source
removal, and migrations are unavailable.

## Search and compatibility

`search(text, top_k, max_rows)` embeds the query with the profile's query task,
then acquires the same guarded transaction for a coherent metadata/data view.
Eight-row pages validate each vector's binding, source/chunk references, model,
width, and normalization **before scoring it**. Source plan metadata, the original
planning receipt, effective token ceiling, and agreement between each chunk and
its plan are checked too. Provenance is not an AQL filter: an incompatible or
legacy row fails the whole search.

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
mid-planning/inference profile drift, rejected plans before inference, stored token
evidence corruption, compatible reloads, cross-database isolation,
incompatible stored provenance, scan limits, disabled/unsupported policy,
unknown-record preservation, and inference cancellation.

This closes the synthetic document round trip in stage 1 of the
[implementation plan](database-embedding-profiles.md). Real-Jina qualification and
conventional token-aware planning now have dedicated protocols. All-entry-point
enforcement, portable job pins, recoverable provisioning, and frontend setup
remain required. No public pairing feature should be enabled until those applicable paths share the same admission rules.
