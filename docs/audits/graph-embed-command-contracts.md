# Graph embedding command contracts

Epic #12 source review at `7bd9f64a3a491ace6798294427d9b70aa5a224a8`.
Four leaves are traced through declarations, main routing, adapters and their
shared query/export/session boundaries. This is not full runtime certification.

## Routes and observable behavior

| Leaf | Native route | Contract |
|---|---|---|
| `graph-embed embed` | `graph_embed_query::run_embed` → dispatch | Reads an existing structural vector; does not generate one despite the CLI help wording. Emits `graph-embed.query`. |
| `graph-embed neighbors` | `graph_embed_query::run_neighbors` → dispatch | Stored-vector dot-product ranking; emits `graph-embed.similar`. |
| `graph-embed train` | `graph_embed_train::run` | Loads schema/graph, trains through the provider, normally exports vectors, then releases its session. |
| `graph-embed update` | `graph_embed_update::run` | Loads a checkpoint and current graph for forward inference/export; optional missing-only selection. |

The query routes require qualified node IDs. Missing nodes/vectors and empty or
non-numeric/non-finite/out-of-float32-range target vectors fail. Neighbors reject
zero limit and clamp values above 1,000. They discover document collections,
query each sequentially, filter candidate dimensions/numeric values, exclude the
target, merge local top-k results, then round scores after global ordering.
This is a cross-collection scan using dot products, not a normalized cosine or
indexed approximate search. Any collection-query error fails the operation.
Malformed returned neighbor objects are not strictly decoded: missing similarity
sorts as zero. Stored-vector generation/freshness is not checked here.

## Training and update boundaries

Main requires explicit `--gpu` for train/update, even a missing-only no-op.
Training validates positive epochs/cadence and positive finite split ratios
whose sum is below one. Defaults include 200 epochs, dimension 128, hidden 256,
21 bases, dropout 0.2, learning rate 0.01, patience 20 and seed zero. The seed
controls partitions/sampling, not complete model reproducibility. Other parameter
validation is delegated downstream and is not established by these adapter guards.
Schema selects architecture. Both adapters construct the default training client
at `/run/hades/training.sock`; its defaults are 10-second connection, 60-second
fast-operation and 600-second slow-operation deadlines.

Export-target “preflight” constructs clients and validates configuration; it does
not issue a request proving database existence, credentials or write permissions.
`--no-export` omits the export target entirely. A training run creates fixed
`graph.safetensors`, `embeddings.bin` and checkpoint paths in the chosen directory
(default `/tmp/hades-train`). Update uses `best.pt`, `graph_inference.safetensors`
and `embeddings.bin`. Local checkpoint validation checks existence only; provider
loading supplies further validation. Shared path permissions, concurrent callers,
symlinks and crash recovery still need dedicated operational verification.

Full update validates checkpoint existence/connects before graph loading.
`--new-nodes` first loads the schema/graph and requires `hetero_sage`; it selects
existing destination documents with null/missing `structural_embedding`, in
5,000-key batches with at most eight queries in flight. Absent destination nodes
are counted separately. Selection indices are sorted/deduplicated. No selected
nodes returns a successful no-op explicitly reporting no checkpoint validation or
service contact; existing non-null vectors are left untouched regardless of age
or validity. With work, checkpoint architecture must also be `hetero_sage`.

A source-level candidate remains in selection parsing: absent/missing booleans
are optional lookups, so malformed rows can be ignored as already embedded.
The parser also does not compare response cardinality with requested keys. This
needs a private malformed-response reproduction; no production incident is claimed.

Update checks provider row count against its selected/full ID map. Shared export
validates dimensions, finite values and subset identity before writing, requires
one acknowledgment per update, and reports acknowledged earlier batches on error.
Exports are not atomic across batches. Training delegates matrix-size validation
to export and does not independently compare the provider's `num_nodes` field.
Both adapters await session release before printing success; release failure turns
an otherwise successful operation into an error after its side effects. An earlier
operation error remains primary if release also fails.

## Existing evidence and remaining acceptance

[CLI lifecycle evidence](cli-training-lifecycle.md) covers a private CPU provider,
real disposable database, two training epochs and subsequent update/export. It
also establishes different training-edge versus full inference-edge contexts;
those metrics are synthetic, not representative retrieval scores. See
[provenance boundaries](structural-vector-provenance.md) for missing-only refresh
and mixed-generation limits. These historical reports retain their own source pins.

This map adds four leaves: combined maps cover 72/80; codebase (seven) and daemon
(one) remain. Full acceptance still includes runtime candidates, GPU/flag/alternate
export-target cases, concurrent freshness, learned-graph quality, claim-to-code
review, deployed provenance and recovery. No services, models or databases were
contacted for this source review and no runtime changes were made.
