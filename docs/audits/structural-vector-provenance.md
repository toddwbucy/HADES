# Structural-vector provenance and refresh boundaries

Source review at `a2236f385dd72ad7f77098ae84166df53f6b4cf2`, for epic #12.
This report traces implementation contracts. It does not inspect live vectors,
reproduce mixed checkpoints in production, or measure retrieval quality.

## Checkpoint compatibility is semantic identity

`services/training/contract.py::FIELDS` identifies contract version, relation
order, collection order, feature dimension, architecture, feature-construction
policy and feature-model names. `TrainingServicer.LoadGraph` in
`services/training/server.py` rejects a contract different from the bound
model/checkpoint. Checkpoint load validates its saved contract before adopting
candidate state and clears the graph when its device or contract differs.
These checks protect tensor meaning; they do not identify weight bytes, graph
snapshot revision or per-node source revisions. Two different trained models
can share these semantic fields. Feature model names are not model-file hashes.

## Missing-only update is an explicit contract

The `Update.new_nodes` help in `crates/hades-cli/src/commands/graph_embed.rs` and
`docs/declarative-schema.md` explicitly leave existing vectors untouched.
`graph_embed_update.rs` selects destination documents lacking
`structural_embedding`, loads the checkpoint and graph when work exists, and
exports compact rows for selected nodes. No selected nodes means a documented
successful no-op without contacting the training service or validating a
checkpoint. Omitting `--new-nodes` performs the full graph forward pass/export.

A missing-only run is therefore not a freshness check. Replacing checkpoint
weights or changing graph neighbors does not make a non-null vector eligible.
This is a limitation of the documented mode, not evidence that its selection
violates the command contract.

## Ingestion does not give one universal invalidation rule

| Path | Source behavior | Structural-vector consequence from source |
|---|---|---|
| Parsed code ingestion | `codebase_ingest.rs` constructs `Replacement` with `merge_file: false`, `purge_symbols: true`; new payloads omit structural vectors | Replaced file and symbol documents do not preserve their prior vector fields |
| Parser-free text ingestion | Same file constructs `Replacement` with `merge_file: true`; payload does not name `structural_embedding` | Existing file fields outside the payload, including a structural vector, survive the update |
| Document pipeline overwrite | `pipeline/orchestrator.rs::store` deletes old chunks/embeddings, then imports metadata/chunk/embedding payloads; `db/crud.rs::insert_documents` uses `onDuplicate=replace` | Replaced documents retain only the supplied payload, not arbitrary old structural fields |

`codebase_persist.rs::Replacement::store` implements the file update/replace
choice and replaces supplied symbol documents. These paths do not traverse the
whole graph invalidating structural vectors on other affected nodes. The table
is a source trace, not a newly executed ingestion fixture. It makes no claim
about every adapter or concurrent ingestion/update behavior.

## Export and retrieval do not enforce a shared generation

`graph/export.rs::export_grouped_embeddings` updates only
`structural_embedding`. It does not attach a checkpoint hash or source revision.
Its acknowledged batches may persist before a later failure (#83/PR #84), so a
failed full export is not a database-wide atomic replacement.

`dispatch.rs::graph_embed_neighbors` and `structural_rerank` consume stored
vectors without a shared generation identity comparison. Dimension and numeric
checks cannot establish that vectors share a checkpoint or graph snapshot.
#85/PR #86 separately addresses malformed numeric values; it does not add
provenance. The semantic checkpoint compatibility checks above do not reach
back into previously exported database vectors.

## Remaining verification and disposition

No new production defect or quality score is asserted here. The original audit
acceptance still requires representative learned-graph evaluation and matching
claims to actual implementation. A future freshness/generation contract needs
private fixtures for changed text, changed neighbors, two compatible checkpoints
with different weights, and interrupted export. It must define legacy-vector
handling and read behavior during partial refresh before any migration.

For evaluation, retain the exact checkpoint, graph/corpus snapshot and export
outcome with the experiment. Do not label missing-only success as a full refresh
or infer current provenance from vector dimension. Any production refresh is a
separate maintenance operation subject to the owner's verified pre-risk snapshot
and arranged-downtime policy; none was executed for this review.
