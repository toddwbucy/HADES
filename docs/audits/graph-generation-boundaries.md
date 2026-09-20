# Executed structural-vector generation boundaries

This extends the private [training alignment contract](training-alignment.md).
The initial runtime source is `bd2aefa`; the fixture builds on PR #111. No live
database, training service, GPU or production checkpoint was used.

## Observations

The real CPU HeteroSAGE servicer saves a checkpoint before and after one optimizer
step. The saved semantic contracts are equal and at least one weight tensor
differs. Both checkpoints load successfully; each reproduces its original
embedding bytes exactly through GetEmbeddings. Compatibility therefore protects
input meaning, but does not require identical trained weights.

Using the later checkpoint, two separately Rust-serialized inference graphs
retain that semantic contract and all original node indices:

- Changing `papers/same`'s feature vector from `[3, 1]` to `[9, -3]` changes
  output vectors for `papers/same` and its neighbor `concepts/same`.
- Removing the two edges while retaining the three nodes changes the output
  vector for `concepts/same`.

These are controlled tensor-input changes, not executed text ingestion or a
database mutation/invalidation test. They establish that semantic compatibility
does not identify graph feature or adjacency generations.

The real Rust exporter first writes all earlier-checkpoint vectors into a
disposable ArangoDB 3.12.11 graph. A reversed subset export from the later
checkpoint updates only the requested two nodes, preserving the third. After
resetting the graph to the earlier vectors, another export uses one-document
batches in one collection: the first existing target succeeds, then an absent
target fails. The error reports one acknowledged update. Reading all three real
documents confirms one later-checkpoint vector and two earlier-checkpoint vectors
remain. This is explicit partial persistence, not false success or rollback.

## Evidence and interpretation

The final fixture and focused lint pass. The
[retained result](graph-generation-result.json) records the exact test output,
runner outcome and source hashes. Reproduce with the opt-in command in the
alignment report; no additional service installation is required. The owned
private server stopped with exit zero. The checkpoints and tensors are temporary
test artifacts, not a retained evaluation model or corpus.

This confirms specific limitations previously established by the
[provenance source review](structural-vector-provenance.md). It does not measure
ranking harm or imply the documented missing-only mode promises full freshness.
Readers currently have no enforced shared generation identity; a successful
subset update or a failed full update must not be described as a complete refresh.

Remaining acceptance work includes actual changed-text ingestion, missing-only
selection through the CLI, query behavior during partial generations,
representative learned-graph quality, and a reviewed generation/legacy policy
before any production migration. No runtime behavior is changed by this audit.
