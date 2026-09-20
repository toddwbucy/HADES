# Actual CLI training lifecycle audit

The opt-in `training_alignment` target now includes a successful real CLI
`graph-embed train` followed immediately by `graph-embed update` against one
private Python provider and a disposable ArangoDB. Bubblewrap hides live sockets
and GPU devices. The private provider explicitly maps declared `cuda:0` to CPU.

The fixture creates eight synthetic feature documents and twelve directed edges;
the loader includes seven edge-connected nodes. With seed 1729, two epochs,
four output dimensions, and 25% validation/test splits, training returns finite
metrics, saves `best.pt`, and exports seven vectors. A separate CPU backend loads
that saved checkpoint and the CLI-produced training graph; every exported vector
matches exactly by qualified document ID. Both CLI sessions release ownership.

## Graph context changes the export

An immediate full update with the same checkpoint changes the three connected
concept vectors. This is explained by `TrainingServicer._adjacency`: a graph with
`train_idx` uses only training edges, while an inference graph without splits
uses all edges. Loading the CLI-produced inference graph into the verifier
reproduces the update vectors exactly. The test therefore checks each operation
against its actual graph context, and explicitly requires the fixture to expose
the difference. It does not assume unchanged checkpoint weights imply unchanged
embeddings. No runtime behavior was changed for this audit.

[Retained evidence](cli-training-lifecycle-result.json) includes the passing log,
source/probe hashes, and clean private-server shutdown. The original alignment,
generation and session-error cases also pass in the same opt-in target.
Run through `scripts/test_isolated_database.py --arangod <private-binary>
--training-alignment-python <CPU-venv-python>` after generating protobuf bindings;
the runner owns a fresh Unix-only database server. Bubblewrap is required.

These synthetic metrics are not representative quality scores or service latency.
The fixture does not establish GPU correctness, all training flags, export-to
another database, no-export, early stopping, cancellation during an epoch,
production deployment, or the desired long-term policy for training-versus-full
graph export context. Those limits remain part of epic #12.
