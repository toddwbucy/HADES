# Training feature and export alignment

## Executed boundary

The opt-in `training_alignment` integration test loads three named documents
and two edges from disposable ArangoDB 3.12.11 using the real Rust graph loader.
Rust serializes the training graph; the Python CPU training servicer receives
InitModel, LoadGraph, GetEmbeddings and TrainStep over private Unix gRPC sockets.
Rust decodes and exports the resulting float files into that same private graph.

The fixture uses `papers/same` and `concepts/same` with distinct sentinel feature
vectors, plus a missing-feature node expected to receive zeros. Expected features
are derived from named fixture inputs, independently of the serialized tensor.
It checks collection indices, exact named edge identities and node row order.
Full export must match the pre-training vectors for each qualified document ID.
After one real optimizer step, a reversed, non-contiguous `[2, 0]` request must
match those rows of full inference. Subset export must update those two IDs and
preserve the unselected node's earlier vector.

## Reproduction

Use an isolated checkout and an existing CPU Python environment with the locked
test dependencies. Generate bindings locally first:

```sh
make -C services proto-gen PROTO_DIR=../proto PYTHON=/path/to/cpu-venv/bin/python
python3 scripts/test_isolated_database.py --arangod /path/to/private-arangod \
  --training-alignment-python /path/to/cpu-venv/bin/python
```

The runner starts its own database on a private Unix socket, supplies isolated
configuration, restricts CPU/memory, enforces a command deadline and stops its
owned server on success or failure. The Python peer has a 90-second watchdog and
uses three-second RPC deadlines. This test is explicitly ignored by ordinary
Cargo runs; the opt-in runner executes it with `--ignored`.

## Result and limits

The final fixture passed; see [retained result](training-alignment-result.json).
Earlier harness attempts failed because a resolved interpreter symlink bypassed
the virtual environment, checkout-local protobuf bindings were missing, and a
new assertion was placed before its variable declaration. These were fixture
setup defects, not production findings; each owned database stopped cleanly.

This verifies one small HeteroSAGE CPU training/export path using real component
implementations. It does not invoke the full CLI orchestrator or Rust gRPC client,
measure retrieval relevance, validate large/GPU graphs or RGCN, establish deployed
source identity, or certify checkpoint-generation freshness. The subset case
deliberately demonstrates retained older vectors for unselected nodes; it does
not prove whole-graph generation consistency. Checkpoint switching, changed graph
inputs, interrupted exports and representative learned-graph evaluation remain
required by the broader audit. No production service, data or dependency changed.
