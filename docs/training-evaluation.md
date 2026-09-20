# Training and evaluation protocol

HADES evaluates transductive link prediction: node features and all nodes are
visible, but validation/test relationships are absent from encoder adjacency.
Training targets are drawn from `train_idx`. Training, validation, test, and
embedding export over a training graph all encode **only the training edges**.
Evaluation uses the held-out edges as scoring targets, never as message-passing
edges. Negative sampling excludes observed positives from every split.

The decoder scores endpoint pairs without a relation argument. Split unordered
endpoint pairs as units: duplicate edges, inverse edges, and different relation
types between the same two nodes belong to the same partition. Ratios therefore
apply to pair groups, not necessarily individual edge rows. This avoids reverse
or duplicate relationships revealing held-out topology. Python rejects files
with overlapping/incomplete partitions or pairs crossing partitions.

Rust returns the exact partition written to safetensors; the orchestrator must
use those returned indices. RPC callers cannot train on indices outside that
file's training split. Files without split tensors are inference-only: they can
use full adjacency for embedding updates, but TrainStep/Evaluate reject them.
Do not report full-graph inference results as held-out link-prediction metrics.

## Reproducing the CPU baseline

Generate Python protobuf bindings and use an environment with the training
service dependencies installed, then run:

```bash
make -C services proto-gen PROTO_DIR=../proto
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python scripts/rebaseline_training_topology.py
python -m pytest services/tests/test_training_topology.py
cargo test -p hades-prefetch --lib
```

`docs/audits/training-topology-baseline.json` records a bounded synthetic
comparison of legacy leaked adjacency and corrected training-only adjacency:
two architectures, three fixed seeds, 20 CPU epochs per run. The script constructs
all data locally, loads no external model, and contacts no database/service.
This is a regression baseline, not evidence of production retrieval quality.
Prior held-out metrics used leaked topology and must not be reused to claim
unseen-link performance. The baseline has been rerun with tie-correct AUC (#17). Production
retraining requires a separate resource and deployment plan.

## Metrics, sampling, and unsupported runs

ROC-AUC counts a tie as half a correct positive/negative ordering. Both score
classes must be nonempty and finite; unavailable metrics are errors, not zero
or NaN. Training requires nonempty train/validation/test partitions, positive
epochs and validation cadence, and nonzero negative counts. Tiny graphs or
ratios that floor a split to zero fail with an actionable error before training.

Use `graph-embed train --seed 29` to reproduce pair partitions and negative
sampling for the same graph ordering and pinned dependency versions. The seed
is recorded in graph metadata and CLI output. Validation negatives stay fixed
across epochs; training negatives change deterministically per epoch; test
negatives use a separate seed. This flag does not seed model initialization or
promise deterministic GPU kernels. Retain the graph artifact for comparisons.

Negative sampling excludes observed pairs in either direction and self-loops,
with replacement. A bounded rejection budget must produce every requested
sample or fail explicitly; dense graphs are not silently evaluated on a short
or empty sample. Failures propagate through prefetching to the orchestrator.
Early stopping saves only finite improving validation losses, leaves the best
checkpoint unchanged on ties, and restores that checkpoint before test scoring.
Nonfinite metrics or failure to produce a checkpoint abort the run.
