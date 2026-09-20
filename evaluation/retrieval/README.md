# Repository retrieval evaluation, version 2

`repository-v2.json` freezes 24 code-navigation questions and 24 function excerpts
from HADES before inference. Each excerpt records its source path, revision,
line range, content hash, and truncation status. These are author-created seed
judgments: grade 3 names the function directly implementing the requested behavior;
other functions are unjudged and treated as zero. This is neither an exhaustive
relevance set nor an independent production benchmark. Do not tune questions or
labels after seeing scores; publish reviewed corrections as a new dataset version.

The evaluator uses the existing Jina v4 weights offline with the code adapter and
Passage prefix on both sides, matching the code profile's prompt convention. CPU
bfloat16 and a 2,048-token input cap differ from the live GPU runtime. No installed
service is contacted and no GPU is used. Model loading executes the local model's
custom Python code: inspect that code before evaluating another model directory.

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 1200 /path/to/test/python \
  scripts/evaluate_retrieval_quality.py --model /path/to/local/jina-v4 \
  --output /tmp/hades-quality-v2
```

The evaluator sets one CPU affinity, lower priority, a 32 GiB address-space limit,
private caches, and offline settings. Use a host with sufficient free RAM and
compatible preinstalled dependencies; never install into the live service's venv.
Outputs record dependency versions and hashes of local model/configuration files.

Vector ranking uses cosine similarity. The graph-assisted baseline reranks the
top ten with the handler's 0.7/0.3 blend and top-three centroid, using same-source-file
membership vectors. That graph is reproducible from corpus provenance, but is
**not** a trained production RGCN/GraphSAGE model. It measures whether this simple
structural signal helps these questions; it cannot certify learned-graph relevance.
Metrics include recall@5, MRR@10, nDCG@10, and complete per-query rankings. Aggregate
scores must be read alongside failures and the small, incomplete judgment set.

`vectors.f32`, when captured, is little-endian float32: document rows in dataset
order, then query rows, each 2,048 values. It allows ranking regression checks
without reloading the model. `vectors.npz` contains the same matrices with explicit
names. Preserve provenance and dataset hashes with any retained vectors.

## Captured baseline

The offline run completed in 223 seconds with peak process RSS of about 8.1 GiB.
Its 48 vectors and provenance are retained in `results-v2/`.

| Method | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|
| Vector only | 0.9583 | 0.7851 | 0.8380 |
| File-membership graph baseline | 1.0000 | 0.7910 | 0.8428 |

The graph result is a small seed-set improvement, not evidence that graph ranking
helps every corpus. Per-query rankings are retained so failures can be reviewed.
The installed tokenizer emitted a regex warning; tokenizer settings were kept
unchanged and their file hashes are recorded rather than silently patched.

Rescore without a model, GPU, network, or running database:

```bash
python scripts/evaluate_retrieval_quality.py \
  --vectors evaluation/retrieval/results-v2/vectors.f32 \
  --output /tmp/hades-quality-rescore
```

The captured output reproduces exactly with the pinned CPU test environment.
Ranking implementations must retain recall and should not lose more than 0.01
absolute MRR/nDCG on this fixed seed without an explicit reviewed explanation.
These are seed regression thresholds, not statistical guarantees or deployment
acceptance thresholds for private production corpora.

## Dataset correction history

Version 2 corrects the grade-3 excerpt for `q-train-adjacency`: it now contains
`_adjacency`, `_encode`, and `TrainStep`, which implement adjacency filtering,
rather than `_training_split`, which validates split indices. Its separately
hashed segments retain their exact lines from the pinned source revision. Queries
and relevance grades are unchanged. All vectors were regenerated offline; the
aggregate scores above happen to be unchanged, though vector hashes differ.
`repository-v1.json` and `results-v1/` remain historical evidence with that known
judgment error. Use version 2 for regressions. Provenance now records evaluator-
generated SHA256 hashes for the dataset, float32 vectors, and metrics.
