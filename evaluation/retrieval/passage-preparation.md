# Private evaluation passage preparation

The user-selected `Bastion/` drafts are excluded from Git. Prepare passages from
the private, hash-verified snapshot, rather than reading changing working files
during an evaluation. Keep the snapshot and outputs in a private directory.

```bash
python3 scripts/prepare_retrieval_passages.py \
  --snapshot /path/to/private/snapshot \
  --output /path/to/private/snapshot/passages-v1.json
```

The snapshot has a `manifest.json` containing a `documents` list of relative
`path`, `bytes` and `sha256` values, with matching files under `documents/`.
The preparer validates every size and hash, rejects escaping paths and invalid
UTF-8, and creates its output exclusively with mode 0600. It never overwrites
an earlier artifact or contacts a model, database or service.

The versioned policy partitions each file into contiguous passages of at most
4,096 UTF-8 bytes, preferring the last newline within the cap. It preserves all
source bytes without overlap, trimming or replacement. Each passage records a
stable ID, source hash, text hash, zero-based end-exclusive byte offsets and
one-based inclusive line offsets. The artifact binds both the source manifest
and preparer code by SHA-256. Limits are 16 MiB per document, 64 MiB per corpus
and 20,000 passages; preparation fails rather than silently dropping excess data.

The captured paper snapshot produced 34 passages across nine drafts. Every source
was reconstructed exactly from its passages, and all offsets/hashes were checked.
Only this aggregate evidence is public; draft text and detailed manifests remain
private. These boundaries are reproducible evaluation units, **not** a claim of
production extraction or chunking parity. A byte cap is not a tokenizer budget;
validate actual prefixed token lengths before encoding to prevent silent truncation.

The output intentionally has no queries or relevance labels. Add the reviewed
candidate questions and collect independent judgments as separately versioned
artifacts. The strict scoring policy is declared for that later step. Preparation
alone establishes neither retrieval quality nor whether a draft claim is implemented.
