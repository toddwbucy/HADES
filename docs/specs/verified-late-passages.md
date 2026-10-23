# Verified contextual passage planning

Issue #225 requires tokenizer-derived late chunking without silently replacing it
with conventional chunk embeddings. `VerifiedJinaLateProvider` is an explicitly
selected adapter over the existing approved local Jina registration. It retains
the artifact MD5 and gives its pooling behavior a separate embedding-space ID.
Existing `VerifiedJinaProvider` spaces and conventional vectors are unchanged.

## Policy and coordinates

Document late policies declare both retrieval passages and encoding windows:

```json
{
  "mode": "late",
  "context_cap_tokens": 1024,
  "minimum_window_overlap_tokens": 128,
  "passages": {"passage_tokens": 256, "overlap_tokens": 32}
}
```

Existing late policies without `passages` remain readable for
caller-supplied boundaries; document planning requires the explicit passage policy.
No default is inferred from the model name.

`POST /v1/late-passages` counts the real serving tokenizer, task prompt and special
tokens without truncation. Plans contain original UTF-8 byte spans for passages
and windows, exact body/input/overlap counts, source SHA-256, the provider receipt,
and one containing window assignment per passage. The effective context ceiling
is the smaller profile/running limit. Window overlap must retain its declared
minimum. Failure to cover a passage is an explicit error, never clipped text.

Rust validates complete source coverage, UTF-8 boundaries, identity, token policy,
resource limits and assignments before inference. It converts each assigned
passage to character offsets within its window. `POST /v1/late-embeddings` encodes
that window once, then averages the token states whose original-character offsets
intersect each passage. Pooling uses FP32 accumulation and L2 normalization;
prompt/special tokens outside the passage are excluded. Query pooling continues
to average its complete prompted input in the same declared space.

Responses include exact source character ranges and observed token ranges. Rust
checks receipt, model, source, counts, coordinates, dimensions, finiteness and
normalization. A failed or reloaded later window discards the entire result.
There is no automatic fallback to conventional passage embedding.

## Limits and integration

Source text is limited to 2 MiB, each plan to 128 passages and 128 windows, and
overlap bytes to 4 MiB per plan. Both planning passes share count-call, scanned-byte
and elapsed-time budgets. Oversized work must be split explicitly.

Managed setup, foreground files, document jobs and public document search now
support this explicit late policy through the same profile guard. Setup still
requires a trusted internal caller. This does not enable production services,
code ingestion, a public setup endpoint, or a frontend model selector.
Use a distinct generation/database for this space;
conventional stored vectors cannot be relabeled as contextual vectors.

## Stored provenance and search

Conventional records retain `hades-bound-document-v2`. Contextual records use
`hades-bound-document-v3`: the source retains its complete late passage plan and
ordered `late_token_ranges`; each chunk and vector repeats its matching
`late_token_range`. All three records retain the same artifact, space, profile,
generation and serving receipt. File snapshots retain their existing provenance.
The entire document is prepared before one guarded transaction publishes it.
Failure in a later context window leaves prior stored records intact.

Search validates all stored context metadata before scoring: complete assignments,
window/passage budgets, minimum overlap, forward coverage, source extent and token
range agreement. Selected source text is still hashed and checked against exact
UTF-8 spans under the same transaction. Hits add `late_context` with the selected
window, its index and the observed token range, while `span` remains the original
retrieval passage. Stored token counts are provider observations; storage does
not independently run the tokenizer.

Replacement validates the old complete source and children before removing any
records. Missing, foreign or damaged provenance is an error, not automatic repair.
An identical verified provider reload can query older vectors; a changed space
or profile cannot. Late policies without explicit `passages` remain unavailable
to document setup/ingestion. Conventional data receives no automatic migration.

## Qualification

CPU tests use actual tokenizer/receipt logic and tiny deterministic weights:

```bash
python -m pytest services/tests/test_verified_late_provider.py -v
cargo test -p hades-core --test bound_embedding_contract
```

For an already approved local Jina registration, run in the installed ML Python
environment with an explicit device and a new evidence directory:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/qualify_verified_late.py \
  --registry /private/model-registry --provider approved-jina \
  --device cuda:0 --output-dir /private/new-late-evidence \
  --arangod /private/arangodb/sbin/arangod
```

The opt-in suite verifies loaded adapter tensors, then sends eight fixed documents,
six expected retrieval matches, repeat queries and a long Unicode source through
the real Rust client and a private Unix HTTP socket. Direct tokenizer calls recount
all windows and passage token coordinates. Reload must invalidate the old receipt.
Reports retain implementation identity, suite digest, runtime, artifact manifest,
receipts and failure stage. With `--arangod` or `--docker`, the strict isolated
runner additionally verifies recoverable setup, file snapshots/extraction transport,
atomic storage and public search, then recounts the persisted long-document plan.
Without those options it checks provider/transport only. No model download, live
corpus or installed service is involved. This is a fixed smoke qualification, not
a general retrieval benchmark; its text passthrough extractor is a test fixture.

The [ThinkPad run on 2026-10-21](../evidence/jina-late-thinkpad-2026-10-21.json)
passed against clean implementation `d0b63f9`: all six expected documents ranked
first; the 12,384-byte Unicode source yielded 11 passages over four windows, with
every token range independently checked. Peak allocated GPU memory was 8.27 GiB.
The separate [initial preflight failure](../evidence/jina-late-thinkpad-2026-10-21-preflight.json)
records a qualification-runner CUDA initialization mistake corrected before the
successful run. These results qualify provider/transport behavior, not database
late-document storage or extraction quality.

The subsequent [database qualification](../evidence/jina-late-documents-thinkpad-2026-10-21.json)
passed at clean implementation `419b503`: recoverable setup and identical retry,
eight real-model file ingestions, all six expected public-search matches at rank
one, then the same 11-passage/four-window Unicode source persisted and recounted.
Search validated all 19 stored passage/vector pairs. This run used the strict
private ArangoDB harness and a text passthrough extractor; it establishes database
integration, not extraction quality or a broad retrieval benchmark.
