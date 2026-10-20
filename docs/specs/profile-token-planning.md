# Token-aware conventional passage planning

This internal [#225](https://github.com/toddwbucy/HADES/issues/225) slice makes
`ProfiledDocuments::prepare` obtain a verified token plan before embedding.
`DocumentInput` accepts only a source identifier and extracted text; callers
cannot supply their own spans. CLI, MCP, jobs, extraction, and frontend setup
are not connected to this path yet. Installed services are unchanged.

## Policy and source coordinates

The saved conventional policy supplies two **upper bounds**:

- `passage_tokens`: body tokens, counted without the passage prompt or special
  tokens.
- `overlap_tokens`: tokens in the exact source substring shared with the previous
  passage, also counted without prompt or special tokens. Actual overlap may be
  smaller, including zero, to fit the budget and make forward progress.

Each candidate is also tokenized with the serving passage prompt and special
tokens. That input must fit the smaller of the saved profile ceiling and the
live provider ceiling. Counting performs no model inference or truncation.

Spans use original UTF-8 byte positions with exclusive ends. Every span is
nonblank, advances beyond the preceding span, and together they cover all source
bytes without gaps. The tokenizer may normalize each candidate (Jina uses NFC),
but the source and stored passages retain their original bytes. This API does
not provide token-offset mappings, grapheme/paragraph boundaries, or late chunking.

## Bounded conservative search

The planner counts each proposed substring with the actual serving tokenizer.
It halves oversized proposals, refines between observed candidates, and grows
later proposals from the preceding fitted span. Overlap uses checked suffixes.
BPE merges and normalization can make counts nonmonotone: the planner never
infers that an untested substring fits and does not promise maximal packing.
It may refuse a source even if a different partition could fit. Blank tails or
extreme policies can also prevent a complete nonblank plan.

Limits per operation are:

| Limit | Bound |
| --- | --- |
| Original UTF-8 source | 2 MiB |
| Passages | 128 |
| Aggregate passage bytes, including overlap | 4 MiB |
| Aggregate source-substring bytes submitted for counting | 64 MiB |
| Tokenizer count calls | 8,192 |
| Planning deadline | 30 seconds, checked before and after each count |

The deadline cannot interrupt a tokenizer call already executing. Requests retain
worker ownership until completion, including after cancellation. Any limit or
fit failure discards the entire plan with HTTP 400 `PE_PASSAGE_PLAN_LIMIT`; the
caller must split the source explicitly. There is no partial plan or fallback.

## Receipt and wire contract

`POST /v1/passages` accepts one string and the existing expected receipt:

```json
{
  "model": "registered/model-id",
  "task": "retrieval.passage",
  "input": "Original extracted text…",
  "expected_receipt": {
    "format": "hades-provider-receipt-v1",
    "artifact_md5": "36ad00d2e76767b48b3a55053472dcf5",
    "space_id": "ec03fc217ec93d932741ef76c3bc39b91fbf35176c3e563177749447950db6c4",
    "serving_generation": "load-fixture-1"
  },
  "passage_tokens": 500,
  "overlap_tokens": 100,
  "max_input_tokens": 2048
}
```

Identity values above are illustrative fixture values. The response format is
`hades-token-passages-v1`, with model/task, the actual checked receipt, source
SHA-256, requested policy, effective ceiling, and `spans`. Each span contains
`start_byte`, `end_byte`, `body_tokens`, `input_tokens`, and `overlap_tokens`.
Unknown/duplicate fields and type coercions are refused. Existing bounded HTTP
transport and receipt checks apply to both planning and embedding.

Planning owns the same provider lifecycle lock and final drift checks as
inference. Rust verifies identity, policy, counts against limits, source digest,
UTF-8 boundaries, coverage, and allocation bounds. Counts remain observations
from the trusted provider, not an independent tokenizer attestation. A reload
between planning and embedding fails preparation under the original receipt.
The database profile is rechecked after planning and again at commit.

## Persistence and validation

`hades-bound-document-v2` stores the full plan on the document and counted spans
on chunks. Replacement validates the full source plan; search checks its metadata,
stored receipt, effective ceiling, and each chunk's agreement with its plan.
The previous internal v1 format only recorded caller-supplied spans and is
refused without automatic adoption. No production records use this internal path.

CPU tests cover real local tokenizer accounting, Unicode, nonmonotone counts,
resource limits, strict parsing, drift, cancellation, and independent recounts.
Rust wire tests cover malformed plans and overlap amplification. The strict
`profiled_documents` database contract covers rejection before inference/writes,
profile changes during planning, stored-plan corruption, and compatible reloads.
See [document validation](profile-bound-documents.md#validation-and-activation-gate)
for isolated database commands.

The [real-Jina protocol](verified-jina-qualification.md) adds the fixed
`services/tests/fixtures/jina_passage_planning.txt` after its original retrieval
checks. It persists multiple passages, searches all stored chunks, and independently
recounts every persisted body/input/overlap using the same actual serving load.
The evidence records the original source digest, full plan, and recount result.

The [ThinkPad qualification](verified-jina-qualification.md#thinkpad-token-planning-result-2026-10-20)
passed with a 2,402-token Unicode source, six stored passages, and exact agreement
for every independently recounted token count.
