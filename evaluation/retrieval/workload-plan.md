# Code and paper-draft retrieval audit

## Scope and evidence status

Epic #12 covers both code search and document research, with equal query counts
and separate results. The user identifies the research collection as evolving
drafts of public-facing papers describing HADES **as intended when built**.
A paper's present-tense claim therefore does not establish implemented behavior.

The user confirmed `Bastion/` in the server checkout as the research corpus.
Its nine Markdown drafts are excluded from Git. A private evaluation snapshot
records each relative path, byte count, SHA-256 and source modification time;
copied bytes were verified against those hashes. Reads were checked for per-file
stability, not taken as an atomic snapshot of the whole directory. Preserve this
snapshot for the run instead of mixing later edits into its judgments.

The questions below remain proposed audit seeds, not user-supplied workload
examples or frozen relevance judgments. The corpus location is resolved.
The [private passage preparation](passage-preparation.md) freezes 34 evaluation
passages across the nine drafts, with exact source offsets and hashes. Relevance
judgments and measured retrieval results are still outstanding. Draft text and the
detailed manifest remain private. No production database collection was queried
or changed to capture the source documents.

## Candidate questions

| ID | Code-search question | Required evidence |
|---|---|---|
| C1 | Where does HADES select the database for a request, and prevent access to a database outside the allowed set? | Entry point, authorization and selection path at a pinned commit. |
| C2 | What happens when replacing an ingested file fails halfway through? | Transaction boundaries, failure handling and relevant contracts. |
| C3 | How are vector results combined with graph information, and what happens when graph embeddings are missing or stale? | Actual scoring and fallback paths; distinguish different search modes. |
| C4 | How does HADES stop a cancelled training job, and which work may continue? | Session ownership, cancellation boundaries and tested limitations. |
| C5 | Which code and tests would change to support another document source? | Adapter interface, ingestion routing, configuration and tests. |

| ID | Paper-research question | Required evidence |
|---|---|---|
| D1 | How do the drafts define HADES's purpose and intended users? | Passages with draft identity and version. |
| D2 | What role is the knowledge graph intended to play alongside vector retrieval? | Architectural claims and their stated rationale. |
| D3 | What benefits do the drafts claim, and which have reported measurements rather than proposed evaluation? | Separate claims, methods, results and missing support. |
| D4 | Where do the drafts disagree about planned capabilities or terminology? | Both sides with version-specific citations; do not silently reconcile them. |
| D5 | What limitations and unfinished work do the drafts acknowledge? | Explicit limitations; distinguish silence from an asserted guarantee. |

Add a paired claim-to-code check after both corpora are frozen: select a paper
claim and retrieve implementation evidence supporting, qualifying or contradicting
it. Do not label an unlocated implementation as absent without a sufficient code
review. Report this mixed task separately from the two retrieval workloads.

## Corpus and judgment protocol

Freeze code commit, collection identifier, document/version identifiers, content
hashes and chunk boundaries. Retain titles, sections, dates and passage offsets.
Include older drafts and plausible distractors; record supersession explicitly.
Keep unpublished text and private corpus manifests outside public issue artifacts.

Have an independent reviewer judge pooled results from each retrieval method,
with method names and scores hidden. Use grades 0 (irrelevant), 1 (context),
2 (partial evidence), 3 (direct evidence). A contrary passage may be directly
relevant. Record judgment rationale, disagreements and unjudged candidates;
do not silently treat missing judgments as proven irrelevance. Freeze before
scoring; publish corrections as a new dataset version.

## Evaluation and acceptance

Compare vector-only retrieval, the existing file-membership comparator and a
separately identified learned-graph model where available. Record model/checkpoint,
task adapter, query/passage prompts, tokenizer, precision and index versions.
The evaluator selects the code adapter with passage prompts for code search,
and the retrieval adapter with passage/query prompts for document research.
It preflights all prefixed inputs through the loaded processor before encoding.

Report recall@5, MRR@10 and nDCG@10 per workload, per query and as an equal-weight
workload average. Require judgments for every top-ten result from every compared
method before publishing MRR@10 or nDCG@10. Compute nDCG's ideal ranking from the
frozen judged pool and label it pooled nDCG, since relevant passages may remain
outside that pool. Label recall as judged-pool recall for the same reason. If any
top-ten result is unjudged, withhold that query's ranking metrics and report the
missing judgments; do not drop unjudged results and compress ranks. Report both
top-ten judgment coverage and the count of fully judged queries. Withhold the
workload aggregate until all included queries meet this rule, preventing selective
reporting of easier queries. Track citation correctness, draft-version accuracy,
claim support and confusion between intended and implemented behavior separately
from retrieval ranking. Use explicit no-evidence cases for abstention assessment;
the current positive-relevance scorer cannot score those cases unchanged.

Capture latency and memory in isolated fixtures with the measurement boundary
stated. Set acceptance thresholds from the reviewed representative baseline,
not the existing 24-question author-judged seed. This plan supplies no new quality
score and does not complete the retrieval acceptance criterion in epic #12.

## Private research candidate run (2026-09-20)

The frozen nine-draft Bastion snapshot produced 34 passages and five questions.
An isolated, offline CPU run using the local Jina v4 model encoded all 39 inputs
with the document-research profile; processor counts were 12–1,091 tokens against
a 2,048-token ceiling, without truncation. Recorded elapsed time was 813.04 s
(model loading, preflight, encoding and scoring included), with peak RSS
8,743,188 KiB. This is a batch resource observation, not serving latency.

Dataset/vector/metrics hashes were verified and frozen-vector rescoring matched
the initial report exactly. Both methods' quality metrics remain withheld: all
relevance labels are empty. The comparator is source-file membership, not a
learned graph. These results do not establish production quality.

The owner and a separately designated third party will independently judge a
private blinded packet containing five questions and 50 pooled candidates. The
version-2 packet uses the 0–3 rubric above; it supersedes an unjudged version-1
packet whose rubric differed. Separate answer sheets preserve reviewer identity,
date, rationale and unresolved judgments. Keep each review hidden from the other
until submission; retain original ratings and record adjudication separately.
No draft text or raw ranking artifact is included in the public repository.
