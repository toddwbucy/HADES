# Code and paper-draft retrieval audit

## Scope and evidence status

Epic #12 covers both code search and document research, with equal query counts
and separate results. The user identifies the research collection as evolving
drafts of public-facing papers describing HADES **as intended when built**.
A paper's present-tense claim therefore does not establish implemented behavior.

The questions below are proposed audit seeds, not user-supplied workload examples
or frozen relevance judgments. The exact collection, draft versions and independent
judgments are still outstanding. No paper content or production collection was
accessed to prepare this plan.

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
The existing evaluator hardcodes the code task and passage prompt on both sides;
it must be adapted and validated before claiming document-research results.

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
