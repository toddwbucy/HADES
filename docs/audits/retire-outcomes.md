# Retirement failure outcomes

Epic #12, P2 #147. Baseline `cd4dffd` reproduces an actual CLI false success on
source `e8e2426`: a private HTTP peer acknowledges the codebase deletion, then
rejects authored-edge deletion with HTTP 503. The command exits zero with
`success=true`, `retired=1`, and `removed=0`. A successful control reports one
removed edge. The preserved baseline uses synthetic acknowledgments and does
not establish actual persistence or a production incident.

## Fix and behavior

Authored-edge deletion errors now propagate with the affected collection,
acknowledged counts for earlier authored collections, and a warning that the
codebase sweep and earlier deletions may have committed. The CLI exits nonzero
without a success envelope. Processing stops on the failed collection.

Missing, null, negative, non-integer or excessive deletion counts are rejected.
Counts from zero through the scanned key count remain valid: a scanned edge may
have disappeared before deletion. The existing explicit-key deletion and
confirmation behavior are unchanged.

## Verification and limits

The complete private CLI response suite passes 13 tests. The new test exercises
HTTP failure, missing/null/string/negative/excess counts, successful deletion and
legitimate zero-count deletion. Every failure requires an empty stdout, nonzero
exit and partial-effect diagnostic. CLI all-target Clippy with warnings denied,
formatting and diff checks pass.

The disposable ArangoDB 3.12.11 lifecycle suite also passes 18 tests (64.08s),
including successful file retirement and killed-process recovery; the private
server stops cleanly. This suite does not inject a real authored-edge backend
failure, which remains covered by the private HTTP fixture.

This does not make retirement atomic. Initial target/edge scanning, the codebase
sweep and each authored collection remain separate requests. Initial response
shape validation, authorization races, whole-operation recovery and ambiguity
of interrupted backend writes remain separate audit concerns. Operators must
inspect affected state before retrying a partial failure; the original target
may already be gone. No production retirement or deployment was performed.
