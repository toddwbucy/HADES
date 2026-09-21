# Compliance report incomplete-evidence outcomes

## Reproduced baseline

At source revision `b5fec8c1672194e036ff09ff98aa479e7ebc89db`, the actual CLI
reports `data.passed: true`, outer `success: true`, and exit 0 in both cases:

- A local Rust file claims CS-32, but the database lookup finds no smell definition.
  The report correctly lists the missing definition and nevertheless passes.
- A definition and compliance edge exist, but the private embedding peer returns
  HTTP 503. The report records a probe error with `pass: null` and nevertheless passes.

The adjacent JSON preserves normalized actual responses and four source/probe
hashes. The regression test executes both cases before asserting that neither
may pass. Its expected baseline failure reports two incorrect passes (0.02s).
The database fixture receives only two/three read cursor requests, respectively.
All configuration and environment are isolated; no production database, model,
GPU, or daemon was contacted. The fixture embedding socket is private and returns
503; the test does not infer this behavior from source alone.

An initial fixture with an absent socket instead produced a connection error
and exited 1. That is a distinct, correctly failing boundary and was not evidence
of this defect. The corrected fixture isolates probe failure after construction.

## Impact and required correction

Automation or reviewers consuming the overall compliance verdict may accept a
claim with absent or unavailable evidence. Make missing definitions and failed
required probes prevent a passing verdict, while preserving diagnostics and
valid success behavior. Define partial/unavailable evidence explicitly; do not
silently omit it. Test missing and unlinked references, negative static/probe
results, successful probes, empty controls, and CLI/shared-service behavior.
The existing command-execution success convention is separate from the verdict
and needs explicit disposition rather than accidental behavior changes.

Unreadable-file omissions, filesystem traversal limits, claim identity ambiguity,
and sequential linking partial writes remain separate audit candidates. This
reproduction establishes neither a production incident nor full compliance safety.

## Remediation

The shared report handler now requires zero missing definitions and an affirmative
boolean true from every generated probe, in addition to the existing static and
unlinked-claim checks. Null/error probes retain their diagnostics and prevent a
passing verdict. Empty reports retain their existing passing behavior.

The CLI and daemon still report successful execution when they successfully
produce a report: outer success and exit 0 mean report generation completed.
Consumers must use `data.passed` for the compliance verdict. This preserves the
same convention for static violations and low similarity; connection/query
errors that prevent generation remain operational failures. No field was removed.

Nine actual-CLI response tests passed, including both reproduced defects and five
controls (empty, static violation, unlinked reference, valid identical vectors,
and orthogonal vectors). One shared-service test passed for missing, unlinked,
and empty reports, preserving request ID and detailed evidence. Successful probe
controls use synthetic 2,048-dimensional vectors, not a live model or calibrated
quality benchmark. Remediation hashes and test summaries are recorded separately;
the baseline response artifact is unchanged.

Unreadable inputs skipped before verification remain a separate audit gap. This
fix does not establish filesystem completeness, relevance of the fixed similarity
threshold, or atomicity of linking operations. Nothing was deployed.
