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
