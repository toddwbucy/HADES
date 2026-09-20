# Frozen code-search workload

The candidate workload uses C1–C5 from [the workload plan](workload-plan.md),
retained without labels in [code-queries-v1.json](code-queries-v1.json). These
remain proposed audit questions, not owner-supplied usage observations.

## Source and scope

`scripts/prepare_code_retrieval.py` resolves a commit and reads its immutable Git
blobs. Working changes, untracked files and the ignored Bastion drafts cannot
enter the snapshot. The manifest records the commit, file/blob identities, sizes,
SHA-256 hashes and exact selection policy. Selected symlinks are rejected.

The policy includes regular tracked files with the listed source/configuration
suffixes under `crates`, `services`, `proto`, `config`, `deploy`, `scripts`, `web`
and `.github`, plus root Cargo/toolchain manifests. Implementation and tests are
included without query-specific selection. Documentation, lockfiles, extensionless
build files, binary assets, external dependencies and untracked generated files
are outside this corpus. This is a versioned code-search scope, not the complete
set of files needed to reproduce a deployment.

## Reproduction

Use a new output directory; the command refuses to replace an earlier snapshot:

```bash
python3 scripts/prepare_code_retrieval.py \
  --repo /path/to/HADES \
  --revision 92f9f829170937da537c6f38b9869b0d9488a3a6 \
  --queries evaluation/retrieval/code-queries-v1.json \
  --max-bytes 3072 --output /path/to/new-private-snapshot
```

The [passage preparer](passage-preparation.md) verifies hashes and preserves
contiguous UTF-8 byte/line offsets. The snapshot is private (directory 0700,
files 0600). Preparation makes no service calls and performs no inference.

At the pinned revision, 236 files contain 3,162,904 source bytes. The initial
4,096-byte passage policy produced 905 passages but processor preflight rejected
an input at 2,080 tokens. That rejected artifact remains separate. The revised
3,072-byte policy produces 1,153 passages; independent reconstruction verifies
every source byte, passage hash and offset. With five queries, all 1,158 prefixed
inputs pass the local Jina v4 processor preflight at 15–1,727 tokens, below 2,048.
The existing tokenizer-regex warning remained; settings were not silently changed.
Actual model-loaded preflight still runs before encoding.

The revised dataset SHA-256 is
`03967fe26ffff45efe54056fef00273e6e282f5e1f60468df12f8b4af86026cf`.
This is the frozen run artifact produced by builder commit `178095e`; a later
cleanup-only builder fix changes its builder hash when replayed, while source
and passage content remain identical. Its artifact binds both the builder and
passage-preparer source hashes. Later
source, query or passage-policy changes require a new snapshot and new judgments.

## Evidence limits and next stage

Four private Git-fixture contracts verify reproducibility, exclusion of working
changes/untracked files, output permissions, refusal to overwrite, rejection of
selected symlinks, rejection of prefilled/duplicate judgments and cleanup of
newly created incomplete snapshots after failed preparation. Existing outputs
are preserved. These run in
the existing operational unittest gate.

An offline encoding run uses a frozen evaluator copy, the existing local model,
one CPU, lowered priority, a 32 GiB address-space ceiling and a 12-hour deadline.
No production database, socket or GPU is used. Completion, frozen-vector rescore,
blinded owner/third-party review and quality thresholds remain outstanding.
Scores must stay withheld until the strict judgment requirements are satisfied.
The existing comparator uses file membership; learned structural retrieval and
the paired paper-claim-to-code check remain separate acceptance work.
