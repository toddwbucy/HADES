# Compliance and linking command contracts

## Scope and evidence

Source-only audit for epic #12 at `09ddd5f208f9e25c0a309068fc7a46857b9d1d9b`.
The adjacent JSON pins inspected files. Four native leaves are traced from CLI
declarations through adapters and shared dispatch. No command was run against
production, and these observations are not runtime reproductions or certification.
Together with the database, system, task, and embedding/extraction maps, this
covers 66 of 80 native leaves. Fourteen remain: ingest, seven codebase commands,
four graph-embed commands, schema apply, and daemon.

## Command map

| Leaf | Inputs, operations, and result |
| --- | --- |
| `smell check` | Required path; format defaults to JSON, optional verbose. Loads definitions from `smell_specs`, scans local files for literal substrings, returns counts and `passed`. Only a violation with tier exactly `static` fails the verdict. Scope is selected from the database but not applied during matching. CS-13 skips recognized whole comment lines. |
| `smell verify` | Required path; optional claims filter. Extracts uppercase CS-number references from recognized whole comment lines, looks up each distinct number/name prefix, then seeks one compliance edge from candidate `codebase_files` IDs. Returns verified, missing, and unlinked counts plus details; no overall verdict. Requested claims absent from the file are not checked. |
| `smell report` | Required path; optional output file and format. Runs check and verify, then embeds file text and smell name for every verified reference. Returns combined data and an overall verdict; output file contains pretty JSON data without the CLI envelope, regardless of stdout format. |
| `link` | Required source ID, optional claims vector. Rejects `--force` as unimplemented despite its skip-confirmation help text. Dispatches claims sequentially with enforcement `static`, empty methods, no summary; prints one JSON array only after all claims succeed. Empty claims yields an empty successful result. |

## Output, access, and filesystem boundaries

Check/report accept JSON, JSONL, and table and validate format before database
work. Verify/link always print JSON. The adapters call dispatch directly with
the configured pool; dispatch routing itself does not enforce daemon access
tiers. Check/verify/report are classified Admin and link Agent for the daemon
boundary. Database permissions remain a separate requirement.

A returned check/report verdict of `passed: false` does not make its CLI adapter
return an error: the outer envelope still says `success: true`. Operational
errors propagated by handlers become CLI errors. Consumers must inspect the
verdict/counts, not equate command execution with compliance.

A directly supplied file bypasses extension filtering. Directory scanning uses
15 configured extensions, skips dot-prefixed entries, `__pycache__`, and
`Acheron`; initial path/directory failures propagate. Child traversal errors,
entry errors, and unreadable/non-UTF-8 file contents are silently skipped.
`files_checked` counts collected paths before successful reads. Directory
symlinks are followed through `is_dir`; no visited-directory set, depth budget,
file-size bound, or total-work budget is present in this helper. Comment
recognition is prefix-based, not a language parser.

Verification generates modern root-scoped and legacy/basename candidate keys.
The query uses membership in all candidates and `LIMIT 1` without ranking them;
modern-first vector order does not establish preferred query selection. File
claims cannot be satisfied by symbol-collection edges. Lookup of a matching
smell also uses `LIMIT 1` without ambiguity rejection.

## Verdict and embedding limitations

The report passes when the static check passes, there are no unlinked claims,
and no probe has boolean `pass: false`. Missing smell nodes are excluded from
that formula. File-read and embedding errors produce probe `pass: null`, which
also does not fail it. These are source-confirmed gaps requiring isolated
runtime reproduction and disposition before audit completion.

The embedding client is constructed even with no verified references. Each
probe reads the entire file before keeping its first 8,000 characters; embeds
that text as retrieval passage and the smell name as retrieval query; uses a
fixed cosine threshold of 0.5, with no calibration established by this review.
Report file writes replace existing contents without atomic replacement or a
same-input guard. Failure to write the report propagates before stdout output.

## Linking and partial effects

The handler trims the source ID, requires exactly one slash and nonempty parts,
and reads the source document before resolving a smell. CS-number inputs use a
case-insensitive numeric parser and a zero-padded smell-key prefix lookup;
other inputs use an exact key. Enforcement is validated against five allowed
values. No source-collection schema restriction is applied here.

Edge identity includes a 40-character readable prefix and eight hex characters
of SHA-256 over null-separated source collection, source key, and smell key.
Insertion conflict is treated as `already_exists: true` without reading back the
existing edge or updating its enforcement/methods/summary. The returned metadata
therefore describes this request, not necessarily stored metadata. Endpoint
reads and insertion are not a transaction. Multiple claims have no rollback:
a later failure can leave earlier edges committed while preventing the final
success array from being printed.

## Remaining validation

Reproduce missing-definition, unavailable-probe, and unreadable-file verdicts
with private fixtures. Verify partial-link outcomes and conflicting stored
metadata. Exercise symlink/traversal bounds and legacy-key ambiguity. Establish
whether policy violations should affect process status separately from
operational errors. These source candidates do not prove production incidents
or satisfy the broader retrieval, provenance, and recovery acceptance criteria.
