# CLI database response-shape baseline

At runtime source `d751f7e`, a private Unix HTTP peer and the actual isolated
CLI reproduce false success for malformed HTTP 200 database responses. The
probe runs under bubblewrap with an unshared network and explicit fixture
sockets/configuration; no live database is discovered or contacted.

| Command / response | Actual result |
|---|---|
| `db databases`, `{}` | Exit 0, success envelope, empty database list |
| `db databases`, valid empty `result` array | Exit 0, empty list (control) |
| `db export fixture`, `{}` | Exit 0, reports zero exported documents |
| Export with empty `result` and `hasMore:false` | Exit 0, empty export (control) |
| Export with one document, string `hasMore:"true"`, cursor ID | Exit 0, reports one exported document; deletes cursor without requesting another page |

`run_databases` defaults an absent/non-array result to an empty vector.
`run_export` silently skips non-array result fields and defaults missing or
non-boolean `hasMore` to false, on initial and subsequent pages. A malformed
successful response can therefore be indistinguishable from a genuinely empty
result or completed export. This is a P2 protocol-correctness finding, not an
observed production data loss or a claim that normal ArangoDB emits these shapes.

[Retained evidence](cli-db-response-shapes-result.json) includes requests,
responses, exit codes, output and binary/source identities. Replay with
`python3 docs/audits/repros/cli_db_response_shapes.py --binary <isolated-hades>`;
bubblewrap/user namespaces are required. The mock owns only a temporary socket.

Remediation should require typed result/pagination fields, reject invalid cursor
state before claiming completion, and preserve best-effort cursor cleanup on
errors. Add valid empty/multiple-page controls and malformed first/later-page
cases. Streamed output may already contain earlier valid rows on a later failure;
that must terminate nonzero rather than report a complete export. This baseline
does not establish atomic replacement of an export destination or resolve the
remaining 80-leaf command-contract audit.


## Remediation verification

The database list now requires an array of strings. Export validates every page's
result array, document objects, boolean pagination and safe/stable cursor ID
before emitting that page. A missing cursor with further pages is an error;
a changed safe ID is retained alongside the original for cleanup. Unsafe IDs
are never interpolated into request paths. Output is flushed before reporting
completion. Errors still trigger best-effort cleanup for known safe cursors.

[Separate remediation evidence](cli-db-response-shapes-remediation.json) records
the original five cases on the fixed binary: all three malformed responses now
exit 1 without success output, and both valid empty controls still exit 0. The
original baseline is unchanged. Four actual CLI contract tests pass against
private Unix peers, covering malformed lists/pages, missing/unsafe/changing IDs,
valid empty/multiple-page exports, partial output on later failure, and cursor
cleanup after output failure (`/dev/full`). The new service-free target is wired
into ordinary Rust CI; focused all-target Clippy and formatting checks pass.

These contracts verify selected response handling, not every DB command, complete
transport validation, export cancellation on process death, atomic output-file
replacement, or a production deployment. Existing output files are still opened
before the cursor request and streaming output can contain earlier valid pages
when a later operation fails.
