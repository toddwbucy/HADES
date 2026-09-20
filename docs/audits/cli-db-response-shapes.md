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
