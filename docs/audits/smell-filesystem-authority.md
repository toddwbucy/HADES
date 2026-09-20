# Smell filesystem authority (#49)

## Confirmed boundary failure

A private service-layer fixture at `979fe97` used `ConnectionPolicy::agent_only()`
with no permitted ingest roots. `smell.report` succeeded and returned a matching
line from a synthetic local file. The fixture supplied one forbidden-pattern
record through a private database socket and an unused private embedding listener.
No production file, database or inference service was contacted. Disclosure
required an authorized caller and a matching configured pattern; this is not
evidence of unauthenticated access or observed production disclosure.

The shared command tier exposed disk-scanning `smell.check`, `smell.verify` and
`smell.report` to agents. The recursive scanner also follows symlink descendants.
A top-level path check alone would not contain all reads.

## Implemented boundary

All three filesystem commands require Admin authority before dispatch. Trusted
local CLI behavior remains available. The MCP `smell_report` tool now dispatches
`smell.stored_report`, an Agent command that reads only the selected database.
No canonicalization, stat, file read or embedder connection occurs in that handler.

The new command matches exact stored `codebase_files.path` or a full file `_id`,
then follows recorded `compliance_edges` to `smell_specs`. An existing full file
ID takes precedence over identical relative paths; absent IDs fall back to exact
stored-path matching. Repeated relative paths
across roots retain distinct file IDs. Missing associations yield an empty result,
which does not certify that source is free of smells. Symbol-level associations
and fresh disk analysis are outside this file-report contract.

Input is limited to 4096 bytes. The query requests at most 101 rows to return 100
with an explicit truncation flag; names and enforcement strings are bounded in
AQL. Streaming cursor ownership limits each response to 256 KiB, accumulated rows
to 1 MiB, and server query memory to 32 MiB, with the shared cursor lifetime and
cancellation cleanup. Database ACLs and the MCP database allowlist still apply.

## Maintained verification

- `filesystem_policy`: deny all three agent scan commands for a synthetic file
  and a directory with a symlink descendant; assert no database requests. Verify
  stored-only lookup, parameter binding, truncation, invalid input and oversized
  response rejection, plus successful trusted Admin scanning.
- MCP unit contract: invoke the actual tool under Agent policy with a private
  cached pool and assert its stored-graph response and bound database request.
- `graph_contract`: execute the real AQL against disposable collections with
  identical paths across roots, full file IDs, ID/path collisions, absent IDs
  falling back to paths, absent paths and quoted input.

The service-free and real database fixtures run in their existing isolated CI
jobs. These checks do not certify other path-taking commands or replace the
broader security workstream. No deployment is included.
