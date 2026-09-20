# Transport authority and response contracts

Reviewed source: `66ebc4d6c6c88c41d5e5c5ba3371d3521b88aa10`, September 20, 2026.
The [machine-readable inventory](api-command-inventory.json) maps all 59
`DaemonCommand` wire names to tiers and MCP tool exposure: 37 Agent, 13 Admin,
six Internal and three Provisioning commands; 18 curated MCP tools. These counts
are declaration coverage, not proof of every handler's behavior. The retained
[extractor](repros/api_inventory.py) reads source only and asserts matching sets
of wire names, tier arms and curated tool methods. Run it from the repository root:

```sh
python3 docs/audits/repros/api_inventory.py > /tmp/hades-api-inventory.json
```

## Boundary matrix

| Entry | Authority | Success/error contract | Reviewed source |
|---|---|---|---|
| Native CLI | Operator's configuration and database credentials; not the network Agent ceiling | Common JSON output is `{success,command,data,timestamp}`. Early `anyhow` failures can terminate nonzero with stderr, without a JSON error envelope. Batch paths can explicitly set `success:false`. Table output is not JSON. | `hades-cli/src/main.rs`, `commands/output.rs` |
| Unix daemon | Transport constructs `local_admin()`; request can self-restrict to Agent | Four-byte big-endian length then `DaemonResponse`: `request_id,success,data,error,error_code`. Oversize request gets an error then close; incomplete/idle frames can close without a response. | `commands/daemon.rs`, core `service.rs` |
| MCP HTTP | Bearer authentication, served-database allowlist/prefixes, Agent ceiling with optional scoped provisioning | Handler envelope serialized inside text content; `isError` follows `resp.success`. Database-selection rejection is instead plain-text error content; serialization failure uses MCP protocol error. HTTP acceptance is not command success. | `commands/mcp_server.rs` |
| Viewer | Its own host/session checks; backend uses explicit database CLI argv under operator credentials | Successful routes return graph/snapshot/delta payloads. Backend failures use HTTP 502 and `{error}`; overload uses 503; whole-request timeout uses 504 with text. Separate routes/middleware supply 400/401/404/421. | frontend `server.rs`, `assemble.rs`, `backend_process.rs`, `contract.rs` |

The CLI and viewer are not simply alternate encodings of every daemon command.
The viewer uses CLI exports/AQL as well as graph reads; it is not restricted to
the 18 MCP tools. The native CLI contains administration, model/tool management,
training and ingest paths outside `DaemonCommand`. A complete CLI subcommand and
parameter-default parity review remains required.

## Shared service error mapping

`service::handle_request` parses, resolves authority, authorizes, dispatches under
a timeout and maps the outcome. `request_id` is echoed when parsed as a string;
malformed JSON cannot reliably supply one. Unknown commands and parameter
serde failures both map to `UNKNOWN_COMMAND`, not universally `INVALID_PARAMS`.

| Condition | Envelope error code |
|---|---|
| Invalid JSON / unsupported session string | `MALFORMED_JSON` / `INVALID_SESSION` |
| Command deserialization fails | `UNKNOWN_COMMAND` |
| Tier, creation prefix or ingest-root denied | `ACCESS_DENIED` |
| Invalid node ID, limit or handler parameter | `INVALID_PARAMS` |
| Missing node/document or database not-found error | `NOT_FOUND` |
| Missing/malformed structural vector or other query failure | `QUERY_FAILED` |
| Search admission exhausted / external service failure | `SEARCH_OVERLOADED` / `SERVICE_ERROR` |
| Dispatch timeout | `INTERNAL` |
| Native implementation absent | `NOT_IMPLEMENTED` with serialized command in `data` |

The last row is an exception to the response field comment saying `data` is null
on error. It is a source contract observation, not evidence that every command
currently reaches that branch. Consumers should test `success` and the error code
rather than infer success from non-null `data`.

## Provisioning and ownership

The three Provisioning commands are database creation, schema initialization and
asynchronous ingest start. MCP advertises them even when policy denies execution.
Agent-tier commands include task writes; Agent does not mean read-only.
Internal commands require Admin just as Admin commands do.

Shared policy requires both the provisioning grant and effective session equal to
the transport ceiling. Thus a local Admin connection self-restricted to Agent
loses provisioning. Prefix and canonical-root checks narrow database creation and
ingest admission; blank prefixes are rejected/ignored rather than granting all.
MCP's pool selection separately limits the default database, explicit extra names
and provisionable prefixes. These boundaries do not replace ArangoDB grants
([database access review](database-access-boundary.md)).

Canonicalization rejects ordinary symlink/parent escapes at the time of checking;
it is not a filesystem snapshot or a proof against later mutable-tree changes.
A successful ingest-start envelope acknowledges admission/child startup, not a
completed ingest. Clients must poll status and distinguish completed, failed and
recovery-required outcomes; [job ownership](ingest-job-ownership.md) retains the
executed failure/cancellation evidence.

## Validation and limits

At the pinned revision, all 22 `service::tests` passed with one test thread:
parsing/correlation, error mapping, transport ceilings, self-restriction, blank
prefixes and canonical root/symlink rejection. Denied dispatch cases use private
mocks and assert no request traffic. No live database endpoint was exercised.
All 19 `commands::mcp_server::tests` also passed: the actual 18-tool router,
private smell-report dispatch, token/host/database policy, SDK initialization and
body limits/deadline. These are CPU/private-router fixtures, not live network
certification. Source hashes in the inventory bind the reviewed transport files.

Viewer parsers currently inspect expected `.data` fields after backend exit-status
validation; they do not independently enforce `success:true` or a protocol version.
This is a compatibility limit requiring a private mismatched-backend fixture before
claiming an observed false-success defect. The snapshot contract has partial/count
metadata, but no negotiated version field. Existing viewer lifecycle tests do not
certify browser asset behavior or every payload shape.

The [provider review](provider-contracts.md) adds gRPC/HTTP status and declared
size/admission limits with a 78-test CPU replay. Outstanding: exhaustive native
CLI mapping/defaults, effective provider size/admission fixtures, viewer
malformed-envelope compatibility, and provisioning races
against mutable trees. This report completes the static daemon-tier/MCP exposure
inventory and shared envelope trace, not the whole architecture/security workstream.
