# Rust test fixture inventory (#12, #20, #47)

Scope: all 25 top-level `crates/*/tests/*.rs` targets at `979fe97`, plus
the service-free `filesystem_policy` target added in PR #50 and viewer
`process_lifecycle` added for #52 and CLI `daemon_shutdown` for #51 (28 total), and
source-unit modules containing database client/pool construction. This inventory
records resource boundaries and maintained execution, not exhaustive behavioral
coverage of every production component.

## Integration targets

| Targets | Resource boundary | Maintained execution |
|---|---|---|
| `arango_crud`, `arango_index`, `arango_query`, `arango_transport`, `arango_cache`, `graph_loader`, `graph_contract`, `cursor_lifecycle`, `transaction` (core) | `with_temp_db`, directly or through `common::with_tasks_db`; private cursor proxies where needed. Transport's `_system` client uses the fixture socket for database listing. | Strict disposable database CI |
| `file_identity`, `codebase_lifecycle` (CLI) | Disposable databases, temporary source trees, synthetic embedding/analyzer mocks; interruption tests kill only fixture processes. | Strict disposable database CI |
| `retrieval_benchmark`, `search_handler_benchmark` (core) | Ignored opt-in benchmarks using disposable databases, synthetic vectors and explicit strict mode. | Runner `--benchmark` / `--handler-benchmark`; not ordinary CI performance gates |
| `proto_types` (proto); `pipeline`, `config_integration` (core) | Serialization/configuration and service-free contracts; temporary configuration files. | Rust CI service-free targets |
| `embedding_client`, `extraction_client`, `training_client` (core) | Types/configuration and explicitly nonexistent temporary sockets. PR #43 adds private training RPC mocks. | Rust CI service-free targets |
| `embedding_contract`, `search_admission`, `retrieval_quality`, `filesystem_policy` (core) | Private mocks, synthetic responses/vectors or frozen local evaluation artifacts. | Rust CI service-free targets |
| `ra_span_agreement`, `gopls_semantic`, `clang_cuda_probe` (core) | Temporary source fixtures; installed analyzer/toolchain prerequisites. CUDA parsing is not GPU inference. Missing prerequisites can skip; a known CUDA call-expression gap is explicitly ignored. | Workstation opt-in; not evidence of complete language coverage |

The first two rows comprise all eleven database-dependent integration targets in
the maintained normal runner. Its separate missing-socket probe requires strict
setup failure rather than a silent pass. The runner starts a new Unix-only server,
uses fresh data/configuration, sanitizes endpoint environment and cleans its own
process groups. A caller manually setting `ARANGO_SOCKET` is still responsible
for supplying a separate test server; a unique database name alone is not server
isolation.

The viewer `process_lifecycle` target runs in the service-free CI step. It starts
only the just-built private viewer and synthetic child scripts, uses ephemeral
loopback HTTP, signals only its owned viewer and verifies descendant cleanup.
It never discovers an installed HADES executable or database endpoint.

The CLI `daemon_shutdown` target also runs in service-free CI. It starts the
just-built daemon with cleared environment, a temporary configuration and explicit
private Unix database sockets. SIGINT/SIGTERM cover idle exit and draining an
ingestion reservation blocked on a synthetic database response. The mock rejects
admission before any ingestion child or job insertion. This does not yet cover
daemon shutdown with a running ingestion child.

The database-gated `codebase_lifecycle` target additionally exercises actual daemon
shutdown with a running ingestion child held at a synthetic embedding response.
Both SIGINT and SIGTERM must reap that child, persist failure in the disposable
database and preserve the previous graph. Its private failure guard verifies the
fixture's unique source path before signalling any recorded process group.
The same target exercises an actual authenticated MCP endpoint serving two private
databases. Concurrent requests compete for the final ingestion slot; exactly one
wins. It counts actual daemon children, rejects excess and cross-database duplicate
tree requests, and checks both child reaping and per-database outcomes on shutdown.

## Source-unit fixtures

- `codebase_ingest`, `codebase_persist`, `codebase_prune`, `codebase_retire` use
  `with_temp_db` for database writes. The purge-failure fixture uses a private
  mock. The runner executes `commands::codebase_` with strict prerequisites.
- `dispatch` accepted-AQL fixtures previously used normal endpoint discovery and
  accepted arbitrary backend failure. #47 changes these to deterministic private
  cursor mocks, exact response/request assertions and bounded execution.
- Rejected dispatch commands and `service`/`daemon` authorization fixtures now
  use private mocks and assert no database request was received. Their safety
  does not depend on the guard under test continuing to work.
- `db::pool` and transport configuration/auth-header tests construct clients but
  do not issue requests. Transport body-limit tests use private sockets.
- MCP tests cover in-process HTTP/auth/body handling and database allowlist pool
  selection. The reviewed allowlist paths construct cached pools; the HTTP
  fixtures initialize MCP. PR #50 also exercises the actual smell-report tool
  through a cached private cursor mock.

Remaining limits: source scanning is a reviewed snapshot, not an automated proof
that future fixtures are isolated. Analyzer prerequisite skips and ignored probes
must be reported separately from passing tests. This inventory does not certify
all parser, adapter, provisioning or production ACL behavior.
