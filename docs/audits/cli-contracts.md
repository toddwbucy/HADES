# Native CLI inventory and embedding response caps

Source/build cut: `7480530e0a0bba91286285a478701a321a090c14`.
An isolated offline `cargo build -p hades-cli --bin hades` completed successfully
with one Cargo job and lowered priority. The resulting binary was invoked only
with `--help`, before configuration loading in `main.rs`.

The [inventory](cli-help-inventory.json) retains the binary SHA-256 and complete
help output for **93 nodes / 80 visible leaf commands**. Every invocation exited
zero with no stderr, using an empty temporary home/cwd and an allowlisted
environment without HADES/database credentials. No operational command, service
request, GPU query or database connection was executed.

## Declared command tree

| Root command | Leaves | Leaf paths below root |
|---|---:|---|
| `status` | 1 | `(direct command)` |
| `orient` | 1 | `(direct command)` |
| `extract` | 1 | `(direct command)` |
| `ingest` | 1 | `(direct command)` |
| `link` | 1 | `(direct command)` |
| `db` | 33 | `query`, `aql`, `list`, `stats`, `recent`, `health`, `check`, `purge`, `create`, `delete`, `collections`, `databases`, `create-database`, `truncate`, `drop-collection`, `count`, `get`, `insert`, `update`, `export`, `create-index`, `index-status`, `graph create`, `graph list`, `graph drop`, `graph traverse`, `graph shortest-path`, `graph neighbors`, `graph materialize`, `schema init`, `schema list`, `schema show`, `schema version` |
| `embed` | 6 | `text`, `service status`, `service start`, `service stop`, `gpu status`, `gpu list` |
| `codebase` | 7 | `ingest`, `update`, `stats`, `validate`, `prune-orphans`, `drift`, `retire` |
| `task` | 18 | `create`, `list`, `show`, `update`, `close`, `start`, `review`, `approve`, `block`, `unblock`, `handoff`, `handoff-show`, `context`, `log`, `sessions`, `dep`, `usage`, `graph-integration` |
| `smell` | 3 | `check`, `verify`, `report` |
| `graph-embed` | 4 | `train`, `embed`, `neighbors`, `update` |
| `schema` | 1 | `apply` |
| `tools` | 2 | `status`, `install` |
| `daemon` | 1 | `(direct command)` |

This exhaustively enumerates visible help children at the pinned build. Hidden
options/aliases, valid argument combinations, parameter forwarding and behavior
are not proven by help text. The parser counts exclude the recursive `help`
command itself. The [walker](repros/cli_help_inventory.py) follows only declared
command names and always appends `--help`; it bounds count, depth, per-call time
and accepted output size. It does not execute a shell command assembled from help.

Global `--database` has alias `--db`; `-g` is GPU, not graph. Source also declares
hidden `--resolved-config-fd` for the daemon's sealed child configuration. That
internal flag is absent from ordinary help and must not be inferred absent from
the source. The sealed configuration's security/lifecycle evidence is recorded
in [ingest ownership](ingest-job-ownership.md).

The [daemon/MCP inventory](api-command-inventory.json) has 59 wire commands and
18 curated MCP methods. These counts are not expected to equal the native CLI:
service control, tool installation, graph training, schema application, file
extraction and many graph maintenance operations are CLI surfaces. A matching
spelling is not evidence that defaults, validation, result shape or authority
are identical. The [transport report](api-boundaries.md) traces their envelopes.

## Embedding client response-cap call sites

At this source cut, `EmbeddingClient::connect_at` uses default configuration and
initializes `response_limit` to `None`. Its HTTP response collection uses
`usize::MAX` unless the caller opts in. The following production source call sites
were read; test constructors are separate and do not set production defaults.

| Caller | Operation | Explicit embedding response cap |
|---|---|---|
| core `dispatch::handlers::db_query` | Query vector for bounded search | 256 KiB before embedding |
| core dispatch status | Metadata probe inside a five-second timeout | None |
| core dispatch `embed_text` | One passage vector | None |
| core dispatch smell report | Embedding verified references | None; Admin filesystem path |
| CLI `embed_mgmt::run_embed_text` | One passage vector | None |
| CLI embed service status/start and GPU status | Provider info/health probes | None |
| CLI `codebase_ingest::run_phase` | Code windows/batches and provider info | None |
| CLI document ingestion | Pipeline extraction/embedding batches | None |

`with_response_limit` on ArangoDB readers/writers is a different transport and
does not bound embedding bodies. Likewise input batching/token windows do not
bound an arbitrary provider response. The search-specific cap cannot be cited as
a guarantee for the other paths. This is a static call-site limitation, not a
measured OOM or evidence of a malicious production provider. A proposed universal
cap needs legitimate response-size and batch/window compatibility evidence first.

## Replay and remaining review

```sh
python3 docs/audits/repros/cli_help_inventory.py \
  --binary /tmp/hades-audit-target/debug/hades \
  --source-revision 7480530e0a0bba91286285a478701a321a090c14
```

The caller-supplied source identifier is a build record, not embedded provenance
in the executable. Compare the retained binary and source-file hashes when
replaying; rebuilding may change artifact bytes. The help-only result does not
identify the deployed binary's source revision. The [source manifest](cli-contract-source.json)
binds CLI Rust files and embedding/dispatch source inspected here.

Remaining architecture acceptance: trace all 80 leaf argument/default/result paths,
exercise mismatched viewer envelopes, and measure/apply appropriate provider caps
where required. This inventory completes declared CLI navigation coverage, not
80 operational end-to-end tests or deployment certification.
