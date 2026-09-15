# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HADES — a knowledge-graph CLI over ArangoDB: semantic search, graph traversal,
document ingest, and multi-language codebase ingest. v0.3.0, workspace-wide.
Fully native Rust; the earlier Python CLI is gone (see the `main.rs` crate doc).
Python remains only for the GPU-bound services under `services/`.

## Build, lint, test

```bash
cargo build                      # debug -> target/debug/hades
cargo build --release
cargo fmt --all -- --check       # CI gate
cargo clippy --workspace --all-targets -- -D warnings
```

- The toolchain is pinned exactly in `rust-toolchain.toml` (1.98.1) and asserted
  in CI; `Cargo.toml`'s `rust-version` mirrors it. Bump both together and land
  `cargo fmt --all` plus clippy fixes in the same commit.
- `protoc` must be on PATH: `hades-proto`'s `build.rs` shells out through
  `tonic-build` to compile `proto/**`.

Tests fall into three classes, which is why CI does not run a bare
`cargo test --workspace`:

```bash
cargo test --workspace --lib --bins          # what CI gates on (unit tests, incl. the CLI binary)
cargo test -p hades-core --test pipeline     # one integration target
cargo test -p hades-core --lib config::      # filter by path within a crate
```

- Unit tests live in `src/` next to the code (~630 `#[test]`/`#[tokio::test]`
  across the workspace).
- `crates/*/tests/*.rs` integration targets need external resources. They
  self-skip when the *socket* is missing (`ARANGO_SOCKET` names it; the default
  `/run/arangodb3/arangodb.sock` is the system install's, and a user-level
  arangod binds elsewhere) and honor a strict flag that turns a skip into a
  panic (`ARANGO_TESTS=1` for the `arango_*` and `graph_loader` targets;
  `HADES_CUDA_FIXTURE` for the libclang probe). The convention is specified in
  `docs/specs/workstation-specific-tests.md`. A missing *database* is not a
  skip: `arango_transport`, `arango_crud`, `arango_index` and `arango_query`
  still name `bident_burn` and a seeded `persephone_tasks`, neither of which
  exists, so they fail until the Persephone pass moves them onto
  `hades_core::test_support::with_temp_db` the way `arango_cache` already is.
- Service-dependent targets with no skip guard — `embedding_client`,
  `extraction_client`, `training_client` — and analyzer-dependent ones
  (`clang_cuda_probe`, `gopls_semantic`, `ra_span_agreement`) fail on a bare
  machine by design. Run them only where the service or binary exists.
- End-to-end graph invariants: `HADES_BIN=target/release/hades
  ./scripts/bident_burn_smoke.sh` (needs live ArangoDB + embedder; uses a
  dedicated `bident_burn_smoke` database). `scripts/cli_audit.sh` is the
  weaker "does every command run" companion.

## Architecture

Five workspace crates, deliberately layered:

- **hades-cli** — the `hades` binary. 14 top-level command groups
  (`status`, `orient`, `extract`, `ingest`, `link`, `db`, `embed`, `codebase`,
  `task`, `smell`, `graph-embed`, `schema`, `tools`, `daemon`). One module per
  group under `src/commands/`; large groups split further
  (`codebase_ingest.rs`, `db_search.rs`, …). `hades ingest` is the single front
  door for ingesting any tree: it routes every file by extension and runs both
  pipelines under one envelope (see *Ingest paths*). `hades codebase ingest` is
  the code-only path it delegates to, kept for single-language work.
- **hades-core** — all logic. Config, ArangoDB client, dispatch, daemon
  service, code analysis, chunking, pipeline, graph/schema, Persephone clients.
- **hades-proto** — generated gRPC stubs for the extraction and training
  services. The embedding service is *not* gRPC (see below).
- **hades-prefetch** — safetensors serialization, edge splitting, negative
  sampling, mmap graph access, and the async double-buffered prefetcher that
  feeds RGCN/GraphSAGE training.
- **hades-frontend** — the `hades-viewer` binary (WebGL graph viewer). Depends
  on **no** other HADES crate on purpose: it shells out to `hades` and consumes
  the CLI's JSON/jsonl contract. Keep it that way.

### One dispatch layer, two front doors

`hades-core::dispatch` owns the command set (`DaemonCommand`) and its handlers;
`dispatch()` matches exhaustively with no catch-all, so adding a variant without
a handler is a compile error rather than a runtime `NOT_IMPLEMENTED`.
`hades-core::service` sits above it and owns everything between raw payload and
response envelope: parsing, session resolution, tier authorization, error
mapping. Transports (the Unix-socket daemon listener, the LAN MCP endpoint in
`commands/mcp_server.rs`) own only framing, and hand each payload to
`handle_request` with the `ConnectionPolicy` their trust boundary implies.

Authority comes from the transport, never from request bytes: a local Unix peer
gets an admin ceiling, a network transport gets an agent ceiling, and a
request's `session` field may only restrict itself downward. Every command
carries an `AccessTier` (`Agent` / `Admin` / `Internal` / `Provisioning`) via
`DaemonCommand::access_tier` — raw AQL, writes, and DDL are `Admin`. Adding a
command means choosing its tier. See `docs/daemon-protocol.md` and
`docs/model-operation-vocabulary.md`.

`Provisioning` covers `db.create_database`, `db.schema_init`, `ingest.start` and
`ingest.status`: what a remote agent needs to build its own graph, and nothing
more. It is deliberately *not* folded into `Admin`, which would have handed over
raw AQL and `db.purge` along with it. A transport granting it must supply
`ProvisioningLimits` — the database name prefixes it may create under and the
directories an ingest may read. The MCP endpoint enables it only when both
`--mcp-db-prefix` and `--mcp-ingest-root` are given; one alone leaves it off, and
an empty prefix is refused at startup because it is a prefix of every name.

### Ontology is data, not code

Each database stores its own edge definitions and named graphs as documents in
its `hades_schema` collection, loaded at runtime by `graph::runtime_schema`.
There is **no compile-time fallback** — a database must be seeded (e.g.
`hades db schema init --seed empty`) before `graph::load` works. `hades schema` applies declarative
YAML at bootstrap; `hades db schema {init,list,show,version}` inspects the live
collection. Don't add domain-specific schema to Rust.

The one compiled-in piece is `db::collections::CollectionProfile` — the generic
`metadata`/`chunks`/`embeddings` triple, its foreign-key field name, and the
`query_task`/`passage_task` pair naming which Jina LoRA adapter its vectors were
written with. Two profiles exist: `default` (`documents`/`chunks`/`embeddings`,
fk `parent_key`, `retrieval.query`/`retrieval.passage`) and `codebase`
(`codebase_*`, fk `file_key`, `code`/`code`, plus symbols and one edge collection
per relation). The task pair lives on the profile so a search cannot query a
collection with the wrong adapter.

### ArangoDB access

`db::ArangoClient` speaks HTTP over a Unix socket, falling back to TCP.
Discovery order: configured `readonly`/`readwrite` socket → the direct socket at
`/run/arangodb3/arangodb.sock` → `http://{host}:{port}`. Both configured
sockets normally point at the same file; read/write separation is an ArangoDB
ACL matter, not a HADES one. `ArangoPool` layers health and pooling on top;
`db::query`, `db::crud`, `db::vector`, `db::index`, `db::cache` are the call
surfaces. `db::keys` canonicalizes raw identifiers into ArangoDB keys (strips a
trailing `v\d+`, maps `.` and `/` to `_`) — go through it rather than
hand-rolling keys.

### Ingest paths

**The extension decides, in one place.** `core::ingest_routing::route_for`
returns `Code(Language)`, `Document`, or `Unrouted`, and it is the only thing
that chooses a pipeline. Code wins any tie, because an analyzer produces symbols
and edges extraction cannot. `Unrouted` files are *reported* in the ingest
envelope rather than dropped — silently skipping half a mixed tree is the failure
this module exists to end. Adding an extension to `DOCUMENT_EXTENSIONS` means
teaching the extraction service first: it accepts only `md`, `markdown`, `txt`,
`text`, `rst`, `pdf`, `tex`, `gz`, and claiming a file it will refuse is worse
than declining it.

Ingest is incremental by content hash. Both code file rows and document rows
carry `content_hash`; an unchanged file is skipped, a changed one is re-ingested
in place with `overwriteMode=replace`. So re-running `hades ingest` (or MCP
`ingest_start`) over a tree is the resync — no separate command, and no need to
name what changed.

Documents: `pipeline::Pipeline` runs extract → chunk → embed → store, with
single and batch modes; `batch/` adds checkpointed state, rate limiting, and
progress so `--resume` works.

Code: `code::analyze*` dispatches per file and records a provenance tier on
every artifact — `semantic` (rust: `syn` + rust-analyzer; go: tree-sitter +
gopls; python: `rustpython-parser`; C/C++/CUDA: libclang, optionally reading
`compile_commands.json`), `structural` (registered tree-sitter grammars), or
`text`. Re-ingest refuses to replace a higher tier with a lower one unless
`--allow-analysis-downgrade` is passed. libclang is `dlopen`ed at runtime, so
its absence degrades rather than breaking the build.

Chunking: `chunking/` holds only client-side strategies (sentence, sliding
window, token) plus `code::AstChunking`. **Late chunking is server-side** —
token-level tensors are too large to cross the wire, so windows are encoded and
pooled next to the model (`persephone::embedding::embed_late_chunked` +
`services/embedding/jina_v4.py`). `HADES_DISABLE_LATE_CHUNKING=1` falls back to
per-chunk embedding. Don't reintroduce a client-side late-chunking module.

Window size is **read from the backend, never hardcoded**: `embed_window_chars`
asks `/v1/models` for `max_seq_length` and multiplies by a 2.0 chars-per-token
floor, falling back to 12,000 characters only when the field is absent. That is
what lets the same client run against a 32,768-token profile and an
11,900-token one without a config change. A chunk that cannot fit even alone is
partitioned out and retried per window rather than failing the file, and window
positions map back to file-order chunk indices through an explicit
`chunk_indices` list — not `base + offset`, which put vectors under the wrong
keys with matching counts.

## Configuration

Priority, highest first: CLI args (`--db`, `--gpu`) → env (`ARANGO_*`,
`HADES_*`) → YAML → compiled-in defaults.

YAML search order: `$HADES_CONFIG` (must exist if set) → `./hades.yaml` →
`./core/config/hades.yaml` → `~/.config/hades/hades.yaml` →
`/etc/hades/hades.yaml`. Note that the repo's own `config/hades.yaml` is **not**
in that list — it is the documented reference/install template, so a run from
the repo root uses compiled-in defaults unless you point `HADES_CONFIG` at it.

There is no default database: `effective_database()` errors unless `--db`
(alias of `--database`) or `HADES_DATABASE` is set. Passwords come from
`ARANGO_PASSWORD` only.

## External services

| Service | Default endpoint | Transport |
| --- | --- | --- |
| ArangoDB | `/run/arangodb3/arangodb.sock` | existing socks proxy — do not create new socket infrastructure |
| Embedder (Jina V4) | `http://localhost:8087/v1` | OpenAI-compatible HTTP, contract in `docs/persephone-embedding-api.md` (PE-API v1) |
| Extractor (Docling) | `/run/hades/extractor.sock`, override `HADES_EXTRACTOR_SOCKET` | gRPC |
| Trainer | `/run/hades/training.sock` | gRPC |
| HADES daemon | `/run/hades/hades.sock` | length-prefixed JSON, `docs/daemon-protocol.md` |

The embedder's client-side endpoint (`embedding.service.socket` or
`HADES_EMBEDDER_SOCKET`) and the service's own listen address (`embedder.conf`)
are two separate settings that must agree. The pre-PR-#70 gRPC embedder socket
at `/run/hades/embedder.sock` was removed deliberately — do not reintroduce it.
Weaver's per-agent embedders under `/run/weaver/...` belong to WeaverTools;
leave them alone.

The extractor's compiled-in default lives under `/run/hades`, which a user-level
deployment cannot write, so `ExtractionClient::from_env` honors
`HADES_EXTRACTOR_SOCKET`. The path must be absolute — a relative `unix://` is
refused rather than resolved against the cwd. The unit runs with `PrivateTmp`,
so it cannot see `/tmp` paths a client hands it; put test trees under `~/.cache`.

### Embedder load profiles

The model is ~9 GiB of weights and cannot run on CPU, so exactly one embedder is
live at a time and the card it sits on sets the sequence ceiling. Profiles live
at `~/.config/hades/embedder-profiles/gpu{0,1,2}.conf` (templates and the
switcher script are in `deploy/systemd/`, documented in its `README.md`), and are
switched with `hades-embedder-profile <name>` against the templated user unit
`hades-embedder@.service`:

| Profile | Card | Ceiling | Notes |
| --- | --- | --- | --- |
| `gpu0`, `gpu1` | RTX A6000, 48 GiB | 32,768 tokens | the documented Jina v4 length; batch 1; 28 GiB free-VRAM floor so it refuses a card WeaverTools is using |
| `gpu2` | RTX 2000 Ada, 16 GiB | 11,900 tokens | bisected with real documents; the earlier 15,000 came from a synthetic probe and is not reachable |

Every profile binds the same port, so switching cards changes nothing for a
client: it re-reads the ceiling from `/v1/models`. Two profiles cannot be live at
once — the second fails to bind rather than loading a second copy of the model.
Over-ceiling input is refused by name (`PE_INPUT_TOO_LARGE`) rather than
truncated and reported as success; pre-chunking oversized documents is separate,
unbuilt work.

## Critical rules

1. **Production data is sacrosanct.** HADES connects as the dedicated `hades`
   ArangoDB user and has *no* compiled-in writable-database allowlist — the
   ArangoDB ACL grants on that user are the authoritative gate (production
   research databases are granted `ro`). The allowlists that do exist in code
   are unrelated: MCP read scoping and the unparsed-extension list.
2. **No database is a default, including for HADES's own state.**
   `bident_burn` held the `persephone_tasks` kanban and was dropped on
   2026-09-14 with the other superseded databases. The `task` commands run
   against whatever `--db` names and need both a database and the
   `persephone_tasks`, `persephone_logs`, `persephone_handoffs` and
   `persephone_edges` collections, which nothing in the binary creates: the
   only creator today is the `hades db create persephone_*` loop in
   `scripts/bident_burn_smoke.sh`. Treat Persephone as deferred work rather
   than a missing file. `effective_database()` errors without `--db` or
   `HADES_DATABASE` by design, so do not reintroduce a hardcoded target.
3. **A write test gets a database created for the test**, never a corpus
   somebody is querying. The pattern is `hades_core::test_support::with_temp_db`
   behind the `test-support` feature, enabled through a dev-dependency: a
   per-process name, delete before create, fixture collections from
   `CODEBASE.all_collections()`, dropped even when the test panics. One
   definition rather than a copy per crate, because two harnesses to keep in
   step is the defect this file's other rules are about.
   `scripts/bident_burn_smoke.sh` is the script-level version and is weaker
   by design: it creates `bident_burn_smoke` once and truncates on later runs,
   since there is deliberately no `drop-database` command (#118).
4. **A database seeded with `db schema init --seed empty` has an empty
   `relation_order`**, and that seed is what the MCP `db_schema_init` writes.
   `graph::loader` scans exactly the collections `relation_order` names, and
   nodes are discovered only through edge scans, so it returns a graph with no
   edges *and* no nodes. `graph-embed train` then fails at the tensor step with
   "graph has no edges", after the loader has already returned that empty
   graph. `graph-embed update` fails earlier on a fresh machine, at its
   checkpoint preflight ("no trained model found"), and reports success over
   the empty graph only where a checkpoint from some earlier run already sits
   in the shared default `--checkpoint-dir` of `/tmp/hades-train`. Neither
   failure names the schema, which is why this is a rule. Apply a schema file
   when creating a database: `config/schemas/codebase.yaml` for
   the universal code layer, or a domain file carrying it plus its own
   relations (`services/adapters/weavertools/schema.yaml` is the worked
   example). There is no MCP operation for that yet, so a remotely
   provisioned graph needs the file applied from the CLI before training.

## Conventions

- Edition 2024, resolver 2; every dependency version lives in the root
  `Cargo.toml` `[workspace.dependencies]` and crates use `dep.workspace = true`.
- `thiserror` for library errors, `anyhow` for the binaries. Handler and
  dispatch errors are typed enums that map to stable string error codes.
- `tracing` for all logging — never `log` or `println!` for diagnostics.
- CLI output contract: the JSON envelope `{success, command, data, timestamp}`
  on stdout, logs and progress on stderr. Every command takes
  `-f json|jsonl|table`. Batch commands must report their real outcome via
  `print_output_with_success` — a batch that ran is not a batch that succeeded.
- `CHANGELOG.md`: each PR adds its entry under `[Unreleased]` in the same
  commit, categorized Keep-a-Changelog style.
- Comments in this codebase explain *why* a decision was made, often citing the
  issue number. Match that when touching the same code.

## Docs worth reading before changing related code

`docs/daemon-protocol.md` (wire format, tiers) ·
`docs/model-operation-vocabulary.md` (the closed agent-facing operation set) ·
`docs/persephone-embedding-api.md` (PE-API v1) ·
`docs/codebase-graph-ontology.md` (code graph primitives) ·
`docs/declarative-schema.md` (schema lifecycle) ·
`docs/mcp-deployment.md` (LAN endpoint, GPU preflight) ·
`Bastion/graph-methodology.md` (the axiom-gated graph methodology the schema
encodes) · `deploy/systemd/README.md` (user units, linger, load profiles) ·
`README.md` (install, ArangoDB user setup, systemd).
