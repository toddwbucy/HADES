# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is `0.x` the API and CLI surface should be considered unstable;
breaking changes may land in any minor bump and are noted under **Changed** when
they affect operator-facing behavior.

## Convention

Every PR adds its entry to the `[Unreleased]` section in the same commit, under
one of: `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.
When a release ships, the `[Unreleased]` section is renamed to the new version
with the release date, and a fresh `[Unreleased]` is opened above it.

## [Unreleased]

### Security

- Reject WeaverTools adapter HTTP redirects before forwarding credentials or
  graph writes beyond the configured database endpoint (#66).

- Require a scoped ArangoDB signing key in the fresh-VPS harness and installation
  instructions; remove the package-authentication bypass recommendation (#60).

- Enforce atomic detached-ingest admission across served databases and overlapping
  source trees; fail closed on admission errors and retain slots through cleanup (#51).
- Own ingestion process groups through startup, request disconnects, PID-write
  failures, supervision panics and daemon shutdown; reap children before releasing
  admission, with private concurrency and descendant-cleanup contracts (#51).
- Capture ingestion output through bounded asynchronous pipes instead of shared
  temporary files; fail explicitly on overflow, malformed JSON and runtime expiry (#51).
- Persist ingestion phases with per-daemon ownership and bounded retries; report
  unowned unfinished records as requiring reconciliation, including after restart,
  while refreshing completed jobs to avoid a stale-status race (#51).
- Preserve resolved ingestion configuration and credentials through bounded sealed
  descriptors; reject invalid canonical paths before job-record serialization (#51).

- Bound viewer CLI child output and runtime, retain process admission through
  cancellation cleanup, and prevent child diagnostics from entering HTTP errors (#52).
- Bound viewer graph accumulation, JSON serialization, concurrent connections
  and request/socket lifetimes (#52).
- Handle viewer SIGINT/SIGTERM before discovery; cancel and reap owned child
  groups on shutdown, with real private HTTP/disconnect and router contracts (#52).
- Verify viewer cumulative graph budgets, HTTP 503 admission/readmission and
  whole-request deadlines; preserve a resource-limited slow-reader measurement (#52).
- Require Admin authority for filesystem smell scans; route the MCP smell report
  to bounded, database-only recorded associations (#49).

### Documentation

- Freeze reproducible, hash-bound passages for the private paper-research corpus;
  add a bounded preparation utility and source-offset/privacy contracts (#12).

- Confirm `Bastion/` as the paper-research audit corpus and record the private,
  versioned snapshot protocol without publishing draft contents (#12).

- Define candidate code-search and paper-draft research workloads, version-aware
  evidence judgments and remaining retrieval acceptance requirements (#12).

- Record the live daemon artifact/compiler provenance and explicit source-mapping
  limitation; refresh packaging and additional audit-remediation status (#12).

- Preserve the epic #12 baseline audit reports and reproductions, and add an
  evidence register distinguishing merged fixes, deployment state and remaining
  full-audit acceptance work.

### Changed

- Refresh epic #12 evidence dispositions, revision-bound coverage and remaining
  independent retrieval, deployment and operational acceptance work.

- Require coordinated trainer/client upgrades for the session-ownership RPC
  contract; legacy unowned requests fail closed (#42).

- Calibrate search admission against full-handler memory pilots; release normal
  MCP request reservations after cleanup acknowledgment, retaining the replay
  window only for cancellation. Add creation-cancellation and body-timeout
  regressions (#22).

- Enforce MCP's 16 MiB body limit before SDK parsing, including chunked requests;
  cap concurrent body handling and apply a 15-second body-read deadline (#22).
  The previous Axum extractor limit did not cover the SDK's raw body collector.

- Share a 64-connection cap across Unix daemon and MCP sockets, limit Unix
  response writes to 15 seconds, and expire MCP TCP connections after five minutes
  without deleting their resumable sessions (#22).

- Exact vector search folds bounded pages into top-K results with shared
  admission and bounded embedding, detail, and structural responses (#22).
  Invalid stored vectors and incompatible model/dimension metadata fail explicitly.

- Code-file keys now include the canonical ingest root and a full path digest,
  preserving distinct dotted, underscored, Unicode, long, and multi-root paths
  (#13). Ingest rejects legacy or conflicting ownership before replacement;
  drift rejects legacy comparisons. Existing corpora require the explicit rebuild
  and rollback procedure in `docs/code-file-identities.md`.

### Added

- Record the larger full-handler allocation pilot and near-limit MCP slow-reader
  memory measurement, using resource-limited isolated fixtures (#22).

- Add an opt-in disposable-database full-search-handler memory benchmark, including
  maximum result count/query text, hybrid/structural reranking, and serialization;
  cover MCP request/stream overload and session initialization/idle expiry (#22).

- Bound MCP sessions, retained requests, GET/resume streams, and serialized SDK
  messages; preserve active-request resume and clean up cancelled request caches
  independently of HTTP clients (#22). Oversized responses return an explicit
  `MCP_RESPONSE_TOO_LARGE` error. Completed-response replay remains unavailable
  with the installed SDK.

- Database clients can bound each HTTP response before JSON parsing; response
  bodies now share the request deadline instead of waiting indefinitely after
  headers. This supports the bounded retrieval work in #22.

- Gate CPU Python contracts with a hashed dependency lock and database workflows
  with a private resource-limited ArangoDB runner (#20). Replace named-corpus
  fixtures with disposable seeded databases, generate adapter fixtures locally,
  and cover the CLI graph/retrieval lifecycle with partial failure and retry.

- **`config/schemas/codebase.yaml`**, the universal code graph as data. `hades
  ingest` creates its collections and gharial edge definitions directly and leaves
  `hades_schema` holding at most the empty `meta` that `--seed empty` writes, so
  `relation_order` is `[]` and `graph::loader` scans nothing, returning a graph
  with no edges and no nodes. `graph-embed train` fails at the tensor step with
  "graph has no edges" after being handed it; `graph-embed update` fails earlier
  at its checkpoint preflight on a fresh machine, and reports success over the
  empty graph wherever an earlier run left a checkpoint in the shared default
  `--checkpoint-dir`. Neither failure names the schema. That was found in a
  domain graph and applies to every database seeded empty and filled by ingest,
  so the generic layer now has a file of its own.
  `services/adapters/weavertools/schema.yaml` stays as the worked example of a
  domain layer on top of it. Applied to a fresh `bident_v4`: 12 collections, 4 edge
  definitions, one named graph, `num_relations` 4 rather than 0.


- **`extraction.service.socket` in `hades.yaml`.** The extraction endpoint could
  only be named by `HADES_EXTRACTOR_SOCKET` or the compiled-in
  `/run/hades/extractor.sock`, which a user-level deployment cannot write. The
  variable was set in the daemon's unit file, so the daemon reached the extractor
  and a CLI run from a plain shell did not: `hades ingest` over a mixed tree
  finished its code half and returned `document_phase_error: failed to connect to
  extraction service`, with nowhere to send 83 documents. The embedder's endpoint
  has always been configurable; this is the same setting for the other service.
  Env still wins over YAML, an empty value is ignored rather than overwriting a
  working path, and `ExtractionClient::connect_at` takes the resolved value.


- **`services/adapters/weavertools/schema.yaml`**, the WeaverTools graph as data.
  `hades_schema` in that database held nothing but an empty `meta` document, so
  `relation_order` was `[]` and `num_relations` was `0`. `graph::loader` scans
  exactly the collections `relation_order` names, so structural training over a
  graph holding 13,000 edges would have loaded **zero** of them and reported
  success. The file registers all 18 edge definitions, the named graph, and the
  relation order, so the graph is reproducible from the repository and trainable.
  Applying it to the live database moved `num_relations` from 0 to 18.
- `via` and `tag` on the WeaverTools `Edge` record, and `via` in the edge key.
  `weaver-spu --seam--> weaver-harness` is declared three times in one PRD, once
  per contract papering a separate socket seam, and a key built from source,
  relation and target alone collapsed all three into one row. Three real seams
  became one, and the stored total came up two short of the corpus's 665 `edge:`
  declarations, which is how it was found. All 665 are now distinct.


- **Discovery on the MCP surface**, from a session that surveyed a fresh graph and
  reported it as the weakest part. `db_collections` enumerates collections with
  counts and types, and `graph_list` names the graphs with their edge definitions.
  Neither existed, `orient` reports collection *profiles* and so never named
  `codebase_symbols` or any edge collection, and the one call documented to
  enumerate (`db_list` with no collection) returned documents from the default
  profile instead. Both commands existed in the dispatch layer at `Internal` tier,
  which put the only two answers to "what is in this database" out of reach of the
  sessions that need them most; enumeration returns names and counts, never
  bodies, so both are now `Agent` tier.
- `db_list` takes a `fields` projection, and omits bulk text and vector fields by
  default. Three rows of one real corpus came to 53,555 characters because each
  carries the whole document's `full_text`, and eighty-three came to 2.8 million,
  which exceeded the caller's context and made the call unusable for the listing
  it was reached for. The same three rows are now 1,550 characters, and
  `fields: ["_key", "status"]` is 286.
- `db_schema_init` is exposed and moved to the `Provisioning` tier.
  `create_database` told the caller to seed with it next, and at `Admin` tier that
  instruction named a step no MCP client could take, so a provisioned client could
  create a database and not finish it.
- `services/adapters/weavertools/write_graph.py`, the half of the conformance chain
  that reaches a database. The extractor in that package computes assertions,
  terms, axioms and the `cites` edges joining code to the documents claiming it,
  and imports no database driver by design. Nothing consumed it, so `hades ingest`
  produced code nodes and document nodes with no edge between them: a graph that
  can find code and can find prose and cannot ask whether one satisfies the other.
  The writer attaches each `cites` edge to the `codebase_files` node HADES already
  created rather than to a private copy of the source list, so the result is one
  graph instead of three collections sharing a database. Node kinds get one
  collection each and relations one each, which is the notation the 2026-08-08
  graph used. The extractor's notes land in `wt_ingest_report`, because 48 sources
  owing a header and a malformed node id left in a terminal become a graph that
  reads as complete.
- The MCP server's `instructions`, which a client reads on `initialize` before it
  looks at any tool, now carry the three things that otherwise cost a round trip
  each: that `db_query` takes a collection *profile* and code and documents are
  embedded with different adapters, that traversal needs an explicit `graph` name
  because omitting it targets a `default` that will not exist, and that ingest is
  a job to poll rather than a call to wait on. The previous text predated all of
  this and said writes were "governed by database ACLs", which is incomplete now
  that a Provisioning tier exists and misleading on an instance running with
  ArangoDB authentication disabled.
- **An MCP client can build its own graph.** Three tools, `create_database`,
  `ingest_start` and `ingest_status`, behind a new `Provisioning` access tier that
  a transport can grant on its own. A local Unix peer has it by being admin; the
  network endpoint has it only when the daemon is started with both
  `--mcp-db-prefix` and `--mcp-ingest-root`, and then only inside those bounds.
  Empty bounds permit nothing rather than everything, and both refusals name what
  would have been allowed. The ceiling stays at Agent, so `db.aql`, `db.purge`,
  `db.insert` and `db.graph.drop` are still unreachable from the network: this
  grants two commands, not a promotion.
  `ingest_start` returns a job id instead of blocking, because ingests run for
  minutes and the transport caps a request at 60 seconds, so a blocking call
  would report failure for work that was still succeeding. Job rows live in the
  database being built, so they survive a daemon restart, and a row that never
  leaves `running` is reported as orphaned rather than as progress. Ingest paths
  are canonicalized before the bounds check, so `..` and symlinks cannot walk out
  of a permitted root.
- **One ingest command.** `hades ingest <dir>` walks a tree once, routes every
  file by extension through `hades_core::ingest_routing`, runs the code phase and
  the document phase against the same root, and emits a single envelope. Code
  keeps its analyzers, symbols, edges and late-chunked embeddings; documents keep
  extraction and the document profile; both land in one graph from one command.
  Files nothing claims are listed by path and reason under `unrouted` rather than
  silently skipped, which is what `codebase ingest` reporting markdown as "no
  handler for extension" and `hades ingest` handing a `.py` to docling used to
  cost. Document keys come out root-relative for free, since the root is the
  directory given. Naming files explicitly keeps the document-only behaviour,
  which is what a single paper wants, and `codebase ingest` remains for
  tree-level graph operations.
- Per-GPU embedder **load profiles**. The embedder is one templated systemd unit
  instantiated per profile, each naming the card it loads on and fixing that
  card's measured sequence ceiling, batch size and VRAM floor. Every profile
  binds the same port, so switching cards is a stop and a start with no client
  configuration change, and two profiles cannot be live at once because the
  second fails to bind. `GET /v1/models` now reports `profile`, `device` and
  `physical_device` alongside `max_seq_length`, so a client can see which card
  answered and what it will accept. `physical_device` exists because
  `CUDA_VISIBLE_DEVICES` renumbers from zero: a model pinned to the second
  A6000 reports `device: cuda:0`, and an operator reading that concludes GPU 0.
  Templates and a `hades-embedder-profile` switch command are in
  `deploy/systemd/`.
- A non-blocking `stable drift` CI job, weekly and on pushes to `main`, that
  runs fmt and clippy under current stable and warns rather than fails.
  Pinning the toolchain removed the only mechanism that ever told this
  repository a new release wanted changes, which is how CI came to be red
  from the first push and stayed there. This keeps the eventual bump one
  release wide.
- Late chunking on the code ingest path. A file's chunks are grouped into
  windows that fit the embedder's context, each window is encoded in one
  forward pass, and a vector is pooled per AST boundary, so every chunk
  vector carries the surrounding file's context instead of being encoded
  blind. `PE-API` gains an opt-in `late_chunk` request object and per-chunk
  response metadata (`chunk_index`, `token_start`/`token_end`,
  `char_start`/`char_end`). Set `HADES_DISABLE_LATE_CHUNKING=1` to fall back
  to per-chunk embedding.
- `hades-viewer` (`crates/hades-frontend`) — a local WebGL graph viewer for
  any HADES graph, plus a shared-reference channel between a human and an
  agent. Renders a named graph as a force-directed view with styling driven
  by attributes discovered from the data, expands neighborhoods on demand,
  and makes every view addressable (`?db=&graph=&node=`) so an agent can
  hand back a link to the exact node it means; right-click copies a
  briefing with the node's attributes, connections, and runnable `hades`
  commands. Depends on no other HADES crate — it consumes the CLI's
  JSON/jsonl output only. Binds loopback or a private LAN range, validates
  the Host header, and requires `--password` for any non-loopback
  bind. (#182)

- `scripts/install/test/` — container-based install validation harness.
  Builds a fresh Ubuntu 24.04 image with ArangoDB pre-installed, then
  runs the README install steps end-to-end. Catches packaging,
  ordering, and prerequisites issues without requiring a real VPS.
  Two real issues caught and fixed in the README this round: the
  ArangoDB GPG signing key is currently expired upstream, and the
  README's step ordering required the `hades` group before
  `systemd-sysusers` had created it. (#96)

- `AGENTS.md` — a self-contained onboarding guide for an agent that has
  never seen HADES, usable as a Codex `AGENTS.md` entry point or as a
  pasted system prompt. It covers what `--help` cannot say: that `db query`
  searches a collection profile rather than "the database", how the drift
  buckets relate to what ingest actually skips, and which remedies are safe
  to narrow. (#186)

### Changed

- `CLAUDE.md` rewritten against the current tree. It described two ingest
  commands, a hardcoded window budget, an extractor endpoint with no override,
  and three access tiers. It now records the single `hades ingest` front door
  and the routing table behind it, content-hash incrementality as the resync
  mechanism, the window budget read from the backend's `/v1/models`, the per-GPU
  load profiles with their measured 32,768 and 11,900 token ceilings, the
  `Provisioning` tier and the two flags that enable it,
  `HADES_EXTRACTOR_SOCKET`, and the embedding task pair on `CollectionProfile`.

- `codebase ingest` starts one rust-analyzer session per Cargo *workspace*
  rather than per member crate. Grouping keyed on the nearest `Cargo.toml`,
  but rust-analyzer runs `cargo metadata` on startup, which from any member
  directory resolves the whole workspace and returns every member package.
  Each session therefore loaded the entire workspace anyway, with its own
  cold salsa database, and the phase paid that N times sequentially -- five
  times on this repository. Grouping now walks up to the manifest declaring
  `[workspace]`, matching Cargo's own resolution, including a nested crate
  that opts out with an empty `[workspace]` table and a standalone crate
  belonging to no workspace.


- **Breaking (daemon/MCP):** `db.query` rejects `limit: 0` and `limit` above
  1000 with `INVALID_PARAMS` instead of returning an empty success or
  silently clamping. Zero previously produced `{"success": true,
  "result_count": 0}`, which an agent cannot distinguish from "nothing
  matched"; the clamp silently truncated a CLI `-n` that the pre-#187
  implementation honoured. (#187)

- **Breaking (MCP):** the `db_query` tool no longer accepts `rerank`. It is
  absent from the advertised schema and a client that still sends it gets
  `INVALID_PARAMS` naming `hybrid`/`structural` as the alternatives, rather
  than a success whose results were never reranked. (#187)

- `db.query`'s vector search refuses collections holding more than 100,000
  embeddings. The search has no vector index and materializes every stored
  vector to score it, which was the operator's own cost as a one-shot CLI
  and is a shared long-lived process's cost now that the daemon and MCP host
  the same handler. An oversized collection is a clear error rather than a
  60s request timeout that also leaks the ArangoDB cursor. (#187)

- `HADES_DEFAULT_COLLECTION` is read only by the CLI, never by the shared
  handler. Inside the daemon it is a process-wide global spanning every
  database and agent, so an operator's unit-file setting would silently
  redirect an unrelated database's search to collections that do not exist
  there and return zero results as a success. (#187)


- **Breaking (CLI):** `-g` is no longer an alias for `--graph` on
  `db graph traverse`, `db graph shortest-path`, and `db graph neighbors`.
  It collided with the global `--gpu -g`, which made clap's uniqueness
  assertion fire on every invocation of those three commands in debug
  builds. Release builds resolved `-g` to `--graph`, so scripts using the
  short form worked there and now need the long `--graph` form; `-g` is
  the global `--gpu` everywhere. `scripts/cli_audit.sh` updated. (#182)

- README **Install** section rewritten end-to-end. Drop the WIP banner
  (the procedure has been validated via the harness), add prerequisites
  (ArangoDB, Rust toolchain, protoc), fix the step ordering so
  systemd-sysusers runs before any command that references the `hades`
  group, add explicit `mkdir -p /etc/hades` and `sudo` invocations
  where they were missing, and add a verification step at the end. (#96)
- Verified by full bident_burn re-ingest after #110 landed: coverage
  reached 99.95% (2109/2110 chunks embedded). Single outlier is a
  stale chunk record from a deleted file pre-fix; AC for #98 satisfied.

### Removed

- `archive/`, which held one file: `python-hades-requirements.txt`, the dependency
  list for the Python CLI that was retired when the tree went fully native. A
  directory named for what it no longer contains is worse than no directory, and
  the file stays in git history for anyone who needs it.
- `RESEARCH_GOALS.md`, and the two references README.md made to it, one in prose
  and one in the documentation table. Deleting the file without them would have
  left a public repository with two broken links.
- `AGENTS.md`, which had been moved to `/opt/AGENTS.md` outside the repository.
  The copy left here was a second agent-guidance file with no owner, and two of
  those drift apart rather than agree.

### Fixed

- Keep WeaverTools node and edge metadata within their own declaration records,
  including adjacent mixed records without blank separators (#70).

- Verify the actual daemon socket and database health in the fresh-install
  harness with bounded framing, response sizes and startup deadlines (#64).

- Select distinct document-research adapter/prompts in offline retrieval
  evaluation and reject oversized prefixed processor inputs before inference;
  record profile and token-count provenance while preserving code defaults (#12).

- Add strict retrieval judgment coverage for representative datasets: preserve
  unjudged rankings, withhold incomplete scores and aggregates, and distinguish
  unsupported no-positive cases while preserving historical seed results (#12).

- Isolate dispatch and authorization unit tests with private cursor mocks; require
  exact allowed-query results and no requests from rejected commands (#47).
- Fence the complete training-provider lifecycle with expiring session ownership;
  reject competing/stale clients and renew/release shared Rust client leases.
  Allow renewal to queue behind long operations while bounding stalled renewals
  and retaining provider-side expiry checks (#42).

- Encode bootstrap credentials safely, keep passwords and responses in memory,
  bound Unix HTTP requests, and preserve existing user grants (#44).

- Run enrichment source rechecks on blocking workers while preserving pre-write
  and pre-commit validation and transaction cancellation responsiveness (#40).

- Stage parsed and fallback file replacements before a serial database
  transaction; retain previous graph state on persistence failure and reject
  stale prepared writes. Independent transaction ownership aborts cancellation
  and callback failures. LSP enrichment commits symbols, edges and metadata
  atomically, rejects stale preparations, and embedding preparation errors retain
  the prior graph. Remap moved symbols' inbound edges in the file transaction;
  remap failures roll back replacement and overlapping moves retain their targets.
  Store cross-file relationship batches atomically, reject stale file revisions
  and missing endpoints, and propagate stage failures with their JSON summary.
  Pending parsed files retry relationships without `--force`; acknowledge stage
  completion only after its transaction commits. Keep unchanged analyzed files
  in relationship resolution with revision guards, and clear pending recovery
  after explicit raw-text downgrade commits. Verify CLI interruption during
  embedding preserves the graph and permits retry. Reject LSP enrichment when
  captured source bytes differ from the stored content hash before analysis or
  during persistence. Verify an abandoned transaction rolls back and releases
  its exclusive lock through server expiry after writer process death. Load
  preserved higher-fidelity symbols as guarded resolution targets without
  replaying their outgoing calls, including explicitly unparsed files whose
  stored language identifies the retained analysis. Preserve JSON summaries on
  Rust enrichment failure and report direct codebase-ingest failure accurately.
  Distinguish failed extraction from empty success and report rewritten files
  affected by partial workspace or file-level enrichment failure. Verify that
  racing replacement and relationship stages cannot both commit from one revision.
  Exercise full-CLI process death after acknowledged transactional chunk writes,
  checking all graph collections after server expiry and successful retry. Cover
  the real operation deadline and ensure purge errors abort before replacement writes.
  Further recovery and cancellation
  coverage remains under audit (#40).

- Build complete Python service wheels with generated RPC bindings, trainer
  modules and adapter resources; include canonical protos in source archives
  and verify installed-wheel imports outside the checkout in CI (#38).

- Allow detail rows up to the aggregate search result budget by paging one row
  with bounded envelope headroom; regenerate vector/metric provenance hashes and
  correct the retrieval seed's training-adjacency excerpt in version 2 (#22).

- Bound plain/gzipped LaTeX source bytes and the complete decompressed tar stream
  before parsing archive headers. Count rejected entries toward the member limit,
  reject duplicate normalized source paths, and read the main member directly
  without materializing archive-controlled paths or permissions. Validate raw
  PAX/GNU header sizes before metadata allocation, including on older Python
  runtimes; bound metadata bytes and count hidden extended headers (#35).

- Convert embedding tensors to detached CPU float32 before NumPy export,
  supporting bfloat16 model output and tensors requiring gradients. Conversion
  to float32 preserves batch shape and ordering, including lists of tensor rows
  (#33); float64 values can round to float32 precision.

- The embedder profile selector now persists exactly one enabled instance,
  reconciles failed/runtime-enabled profiles without restarting an already-selected
  service, validates responder identity, and attempts rollback on switch failure
  (#19). Mocked switch/reboot/failure tests and a maintenance rollback guide cover
  the procedure; matching metadata is explicitly not proof of inference health.

- AQL cursor ownership now survives caller cancellation, including cancellation
  before the first response supplies its ID. Known cursors are deleted after
  completion, malformed payloads, server errors, or timeout. Query execution and
  cleanup have explicit time budgets, with server runtime/TTL fallbacks (#21).

- Bind graph artifacts and checkpoints to versioned semantic contracts (#15):
  relation order, collection indices, feature width/provenance, construction
  policy, and architecture. Reject incompatible or legacy unverified artifacts
  before changing restored weights or graph state; only fresh initialization
  can adapt its feature width. Reserve schema-declared collection indices and
  reject vectors with unknown or mixed model identity. See training evaluation
  documentation for the explicit legacy retraining procedure.

- Correct tied-score ROC-AUC and reject unavailable/nonfinite training metrics
  (#17). Require nonempty splits and samples, propagate dense-graph sampling
  failures, and validate checkpoint selection. Add `graph-embed train --seed`
  for repeatable splits and fixed validation/test negative samples.

- Separate training adjacency from held-out link-prediction targets (#14). Split
  inverse and duplicate endpoint pairs together, serialize and return one shared
  partition, and reject invalid partitions or training on held-out edges.

- Reject embedding responses with duplicate, missing, or invalid input indices,
  nonnumeric or nonfinite vectors, unexpected dimensions, or incompatible model
  identities before associating vectors with chunks (#16). Apply vector and model
  checks to late chunking too; preserve valid out-of-order response handling.

- Training RPCs now await precondition failures and validate graph tensors, sample indices, model configuration, and checkpoint loads before replacing active state (#18). Invalid requests return explicit gRPC statuses; reinitializing a model clears the previous graph. CPU tests exercise these contracts over temporary Unix sockets.

- **A partial re-ingest no longer dangles a dependent's edges** (#9). `symbol_key`
  hashes the definition line, so a comment inserted above a symbol changes its key
  while its name and meaning stay put. The edited file is rewritten under new
  keys, a dependent whose own content did not change is skipped, and its stored
  import or call edge is left naming a key nothing holds. #8 made this fire far
  more often, since comment edits now re-ingest where they used to skip.

  The ingest pairs each old symbol key with the key that replaced it and
  re-points the inbound edges. Pairing is by `(qualified_name, position among
  symbols sharing that name)`, computed inside the rewrite where both sides are
  still knowable -- after the purge a key cannot be reversed into a name. The
  envelope gains `repointed_inbound_edges` beside the existing
  `dangling_inbound_edges`, which now means "renamed or removed" rather than
  "moved or renamed or removed".

  **A name whose count changed is deliberately left unpaired.** Position stops
  identifying a symbol when one of three `Config::new` is deleted, and a wrong
  pairing would silently attach a dependency to a definition nobody wrote --
  worse than the dangling edge it replaced. 618 groups in one real corpus share a
  qualified name within a file, so this is the common shape rather than a corner.
  A rename is likewise not paired: the edge dangles, which is true.

  **The re-point writes the canonical key rather than updating in place.** An
  edge's `_key` is `edge_key(from, kind, to)`, and the analyzer phases re-resolve
  cross-file `calls` and `implements` edges for every file in a run, skipped ones
  included, writing them under the new target's key. Mutating `_to` therefore left
  two documents for one relation, the phase's and a stale-keyed copy, which
  traversals and neighbour counts double -- worse than the dangling edge it
  replaced, because a duplicate is silent where a dangle was reported. Inserting
  at the canonical key collapses with whatever the phase wrote.

  Every read completes before any write, because the remap can chain: two symbols
  sharing a qualified name can move so that one's new key is another's old key.
  Resolving against a collection being written in the same query would drag an
  edge past its own target. That holds across the whole remap and not merely
  within a chunk -- chunking the reads to bound the bind parameter would otherwise
  reintroduce the hazard at the boundary, with the chunk holding `B -> C` finding
  the edge the chunk holding `A -> B` had just moved. For the same reason a key
  being dropped is never a key just written.

  Verified against the reproduction from #9: a comment growing above a symbol now
  reports `repointed: 2, dangling: 0` with one call edge and one import edge, both
  resolving, where it previously reported `dangling: 2` and neither resolved. A
  genuine rename reports `repointed: 0, dangling: 2`, so the guard holds. Six unit
  tests cover the pairing rule with no database, and two live tests cover the
  collapse and the chain.


- **Tree ingest skipped files whose content changed, leaving stale chunks that
  search served as current** (#7). The incremental gate compared `symbol_hash`,
  which `compute_symbol_hash` builds from sorted symbol *names* and nothing else,
  so an edit touching only comments left it identical and the file was reported
  `skipped` with `files_embedded=0`. Its stored chunks kept the old text while
  the symbol half, which rust-analyzer re-reads every run, moved on: the two
  halves of one file disagreed, and only one of them is what `db_query` reads. In
  the reported case a corpus was serving a sentence it had retired, as if current.
  The gate now compares `content_hash`.

  **Not a split gate, though that was the first plan.** The two hashes are not
  independent -- a byte-identical file has identical symbols, so `content_hash`
  unchanged implies `symbol_hash` unchanged, making it strictly stronger. And the
  decision cannot be divided per artifact, chunks on content and symbols on names,
  because `symbol_key` hashes the line number: a comment adding eight lines moves
  every later symbol's key, and chunk documents reference those keys in
  `overlapping_symbols`, so refreshing one without the other points fresh chunks
  at keys that no longer exist. Within a file the two move together.
  `symbol_hash` keeps its genuine cross-file job, deciding whether a file's
  dependents need their edges re-resolved (#183).

  This also makes an existing promise true rather than aspirational: the
  single-file refusal in `main.rs` already told callers "the ingest is
  incremental, so only files whose content hash changed are re-processed".

  Verified end to end in a throwaway database: first ingest embeds, an unchanged
  re-run skips with zero embeddings, a **comment-only** edit re-embeds and the
  stored chunk carries the new text with zero chunks holding the retired
  sentence, and a further unchanged run skips again. Two regression tests, one at
  the gate and one driving `ingest_file` twice and reading the stored chunk text,
  both confirmed to fail against the pre-fix gate.


- **`scripts/bident_burn_smoke.sh` never truncated anything.** The line ran
  `db truncate "$col" --yes`, and the flag is `-y/--force`, so clap rejected it
  and the bare fallback refused without confirmation. Both arms failed silently
  behind `>/dev/null` from the day it was written, which means every run after
  the first asserted against the previous run's rows. It passes `--force` now,
  and the script header no longer compares itself to `bident_burn`.
- **The throwaway-database harness is one definition, not two.** The first fix on
  this branch left the same shape in `hades-cli`'s test module and
  `hades-core/tests/common`, already drifting in signature, naming and fixtures,
  with "kept in step" as the only mechanism, which is the defect this branch
  exists to remove. It lives in `hades_core::test_support` behind a
  `test-support` feature, off by default and enabled through a dev-dependency
  (including a self dev-dependency so hades-core's own `tests/` can reach it),
  so nothing reaches a release build. The fixture collections come from
  `CODEBASE.all_collections()`, the list `ingest` creates from, instead of a
  hand-written copy that a ninth collection would silently outdate, and the
  third copy of the literal edge-type code `3` goes with it. A failed drop is
  now reported rather than discarded, since a leaked database is otherwise
  invisible, and the fixtures are created *inside* the spawned task so a fixture
  that fails to create cannot leak the database it was meant to be isolated in
  (found by CodeRabbit on PR #5; verified by a probe that panics inside the task
  and leaves nothing behind).


- **The critical rules named a database that no longer exists.** `bident_burn`
  held the `persephone_tasks` kanban and was dropped on 2026-09-14 with the other
  superseded databases, while both CLAUDE.md files still told a session to run
  project management against it and to target it for write tests. The replacement
  states a rule rather than an inventory, since a guidance file listing live
  databases goes stale: no database is a default, the `task` commands need a
  database and the four `persephone_*` collections created for them (only the
  smoke script does that today), and a write test creates and owns its own. A
  fourth rule carries finding 35 forward with the mechanism stated: a database
  seeded with `--seed empty` has an empty `relation_order`, over which the loader
  returns no edges and no nodes, `train` fails at the tensor step, and `update`
  fails at its checkpoint preflight or, given a checkpoint from an earlier run in
  the shared default directory, reports success.
- **Write tests get a throwaway database, dropped even on panic.** The
  `codebase_prune` harness had the right shape and is now the only one, in
  `hades_core::test_support` (see the Fixed entry above for why it is there and
  not copied per crate). Per-process name with a counter (two tests in one binary
  share a pid, and the old delete-before-create was dropping a sibling's database
  mid-run), the user from `HADES_TEST_USER`, which needs `rw` on `_system` and now
  says so; refusal to create is a failure under `ARANGO_TESTS=1` rather than a
  silent skip. `arango_cache`, `codebase_retire` and `codebase_prune` run on it.
  `codebase_ingest`'s live tests drop their dead `bident_burn` default and require
  `HADES_TEST_DB`, failing loudly in strict mode rather than skipping green.
- **The `codebase_retire` regression guard for #158 had never run.** It skipped
  whenever its target database lacked the codebase collections, which was always,
  and reported success. On a real database it failed at once: the sweep binds all
  four codebase edge collections and maps ArangoDB's 1203 (collection not found)
  to an empty result, so a fixture without them makes the sweep report nothing.
  The harness creates them. It passes, for the first time.
- On the two failing test targets: they were **already failing before that
  database was dropped**, under `ARANGO_TESTS=1`, on the socket default
  `/run/arangodb3/arangodb.sock`, which a user-level deployment does not have; a
  bare run returned five green skips. `ARANGO_SOCKET` names the socket.
- The README claimed the smoke script "never touches `bident_burn`", true and
  useless once that database was gone, and `cli_audit.sh` is still hardwired to
  it. The README says so. `arango_transport`, `arango_crud`, `arango_index` and
  `arango_query` also still name it and need a seeded `persephone_tasks`; they
  wait for the Persephone pass.


- **The WeaverTools extractor no longer decides its own scope.** It walked a
  hardcoded `docs/` root with a hardcoded `process/` exclusion while `hades ingest`
  walked `.hadesignore`: two implementations of one rule, agreeing by coincidence of
  intent. The failure that makes it matter is a document under `docs/` that declares
  a node and is later added to `.hadesignore` - the ingest stops writing its
  `documents` row, the extractor keeps reading its declarations, and the
  `declared-in` edge names a row that does not exist. `write_graph` checked for
  exactly that and would have caught it, but at build time, after the run, with the
  bad edges already computed. `ingest()` now takes the path sets the graph holds,
  supplied by the caller because the caller is the half with database access, and a
  file out of scope holding declarations or citations is reported by name rather
  than reflected in a count that came out low. The extractor stays a pure function
  over a repository. Verified against `WeaverTools_v5`: every count unchanged
  (413/491/481/13/665/12), so the change removes a failure class without moving
  data. Five tests in `services/tests/test_weavertools_scope.py` construct the
  case, including the invariant that no `declared-in` target lies outside the scope.
- `write_graph` reads each `cites` edge's source key from `codebase_files` instead
  of re-deriving it from the path. Re-deriving a key the ingest owns is what the
  document half was doing before the same class of mismatch was found there.
- **A graph fence must open and close on its own line.** Unanchored, the pattern
  matched an inline ```` ```graph ```` inside a sentence and ran to the next triple
  backtick anywhere in the file, so a document discussing the block grammar opened a
  phantom fence over its own prose. Two stand in this corpus. Both happen to hold no
  record at a line start and so contributed nothing, which is why nothing found them
  until the scope report named the file; a quoted example inside the swallowed
  region would have injected declarations nobody wrote. Measured across the corpus:
  399 fences before and 397 after, the two lost being exactly the phantoms, with 491
  node records and 665 edge records either way.


- **`db.graph.traverse`, `db.graph.neighbors` and `db.graph.shortest_path` no
  longer invent a graph named `default`.** ArangoDB has no default graph, so
  omitting `graph` sent every traversal at a graph no database here has;
  ArangoDB answered "graph not found", and a session reading that failure
  reasonably concluded the vertex had no neighbours and reported a corpus of
  1,635 edges as having none. The graph is resolved instead: one named graph
  needs no argument, several make the error name them, and none says the database
  is unseeded. A wrong default is worse than a required argument because it fails
  as an absence rather than as an error. The MCP tool descriptions said "omit for
  the default graph" and now say what actually happens, pointing at `graph_list`.
- The WeaverTools `declared-in` edge ends at the ingested `documents` row rather
  than at a node minted per markdown file. `wt_documents` held 68 file nodes
  beside the corpus's 13 declared `kind: document` records — two different things
  in one collection, matching neither count — and the file half carried no
  embedding, so every one of those 491 edges ended on a vertex with no features
  for a GNN to read. It is the argument the adapter already made for source
  files: a node duplicating one the ingest created is a join that proves nothing.
- An edge record's `via` and `tag` are read from that record's own lines. The
  block splitter leaves each edge stanza running to the *next* edge record, so 86
  of this corpus's stanzas carry a following `node:` record inside them, and an
  unbounded search read the next node's tag onto the edge — a field that looks
  right, belongs to something else, and no count would catch.


- Errors report their cause instead of restating the request.
  `HandlerError::Query` has always carried its `ArangoError` as `#[source]`, and
  rendered only its own context, so the response was built from a `Display` that
  dropped the reason at the last step: `graph_neighbors` answered "graph traverse
  from 'codebase_files/x'" while the cause it held said `graph 'default' not
  found`. One survey spent six probes establishing a negative that one error could
  have given. A query that failed because something was absent now also reports
  `NOT_FOUND` rather than `QUERY_FAILED`, because "no such collection" ends a
  question while a query failure invites a retry.
- A single source file named directly is refused rather than stored as a document.
  It took the document path regardless of extension, so `hades ingest src/lib.rs`
  landed a code file in `documents` with no symbols, edges or AST chunks. Routing
  it to the code phase needs a root, since a code file's key derives from the
  ingest root and a lone file bases at its parent, so it would write a second node
  under a re-based key instead of updating the first.
- Documents are re-ingested when their content changes. Code file nodes have
  carried `content_hash` from the start, which is what makes that half
  incremental; document rows carried no hash, so the skip keyed on the document
  merely existing and an edited spec was skipped forever, with `--force` over the
  whole corpus the only way to pick it up. Documents now record `content_hash` and
  `ingested_at`, and one `hades ingest <tree>` is a true resync for both halves.
- Late-chunked vectors are mapped back to their chunk by recorded index rather
  than by `first_chunk_index + position`. Skipping a chunk too large for any
  window broke the assumption that a window's boundaries are consecutive in file
  order, so every vector after such a gap was stored under a later chunk's key,
  with the counts still matching and nothing to show it. Each window now carries
  the file-order index of every boundary.
- `ingest.start` refuses a second job over the same tree and caps concurrent jobs
  at two. Nothing was bounded before: a provisioned token could call it in a loop
  and get one full ingest process per call, each running analyzers and a language
  server over the same tree and writing the same keys. A row whose process is gone
  does not count toward the cap, so a daemon restart does not block later jobs.
- Ingest jobs keep their diagnostics. The child's stderr went to `/dev/null`, so a
  failed job reported only "exit status 1" with the reason gone. It is captured,
  its path recorded on the job row, and its tail stored in `detail` on failure.
- `HADES_EXTRACTOR_SOCKET` requires an absolute path after `unix://` as well as
  bare. A relative socket path resolves from the process working directory, so the
  daemon and the CLI would reach different sockets from one configuration value.
- `--unparsed-ext` is refused with named file inputs instead of ignored. It selects
  extensions during a tree walk and there is no walk for named files, so
  `hades ingest Cargo.toml --unparsed-ext toml` looked like it had asked for the
  parser-free path and had not.
- `hades-embedder-profile` reports a failed switch. `curl` without `--fail` treated
  any response including 4xx and 5xx as readiness, and the status helper returned
  zero when nothing was active, so a switch that never came up looked successful
  and the next command ran against no embedder.
- `write_graph.py` refuses to send `ARANGO_PASSWORD` to a non-loopback host, since
  the endpoint is plain HTTP and the credential would cross the network in the
  clear.
- A single chunk larger than the embed window no longer costs its file every
  vector. `AstChunking` caps a chunk at 8,000 characters by splitting at line
  boundaries, but `split_at_lines` emits one whole line when its accumulator is
  empty, so a minified or generated file with one very long line produces a chunk
  of unbounded size. Sent whole to `embed_late_chunked` it exceeds the backend's
  ceiling, is refused, and takes the file's other windows with it because that
  call returns on its first error. Such chunks now take the plain embed path, one
  vector each without surrounding context, and a batched late-chunk failure is
  retried window by window so a bad window costs its own chunks rather than the
  file. Closes the data-loss half of the OOM-retry gap.
- Uniform late-chunking windows no longer report `char_end = 0`. The walk was
  bounded by the count of non-padding tokens, which includes trailing special
  tokens, and those carry `(0, 0)` offsets, so a final window ending on one
  reported an inverted character range that a caller intersecting against symbol
  spans matches nothing from. Bounded now by the last token covering a real
  character. Affects the `chunk_size_tokens` mode only; HADES always sends
  explicit boundaries.
- The PE-API no longer requires a context ceiling its own reference
  implementation cannot meet. Implementation requirement 1 mandated a "32k
  context" capability profile while the reference backend serves 11,900 tokens on
  a 16 GiB card, so a conforming client trusting the advertised figure would send
  inputs that are refused. `max_seq_length` is now specified as the ceiling the
  backend will accept, distinct from the model's architectural maximum.
- The routing table claimed `html`, `htm`, `csv`, `docx`, `pptx` and `xlsx` as
  documents on the assumption that docling's unknown-format fallback would take
  them. `docling_backend.py:122` refuses anything outside `{pdf, txt, text, md}`,
  so 13 `.html` templates in a real corpus were claimed by the walk and then failed
  extraction, turning a readable `unrouted` entry into a failed document in a
  summary. The list is now what the service verifiably accepts: markdown and text
  through the plain-read path, `pdf` through docling, `tex` and `gz` through the
  LaTeX backend. Adding one back means teaching the extractor first.
- The embed window is sized from the backend's reported ceiling instead of a
  constant, so the full context of whichever card the embedder is loaded on gets
  used. `WINDOW_CHARS = 12_000` was justified by a comment claiming dense Rust
  tokenizes at roughly 1.5 characters per token; measured across 285 files of real
  Rust, Python, CUDA and markdown the median is 4.18 and the lowest 2.18. So that
  budget packed about 2,870 tokens against a 32,768-token card and split **134 of
  those 285 files**, producing 672 windows where 313 suffice. Every extra window
  is a seam where a chunk's vector loses the surrounding file, which is what late
  chunking exists to prevent. The budget is now `max_seq_length` from
  `/v1/models` times a measured chars-per-token floor of 2.0, which is 65,536
  characters on the 32,768-token profile and falls back to the old constant when a
  backend reports no ceiling.
- `EmbeddingClient::info` reads the entry the backend actually serves. It matched
  only on the configured model name, and this deployment configures the alias
  `jinaai/jina-embeddings-v4` while the backend serves a local filesystem path, so
  the id never matched and every vendor field came back `None` — including the
  `max_seq_length` the window budget above depends on, which silently fell back to
  the conservative constant on a card holding three times as much. An exact id
  match is still preferred, with the sole entry of a single-model listing as the
  fallback. Finding #14 was this same mismatch reached from the other side.
- The document pipeline creates its own collections. Ingesting into a fresh
  database failed with "collection or view not found: documents" *after*
  extraction and embedding had run, so the expensive work was done and then
  discarded at the store step. `codebase ingest` had always created its nine
  collections and its named graph on the fly; the document path now creates its
  three.
- `hades ingest` keyed documents by file stem, so files sharing a name in
  different directories collided and every one after the first was reported as
  skipped inside a run whose envelope said success. It cost 3 of this
  repository's 17 markdown files on their first ingest, and would have cost 10
  of WeaverTools' 101, where 11 files are named `README.md`. Two changes: a new
  `--root <dir>` derives keys from each input's path relative to that directory,
  the way `codebase ingest` has always keyed files, and a key already held by a
  *different* `source_path` is now an error naming both paths rather than a
  skip. The collision check runs under `--force` too, where the old behaviour
  was worse than skipping: it overwrote a different document.
- `db.query` embedded every query with `retrieval.query`, including searches of
  a `code` corpus, and the Jina task adapters do not share a vector space.
  Measured on this repository's own code graph, 1,453 vectors: separation
  between the top hit and the median was 0.1405 with `retrieval.query` against
  0.2132 with `code` on both sides, and the mismatched search returned an
  unrelated hex-digit assertion at rank one where the matched one returned the
  chunking strategy. The task is now a property of the collection profile,
  `code` for `codebase` and `retrieval.query` for documents, so it cannot drift
  from what the corpus was embedded with. This answers roadmap question R28 for
  the code half: no query/passage split there.
- The embedder silently truncated any input over its sequence ceiling and
  reported success. A 45,183-token document sent to a 32,768-token profile
  returned HTTP 200 with one vector, roughly 12,400 tokens discarded, and
  nothing in the response or the log said so. `_embed_batch_locked` prefers the
  model's own `encode_text`, which truncates at its configured max_length, and
  the fallback arm passed `truncation=True` explicitly, so both paths dropped
  the tail. The vector count matches the input count exactly in that state,
  which is why no count reconciles it. **Breaking**: inputs above
  `max_seq_length` are now refused with `PE_INPUT_TOO_LARGE`, naming the input
  index, its token count and the ceiling, which is what the PE-API error table
  specified from the start. Callers pre-chunk or load a profile on a bigger
  card. The refusal covers both endpoints: the boundaries path reports which
  boundaries fell past what the model saw, and the uniform path refuses rather
  than returning fewer windows than the text warrants. The count is taken with
  special tokens against a ceiling reduced by `PROMPT_RESERVE_TOKENS`, since
  `encode_text` prepends a task prompt that a bare count cannot see and an input
  of exactly `max_seq_length` would otherwise pass the guard and be truncated.
- The measured ceiling for the 16 GiB card was wrong by 25 percent. The
  15,000-token figure came from a synthetic probe; real documents through the
  running service pass at 11,926 tokens repeatedly and fail at 12,196, 12,215,
  12,417 and 13,382, the last of those also as the first request to a freshly
  started process. The profile is corrected to 11,900. On the 48 GiB card a real
  31,871-token document peaks at 27,526 MiB against the 23,906 MiB a probe
  recorded, so its VRAM floor is raised to 28 GiB.
- `MODEL_RESIDENT_MIB` is measured rather than estimated: 7,993 MiB, read from
  `torch.cuda.memory_allocated` after a bare load, of which 7,162 MiB is 3.755B
  fp16 parameters and 686 MiB is 1,518 fp32 LoRA tensors peft keeps unquantized.
  Set to 8,192. The previous 9,216 was an estimate, and an intermediate value of
  12,288 taken from an OOM traceback was wrong in the other direction: that
  reading came from *during* a failing forward pass, so it counted roughly 4 GiB
  of partial activations along with the model, and it would have made the
  contention preflight demand 14,336 MiB free on a card that has about 15,500.
- `codebase ingest` keyed one tree two ways depending on how its root was
  typed. The base was canonicalized while discovered paths were not, so
  `rel_path_for`'s `strip_prefix` missed and fell back to the whole path as
  given: `crates/hades-proto` ingested relatively keyed three files
  `crates_hades-proto_build_rs` and absolutely keyed the same three `build_rs`,
  producing six file nodes with six sets of chunks, symbols and embeddings, both
  halves stamped with the same `ingest_root` so `codebase drift` could not tell
  them apart. The same cause broke rust-analyzer enrichment from the other end:
  the crate-root walk popped a relative path to empty, and a session spawned
  with an empty working directory failed with ENOENT, losing every call and
  implements edge in the run. The ingest root is now resolved and canonicalized
  once, before anything derives from it.
- The extraction client reads `HADES_EXTRACTOR_SOCKET`, which
  `services/extraction/config.py` has always honoured while the Rust side
  hardcoded `/run/hades/extractor.sock`. That directory is created by
  `tmpfiles.d` as root, so a user-level deployment had no reachable extractor
  and `hades ingest` could not process a document at all. `http://`, `https://`,
  `unix:///path` and bare absolute paths are accepted, matching the embedding
  client; a malformed value is an error naming the value rather than a silent
  fall back to the default.
- CI now passes. It had never passed on this repository, including on `main`.
  `rust-toolchain.toml` pinned `channel = "stable"`, which is a moving target
  that resolves per machine, so CI ran 1.98.1 while a workstation ran whatever
  its last `rustup update` fetched. The channel is now pinned to an exact
  version, so the local gate and CI are the same gate.
- `TrainingError::Status` and `ExtractionError::Status` box their
  `tonic::Status`, which is 176 bytes and set the size of every `Result` in
  those modules. **Breaking**: both variants now hold `Box<tonic::Status>`,
  and a manual `From<tonic::Status>` keeps `?` working. This removes 27
  `clippy::result_large_err` errors at the source rather than allowing them.
  The lint is allowed on the four generated modules in `hades-proto`, where
  the signatures come from `tonic-build` and boxing is not available, and on
  one call site in `hades-frontend` whose error is an axum `Response` by
  design. A third allow in `hades-core::training` was made dead by the boxing
  and is removed, along with its comment, which said `TrainingError` is large
  because of `tonic::Status` after that had stopped being true.
- `decode_f32_embeddings` and the tensor readers use `as_chunks::<4>()`
  instead of `chunks_exact(4)`, which drops three `try_into().unwrap()` calls
  that existed only to convert a slice to an array. The two tensor readers
  state the divisibility they assume at the point `as_chunks` discards the
  remainder. safetensors already rejects a mismatched byte range on
  deserialize, so the branch is unreachable today and is a guard on the
  assumption rather than on the file.
- CI actions are pinned to commit SHAs with a `permissions: contents: read`
  block. A tag can be moved and a branch moves by design, so `@v4` or
  `@master` let upstream change what runs here with no change landing in this
  repository, and the drift job hands its token to a third-party action.
- `ci.yml` reads the pinned toolchain from `rust-toolchain.toml` and asserts
  the running `rustc` matches it. It previously asked the action for `@stable`,
  so the pin held only by rustup's per-command override, and anything setting
  `RUSTUP_TOOLCHAIN` would have restored the drift silently.
- The workspace declares `rust-version`, so a build that bypasses rustup gets
  cargo's version diagnostic rather than a missing-method error.
- PE-API `boundaries` are documented as character offsets and the Rust client
  now sends character offsets. It previously sent `TextChunk` byte offsets,
  which the tokenizer's character-indexed offset mapping read as characters,
  so on any file containing a multibyte character every chunk was pooled from
  a span that drifted further from its code through the file. Nothing failed,
  because the pooling was correct over whatever range it was given. The client
  now compares the returned `char_start` against the boundary it sent and
  errors on a drift token alignment cannot explain.
- A late-chunked response missing `chunk_index` is an error rather than a
  default of 0, which previously collapsed every vector of an input onto its
  first chunk when a backend ignored the `late_chunk` field.
- Vector counts are checked against boundary counts on both sides of the wire,
  and a mismatch fails the file rather than storing it with fewer embeddings
  than chunks.
- Spans are paired with the vectors pooled from them in the embedder rather
  than reconciled by list length afterwards, which mislabelled every vector
  after a skipped span.
- `POST /v1/embeddings` rejects `late_chunk.boundaries` alongside multiple
  inputs instead of applying one input's character ranges to all of them.
- The inference lock covers the plain embedding path as well as the
  late-chunked one. Both reach the Rust-backed fast tokenizer, which raises
  "Already borrowed" when two pooled threads touch it at once.
- Embed windows are sliced with `str::get` and skipped when the offsets do not
  land on character boundaries, instead of panicking mid-ingest.
- `EmbedResult` regained the `Debug` and `Clone` derives and the doc comment it
  lost when `LateChunkVector` was inserted between the attribute and the
  struct it decorated.
- `codebase drift` reported one tree's file nodes as `stale` for another,
  on a delete path. File keys are relative to the ingest root, so they
  carry no evidence of which tree produced them, and the graph side of
  the comparison read the whole `codebase_files` collection unfiltered.
  In a database holding two ingested trees, `codebase drift /repo-a`
  therefore listed every node of repo B as stale while those source files
  sat untouched — and `--full` exists to feed exactly that output to
  `codebase retire`, which removes each target's file node, chunks,
  embeddings, symbols and incident edges. The documented pipeline deleted
  the other tree.

  Ingest now records an `ingest_root` on each file node, for every file
  discovered under the root rather than only the ones a run rewrote, so a
  re-ingest attributes an existing graph even where it skips unchanged
  files. Drift compares only nodes carrying its own root and reports the
  rest as `other_roots`. Nodes predating attribution are still compared —
  dropping them would report an entire existing graph as `uningested` —
  but the stale ones among them are listed separately as
  `stale.unattributed_keys` and held out of `stale.keys`, so the documented
  `drift --full | codebase retire` pipeline cannot delete a node this
  command could not prove belongs here. Re-ingesting a root attributes
  every node whose file still exists; a node whose file is already gone is
  never rediscovered and no re-ingest can attribute it, so a graph built
  before this change keeps a residue that only a reviewed retire clears.

  Scope: this makes drift stop mislabelling another tree's nodes. It does
  not separate trees that share a relative path, because `file_key` is
  still purely root-relative and `src/main.py` in two repositories is one
  document. Non-overlapping trees in one database are handled; colliding
  ones need the root folded into the key, which is a migration. (#192)

- `db.query` was advertised over the daemon and MCP but never implemented.
  Dispatch fell through a catch-all arm to `NOT_IMPLEMENTED`, so the MCP
  `db_query` tool — one of the twelve curated agent-tier tools — failed on
  every call while appearing in the tool list. The search pipeline now lives
  in `dispatch::handlers::db_query`, shared by `hades db query`, the daemon
  and MCP, and the catch-all is gone so a future command cannot reach the
  protocol surface without a handler. (#187)

- Go files were permanently pinned against re-ingest. Go has no per-file
  semantic analyzer, so ingest can only produce `analysis_tier: "structural"`,
  but the post-loop gopls phase then stamped the **file node** `"semantic"`
  without rewriting `symbol_hash` — which it cannot, since that digest belongs
  to the per-file analysis. `preserve_higher_fidelity` compares against exactly
  that field and runs ahead of the `--force` check, so from the second run
  onward every `.go` file lost the comparison and returned skipped with
  `higher-fidelity stored analysis preserved`, `--force` included, leaving
  `codebase drift` reporting the same counts forever. The LSP phases no longer
  overwrite the file node's `analysis_tier`/`analyzer` (the enrichment is
  already recorded under `gopls_analyzed` / `ra_analyzed` and their
  companions, and the symbols and edges it writes carry their own tier), and
  the fidelity guard now yields when the gopls phase is scheduled to re-enrich
  the file later in the same run. Existing graphs recover without
  `--allow-analysis-downgrade`, but they do need
  `codebase ingest --force <the original ingest root>`: once the guard yields,
  the unchanged-digest skip fires next, and gopls never rewrote `symbol_hash`,
  so a plain re-ingest still returns early and leaves the old stamp in place.
  The guard is unchanged where it is still load-bearing — it stays in force for
  Rust, whose `semantic` tier comes from `syn` per file rather than from the
  LSP phase, for an incoming raw-text tier, where nothing re-supplies what the
  purge drops, and for a C++ tree re-ingested without its compilation
  database. (#193)

- Operator-facing help text that contradicted the implementation. `db query
  --rerank` advertised itself as "Enable re-ranking of results" while the
  flag exits non-zero without searching, and now says so. `codebase drift`
  still described the pre-#183 output, and now documents `changed`,
  `unhandled`, `changed.unverifiable` and `clean`, including the fact that
  what `symbol_hash` covers depends on the node's `analysis_tier`. The
  `codebase ingest --force` help and the runtime dangling-edge warning both
  told operators to "re-ingest the dependent files", which either no-ops
  (the dependents' own `symbol_hash` is unchanged, so they are skipped) or
  writes duplicate nodes under re-based keys (a narrower path re-bases every
  key beneath it); both now name `--force` and the original ingest root.
  (#186)
- `codebase drift` reported a clean sweep over partially-covered trees.
  Files with no ingest handler fell outside drift's notion of source
  entirely — neither ingested nor reportable — so `stale=0 uningested=0`
  was returned for a tree ingest had only partly read. Drift now reports
  an `unhandled` bucket with a per-file reason, and a `clean` flag that
  is false whenever anything is stale, uningested, changed or
  unverifiable. `unhandled` deliberately does not gate `clean`, since
  every repository contains files no analyzer handles. (#183)
- `codebase drift` could not see content staleness at all. `symbol_hash`
  is name-only for Python and Rust at tier `semantic` (a rewritten body,
  changed signature, or edited comment leaves it identical), and drift
  compared only file
  existence, so an edited file reported clean while its stored chunks and
  embeddings were stale. Ingest now records a full-source `content_hash`
  alongside it and drift reports a `changed` bucket. Files ingested before
  this change are counted as `unverifiable` rather than assumed clean.
  Incremental re-ingest behavior is unchanged — `--force` still refreshes
  such a file. (#183)
- Extensionless scripts were invisible to ingest. `--unparsed-ext` is
  extension-keyed, so a file named `deploy-thing` with a `#!/bin/bash`
  first line could not be named by any flag. Discovery now sniffs the
  shebang of extensionless files: a recognized interpreter selects the
  analyzer (`#!…python3` → Python), and any other shebang routes the file
  to the raw-text path so its content is at least visible. (#183)
- `codebase ingest` gave no signal when a rebuild left inbound edges
  dangling. A rebuild that drops a symbol (rename, re-qualification,
  analyzer change) leaves `codebase_imports_edges` from *other* files
  pointing at nothing, breaking the `imports_edge_endpoints` invariant
  with nothing in the output to say so. Ingest now reports
  `dangling_inbound_edges` in its JSON summary, scoped to the files it
  rebuilt and to targets that genuinely do not resolve, and `--force`
  documents the repair. They are reported rather than deleted: each edge
  records a real dependency, and removing it would erase the only signal
  that the dependent needs re-ingesting — the dependent is unchanged, so
  every later ingest skips it and never re-derives the relation. Re-ingest
  the dependents, or run `codebase prune-orphans` to drop them. (#183)

- Embedding coverage gap during `codebase ingest`. Two root causes:
  the embedder service OOMed on per-file batches whose padded sequence
  lengths exceeded available GPU memory (typically on large source
  files like `dispatch.rs` with 90+ chunks of variable token-length),
  and the ingest code swallowed the resulting HTTP 500 into a
  `warn!` log line that never reached the user-facing JSON output —
  so files appeared ingested successfully while their embeddings
  were silently dropped. Two coordinated fixes:
  - `EmbeddingClient::embed` now splits requests at `batch_size`
    and progressively halves on OOM-shaped errors, recursively down
    to single chunks. Results reassembled by start index regardless
    of completion order.
  - `codebase ingest` surfaces per-file embedding failures via a new
    `embedding_error` field on each file's JSON result, plus a
    `files_with_embedding_failures` count and `embedding_failure_paths`
    list in the top-level summary. Failures are no longer silent.
  Verified on `dispatch.rs` (97 chunks, was 0/97 → now 96/96) and
  on a full re-ingest of `bident_burn`. (#98)

### Added

- Spec doc `docs/specs/workstation-specific-tests.md` codifying the
  skip-or-strict pattern for integration tests that depend on
  workstation-specific resources (live ArangoDB, embedder service,
  specific database state). Convention is the existing practice in
  `tests/graph_loader.rs` and the `arango_*` integration tests: live
  in `tests/`, skip when prerequisites are absent, panic when
  `ARANGO_TESTS=1` is set and prerequisites are still absent. (#93)
- `crates/hades-core/tests/arango_transport.rs` brought in line with
  the spec: previously skipped without honoring `ARANGO_TESTS=1`
  strict mode; now uses the shared `require_socket()` helper that
  panics under strict mode. (#93)

### Changed

- Strip arxiv- and NestedLearning-specific defaults from test fixtures,
  comments, and operator-surface documentation. `NestedLearning` is no longer
  used as a sample database name in tests; `arxiv_metadata` is no longer used
  as a sample collection name. `CLAUDE.md`'s "Production data is sacrosanct"
  paragraph updated to reflect the ArangoDB-ACL-based security model rather
  than the (since-removed) compile-time allowlist. Obsolete write-guard
  assertion removed from `scripts/cli_audit.sh`. (#92)
- README: replace the "Three-tier access control via SO_PEERCRED" claim with
  an accurate description of what the tier dispatch actually does — opt-in
  client self-restriction, useful as a UX guard for AI-agent harnesses.
  Security against malicious clients is enforced at the ArangoDB layer via
  ACL grants on the `hades` user; the daemon's tier dispatch is not a
  security boundary. (#97)

### Added

- `CHANGELOG.md` (this file), with the convention documented above. (#95)

## [0.3.0] - 2026-05-14

This is the baseline release: the state of `main` at the point CHANGELOG
discipline began. Entries are reconstructed from PR history.

### Added

- Python call-graph extraction via AST. `codebase ingest` of Python source
  now populates `codebase_calls_edges` using a rustpython-parser AST walk
  plus a three-strategy resolver (exact qualified-name match, `self.method`
  → `ParentClass.method` rewrite, bare-name fallback). Parallel to the
  existing rust-analyzer-driven path for Rust. (#91)
- `.gitignore` and `.hadesignore` are honored during codebase ingestion.
  Replaces the hand-curated `SKIP_DIRS` const with `ignore::WalkBuilder`,
  which respects standard ignore files plus a custom HADES-specific
  filename for exclusions that don't belong in version control. The hard
  `SKIP_DIRS` floor (`__pycache__`, `node_modules`, `target`, `venv`,
  `dist`, `build`) is preserved for repos with no ignore files. (#90)
- Code smell `_key` convention and `hades_burn_self` self-analysis
  database for dogfooding-driven schema work. (#86)

### Changed

- ArangoDB ACLs replace the compile-time `WRITABLE_DATABASES` allowlist
  for write-safety. HADES now connects as a dedicated `hades` ArangoDB
  user; write restrictions on specific databases are enforced by
  ArangoDB grants on that user, not by a Rust const checking the
  database name. Operator manages per-database access in arangosh.
  Removed 23 in-process call sites of `require_writable_database()`,
  the `WriteDenied` error variant, and 8 tests that exercised the
  removed guard. README install section documents the new bootstrap
  flow. (#89)
- Multiple README revisions clarifying the research-program context,
  the Persephone Embedding API contract, and the schema-as-data
  architectural commitment. (no PR — direct commits)

[Unreleased]: https://github.com/toddwbucy/HADES/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/toddwbucy/HADES/releases/tag/v0.3.0
