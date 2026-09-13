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

### Added

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

### Fixed

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
  card.
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
