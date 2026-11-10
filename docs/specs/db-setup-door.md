# Design: the first public door into model-bound databases (act 2 of #225)

Status: design accepted by the operator on 2026-11-10, with the Planner's
recommendations adopted as recorded under "Decided, and by whom". The operator's
approval of the pull request that lands this file is the gate for the feat
children listed under "PR stack".

## Intent

Twenty-two merged PRs (#226 through #248) built the machinery for databases that
are bound to a verified embedding model: identity contracts, MD5 manifests, provider
receipts, the profile store, the setup coordinator with its ledger and recovery,
profiled document ingestion, token planning, late chunking and profiled search.
None of it can be reached by a person. `WorkspaceSetup::run` has no caller outside
the test suites, the laptop has no model registry, no registered provider in its
configuration, and the running embedder serves the legacy TCP contract that the
managed path refuses.

This act opens that door at the smallest useful size: an operator can register the
approved Jina snapshot, run one verified provider service, create one managed
database from the CLI, ingest documents into it, and search it from the CLI, the
daemon and MCP. Everything the spec defers (frontend setup, code workloads, vector
indexes, migration of legacy vectors, smaller models) stays deferred.

Authority:
[docs/specs/database-embedding-profiles.md](database-embedding-profiles.md),
sections "Setup experience and provisioning", "Failure and retry boundary" and
"Existing databases and compatibility changes";
[docs/specs/recoverable-workspace-setup.md](recoverable-workspace-setup.md); and
[docs/specs/registered-provider-discovery.md](registered-provider-discovery.md).
CLAUDE.md rules 1 through 4 apply.

Related: [docs/specs/registered-model-loading.md](registered-model-loading.md),
[docs/specs/verified-late-passages.md](verified-late-passages.md) and
[docs/operations/model-bound-recovery.md](../operations/model-bound-recovery.md).

## What exists today, measured on the laptop

- `WorkspaceSetup::new(root, installation, owner, admin_system_client, catalog)` and
  `run(SetupRequest { operation_id, database, profile, manifest })` in
  crates/hades-core/src/workspace_setup/. `inspect` and `diagnose` exist. Callers:
  tests only.
- `VerifiedProvider::document_profile(generation, ChunkPolicy)` builds the
  initializing profile from a verified provider; `ChunkPolicy` is `Conventional
  { passage_tokens, overlap_tokens }` or `Late { context_cap_tokens,
  minimum_window_overlap_tokens, passages }`.
- `config.registered_providers` (YAML key `registered_providers`, fields
  `registry_root` and `providers[] { id, socket, timeout_seconds,
  max_response_bytes }`) supplies the installation catalog. `hades ingest` and the
  daemon's document search already refuse a managed database when this key is
  absent; the refusal is a message string, not a reusable error code. There is no
  configuration surface for the setup ledger root or the installation ID.
- `python -m embedding.registry --registry DIR register --provider ID --source
  SNAPSHOT --revision COMMIT` writes a `hades-model-registration-v1` record. The
  approved snapshot is `models/jinaai--jina-embeddings-v4` under the persistent
  data directory named in [docs/installation-thinkpad.md](../installation-thinkpad.md),
  which `~/.local/share/hades` links to. The 2026-10-21 qualification registered it
  in a temporary registry; manifest MD5 446160668af00248832ed8cf5d8d4432. No
  durable registry exists.
- `embedding.verified_http.create_verified_app(VerifiedJinaProvider)` serves
  `/v1/verify`, `/v1/embeddings`, `/v1/passages`, `/v1/late-passages` and
  `/v1/late-embeddings`. Its module docstring says "no environment/CLI activation
  yet": there is no entry point and no unit. The catalog requires a same-account
  Unix socket with a 0700 parent. The production unit runs `embedding.http_server`
  on TCP 127.0.0.1:8087 with 900 s idle unload, with the ceiling ruled at 20,000
  tokens on 2026-11-09 after a real-prose bisection (23,000 passed, 24,000 ran out
  of memory; PR #250); applying it to embedder.env is an operator step.
- Peak whole-card memory at the 20,000 ceiling was 22,491 MiB on a 24,463 MiB card,
  about 2 GiB of headroom with the desktop on the same card.
- The setup coordinator's inventory (`workspace_setup/database.rs`) accepts only
  `hades_setup`, `hades_workspace`, `documents`, `chunks`, `embeddings` and, once
  ready, `hades_ingest_jobs`. An identical retry of `run` and every `diagnose` call
  re-run that inventory. Any other user collection in the database, including
  `hades_schema` and the reading desk's `hades_reading_*` and per-session
  `reading_s_*` collections, has three effects. `diagnose` concludes
  `InspectRequired` with reason `SchemaNotVerified`. A `run` retry before the
  operation is `Ready` fails with a `Reconciliation` error and downgrades the
  receipt to `NeedsReconciliation`, after which every later `run` is refused as
  held for explicit administrative reconciliation. A `run` retry after `Ready`
  returns the same error but leaves the receipt `Ready`. Ingest and search are
  unaffected, because admission does not run the inventory.
- `Conveyance_KG` holds reading-desk collections only (7 readings and 3 saved
  conversations in the 2026-10-19 preflight record; no documents, chunks or
  embeddings). Reading collections are created lazily by the daemon, so a fresh
  database serves the reading desk without preparation.

## Goals

1. One CLI command creates a managed database end to end and reports the receipt.
2. The same installation configuration lets `hades ingest`, `hades db query`, the
   daemon and MCP operate on that database with no further flags.
3. Every step is recoverable through the existing ledger: a retry with the same
   request resumes; a changed request under the same operation ID is refused; an
   ambiguous creation is reported, never adopted.
4. The laptop ends the act with one managed research database, the reading desk
   pointed at it, and a documented path from empty to first search.

## Non-goals

- No frontend setup page. No daemon or MCP `db.setup` operation. The CLI is the
  only door in this act; the daemon command set is unchanged.
- No code workload. `hades codebase ingest` and `graph-embed` keep refusing managed
  databases.
- No adoption of `Conveyance_KG` or any existing database. The coordinator creates
  databases; it never claims one.
- No vector index. Managed search stays exact, as today.
- No change to the embedding model, ceiling, or Python dependencies.
- No federation, migration tooling or smaller-model adapter.

## Anti-goals

- A `db setup` that succeeds against the legacy TCP endpoint, or that falls back to
  it. The managed path exists to refuse unverified providers.
- Two independently loaded copies of the model competing for the card.
- Secrets or box paths in configuration examples that reach the public repository.
- A deployment where the daemon serves a managed database but its configuration
  lacks the catalog, so search fails only at query time.

## The parts

### Part A: a deployable verified provider service (services/, operator ruling)

This is the one piece that is a new capability rather than a wrapper, so it was a
spec gap needing the operator's election. The options considered:

1. Verified only. Add `python -m embedding.verified_server --registry DIR
   --provider ID --uds PATH` that loads the registered provider once and serves
   `create_verified_app` on the Unix socket. Switch `hades-embedder.service` to it.
   The legacy TCP endpoint disappears. Legacy databases lose semantic ingest and
   search; on this laptop none has vectors, so nothing is lost. `hades status`
   needs a verified-aware probe or will report the embedder as absent.
2. Dual bind, one load. The same entry point also serves the legacy `/v1/models`
   and `/v1/embeddings` on TCP from the same loaded provider. Keeps legacy clients
   alive at the cost of making the legacy contract depend on the verified loader.
   More code, more surface to qualify.
3. Second unit. Leave the legacy unit alone and add a second unit for the verified
   socket. Two model loads when both are warm (about 18.3 GiB of a 23.9 GiB card,
   twice the 9.16 GiB peak reserved in the 2026-10-20 qualification evidence); the
   legacy idle unload makes that rare but not impossible.

Elected: option 1 (operator, 2026-11-10, on the Planner's recommendation). It is
the smallest, it matches the spec's intent that the managed path is the path, and
the laptop has no legacy vectors to protect. The Olympus-era profile switcher is
not installed here, so there is no second unit to coordinate with. The socket
parent must be a 0700 directory owned by the service account, and the socket path,
the ledger root and the registry root must all be given as resolved paths with no
symlink ancestor: the catalog and the ledger open every ancestor without following
links. On this laptop `~/.local/share/hades` is itself a symlink, so the runbook
uses its resolved target (the persistent data directory named in
[docs/installation-thinkpad.md](../installation-thinkpad.md)) for all three, for
example `<data>/run/embedder/`, `<data>/setup-ledger/` and `<data>/model-registry/`.

Configuration contract for the entry point (Planner's call): it reads the existing
`HADES_EMBEDDER_*` variables (device, fp16, batch size, max tokens, idle timeout)
so the laptop's `embedder.env` carries over, plus `--registry`, `--provider` and
`--uds` on the command line. Idle unload is honoured as today; the laptop runbook
sets it to 0 for the verified unit, because after an unload every held client gets
`PE_RECEIPT_MISMATCH` and the daemon re-verifies and reloads on the next query.
The reload costs 6 to 7 s on this laptop (2026-11-09 bisection journal), plus the
first request's cold-cache penalty. The qualification case states the configured
timeout it ran with.

The entry point always constructs `VerifiedJinaLateProvider`, which subclasses the
conventional provider and advertises the late-chunk and tokenizer-offset
capabilities. A conventional policy validates against that observation too, since
it requires no late capability, so one provider class serves both policies and no
class selector is needed. The PR 2 qualification case checks that `/v1/verify`
over the socket reports both late capabilities and that `/v1/late-passages`
answers.

Repository footprint: the repo ships `deploy/systemd/hades-embedder@.service`, the
`hades-embedder-profile` switcher and the gpu0/1/2 profiles from the Olympus era.
Olympus HADES was decommissioned on 2026-10-20 (operator's account), so nothing runs
them. The verified unit is added beside them as `hades-embedder-verified.service`,
and README and CLAUDE.md mark the templated profiles as legacy TCP serving. Under
option 1, `hades status`, `hades embed`, codebase embedding and legacy `hades ingest`
lose their endpoint on this laptop: `status` gets a verified-aware probe in Part B;
the others fail with a connection error until they are profile-aware, which the
runtime docs must say. The operator accepted this footprint and this list of
surfaces allowed to break on 2026-11-10.

### Part B: the CLI door (hades-cli, hades-core config)

Add `hades db setup` as an Admin-tier CLI command group with three verbs:

- `hades db setup run --database NAME --provider ID --operation-id ID --generation G
  --chunking late|conventional [policy flags]`. Builds the `_system` admin client
  from the configured credentials, the catalog from `registered_providers`, verifies
  the provider (this loads the model), constructs the profile through
  `VerifiedProvider::document_profile`, and calls `WorkspaceSetup::run`. Output is
  the JSON envelope with the `SetupReceipt` as `data`; a non-ready status is a
  failed command (`print_output_with_success`).
- `hades db setup inspect --operation-id ID`: the ledger receipt, no model load.
- `hades db setup diagnose --operation-id ID`: the advisory recovery report.

Configuration: a new optional `setup` section, `setup: { ledger_root, installation_id
}`. The ledger root must exist, be absolute and 0700, as the coordinator already
demands; the command does not create it. The owner ID passed to the coordinator is
the ArangoDB username the CLI connects as, so two users on one installation hold
separate operations. Both keys are documented in config/hades.yaml as commented
examples with placeholder paths.

Policy flags: `--chunking` is required (no default policy is silently chosen).
`late` takes `--context-cap-tokens`, `--window-overlap-tokens`, `--passage-tokens`,
`--overlap-tokens`; `conventional` takes `--passage-tokens` and `--overlap-tokens`.
Setup already validates each policy against the provider's observed ceiling and
refuses one that does not fit; the CLI adds nothing on top of that.

`--generation` is required, like `--chunking`: a default such as a timestamp would
change the request digest on every repeat and defeat the ledger's recoverable
retry, which only admits an identical request under the same operation ID. The
receipt echoes the request hash, and an operator retries by repeating the exact
command. Recovering an omitted generation from the saved ledger record would need
a new coordinator accessor and is deferred to the Part B implementation PR.

Tiering: `db setup` is `Admin`. It needs `_system` database-creation authority; the
laptop grants the `hades` user rw on `_system` and `*`
([docs/installation-thinkpad.md](../installation-thinkpad.md), user setup).

Admin client and ledger scope: the ledger scope hashes the `_system` client's socket
path and base URL, and `ArangoClient::from_config` derives the base URL from
`database.host` and `port` even when a socket is used. The CLI therefore builds the
admin client exactly as `hades db create-database` does, and the configuration docs
state that the database endpoint settings are frozen for a ledger's lifetime: a
changed host, port or socket needs a new `ledger_root`, and the old ledger is kept
for the operations it records. No daemon variant: the dispatch enum is untouched
in this act, and the MCP `create_database` tool keeps creating legacy databases
under its prefix, which the docs must now say plainly.

### Part C: the laptop runbook and first corpus (docs, operator steps)

A new section in [docs/installation-thinkpad.md](../installation-thinkpad.md),
written as the act is exercised:

1. Register the snapshot into a durable registry (`<data>/model-registry`, under
   the resolved data directory, never through the `~/.local/share/hades` link) with
   the resolved commit ID. Record the manifest MD5 and
   compare it with the 2026-10-21 evidence.
2. Configure `registered_providers` and `setup` in the HADES YAML. Create the
   ledger root and the socket parent with mode 0700.
3. Switch the embedder unit per the Part A election; verify `/v1/verify` over the
   socket from the service account.
4. `hades db setup run` for the new database. Its name is `Conveyance_KG_v2`
   unless the operator renames it before this step (Planner's default; the only
   constraint is the setup name grammar). Record the receipt, operation ID and
   generation.
5. First combined backup, before any ingestion: with setup quiescent, archive the
   ledger root and the registry root with ownership and permissions preserved, and
   take an arangodump of the new database with system collections, into one dated
   backup directory. A database-only backup cannot restore the setup ownership
   record. This procedure becomes a "Managed workspace" section of
   [docs/operations/thinkpad-backup-restore.md](../operations/thinkpad-backup-restore.md)
   in PR stack item 5.
6. Point the daemon and viewer at the new database (unit edits, operator's lane).
   Move the reading-desk state, after the inventory widening (PR stack item 3) is
   merged and deployed: arangodump from `Conveyance_KG` with
   `--include-system-collections true`, as thinkpad-backup-restore.md already does,
   and arangorestore into the new database the `hades_reading_*` and `reading_s_*`
   collections together with the `_graphs` definitions for the reading graphs,
   because the reading desk resolves each graph through gharial before every write.
   Verify the desk lists the same readings, then open a reading session and make
   one mutation (a pin or a proposal) to prove the graph definitions resolve.
   `Conveyance_KG` is left in place until the operator drops it by hand.
7. First ingest: the repository's own `docs/` tree (113 markdown files and 215,490
   tokens at 52d04d4, the corpus the 2026-11-09 ceiling bisection used; the tree
   has grown since) unless the operator names another tree first. It is real prose
   with known answers, so `hades db query` can be checked with questions such as
   which collections the setup coordinator's inventory accepts. Record the sources
   returned.

The embedder ceiling is 20,000 tokens (ruled 2026-11-09, recorded by PR #250).
The late policy saved in step 4 must fit that ceiling, and setup enforces it. The
defaults for the first managed database (Planner's call, open to override before
step 4): `context_cap_tokens` 16,384, which leaves room under the live ceiling for
the prompt reserve and small future changes; `minimum_window_overlap_tokens`
1,000, matching the laptop's legacy prechunk overlap; passages of 500 tokens with
200 overlap, the legacy chunking precedent. These numbers are frozen into that
database's profile; changing them later means a new generation.

## Decided, and by whom

- Operator, 2026-11-09: resume HADES with deploy first, then this door. Keep feat/,
  fix/, docs/ branch prefixes. Private target dirs under ~/Projects/hades-targets.
  Operating embedder ceiling 20,000 tokens (PR #250).
- Operator, 2026-11-10, adopting the Planner's recommendations: Part A option 1
  (verified-only service, legacy TCP surfaces allowed to break on the laptop as
  listed in Part A); widen the coordinator inventory to tolerate data-owned
  collections (`hades_schema`, `hades_reading_*`, `reading_s_*`) as its own feat
  child with disposable-database tests for retry and diagnose, so a managed
  database can carry a schema and the reading desk; move the reading-desk state
  into the managed database once that widening is deployed.
- Planner's calls, open to override while reading: CLI shape and verbs in Part B;
  `setup` config section; owner = ArangoDB username; `--chunking` and
  `--generation` required with no silent default; the `HADES_EMBEDDER_*` variables
  reused by the verified entry point with idle unload 0 on the laptop; ArangoDB
  endpoint settings frozen for a ledger's lifetime; `hades-embedder-verified.service`
  added beside the legacy templates; database name `Conveyance_KG_v2`; the
  repository `docs/` tree as the first corpus; the late policy defaults in Part C.

## Open for the operator

Nothing blocks the build. Two values can still be changed before the runbook
reaches them: the database name (before step 4 of Part C) and the first corpus
(before step 7).

## Left to the builder

Flag spellings, help text, error-code names for the new failure modes, the exact
JSON field order of the receipt, test layout. Tests: service-free tests for config
parsing and refusal paths; disposable-database tests through `with_temp_db` for
run, identical retry, changed-request refusal and diagnose; the opt-in real-Jina
qualification extended by one case that drives `hades db setup run` end to end on
the laptop and records evidence under docs/evidence/.

## PR stack

#249 and #250 are merged. This stack opens with the design as its anchor:

1. docs (anchor, draft to main): this design, under docs/specs/db-setup-door.md.
2. feat: Part A, the verified-only entry point and `hades-embedder-verified.service`
   under deploy/systemd/, with the qualification case that proves `/v1/verify` over
   the socket.
3. feat: widen the coordinator inventory for data-owned collections, with
   disposable-database tests for retry and diagnose, and the matching change to
   [docs/specs/recoverable-workspace-setup.md](recoverable-workspace-setup.md).
4. feat: Part B, the `hades db setup` commands, `setup` config and the verified-aware
   `hades status` probe, with service-free and disposable-database tests.
5. docs: Part C runbook, written after the operator has run it, which also adds the
   "Managed workspace" combined-backup section to
   docs/operations/thinkpad-backup-restore.md.

Each child targets the previous branch. The Planner undrafts children; the operator
collapses the stack and undrafts the anchor once.
