# Knowledge workspace

The `hades-frontend` crate provides a local reading workspace over existing HADES
graphs. It combines a document outline, glossary, source reader, graph exploration,
and semantic search. It calls the `hades` CLI and depends on no other HADES crate.

## Start the workspace

Build the viewer and provide an explicit database:

```bash
cargo build -p hades-frontend
hades-viewer serve --db research --hades-bin hades --bind 127.0.0.1:8088
```

Run the built binary from Cargo's configured target directory, or install it with
`cargo install --path crates/hades-frontend`. The CLI inherits the server process's
configuration and credentials. Set `HADES_CONFIG` explicitly when using a checkout
configuration and supply credentials through the existing environment.

Open `http://127.0.0.1:8088/?db=research&graph=workspace`. Replace the example
database and graph with existing names. Repeat `--db` to allow additional databases.
The existing viewer authentication and host restrictions apply to every API route.
Omitting all `--db` flags discovers the databases accessible to the configured
credentials at startup. An explicit repeated list keeps a chosen default first
and limits the picker to those databases. Launch settings stay local.

## Read, search, and share references

- **Outline** follows `contains` edges between nodes with `attrs.type` equal to
  `document` or `section`. Source positions order siblings when available.
- **Glossary** lists `concept` and `unresolved_keyword` nodes. Equal labels do not
  merge distinct objects. **All objects** also exposes other node types.
- **Read**, **Split**, and **Graph** change the central view. The reader displays
  stored text literally, including Markdown and equations. It does not execute
  source HTML or fetch referenced illustration files.
- **Layout** offers Auto, Wide, and Compact. Wide places the reader and graph
  beside each other and gives the context pane more room on large displays,
  including 3840-by-2400 monitors. The preference survives reloads in this browser.
  Smaller windows retain the responsive layout, including when display scaling
  reduces the browser's usable width.
- **Search** selects the Documents (`default`) or Code (`codebase`) retrieval
  profile. Search covers that profile in the selected database, not just the
  displayed graph. It requires completed embeddings and a working embedder.
  Saving an authored graph node alone does not make it searchable.
- A search result resolves its actual stored chunk ID. Its connections can lead
  back to source sections when ingestion created those links. If no unique chunk
  resolves, the result opens its parent document or file.
- Select an object or inspect a relationship, then add it to the context tray.
  Selecting text in the reader first supplies that exact excerpt. Copy the packet
  into an agent conversation. It includes database, graph, object IDs, edge
  endpoints, and available source versions, hashes, and locations.

The tray holds at most six references, each with an excerpt of up to 1,200
characters, and caps the serialized packet at 12,000 characters. Truncation is
explicit. The tray lives only in the tab's memory. A copied view link restores
the selected node or edge in the workspace, subject to current access and existence.

## Current boundaries

This slice is read-only. Editing, embedding submission, job progress, durable
conversation sessions, automatic agent notifications, and live change subscriptions
remain future work. Copying a reference does not trigger agent execution. Use
**Reload** to refresh changes made through MCP or the CLI.

The library and connected-object lists describe the loaded graph slice. Partial
snapshots remain labeled. Expand nodes or increase the load limit to see more.
Late search and expansion responses are discarded after database or graph changes.

## Focused validation

```bash
cargo test -p hades-frontend
cargo clippy -p hades-frontend --all-targets -- -D warnings
node crates/hades-frontend/web/tests/workspace.test.cjs
```

The optional browser smoke test requires Playwright and Chromium plus a running
viewer. It intercepts every API request with synthetic fixtures and writes no data:

```bash
WORKSPACE_URL=http://127.0.0.1:8088/ \
  node crates/hades-frontend/web/tests/browser-smoke.cjs
```

`PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` can select existing local installations.
For a password-protected preview, `WORKSPACE_PASSWORD_FILE` supplies a local
credential file and `WORKSPACE_USERNAME` optionally selects the username.
Coverage includes literal source rendering, quotations, edge deep links, source
navigation, database-switch races, failure states, and mobile overflow. Live
read-only checks should separately verify semantic retrieval and source connections
against an explicitly selected, already-ingested corpus.

## Proposed next slices: uploads and agent conversations

These are implementation recommendations, not currently available controls.

### Upload and process a document

Show the target database before submission. Keep a separate write-enabled
database allowlist: permission to browse a database does not grant ingestion.
Accept a bounded file upload into an operator-configured directory, retain the
original filename as metadata, and allocate the stored path on the server.
Use a content hash and database scope to identify repeat submissions. Never use
the browser-supplied filename as a host path or silently force a replacement.

Reuse `ingest.start` and `ingest.status` in
`crates/hades-core/src/dispatch.rs` and the persisted job ownership in
`crates/hades-core/src/ingest_jobs/records.rs`. The viewer currently consumes CLI
JSON, while these detached jobs are exposed through MCP/service dispatch. Add
a narrow CLI adapter to the existing daemon operations before connecting the
upload route. Starting a detached job inside a short-lived CLI process would
lose its owner; the long-lived daemon must own the job. Avoid a second ingestion
implementation inside the frontend.

Display upload receipt, extraction/embedding job state, terminal errors, and
retrieval readiness separately. Preserve the source after failure and offer
explicit retry. Poll the existing job record initially and refresh the relevant
view after success. Verify semantic retrieval before labeling content searchable.

Ordinary file ingestion does not implement the graphable mapping documented in
`docs/graphable-document-ingestion.md`. A document can be embedded without its
heading/concept graph being created. A later graphable import operation should
apply that mapping with source versions, repeat-safe identities, and a review
step for uncertain cross-document equivalence. Do not label an upload as a
completed graphable import until those separate operations finish.

Acceptance: upload a small fixture to an isolated database, observe its job,
find its exact source through semantic search, repeat without duplicate content,
and recover from extractor/embedder failure. Validate scope changes, file/path
limits, write permissions, and retries without touching production corpora.

### Conversation beside the graph

Provisional recommendation: launch dedicated sessions through provider adapters.
Use [Codex App Server](https://learn.chatgpt.com/docs/app-server) for threads,
streamed turns, interruption, and approval requests. Use the
[Claude Agent SDK or Claude Code's streaming interface](https://code.claude.com/docs/en/headless)
for the corresponding Claude session. Both executables are available in the
reviewed local deployment; account access and a successful integration turn
still need verification. Keep their executable paths and authentication local.

HADES owns the pane, provider selection, session-to-workspace association, and
graph reference attachments. Each provider owns its conversation execution and
native session ID. Switching providers creates or resumes that provider's session;
it does not silently give it the other provider's history. Include explicit graph
IDs and bounded passages when the user sends selected context. Never infer agent
context from screen pixels. Show provider, session, and database beside the composer.

The server needs a supervised session bridge, streamed output, cancellation,
permission prompts, and reconnect handling. Keep model credentials server-side
and preserve native permission decisions. Bind HADES MCP access to the intended
database scope. A UI label alone cannot constrain an agent's tools. Starting or
resuming a local session also does not prove access to an existing conversation
in another host, including the conversation used to build this frontend.

Use the wide layout's right pane for chat and a compact attached-context tray.
Graph links in replies should resolve exact references and reveal the object.
An explicit send starts a turn; highlights alone do not. Retain provider session
IDs and message reference attachments separately from canonical graph knowledge.

Acceptance: start one session for each provider, attach a selected source object,
receive a streamed response with a resolvable graph reference, interrupt a turn,
handle a tool approval, reconnect without duplicate submission, and switch
databases without silently carrying a session into another write scope.
