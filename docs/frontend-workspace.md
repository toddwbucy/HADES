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

## Read, search, and share references

- **Outline** follows `contains` edges between nodes with `attrs.type` equal to
  `document` or `section`. Source positions order siblings when available.
- **Glossary** lists `concept` and `unresolved_keyword` nodes. Equal labels do not
  merge distinct objects. **All objects** also exposes other node types.
- **Read**, **Split**, and **Graph** change the central view. The reader displays
  stored text literally, including Markdown and equations. It does not execute
  source HTML or fetch referenced illustration files.
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
Coverage includes literal source rendering, quotations, edge deep links, source
navigation, database-switch races, failure states, and mobile overflow. Live
read-only checks should separately verify semantic retrieval and source connections
against an explicitly selected, already-ingested corpus.
