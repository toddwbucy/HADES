# Knowledge workspace

H.A.D.E.S. stands for **Human Augmented Data Exploration System**.

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

## Adopted direction: guided reading

Owner decisions recorded on 2026-10-17. These describe future implementation,
not controls shipped in this read-only workspace. They supersede the previous
embedded-chat recommendation.

The workspace supports deliberate reading and development of theory. Claude Code
runs in its own terminal. HADES supplies shared references, durable pin sets,
reviewable proposals, and current focus. Pins do not move the user's view.
No interpretive note or substantive relationship becomes accepted knowledge
without user acceptance. Mechanical source archiving, extraction, and indexing
are allowed.

### Human-led reading and node boundaries

The [PDF layout probe](pdf-layout-probe.md) records the initial extraction and
browser-selection evidence, remaining anchor work, and reproduction commands.

The human identifies important passages by highlighting and classifying them.
The agent may then propose connections to existing graph objects, showing the
source passages, proposed relationship, and any inference or uncertainty. A
proposal becomes accepted knowledge only after the human reviews it.

IS and IS-NOT boundaries are general node semantics, not a Bastion-only rule.
Each node states what it represents and excludes in the context of its source.
Store the source revision and evidence with each boundary statement, preserving
whether it is source-stated or proposed and whether the human accepted it.
An unstated boundary remains explicitly unresolved; do not invent an exclusion
to complete a field. A source's candidate or qualification remains visible.
Cross-document equivalence requires comparing these contextual boundaries,
not just matching labels. Existing imported nodes require review, not an
automatic rewrite to satisfy this new convention.

Highlight colors express semantic labels defined by a versioned palette. Show
text labels as well as color. Creating, recoloring, or removing a highlight
records a new event. The proposed node or relationship is a separate revisioned
object, and the human reviews its endpoints and evidence before acceptance.
The event history retains the human selection, model draft, human edits, and
acceptance as distinct contributions. Acceptance does not erase model origin.

### Architecture and ownership

New importer and workspace operation logic belongs in `hades-core`. The CLI stays
as a thin interface, including administration and recovery. A HADES service owns
pins, proposals, and focus. The viewer will use the daemon's Unix-socket protocol
instead of launching a CLI process per poll. This deliberately supersedes the
CLI-only integration rule for that future work. The current Unix daemon serves
one database, so guided reading initially uses the theory database. Start with
scoped polling; common-write-layer change publication comes later.

Keep extraction in the Python Docling service. Layout preservation will be an
additive change to `docling_backend.py` and the extraction proto. Do not migrate
extraction to Rust or add agent hosting, credentials, and chat execution to the
viewer. Version the pin, focus, and proposal formats for later ODE integration.

### Source and review contracts

Own graphable documents use a mechanical importer for headings, glossary,
keywords, illustrations, and declared references. The October 16 script is a
fixture, not the importer. Exclude interpretive assertions unless supplied and
reviewed. Follow `docs/graphable-document-ingestion.md` for revision-scoped
identities and source preservation. Mapped, embedded, and searchable are distinct
states.

External PDFs use guided reading. For text, anchor accepted notes to the immutable
source hash, page, exact quote, and surrounding text from the browser's PDF text
layer, following the W3C Web Annotation selector approach. Docling boxes assist
outlines and spoken-reference resolution; accepted text notes do not depend on
an extractor's element numbering. Figures use page, box, and caption. Ambiguous
or missing matches must remain visible rather than silently moving an anchor.

A proposal has its own revision. Acceptance writes exactly the reviewed revision,
is retry-safe, returns created IDs, and preserves model origin and reviewer
attribution. Guided-reading Claude Code permission settings deny
`mcp__hades__kg_add_node` and `mcp__hades__kg_add_edge`: the agent proposes, and the
HADES service performs the accepted write. This guards against accidents, not
hostile processes. Persistent pins and decisions remain separate from accepted
knowledge; the persistence backend is still open.

Before shipping browser write routes, require a custom header and an Origin
check, use restrictive CORS, and check `Sec-Fetch-Site` where present. Plain-HTTP
LAN browsers may omit Fetch Metadata. Read the viewer password directly from a
file. Native MCP authorization remains separate from browser-origin checks.

### Implementation order and completion criteria

1. Correct the stale server security comment and merge the tested reading
   workspace after addressing its outstanding review findings.
2. Implement the mechanical importer in `hades-core`. Verify identical input
   adds nothing, a new version preserves old rows, conflicting content under an
   existing identity is reported, and interrupted imports resume safely. Run
   end to end on one real graphable document in an isolated database.
3. Only after step 2 completes, compare Docling boxes with pdf.js text selection
   on a two-column paper, an equation-heavy page, and a figure with a split caption.

Pins, focus, and the PDF reader follow this weekend's work. Favor installation,
resource use, and recovery that work for early student users on unfamiliar
machines, including a single RTX 3090. Keep deployment-specific settings local.
