# Guided document reading

HADES provides a human-led reading desk at `/reading`. The human marks source
passages and their meaning. An agent in its own terminal reads those exact
references and proposes notes or connections. Only human acceptance publishes
interpretation to the graph. No agent chat, model credentials, or agent execution
runs in the browser.

## Start a reading workspace

Build the CLI and viewer, and regenerate the extraction bindings before starting
an updated extraction service. Use a dedicated database for a trial. These are
example names, not workstation settings:

```bash
cargo build -p hades-cli -p hades-frontend
make -C services proto-gen PROTO_DIR=../proto PYTHON=/path/to/venv/bin/python
hades --db reading_trial db create-database reading_trial
hades --db reading_trial daemon --socket /private/run/reading.sock \
  --mcp-bind 127.0.0.1:10444 --mcp-token-file /private/reading-token \
  --mcp-dbs reading_trial --mcp-db-prefix reading_trial --mcp-guided-reading
hades-viewer serve --db reading_trial --reading-db reading_trial \
  --reading-socket /private/run/reading.sock --password-file /private/viewer-password \
  --bind 127.0.0.1:8089
```

Run the daemon and viewer in separate terminals or user services. Set `HADES_CONFIG`
explicitly and supply existing database credentials through `ARANGO_PASSWORD`.
The database-creation command requires an account authorized to create it. For an
existing authorized database, omit creation. Do not retarget another running daemon.
The viewer verifies its expected database on the same socket connection used for
an operation, without reconnecting that operation into a different database.

The MCP token is separate from the browser password. Private LAN binds require
viewer authentication. Guided reading refuses command-line passwords: use a file.
Browser mutations require a matching Origin and custom header, deny cross-site
Fetch Metadata when present, and grant no cross-origin CORS access. The Unix socket
is an administrator interface; protect its directory with local permissions.

`--mcp-guided-reading` denies direct graph creation/writes and ingestion through
that MCP endpoint, even when provisioning prefixes are configured. It permits
`reading_propose` inside the authorized prefix. MCP cannot accept proposals or
advance stages. Also merge the relevant deny entries from
`config/examples/guided-reading-claude-settings.json` into the guided Claude Code
session's permission settings, adapting the server alias when necessary. Do not
replace unrelated client settings. These controls prevent accidental bypass; they
do not isolate a hostile process with the operator's filesystem or database access.

## Read and review

1. Add a document. The format picker uses the extraction service's advertised
   capabilities. Original bytes are hashed and archived before extraction.
   The saved conversion opens as the shared reading surface for every format.
   Failures retain the original and appear under **Archived sources and recovery**.
2. Complete **Bound** in your own words: what the source IS and IS NOT.
   Choose **Save boundaries and start reading** to save both fields and
   advance to **Read**, enabling highlighting. **Save boundaries** alone keeps the
   current stage. Failed saves and background refreshes preserve the current draft.
   Bound revisions add graph nodes and events rather than
   replacing earlier boundaries.
3. Select text within a converted passage and choose **New node**, **IS**, **Explicit IS NOT**, or **Question**.
   Labels accompany colors. Highlight creation, recoloring, and removal append
   events. PDF figures can be selected by a page rectangle, with an optional
   source caption; a missing caption does not imply missing figure content was read.
4. Share focus or have the agent poll `reading_context`. The agent uses
   `reading_propose` to post a draft or a named pin set. Pins do not navigate until
   the human chooses one. A notification/poll result does not execute the agent.
5. Review the proposal's evidence, boundaries, standing, and inferred rationale.
   Save edits, then accept that exact revision or reject it. Revision conflicts
   require rereading. Model origin and human editor/reviewer labels remain separate.
6. Complete **Read**, then **Connect**, after resolving pending proposals. Each
   accepted note receives a basis edge to the bound revision used at acceptance.
   Connections can link that note to existing objects from named graphs present
   when the session opened. Their existing records are not rewritten. Missing
   note/basis endpoints are shown as suspect.
7. In **Outline**, arrange accepted notes and export their titles and references.
   Export the JSON authorship record separately. Earlier stages can be revisited
   explicitly; forward progression still checks the stage requirements.

Use *Lost in the Middle* or another source for human testing only after choosing
the intended database. Automated write tests use isolated databases and synthetic
sources. Do not treat test acceptances as reviewed research.

## Source fidelity and anchors

All formats, including PDF, open the immutable `docling-blocks-v1` conversion
for reading and discussion. Numbered passages retain their extraction block IDs.
The header always offers **Converted**, **Original PDF**, and **Side by side**
for PDF sources. Side by side enables Wide layout and gives each surface its own
scroll pane. PDF page changes leave the converted pane in place. On narrow
screens the two panes stack. Turning Wide layout off returns to Converted.
Where PDF page locations exist, **Check page** opens that page in the original,
keeping the conversion visible during side-by-side comparison. Missing picture pixels and empty
passages remain visible as conversion gaps. The original extraction stays unchanged.

The optional original-PDF comparison uses a local, pinned PDF.js renderer and
selectable text layer. The serialization
`pdfjs-text-v1` concatenates that layer's DOM text-node content in order, without
adding whitespace. Quotes retain physical page, source hash, exact text, prefix,
and suffix. Reveal requires a unique matching passage; it never silently moves a
reference. PDF figure boxes use normalized left/top/width/height on the physical
page. The user supplies any caption. The server treats browser PDF selection as a
human-supplied reference, not an independently re-extracted proof of the quotation.

The primary conversion uses `docling-blocks-v1`: ordered blocks, headings, text,
tables,
captions, and available layout locations. The stored representation is immutable
once preparation succeeds, so a converter upgrade cannot move an accepted anchor.
The service checks structured quotes against their archived block. Tables have a
readable grid and an explicit extracted-text selector. Missing picture pixels and
equation text are reported. Open the unchanged original to compare fidelity.
Format routing does not guarantee every upstream optional converter dependency is
installed. Native Office editing and pixel-identical Office rendering are not
provided. Existing PDF text/figure anchors retain their representation and reveal
in the original view. They are not silently remapped into converted passages.
PDF and DOCX have an automated real-converter/browser test.

## Correct conversion errors

Use **Edit passage** beside a converted paragraph, heading, list item, or caption.
Compare it with the original, correct the transcription, and choose **Save correction**
or **Cancel edit**. Tables, equations, pictures, and code blocks have no structural
editor in this slice. Corrections are human actions, separate from interpretive notes.

Each save adds a passage revision scoped to this reading session. Other readings of
that source keep their own text. **History / changes** shows the original extraction,
correction attribution, before/after text, and changed wording. **Restore this revision**
adds another revision. It never removes history or changes the archived original.

New highlights carry the passage revision and correction session. Existing highlights
and accepted notes keep their exact evidence. Changed passages flag those references
for review. **Reveal** displays their historical wording, with **Return to latest** to
resume reading the correction. Even a restore does not silently re-anchor old references.
MCP focus packets and JSON exports report affected references. Correction history is
included in the authorship export. Corrections do not update embeddings.

The service rejects stale edits, validates restore text, and keeps exact retries safe.
An interrupted save remains available through the existing retry control. An open
editor survives polling, so updates from another participant do not erase a draft.
Each passage allows 100 corrections, with 64 KiB of text per revision, within the
session size limits below.

## Persistence and attribution

Domain logic lives in `crates/hades-core/src/reading/`. CLI, MCP, and the viewer use
those operations. The viewer uses the daemon Unix protocol, not a CLI process per
poll. It polls only its selected session every three seconds, recovers from failed
polls, and retains an interrupted mutation's exact request for explicit retry.

- `hades_reading_sources`: source-hash archives and derived reading surfaces.
- `hades_reading_sessions`: current projections, pins, focus, outline order.
- `hades_reading_events`: append-only mutations and retry receipts, with a hash chain.
- `reading_SESSION_nodes` / `_edges`: accepted boundaries, notes, and connections.

An exclusive transaction publishes a mutation, event, acceptance receipt, and graph
writes together. Duplicate request IDs return their original receipt only for an
identical payload and channel. Stale revisions and missing endpoints abort. Related
external collections are declared read-only within acceptance transactions. Source
re-ingestion owns none of these authored collections.

The `chain_valid` export field checks sequence, hashes, and the stored chain head.
Exports include events only through the sampled session revision, so concurrent
appends cannot invalidate a valid history. It detects changed records relative to
that head; administrators can rewrite both history and the head. Attribution names are declared participant labels, not
cryptographic identities or distinct bearer-token principals.

Archiving, preparing, mapping accepted knowledge, embedding, and semantic readiness
are separate states. This desk does not submit embeddings. Existing HADES semantic
retrieval remains available through the graph workspace for already-indexed content.

## Bounds and recovery

This first slice accepts originals up to 8 MiB and structured surfaces up to 4 MiB.
Sessions allow 2,000 events, 8 MiB of event history, and a 2 MiB current projection, with explicit errors at
those limits. Lists show at most 100 sessions/sources and report partial results.
Focus accepts six references. Agent context is capped at 12,000 serialized bytes;
large packets fall back to identifiers with an explicit expansion notice.

```bash
hades --db reading_trial reading read
hades --db reading_trial reading read --session SESSION --view export
hades --db reading_trial reading read --view sources
hades --db reading_trial reading extract SOURCE_SHA256
hades --db reading_trial reading apply /private/exact-saved-mutation.json
```

Saved readings, review, interrupted-action reconciliation, and export remain
available when the extractor is offline. New uploads stay disabled until
capability discovery succeeds. Refresh to reconnect after restoring the extractor.

An extraction request can take up to ten minutes. A lost response does not justify
changing source identity. Inspect the archived source and retry preparation; a
ready representation is reused. Administrators should back up all three reading
collections and the session graph collections together. Source originals are in
the archive collection, not only on the upload machine.

## Focused validation

```bash
cargo test -p hades-frontend
cargo test -p hades-proto --test proto_types
python scripts/test_isolated_database.py --arangod /path/to/arangod --contract reading
```

The browser flow runs on a private ArangoDB instance, CPU extractor, daemon, and
viewer. It uses synthetic PDF/DOCX fixtures, real extraction, real MCP calls,
human edit/acceptance actions, figures, pin/focus, export, and reload:

```bash
python scripts/test_isolated_database.py --arangod /path/to/arangod \
  --reading-e2e-python /path/to/services/venv/bin/python \
  --reading-node /path/to/node --reading-playwright /path/to/playwright \
  --reading-chromium /path/to/chromium --reading-model-cache /private/hf-cache
```

Build both debug binaries first, set `CARGO_TARGET_DIR`, and populate the Docling
model cache beforehand. This test is offline and never selects an ambient database.
Screenshots and exported fixture records stay in the printed temporary directory.
