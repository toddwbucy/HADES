# Graphable Document Ingestion Mapping

Status: mechanical importer implemented in `hades-core` on 2026-10-17.
The October 16 loader remains a historical fixture, not the importer.
This interprets an existing graphable document without rewriting its argument or
certifying its claims. Here, graphable documents declare heading nodes and
illustrations in Markdown comments, keywords, and a document-local glossary.
The authoring skill supplies the format; HADES performs writes and embedding
separately. The CLI is a thin adapter over `graphable::Plan` and `graphable::import`.
No interpretive note or substantive relationship becomes accepted knowledge
without user acceptance. Mechanical source archiving and structural mapping
are allowed. This importer performs only that mechanical work.

## Run and recover

```bash
hades --db theory graphable import paper.md --graph theory --dry-run
hades --db theory graphable import paper.md --graph theory --create-graph
hades --db theory graphable import paper.md --graph theory
```

The first command parses without connecting to ArangoDB. The second creates an
absent authored graph. The third verifies a repeat and inserts only missing rows.
The database must already exist. Choose it explicitly. No embedder, GPU, or Python
runtime is needed to map Markdown. This command is currently an admin CLI
operation, not a new MCP tool or viewer write route.

For a reviewed Markdown extraction, optionally add
`--source-archive original.docx --extraction-note "How the extraction was checked"`.
The importer records that association as operator-supplied. It does not certify
extraction fidelity. DOCX and PDF extraction remain outside this Rust importer,
in the Python workflow. Do not pass a model-written derivative as an accepted
interpretation without review.

An import commits its nodes, edges, and receipt together through
`db::transaction::run`. Cancellation before commit aborts the transaction. If the
commit response is lost, rerun the same command to reconcile durable state.
Conflicts stop the import without replacing existing content. There is no force
flag. A failed first attempt may leave empty schema collections. Correct the
reported source or destination problem and retry. Do not delete the database.
Retry verification reads at most 500 rows per request through the transaction,
compares every payload, and inserts only missing rows. An edited row is a conflict.

Bounds are 8 MiB per input/archive, 100,000 Markdown lines, 10,000 declared nodes,
20,000 total nodes, 50,000 edges, and 12 MiB of serialized transaction payload.
The shared transaction deadline is 60 seconds. Exceeding a bound fails explicitly.

## Source and identity

Keep the original file unchanged and retain a byte-identical version archive.
Store its SHA-256, document ID, version, title, author, source citation, and source
status. Treat extracted text as a derived representation of that source.

Preserve declared node IDs in `source_node_id` and `logical_id`. Database keys
have a type prefix and the first 32 hexadecimal characters of SHA-256 over the
JSON array `[doc_id, version, source_sha256, logical_id]`. Edge keys hash the JSON
array of endpoint keys and relation. JSON tuple encoding avoids delimiter
collisions. The mapping identifier is `graphable-mechanical-v1`.

`source_sha256` identifies the exact Markdown being mapped. Locations refer to
that representation. When supplied, `original_source.original_source_sha256`
identifies the unchanged original behind the extraction. Both files are retained
byte-identically as base64 in a revision receipt in `<graph>_imports`. The graph
has one document root with the complete parsed source text, not a second
canonical argument. The receipt also records counts, source metadata, mapping
version, and a mapped-payload hash.

A graph-scoped receipt reserves `(doc_id, version)`. An identical repeat compares
every stored row, not just the receipt. A changed source, archive, provenance,
mapping, or row payload under that identity requires review. Earlier source
versions and authored edits are never silently overwritten. A retry can restore
missing rows whose remaining payloads match. Recovery does not reconcile or
remove extra administrator-created rows.
Payload comparison treats an exactly representable integral float and its integer
readback as equal, including inside nested metadata. It does not round integers,
coerce strings, or change source hashes or archived bytes.

The October 16 fixture used delimiter-joined keys and mapping version `0.1`.
The importer refuses to append a different mapping into that existing source
version. Use a separate graph to compare mappings. Adoption of a legacy mapping
requires an explicit future migration, not a blind rerun.

Logical IDs remain stable across versions. Database handles identify a particular
version. A later version receives new handles and retains earlier rows. Resolving
"latest" and deciding which interpretations to supersede remain explicit work.

## Graph mapping

Create the authored graph with `kg_create`, using `<graph>_nodes` and
`<graph>_edges`. Every relationship carries its origin and source evidence.
The MCP handlers require exactly the layout created by this tool: one edge
definition whose source and target collections are `<graph>_nodes`, `documents`,
and `codebase_files`. A custom layout with additional endpoints is rejected.

Keep derived search-chunk links in a separate edge collection. An operator can
register a separate retrieval graph and a combined workspace graph through the
structured CLI graph operations. The combined graph shares the authored graph's
edge definition and adds the retrieval definition; it does not copy canonical
nodes. Use the authored graph name for `kg_add_node` and `kg_add_edge`, and the
combined name for traversal that crosses into chunks. Creating custom retrieval
edge definitions and inserting chunk-alignment links remain admin CLI operations,
not additional MCP tools. Preserve these boundaries when applying a schema.

| Source structure | Graph representation |
|---|---|
| H1 | One `document` node with source metadata and full extracted source text |
| H2 and H3 | Addressable `section` nodes with exact IDs, own content, and standing text |
| Heading hierarchy | Parent-to-child `contains` edges |
| H4 and below | Content inside the enclosing H3, following the format's candidate default |
| Glossary entry | Document-version-scoped `concept` node, exact definition and original term label |
| Glossary membership | `defines` edge from the glossary section |
| Node and front-matter keywords | `uses_keyword` edges to scoped concepts |
| Missing glossary term | `unresolved_keyword` node, without an invented definition |
| Illustration | Numbered, non-normative node with caption, medium, source span, and available media |
| Illustration subject | `illustrates` edge to the uniquely resolved declared subject |
| Numbered illustration citation | `cites_illustration` edge |
| Cross-document declaration | `references` edge, with an unresolved reference node until its source arrives |
| Source claims and standing | Preserved as text; no extracted assertion edges |

Shared keywords connect sections through a concept node. They express topical
association, not causation or unrestricted equivalence. Strip a trailing symbolic
alias such as `(S_M)` only for keyword matching, preserving the original label.
Retain rejected synonyms inside the source definition.

For another document, compare definitions, qualifications, and context before
reusing a concept. Keep each document's definition even if equivalence is accepted.
An uncertain connection belongs in a proposed relationship with rationale and
supporting passages. Do not assert it as established equivalence. This first load
has no cross-document equivalence decisions or inferred substantive edges.

Preserve complete `Standing:` text. A section can contain both rulings and
candidates. A section-level label must not promote every sentence to a ruling.
The importer emits only `contains`, `defines`, `uses_keyword`, `illustrates`,
`cites_illustration`, and `references`. Each carries a declared origin and exact
source evidence. It does not copy the five interpretive predicates hard-coded in
the historical fixture. Future substantive inferences need proposal status,
rationale, supporting passages, and user acceptance.

## Precise references and extraction

The supported input is graphable Markdown with YAML front matter, explicit H1-H3
node headers, blockquoted illustrations, and a final H2 glossary. H4 and deeper
headings remain internal content. Fenced and indented code is not treated as graph
structure. Duplicate IDs, missing heading headers, ambiguous illustration
subjects, malformed definitions, and skipped heading levels fail before writes.
Missing glossary definitions and unavailable declared references produce explicit
unresolved nodes and warnings. They do not produce guessed definitions.

Locations carry zero-based Markdown line bounds and UTF-8 byte offsets with
exclusive ends. Every edge stores exact supporting text. The root retains the
full Markdown and its line manifest. Illustration captions and references remain
source text. Media paths are references only: this importer neither follows them
nor infers image content. Embedded images in an attached DOCX remain recoverable
from its archived bytes, but are not extracted or served by this command.

The historical DOCX fixture recorded direct `w:body` child indices, styles,
numbering XML, and image relationships. Those coordinates are not interchangeable
with the new Markdown coordinates. A reviewed text export must retain the original
archive and state any extraction limitations. PDF layout anchoring remains future
work, using the Python extraction service and browser text selection.

## Search and ownership boundaries

The general document-ingestion DOCX route is unsupported
(`crates/hades-core/src/ingest_routing.rs`, `DOCUMENT_EXTENSIONS`). Feed a faithful
text extraction to `hades ingest` with an explicit version-scoped `--id` and the
`default` profile. This creates a derived search document, chunks, and embeddings;
it does not create a second canonical argument. Store `canonical_graph_node` and
the source hash on the derived document.

Connect the source root to that document with `has_search_representation`, the
document to chunks with `has_chunk`, and chunks to overlapping graph nodes with
`overlaps_source`. Verify stored text and chunk offsets against the extraction
before linking. An overlap is a retrieval locator, not evidence for a new claim.

`db query -c default` searches the derived chunks. Authored concept and section
nodes are not independently embedded. Follow a returned chunk into the graph to
recover its exact source context, definitions, qualifications, and illustrations.
The inspected document pipeline calls `prechunk::embed_chunks` with `late=false`
(`crates/hades-core/src/pipeline/orchestrator.rs`, `Pipeline::chunk_embed_store`).
Do not claim whole-document late chunking or one vector per heading.

The file pipeline owns only its derived document/chunks/embeddings. The mapping
owns authored nodes and provenance edges. Re-embedding must preserve canonical
nodes and check any existing chunk-alignment links. Never force source re-ingestion
over authored interpretations. Retiring a version or resolving an external
reference requires an explicit follow-up operation.

## Validation and operational limits

Check exact readback, one heading parent per section, connected hierarchy,
resolvable endpoints, quotation spans, illustration media, and keyword definition
status. Verify semantic retrieval separately from graph insertion. Run a repeat
load check and confirm unchanged document ingestion skips processing.

`crates/hades-core/tests/graphable_import.rs` verifies identical repeats, version
retention, conflicts, concurrent retries, rollback, and recovery of matching
partial rows. The library tests cancel an import after transactional node writes
and verify that retry completes it. `crates/hades-cli/tests/graphable_import.rs`
runs the CLI, reads back every row, checks source archives and evidence offsets,
and confirms that mapping did not create embeddings.

Run the strict tests on an isolated server:

```bash
python3 scripts/test_isolated_database.py --docker --contract graphable_import
python3 scripts/test_isolated_database.py --docker --graphable-import
```

The CLI test accepts explicit `--graphable-source paper.md` and optional
`--graphable-archive original.docx` for a real-document check. These files are
read only. All database writes occur on the private test server. Its local
`graphable-report.json` records actual counts, repeat behavior, and verification.
No production graph or service configuration is changed.

The efficiency aim is usable shared context: stable references, bounded retrieval,
and visible source qualifications reduce reference-resolution time. Alpha denotes
context amplification only. No numerical Conveyance score is assigned.
