# Graphable Document Ingestion Mapping

Status: initial mapping, exercised in a local document ingestion on 2026-10-16.
This interprets an existing graphable document without rewriting its argument or
certifying its claims. Here, graphable documents declare heading nodes and
illustrations in Markdown comments, keywords, and a document-local glossary.
The authoring skill supplies the format; HADES performs writes and embedding
separately. The mapping is a documented convention, not a built-in importer.

## Source and identity

Keep the original file unchanged and retain a byte-identical version archive.
Store its SHA-256, document ID, version, title, author, source citation, and source
status. Treat extracted text as a derived representation of that source.

Preserve declared node IDs in `source_node_id` and `logical_id`. Database keys are
revision-scoped: a type prefix followed by the first 32 hexadecimal characters of
SHA-256 over `doc_id|version|source_sha256|logical_id`. Edges similarly hash their
endpoints and relation within the source namespace. Preserve the full source hash
in every row. An identical repeat verifies the existing payload and inserts only
missing rows. A changed payload or reused version with a different source hash
requires review. Never replace an earlier interpretation silently.

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
| Selected explicit assertions | Concept-to-concept predicates with exact quotations and source standing |

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
Explicit assertions have `assertion_origin: source_explicit`; structural declarations
and source-alignment edges have separate origin values. Future substantive
inferences must use a distinct origin, proposal status, rationale, and source spans.

## Precise references and extraction

For DOCX, source locations identify zero-based direct children of `w:body` in
`word/document.xml`, with inclusive block bounds. Text spans use UTF-8 byte offsets
in the extraction, with exclusive ends. The block manifest retains Word styles,
list-numbering XML, table structure as row/cell separators, and image relationships.
Generated list numbering is not present in the plain-text extraction. The original
DOCX and numbering properties preserve that presentation detail.

The exercised DOCX uses `Title`, `Heading1`, and `Heading2` for graphable H1, H2,
and H3. Verify the actual styles for each new source. Its equations are literal
LaTeX text, retained unchanged. Embedded media
are extracted by their OOXML relationships. Captions remain source content; no
visual assertions are inferred from the images.

## Search and ownership boundaries

The current DOCX route is unsupported
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

These are conventions exercised through existing structured CLI operations,
not a newly implemented HADES importer. Retain the loader, manifests, verification
results, and version archive as local ingestion artifacts. Do not reapply bootstrap
schema over populated collections. Do not use write AQL.

The efficiency aim is usable shared context: stable references, bounded retrieval,
and visible source qualifications reduce reference-resolution time. Alpha denotes
context amplification only. No numerical Conveyance score is assigned.
