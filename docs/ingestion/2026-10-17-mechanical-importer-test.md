# Mechanical importer verification

Date: 2026-10-17. Branch: `feat/graphable-importer`.
Base revision: `3516a6a4de888397e0d2d29cc358b0f73abb6590`.
Implementation: `crates/hades-core/src/graphable`, with a thin CLI adapter.

## Real document

The test used *State Management Primitives*, `loop-state-mgmt`, version `0.14`.
A mechanical Markdown export reused the historical Python extraction fixture.
All 672 paragraph texts were checked against the original DOCX before adding
Markdown delimiters, heading markers, glossary list syntax, and quotation markers.
The original document was not rewritten. Generated Word numbering and embedded
images remain in the original archive; this importer does not extract or serve them.

- Original DOCX SHA-256: `77dd348f375672f9a1aaa7656baccdedb687d95451b5a594a1d6af5b2e068bc6`.
- Parsed Markdown SHA-256: `2c4b5b22cde1ecdda240947204d20bcdcc2f25d2287b736d9b0a0bb0a8f0e187`.
- Mapping: `graphable-mechanical-v1`.
- Test database: `hades_test_graphable_cli_685911_0` on a disposable ArangoDB 3.12.11 server.
- Graph: `theory`.
- First import: **195 nodes and 593 edges inserted**.
- Identical repeat: **0 nodes and 0 edges inserted**.
- Root: `theory_nodes/document_07fbb5ec187792d3de8801791344890c`.

Every stored row matched its plan. Every edge quotation resolved to its recorded
UTF-8 span. Both archived files were byte-identical. Comparing logical relations
with the historical fixture found exactly its five interpretive predicates absent,
with no unexpected additions. External-reference placeholders are now document-scoped.

Undefined glossary terms remain unresolved: `STE100`, `Unix socket`, `crate`,
`read`, and `write`. Declared references to `deployment-tuple` and
`engineering-model/statelessness` also remain unresolved. No cross-document
concept equivalence was inferred.

**Mapped: yes. Embedded: not requested. Semantically searchable: not verified.**
The test confirmed that no embedding collection was created. The database was
removed by the isolated test harness. The production graph was not changed.

## Acceptance and regression checks

- Eight library contracts passed, including cancellation after actual transactional
  node writes, complete quotation spans, stable logical IDs, and adjacent illustrations.
- Three database contracts passed: repeat/version/conflict behavior, rollback and
  partial-row recovery, and concurrent retries.
- The real-document CLI round trip passed with exact readback and archive checks.
- Workspace unit tests and the `pipeline`, `config_integration`, and `proto_types`
  integration targets passed. Workspace clippy and formatting passed.
- Operational script tests passed: 70 tests, with one existing optional skip.

The isolated runner now sets current ArangoSearch thread limits. Its deprecated
flags did not bound threads on a large workstation and caused the pinned server
to exhaust the container's process limit before startup.

### PR #207 review follow-up

Both reviewers identified parser edge cases: adjacent unbulleted glossary entries,
fences indented by up to three spaces, and YAML comments mistaken for headings.
Regression tests now cover each case, including preserved glossary continuation
lines and exact definition evidence. Retry comparisons now read batches of 500
rows within the transaction, preserving conflicts rather than overwriting edits.

The isolated suite passed all 11 library and four database contracts. The added
database test crosses two batch boundaries, restores a missing row, and rejects
an edited edge in a later batch. Cancellation and atomic rollback still pass.
Formatting and strict `hades-core` clippy checks passed.
The real-document CLI test also passed again in disposable database
`hades_test_graphable_cli_707561_0`: 195 nodes and 593 edges, then zero insertions
on repeat. All rows, evidence spans, and both source archives matched exactly.

The next review added coverage for indented H1-H3 headings and indented code
inside blockquotes. Titles and evidence spans survive heading indentation, while
literal illustration declarations and citations remain source text. Numeric
payload comparisons now tolerate exact integral-float readback without rounding
large integers or relaxing source identity checks. All 14 library and five
database contracts passed, along with formatting and strict clippy. The database
test accepts an unchanged numeric-metadata retry and rejects both changed source
bytes and a stored fractional-value edit.
The real-document CLI round trip passed in disposable database
`hades_test_graphable_cli_723239_0`, again with 195 nodes, 593 edges, zero repeat
insertions, exact row and evidence readback, and byte-identical source archives.

A further review identified mixed tab/space indentation, tabs after heading
markers, and quote markers closing an outer fence. The reader now counts
four-column tab stops and tracks fence container ownership. Regressions cover
literal examples, real headings, and closed or unterminated quoted fences.
All 18 library and five database contracts passed, along with formatting and
strict clippy.
The real-document CLI check passed in disposable database
`hades_test_graphable_cli_741562_0`: 195 nodes and 593 edges, no repeat insertions,
and exact rows, evidence spans, and source archives.

The latest Codex review found ingestion annotations extending glossary definitions
and repeated illustration scans during mapping. An annotation now terminates a
definition, including without a separating blank line. Illustration ownership is
indexed once by source line and reused for content and citations. Adjacent
illustration boundary scans also stop before the next entry. A regression adds
1,200 sections with either zero or 30 adjacent illustrations and checks content
ownership and citation evidence. All 20 library and five database contracts,
formatting, and strict clippy passed.
The real-document CLI test passed in disposable database
`hades_test_graphable_cli_774187_0`: 195 nodes, 593 edges, and no repeat insertions.
All rows, evidence spans, and source archives matched. The complete generated
plan also matched the previous revision's real-document plan exactly.

## Retrieval and recovery

After an import into an operator-selected database, use the returned `root_id`
with `db graph traverse --graph theory`, or use `db get theory_nodes <key>` for
an exact node. `contains` follows the outline, `uses_keyword` reaches a scoped
concept, and `defines` leads from the glossary to its definition. Edge `evidence`
resolves against the root's `source_text`. The returned `source_archive_id`
identifies the immutable source receipt.

Rerun the same import to reconcile a lost commit response or restore missing
matching rows. A changed source version receives new handles and retains the old
rows. Conflicts preserve existing content and require review, not a force import.
