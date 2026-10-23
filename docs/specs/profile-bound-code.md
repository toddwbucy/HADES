# Profile-bound code graph persistence

Part of [#225](https://github.com/toddwbucy/HADES/issues/225), following the
[verified conventional code provider](verified-code-provider.md).
`pipeline::profiled_code::ProfiledCode` supplies an internal conventional
prepare/commit path for one already-analyzed source file. It does not activate
CLI, MCP, background code ingestion or managed code search.

## Admission and preparation

`connect` loads the ready database profile from the writer, requires an enabled
conventional code workload, the eight existing `CODEBASE` collections and a
non-unique, non-sparse persistent `file_key` index on each of the seven child
collections and saved direct edge definitions in `hades_schema`, then
verifies the installation-owned provider catalog. It selects the code workload's
passage task, receipt, dimension and token policy and rechecks the complete
database snapshot after provider verification. It creates no schema or collections.
`config/schemas/codebase.yaml` owns these indexes through `lookup_fields`, including
all four edge collections; `hades --db NAME schema apply` provisions them. The
future managed code setup entry point must apply these declarations before
publishing readiness. The canonical schema also supplies the endpoint definitions.
Missing or incompatible indexes are refused before provider verification.

`CodeInput` contains the exact namespace/path, source text, `FileAnalysis` and
file-local relationships. Callers supply neither vectors nor token counts.
Relationships reference the file or a primitive symbol's input index; arbitrary
database endpoints and duplicate identities are refused. Each edge, including a
generated `defines` edge, must match the database's saved `from_collections` and
`to_collections`. Missing or ambiguous definitions fail admission. The canonical
schema permits calls/implements from symbols to symbols, and imports from files
to files or symbols; endpoint permissions are loaded as data, with no Rust fallback.
The core API trusts the
analyzer's reported semantics. Future filesystem/analyzer admission must pin the
source before analysis and retain the same profile through this preparation.

The selected provider plans the complete original source with its tokenizer.
After another profile check, all passages are embedded with the same pinned
receipt. Every vector must meet the declared width, finite/nonzero and normalization
requirements. A failed later batch discards the entire preparation. Preparation
does not write to the database; its result has private fields and can only be
committed through its original database/profile/provider admission.

This internal API does not skip unchanged source: explicit replacement still
plans, embeds and publishes it. Public resync admission must compare the content
hash, compatible saved model/space/generation binding and requested analysis tier
and provenance before planning or inference. Matching source bytes alone do not
establish that the stored graph is current.

Limits are 2 MiB of source text, 128 tokenizer-planned passages, 1024 input symbols,
4096 explicit relationships, 4 MiB of analyzer metadata and 16 MiB of prepared
records. Known source/graph costs are checked before planning. After planning, a
conservative bound reserves serialized vectors and inventory overhead before the
first embedding request; inputs exceeding that bound must be split even if their
eventual vectors might serialize smaller. Final records are measured without a
deep graph clone. Conventional passages are tokenizer-based, not AST chunk boundaries.
Specialized code and explicitly configured ordinary-text code both work; the
task is taken from the selected workload rather than inferred from the filename.

## Records and atomic publication

The versioned `hades-bound-code-v1` format covers the file, chunks, vectors,
symbols, generated defines edges and declared local call/import/implements edges.
Records retain source hashes and the full profile/generation/artifact/space/load
binding. Files retain exact source text, analyzer provenance, the counted passage
plan and an inventory of their owned records. File keys and initial symbol keys
use existing code identity helpers; passage keys also include the corpus generation.
Replacement retains matched symbols' stored keys even when their lines change.
Consumers must resolve managed endpoints from stored symbols, not recompute their
keys from current line numbers. Legacy code key generation is unchanged.

One transaction locks workspace metadata, `hades_schema` and all eight code collections,
revalidates the profile, endpoint definitions and source revision, checks the old graph inventory, and
publishes all records. Individual and bulk acknowledgments must name the expected
keys and collections. A write failure or cancellation before commit rolls back;
a lost commit acknowledgment requires reconciliation with durable state.
Admission caches index names and IDs. Under the lock, each owner read checks that
exact index directly rather than enumerating the collection's indexes. Child sets
are measured once, and publication checks symbol preservation against a set of
incoming keys. Owner reads and deletes force the validated persistent index; a missing or replaced index
cannot silently become a collection scan under the exclusive locks. Shared
document/code storage helpers enforce bounded queries, revisions and complete
write acknowledgments. `CodeReceipt.relationships` counts all persisted edge
rows, including generated `defines` edges: two symbols and one call report three.

Replacement requires explicit opt-in and retains the original source revision
from before planning. Legacy rows, orphan children, incomplete inventories,
changed child records and unknown file fields are refused. The inventory excludes
only ArangoDB's `_id`/`_rev` envelope and normalizes equivalent integral JSON number
spellings without rounding large integers. It detects edits; it is not an
attestation against an administrator changing both records and inventory.

Compatible fresh provider loads may replace old-load records when the saved
artifact, space and profile still agree; already-connected clients retain their
original serving receipt.

## Symbol identity during replacement

Before planning or embedding, replacement reads the existing symbols through the
admitted owner index and checks them against the saved inventory. Missing, edited
or foreign symbols fail without inference or writes. Every old symbol must have
an unambiguous match in the new analysis:

- Matches use qualified name, language kind and optional compiler identity
  (`metadata.usr`). Without a USR, they also require the same verified lexical
  container chain, recovered from module/type/callable spans. This distinguishes
  Rust impl methods whose qualified names omit their enclosing module. Same-line
  or overlapping spans that cannot prove ownership require unchanged source and
  exact positions. Compiler identities include semantic ownership and allow a
  definition to move out of its lexical class body.
- Optional `metadata.is_definition` distinguishes a C/C++ declaration from its
  definition even when both share a USR. Both existing vertices survive. Repeated
  declarations with the same role, Python property/setter or overload definitions,
  and Rust `cfg` alternatives in the same scope still require unchanged source
  and exact positions. Even adding a comment to these ambiguous files is refused
  until explicit reconciliation exists. Body text and declaration order are not
  identity evidence.
- An analyzer upgrade may add a USR when name/kind/verified scope is unique on
  both sides and any previously known declaration role is unchanged. This does
  not permit overload guessing or changes to existing USRs. If an enclosing
  container's own identity changes, the scope proof must be reconciled explicitly.
  USRs remain intact within the shared 4 MiB total analysis budget; they are never
  truncated or silently discarded.
- Missing matches, ambiguous changes and language changes are refused. Renames,
  removals and analysis downgrades remain unavailable through this path.

For example, adding a comment above `fn apple()` updates its source location and
embeddings while preserving its `_id`. Local edges are built against retained IDs;
endpoint substitutions happen once, so swaps cannot redirect a relationship twice.
Authored references, other files' edges and unrelated collections are never
rewritten. If a new symbol occupies an existing endpoint's historical line, it
receives a deterministic, separately salted key. Allocation reserves all retained
and provisional keys and makes at most 1025 attempts; existing IDs never change.

Publication still checks the original file revision and complete graph inventory
under the transaction locks. Concurrent edits or a later write failure preserve
the previous graph. Reconciliation trusts the analyzer's qualified names and kinds;
it is not a general rename detector or an operation to retire referenced symbols.
Analyzers without an explicit discriminator may need to provide richer identities
before files containing repeated qualified names can be updated.

## Validation and remaining integration

Run against the strict disposable server:

```bash
cargo fetch --locked
python3 scripts/test_isolated_database.py --docker --contract profiled_code
# Or use --arangod /path/to/test/arangod instead of --docker.
```

The default isolated CI matrix includes this target. Synthetic providers cover
two/three-dimensional spaces, asymmetric tasks, code-only and ordinary-text-code
profiles, a mixed document/code load with distinct widths and receipts, exact
source spans, missing/incorrect owner indexes, retained symbol IDs across repeated
line moves, scoped Rust method moves/body swaps, declaration/definition roles,
real Python/Rust ambiguity refusal, unique analyzer upgrades, long USRs,
new-symbol key collisions,
untouched external references, replacement conflicts, rollback, profile/provider
drift, invalid graphs, cross-database admission, corruption, cancellation and
failed later batches. No installed model or live research database is used.

Public analyzer/file/job integration, cross-file relationship publication,
ambiguous symbol reconciliation/retirement, managed code search, contextual code,
broader setup and the frontend remain subsequent work. The current public code-ingestion guard
stays in place. Existing legacy records cannot be adopted by relabeling them.

**Cross-file enrichment blocker:** existing `code/cpp_edges.rs`,
`code/python_calls.rs`, `code/lsp/edges.rs`, `code/rust_imports.rs` and
`code/tree_sitter_edges.rs` still derive endpoints with `symbol_key` from current
source lines. These keys can differ from retained managed IDs. Before admitting
any of these enrichers to managed files, resolve candidates through the verified
stored symbol inventory using scoped identity, USR and declaration role, require
unique live endpoints, and check the inventory again at publication. Add
regressions that move a retained symbol, enrich calls/imports across files, and
verify every endpoint exists and refers to the intended definition. The current
guard must remain until that lookup and publication contract is implemented.
