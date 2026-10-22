# Profile-bound code graph persistence

Part of [#225](https://github.com/toddwbucy/HADES/issues/225), following the
[verified conventional code provider](verified-code-provider.md).
`pipeline::profiled_code::ProfiledCode` supplies an internal conventional
prepare/commit path for one already-analyzed source file. It does not activate
CLI, MCP, background code ingestion or managed code search.

## Admission and preparation

`connect` loads the ready database profile from the writer, requires an enabled
conventional code workload and the eight existing `CODEBASE` collections, then
verifies the installation-owned provider catalog. It selects the code workload's
passage task, receipt, dimension and token policy and rechecks the complete
database snapshot after provider verification. It creates no schema or collections.

`CodeInput` contains the exact namespace/path, source text, `FileAnalysis` and
file-local relationships. Callers supply neither vectors nor token counts.
Relationships reference the file or a primitive symbol's input index; arbitrary
database endpoints and duplicate identities are refused. The core API trusts the
analyzer's reported semantics. Future filesystem/analyzer admission must pin the
source before analysis and retain the same profile through this preparation.

The selected provider plans the complete original source with its tokenizer.
After another profile check, all passages are embedded with the same pinned
receipt. Every vector must meet the declared width, finite/nonzero and normalization
requirements. A failed later batch discards the entire preparation. Preparation
does not write to the database; its result has private fields and can only be
committed through its original database/profile/provider admission.

Limits are 2 MiB of source text, 128 tokenizer-planned passages, 1024 input symbols,
4096 explicit relationships, 4 MiB of analyzer metadata and 16 MiB of prepared
records. Conventional passages are tokenizer-based, not AST chunk boundaries.
Specialized code and explicitly configured ordinary-text code both work; the
task is taken from the selected workload rather than inferred from the filename.

## Records and atomic publication

The versioned `hades-bound-code-v1` format covers the file, chunks, vectors,
symbols, generated defines edges and declared local call/import/implements edges.
Records retain source hashes and the full profile/generation/artifact/space/load
binding. Files retain exact source text, analyzer provenance, the counted passage
plan and an inventory of their owned records. File/symbol keys reuse existing
code identity helpers; passage keys also include the corpus generation.

One transaction locks workspace metadata and all eight code collections,
revalidates the profile and source revision, checks the old graph inventory, and
publishes all records. Individual and bulk acknowledgments must name the expected
keys and collections. A write failure or cancellation before commit rolls back;
a lost commit acknowledgment requires reconciliation with durable state.

Replacement requires explicit opt-in and retains the original source revision
from before planning. Legacy rows, orphan children, incomplete inventories,
changed child records and unknown file fields are refused. The inventory excludes
only ArangoDB's `_id`/`_rev` envelope and normalizes equivalent integral JSON number
spellings without rounding large integers. It detects edits; it is not an
attestation against an administrator changing both records and inventory.

Replacements cannot downgrade analysis or remove/move existing symbol keys.
This deliberately preserves endpoints used by authored or cross-file references
until an explicit retirement/remapping operation exists. Existing unrelated rows
and externally owned relationships remain intact. Compatible fresh provider loads
may replace old-load records when the saved artifact, space and profile still
agree; already-connected clients retain their original serving receipt.

## Validation and remaining integration

Run against the strict disposable server:

```bash
cargo fetch --locked
python3 scripts/test_isolated_database.py --docker --contract profiled_code
# Or use --arangod /path/to/test/arangod instead of --docker.
```

The default isolated CI matrix includes this target. Synthetic providers cover
two/three-dimensional spaces, asymmetric tasks, code-only and ordinary-text-code
profiles, exact source spans, replacement conflicts, rollback, profile/provider
drift, invalid graphs, cross-database admission, corruption, cancellation and
failed later batches. No installed model or live research database is used.

Public analyzer/file/job integration, cross-file relationship publication,
symbol retirement/remapping, managed code search, contextual code, broader setup
and the frontend remain subsequent work. The current public code-ingestion guard
stays in place. Existing legacy records cannot be adopted by relabeling them.
