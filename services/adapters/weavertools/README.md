# adapters/weavertools — a project-specific adapter, not a HADES feature

**Scope: WeaverTools only. Nothing here is general.** If you are evaluating
HADES for your own project, you can ignore this directory entirely.

## Why it is here rather than in WeaverTools

It probably belongs in WeaverTools. The conventions it parses are
WeaverTools' own, WeaverTools already has `process/gates/census.py` doing the
same parse for CI, and a general knowledge-graph tool should not carry one
project's documentation format.

It lives here anyway because moving it would be disruptive to WeaverTools at
its current stage. The intent, recorded so it is not forgotten: **once both
projects are stable, this moves to WeaverTools and inverts.** Same boundary,
opposite side — instead of HADES adapting WeaverTools' conventions inward,
WeaverTools emits HADES records outward. The extractor emits storage-independent
records; the separate `write_graph.py` command persists them to ArangoDB.

## What it adapts

WeaverTools declares conformance in two directions.

Specs declare assertions in fenced ` ```graph ` blocks, one `key: value` per
line. Source files declare which assertions they satisfy with
`//! conforms: <crate>-<slug>` at the top. Joining them gives `cites` edges,
and a citation whose target is declared nowhere is a **dangling** edge, which
is the defect the whole thing exists to surface.

This answers the question HADES was wanted for in the first place: has the
code drifted from the claim it makes about itself.

## Why it is trustworthy

The original corpus comparison reproduced WeaverTools' own census from an
independent implementation. These are historical measurements, not a current
audit or a guarantee for another source revision:

| metric | this | census baseline |
|---|---|---|
| `dangling_citations` | 0 | 0 |
| `sources_without_a_header` | 48 | 48 |
| `uncited_perturbations` | 33 | 33 |

413 assertions, 478 `cites` edges. That agreement is the test. It caught two
real bugs that nothing else would have, both recorded in `extractor.py` with
their reasons. The subtle one is worth reading before editing: **citing and
owing a header are different questions.** WeaverTools cites at file level with
`//!` (436 occurrences, the form its Document Format specifies) and at item
level with `///` or `//` (77 more). The cited set takes both, the header
obligation takes only the first. Conflating them parks 32 perturbations in
the uncited column.

`scip_reader.py` walks the SCIP protobuf wire format directly with no protobuf
dependency, because the question was whether the index is usable and a
dependency is a poor way to answer that. Over WeaverTools it finds 156
documents and 13,340 symbols across twelve crates. `rust-analyzer scip` takes
10 seconds and must be invoked as the stable-toolchain binary by absolute
path, since the rustup shim fails inside a workspace whose
`rust-toolchain.toml` pins a nightly without the component installed.

## Deliberate constraints

**Extraction and persistence are separate.** `extractor.py` reads repository
files and emits `Node` and `Edge` records without database access. The optional
`write_graph.py` command reads database scope and writes collections, rows and
an ingestion report over HTTP. `--dry-run` still reads database scope. It is not
a whole-run transaction, and repeated imports do not remove stale records.
See `docs/declarative-schema.md` for the declared-edge mechanism.

**Local repository inputs.** The extractor walks local files, reads them whole
and follows file symlinks; it is not a sandbox for untrusted repositories.
`scip_reader.py` is a standalone exploratory wire-format reader, not a validated
SCIP ingestion endpoint. Neither script is wired into the Rust daemon dispatch.
The September 2026 adapter audit checked these boundaries statically; it did not
rerun the historical WeaverTools census or certify arbitrary input sizes.

**Python, in a tree that does extraction in Rust.** On architecture this
belongs in `hades-core`. It is Python because it was written for a separate
tool before that plan was set down in favour of recovering HADES. Porting it
would be the wrong move while its eventual home is WeaverTools.

## Known gap

`crates/weaver-spu/kernels/transformer.cu` is not covered. It carries a
conformance trace naming a node that exists in neither the current corpus nor
the older `WeaverTools_v3` graph, its collection names predate that graph, and
`launch_` appears zero times in the Rust — so the kernels are build-wired but
not yet called. Until they are, a kernel-to-assertion link has to be declared
rather than derived. The census cannot see the file either, since it walks
`.rs` only.

## Identity version 2 and existing graphs

The writer uses bounded SHA-256 keys over framed identities. Node keys bind the
full declared identifier; edge keys bind persisted endpoints, relation, basis,
via and tag. Readable identifiers remain in document attributes. Exact duplicate
input records are deduplicated; conflicting declarations are refused before
collection creation or imports.

Every adapter row and report carries adapter_identity_version: 2. Before writing,
the adapter inspects its existing node, relation and report collections. Legacy,
mixed-version rows or a changed kind for an existing declared node cause refusal
before the write stage. There is no automatic migration or bypass flag. Do not
stamp old rows as version 2: that does not repair their keys or lost relationships.

For a legacy graph, prepare a separately reviewed rebuild into a fresh database:
ingest its code/documents first, run this writer, verify relationship counts and
endpoint integrity, then plan consumer cutover and rollback. Preserve the old
database until that plan permits retirement. A live rebuild/cutover needs the
owner's arranged downtime and verified active-data snapshot; the audit does not
perform it. Stop legacy writers before changing the writer version. The preflight
is not a database-wide concurrency lock against unrelated or older writers.

This remains an incremental writer with partial-run persistence, not a complete
snapshot replacement. Rows absent from a later extraction are retained; the report
explicitly sets stale_retirement_performed to false and describes its coverage as
present extracted rows only. Successful imports do not certify removal of stale
declarations. In-place retirement/migration requires a separate reviewed procedure.

### Source revision

`wt_ingest_report/latest.source_git` records `{commit, dirty}` observed before
extraction (#171), matching the Rust ingest field. Non-Git inputs carry explicit
null; an unborn Git branch has a null commit. Dirty includes untracked files and
submodule changes across the worktree. This is an observation, not an atomic
snapshot or a guarantee that files stayed unchanged during ingestion.

File/document rows record the observation when rewritten; incremental skips keep
their prior provenance. The CLI envelope describes the current invocation, while
MCP job provenance describes admission and its result contains the child output.
Explicit-file batches retain per-item provenance and use null at envelope level
when observations differ. Older rows are not backfilled.
