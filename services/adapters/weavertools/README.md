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
WeaverTools emits HADES records outward. That is why nothing in here writes to
a database.

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

It reproduces WeaverTools' own census exactly, from an independent
implementation:

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

**No database driver is imported.** The modules emit `Node` and `Edge`
records and persist nothing. That is what makes the eventual move to
WeaverTools possible, and it is also why something else still has to write
the records — see `docs/declarative-schema.md`, since declared edges are what
that mechanism is for.

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
