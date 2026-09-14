"""The records every extractor emits.

Deliberately store-agnostic. The HADES survey found its extractors touched
ArangoDB in zero of eight files and its embedders in zero of five, which is
why both lift to a new backend unchanged while the workflow and storage
layers do not. That property is worth keeping on purpose rather than by
accident, so nothing in `seshat.ingest` may import a database driver.

A `Node` is identified by `ident`, never by a line number. Lines move, and an
identity that moves is not an identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Basis, as the schema's CHECK constraint enumerates it.
#
#   declared   the source said so: a `graph` block, a `//! conforms:` header,
#              a paper's bibliography
#   derived    a tool computed it: SCIP calls, libclang kernel edges, docling
#              structure
#   asserted   a person said so
#   extracted  a model proposed it. Nothing writes this yet. It exists so a
#              guessed edge can never wear the same label as a declared one.
DECLARED = "declared"
DERIVED = "derived"
ASSERTED = "asserted"
EXTRACTED = "extracted"


@dataclass(frozen=True)
class Node:
    ident: str
    kind: str
    path: str | None = None
    line: int | None = None
    title: str | None = None
    body: str | None = None
    tag: str | None = None
    lang: str | None = None


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    relation: str
    basis: str
    # What papers the edge, and its tag. Both are part of the *identity* of a
    # declared edge, not decoration on it: `weaver-spu --seam--> weaver-harness`
    # is declared three times in one PRD, once per contract that papers a
    # separate socket seam, and a key built from source, relation and target
    # alone collapsed all three into one row. Three real seams became one and
    # the count came up two short of the corpus, which is how it was found.
    via: str | None = None
    tag: str | None = None


@dataclass
class Extraction:
    """What one extractor produced, plus what it could not resolve.

    `dangling` is a first-class result rather than a dropped edge. "This file
    cites something that does not exist" is the question the graph exists to
    answer, so it cannot be represented as a missing row.

    `unresolved_reported` guards a specific failure the Yeomna handoff
    recorded: a documents-only ingest once saw 501 conformance headers and
    wrote zero edges while reporting nothing wrong. A pass that sees many of
    something and writes none of it has to say why.
    """

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    dangling: list[Edge] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def extend(self, other: "Extraction") -> "Extraction":
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)
        self.dangling.extend(other.dangling)
        self.notes.extend(other.notes)
        return self

    def summary(self) -> dict[str, int]:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "dangling": len(self.dangling),
        }
