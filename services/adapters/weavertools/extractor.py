"""WeaverTools ingest: declared graph blocks, and conformance headers.

The parsing rules here are not invented. They are lifted from
`process/gates/census.py` on PR #559, which had nine of its own bugs burned
out of it across two review passes, each of which had printed a confident
wrong number. The ones that matter are carried with their reasons:

- **Split stanzas on the record's own keyword, not on a blank line.** Two
  records written back to back are one stanza to a blank-line splitter, which
  silently drops all but the first.
- **Measure citations against every node kind, not assertions alone.** The
  corpus declares vocabulary, document, crate, term and axiom nodes too, and
  the Document Format names one a source file may cite. Measuring against
  assertions calls a sound header dangling.
- **`\\w` does not match a hyphen**, so `compile-pin` and `compile-fail` read
  as untagged under a naive pattern. Two of the five tags.
- **The header obligation follows the unit, never a directory.** A count over
  `src` alone excludes citations sitting in integration tests.
- **Prune `archive/`**, since a frozen copy of a Spec declares every node the
  live one does and doubles them.

See `docs/yeomna-handoff.md` on why a rule with its reason attached survives a
rewrite and a bare rule does not.
"""

from __future__ import annotations

import re
from pathlib import Path

from .records import DECLARED, Edge, Extraction, Node

# Fixed by WeaverTools-Document-Format. A tag outside this set is a finding
# rather than a silent miss.
TAGS = {"compile-pin", "compile-fail", "perturbation", "manifest", "review"}

GRAPH = re.compile(r"```graph(.*?)```", re.S)
NODE_LINE = re.compile(r"^node: (.*)$", re.M)
KIND = re.compile(r"^kind: ([\w-]+)$", re.M)
TAG = re.compile(r"^tag: ([\w-]+)$", re.M)
EDGE_REL = re.compile(r"^edge: ([\w-]+)$", re.M)
EDGE_FROM = re.compile(r"^from: (.*)$", re.M)
EDGE_TO = re.compile(r"^to: (.*)$", re.M)
# Identifiers are kebab-case, always.
IDENT_OK = re.compile(r"^[a-z0-9-]+$")

# **Citing and owing a header are different questions**, and one walk answers
# both. The census records this as one of the nine bugs it paid for.
#
# HEADER_CITE is the file-level obligation: `//!` is an inner doc comment, so
# it applies to the enclosing module, which is the unit the Document Format
# names. A file lacking one owes a header.
#
# ITEM_CITE is membership in the cited set. The corpus also cites at item
# level with `///` and plain `//`, 77 occurrences beyond the 436 file-level
# headers, and an assertion cited only that way is cited. Measuring the cited
# set with the anchored pattern alone moved 32 perturbations into the uncited
# column here, against a baseline with a known answer.
#
# The trailing token is anchored: unanchored, `conforms: weaver_types-x`
# captures `weaver` and the gate names an identifier present in no file, so a
# reader grepping for the offender finds nothing.
HEADER_CITE = re.compile(r"^//! conforms: ([a-z0-9-]+)\s*$", re.M)
ITEM_CITE = re.compile(r"conforms: (\S+)")

# A build script is cargo's unit, not the crate's, and conforms to nothing.
NO_HEADER_OWED = {"build.rs"}
PRUNE_DIRS = {".git", "archive", "target", "node_modules"}


def _walk(root: Path, suffix: str):
    for path in sorted(root.rglob(f"*{suffix}")):
        if PRUNE_DIRS & set(path.parts):
            continue
        yield path


def read_documents(repo: Path) -> Extraction:
    """Every fenced `graph` block under `docs/`.

    **`process/` is deliberately excluded.** It holds
    `WeaverTools-Document-Format.md`, which illustrates the block grammar with
    worked `graph` examples, and a mapper that reads them files documentation
    samples as live assertions. Including it inflated the assertion count from
    355 to 413 here and moved `uncited_perturbations` from 33 to 65 against a
    baseline with a known answer. The census scopes to `docs/` for this
    reason, and the agreement is the test that caught it.
    """
    out = Extraction()
    seen: dict[str, str] = {}

    for base in ("docs",):
        root = repo / base
        if not root.is_dir():
            continue
        for path in _walk(root, ".md"):
            rel = str(path.relative_to(repo))
            text = path.read_text(encoding="utf-8", errors="replace")
            out.nodes.append(Node(ident=rel, kind="document", path=rel, title=path.stem))

            for block in GRAPH.findall(text):
                # Split on the record's own keyword. See module docstring.
                for stanza in re.split(r"(?=^node: )", block, flags=re.M):
                    line = NODE_LINE.search(stanza)
                    if not line:
                        continue
                    name = line.group(1).strip()
                    kind = KIND.search(stanza)
                    kind = kind.group(1) if kind else "unknown"
                    tag = TAG.search(stanza)
                    tag = tag.group(1) if tag else None

                    if not IDENT_OK.match(name):
                        out.notes.append(f"malformed node id in {rel}: {name!r}")
                    if tag is not None and tag not in TAGS:
                        out.notes.append(f"unknown tag in {rel}: {name} ({tag})")
                    if name in seen and seen[name] != rel:
                        out.notes.append(
                            f"duplicate node id {name}: {seen[name]} and {rel}"
                        )
                    seen[name] = rel

                    out.nodes.append(
                        Node(ident=name, kind=kind, path=rel, tag=tag, body=stanza.strip())
                    )
                    out.edges.append(
                        Edge(src=name, dst=rel, relation="declared-in", basis=DECLARED)
                    )

                # Edge records within the same block.
                for stanza in re.split(r"(?=^edge: )", block, flags=re.M):
                    rel_m = EDGE_REL.search(stanza)
                    src_m = EDGE_FROM.search(stanza)
                    dst_m = EDGE_TO.search(stanza)
                    if not (rel_m and src_m and dst_m):
                        continue
                    out.edges.append(
                        Edge(
                            src=src_m.group(1).strip(),
                            dst=dst_m.group(1).strip(),
                            relation=rel_m.group(1),
                            basis=DECLARED,
                        )
                    )
    return out


def read_conformance(repo: Path, declared: set[str]) -> Extraction:
    """`//! conforms:` headers across workspace crates, as `cites` edges.

    The obligation follows the unit: every `.rs` a member owns, tests
    included, `build.rs` excepted. Resolution is against *every* declared
    identifier, not assertions alone.
    """
    out = Extraction()
    crates = repo / "crates"
    if not crates.is_dir():
        out.notes.append("no crates/ directory, conformance pass saw nothing")
        return out

    headers_seen = 0
    files_owing = 0
    headerless: list[str] = []

    for path in _walk(crates, ".rs"):
        rel = str(path.relative_to(repo))
        text = path.read_text(encoding="utf-8", errors="replace")
        out.nodes.append(Node(ident=rel, kind="source", path=rel, lang="rust"))

        headers = HEADER_CITE.findall(text)
        headers_seen += len(headers)

        # Every citation, file-level or item-level, becomes an edge. Malformed
        # targets are reported under the identifier as written, never
        # truncated to a prefix, so grepping for the offender finds it.
        for target in dict.fromkeys(ITEM_CITE.findall(text)):
            if not IDENT_OK.match(target):
                out.notes.append(f"malformed citation in {rel}: conforms: {target}")
                continue
            edge = Edge(src=rel, dst=target, relation="cites", basis=DECLARED)
            (out.edges if target in declared else out.dangling).append(edge)

        # The obligation is separate, and it follows the unit.
        if path.name not in NO_HEADER_OWED:
            files_owing += 1
            if not headers:
                headerless.append(rel)

    # The reporting rule from the handoff: a pass that sees many of something
    # and writes none of it has to say why.
    if headers_seen and not out.edges:
        out.notes.append(
            f"saw {headers_seen} conformance headers and resolved none of them, "
            f"which means the declared set was empty or came from elsewhere"
        )

    out.notes.append(f"headers_seen={headers_seen}")
    out.notes.append(f"sources_without_a_header={len(headerless)}")
    out.notes.append(f"sources_owing_a_header={files_owing}")
    return out


def ingest(repo: Path) -> tuple[Extraction, Extraction]:
    docs = read_documents(repo)
    declared = {n.ident for n in docs.nodes}
    code = read_conformance(repo, declared)
    return docs, code
