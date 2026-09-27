"""WeaverTools ingest: declared graph blocks, and conformance headers.

The parsing rules here are not invented. They are lifted from
`process/gates/census.py` on PR #559, which had nine of its own bugs burned
out of it across two review passes, each of which had printed a confident
wrong number. The ones that matter are carried with their reasons:

- **Split stanzas on the record's own keyword, not on a blank line.** Two
  records written back to back are one stanza to a blank-line splitter, which
  silently drops all but the first.
- **Measure citations against assertions, not every node kind.** A header
  names an assertion, per the Document Format, and the census says the
  widening that once measured against every kind was a misreading
  (`census.py` `take()`, "`declared` holds every node whatever its kind";
  `dangling_citations` is a citation not among the assertions). This module
  kept the widened reading after the census retired it, so a header naming a
  crate resolved here and dangled there (#198). Duplicates are still found
  across every kind; that is a different question.
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

import io
import re
import tokenize
from bisect import bisect_right
from pathlib import Path

from .records import DECLARED, Edge, Extraction, Node

# Fixed by WeaverTools-Document-Format. A tag outside this set is a finding
# rather than a silent miss.
TAGS = {"compile-pin", "compile-fail", "perturbation", "manifest", "review"}
SYSTEM_NODE = "WeaverTools"

# A fence opens and closes on its own line.
#
# Unanchored, this matched an inline ```` ```graph ```` inside a sentence and then
# ran to the next triple backtick anywhere in the file, so a document *discussing*
# the block grammar opened a phantom fence over its own prose. Two of those stand
# in this corpus. Both happen to hold no `node:` or `edge:` line at the start of a
# line, so they contributed nothing and were invisible: a quoted example in
# prose would have injected declarations nobody wrote.
#
# Measured across the corpus before the change and after: 399 fences to 397, the
# two lost being exactly the phantoms, with 491 node records and 665 edge records
# either way and an identical kind tally.
GRAPH = re.compile(r"^```graph[ \t]*$(.*?)^```", re.S | re.M)
NODE_LINE = re.compile(r"^node: (.*)$", re.M)
KIND = re.compile(r"^kind: ([\w-]+)$", re.M)
TAG = re.compile(r"^tag: ([\w-]+)$", re.M)
EDGE_REL = re.compile(r"^edge: ([\w-]+)$", re.M)
EDGE_FROM = re.compile(r"^from: (.*)$", re.M)
EDGE_TO = re.compile(r"^to: (.*)$", re.M)
# The contract that papers a seam. Read because it distinguishes edges that are
# otherwise identical -- see the note on `Edge.via`.
EDGE_VIA = re.compile(r"^via: (.*)$", re.M)
# Identifiers are kebab-case, always.
IDENT_OK = re.compile(r"^[a-z0-9-]+$")

# Document Format sections 1, 3 and 4 distinguish citation forms from the
# file-header obligation. Python's leading comments head a unit; strings and
# comments trailing code do not. Rust/CUDA use their file-level //! marker.
HEADER_CITE = re.compile(r"^\s*//!\s*conforms:\s*([a-z0-9-]+)\s*$", re.M)
HASH_HEADER = re.compile(r"^#[ \t]*conforms:[ \t]*([a-z0-9-]+)[ \t]*$", re.M)
ITEM_CITE = re.compile(r"^[ \t]*(?://[/!]?|#)[ \t]*conforms:(.*)$", re.M)
COMMENT_CITE = re.compile(r"^#[ \t]*conforms:(.*)$")
NO_HEADER_OWED = {"build.rs"}
# Manifests are read for citations but have no module-header obligation.
NO_HEADER_OWED_SUFFIXES = {".toml"}
CONFORMANCE_SUFFIXES = (".rs", ".toml", ".cu", ".py")
PRUNE_DIRS = {".git", "archive", "target", "node_modules"}


# A record is the run of `key: value` lines it opens with, and nothing after.
#
# Splitting a block on `(?=^edge: )` leaves each piece running to the *next*
# edge record, so 86 of this corpus's edge stanzas carry a following `node:`
# record inside them. An unbounded search for `tag:` or `via:` therefore read
# the next node's tag onto the edge -- a field that looks right, belongs to
# something else, and no count would catch. Bounded here instead.
RECORD_LINE = re.compile(r"^[a-z][a-z-]*: ")



FRONT_MATTER_MAX_LINES = 200
# A line that can occur inside a YAML front-matter block: a comment, a mapping
# key (plain, double- or single-quoted) followed by `:`, an explicit `? ` key
# and its `: ` value, a list item, a flow collection, or an indented continuation of any of those.
YAML_LINE = re.compile(r"""^(?:
    \#                                       # comment
  | [ \t]                                    # indented continuation
  | -(?:[ \t]|$)                             # list item
  | [?:](?:[ \t]|$)                          # explicit key and its value
  | [\[{]                                    # flow collection
  | (?:"(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[^\s\#:\[\]{},][^:]*?)[ \t]*:(?:[ \t]|$)   # key:
)""", re.VERBOSE)


def _headings(text: str):
    """Markdown headings outside fenced code, with source offsets (#174)."""
    headings = []
    fence = None
    previous = None
    offset = 0
    lines = text.splitlines(keepends=True)
    # YAML's closing --- is not a Setext underline for its last field (#174).
    # Keep original offsets so declarations retain their source locations.
    #
    # A leading `---` opens front matter when a `---` or `...` closer follows
    # within FRONT_MATTER_MAX_LINES and every non-blank line between could be
    # YAML (#189). The test rejects on the first line that cannot be YAML
    # rather than requiring one of a list of key shapes, which is what missed
    # quoted keys and flow mappings before. Blank lines and `#` comments are
    # YAML, so spaced and commented metadata is recognized. A thematic break
    # followed by prose meets a prose line and stays prose. Accepted loss: a
    # thematic break followed only by headings, list items, indented lines or
    # lines containing a colon, then a later `---`, reads as front matter.
    if lines and lines[0].lstrip("\ufeff").rstrip() == "---":
        for end in range(1, min(len(lines), FRONT_MATTER_MAX_LINES + 1)):
            body = lines[end].rstrip()
            if body in ("---", "..."):
                offset = sum(map(len, lines[:end + 1]))
                lines = lines[end + 1:]
                break
            if body and not YAML_LINE.match(body):
                break
    for line in lines:
        stripped = line.rstrip("\r\n")
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", stripped)
        if fence:
            if re.fullmatch(r" {0,3}" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*", stripped):
                fence = None
            previous = None
        elif marker:
            fence = marker.group(1)
            previous = None
        else:
            atx = re.match(r"^ {0,3}#{1,6}(?:[ \t]+(.*)|$)", stripped)
            if atx:
                title = re.sub(r"[ \t]+#+[ \t]*$", "", atx.group(1) or "").strip()
                headings.append((offset, title))
                previous = None
            elif previous and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", stripped):
                headings.append(previous)
                previous = None
            else:
                previous = (offset, stripped.strip()) if stripped.strip() and not stripped.startswith(("    ", "\t")) else None
        offset += len(line)
    return headings

def _record_head(stanza: str) -> str:
    lines = []
    for line in stanza.splitlines():
        if not RECORD_LINE.match(line):
            break
        lines.append(line)
    return "\n".join(lines)


def _walk(root: Path, suffix: str, prune: frozenset[str] = frozenset()):
    # Pruned on the directories below `root` only. Matching the absolute path
    # would prune a whole tree that happens to sit under a directory named
    # `results` or `target`, the way a checkout or a test's temp dir can (#198).
    for path in sorted(root.rglob(f"*{suffix}")):
        if (PRUNE_DIRS | prune) & set(path.relative_to(root).parts[:-1]):
            continue
        yield path


# `experiments/<e>/` is a corpus container beside `docs/` (Document Format
# section 2, 2026-09-25): the charter at its root, each probe's Spec, `code/`
# and `results/` under `<arm>/<probe>/`. `results/` answers to its own clock and
# no gate or ingest reads it, so it is pruned at any depth, which is the set
# WeaverTools' `.hadesignore` excludes with `experiments/**/results/` and the
# census prunes in both `docs()` and `probes()`. One rule, three readers (#198).
EXPERIMENTS = "experiments"
EXPERIMENT_PRUNE = frozenset({"results"})
DOCUMENT_BASES = (("docs", frozenset()), (EXPERIMENTS, EXPERIMENT_PRUNE))


def _probe_code_roots(repo: Path) -> list[Path]:
    """Every probe's `code/`, found the way the census's `probes()` finds it.

    By walking rather than a fixed-depth glob, so a `code/` one level off the
    documented depth is read instead of silently skipped, and never descending
    into `results/`. The first `code/` on a path is the root; nothing below it is
    searched for another.
    """
    top = repo / EXPERIMENTS
    if not top.is_dir():
        return []
    found = []

    def visit(directory: Path):
        for child in sorted(p for p in directory.iterdir() if p.is_dir()):
            if child.name in PRUNE_DIRS | EXPERIMENT_PRUNE:
                continue
            if child.name == "code":
                found.append(child)
            else:
                visit(child)

    visit(top)
    return found


def _container_note(rel: str, name: str, kind: str) -> str | None:
    """Where the Format puts an experiment or probe record, checked (#198).

    Section 2 derives containers from the directory: an experiment is declared
    by `experiments/<e>/README.md`, a probe by a `-Spec.md` in
    `experiments/<e>/<arm>/<probe>/`. The Spec's stem is not matched against
    the directory: the first probe's is `blackwell/blackwell-probe-Spec.md`,
    named for the node. The records are read as declared; a
    declaration standing anywhere else is reported rather than trusted or
    dropped, since the directory and the record would then disagree.
    """
    parts = rel.split("/")
    if kind == "experiment" and not (len(parts) == 3 and parts[0] == EXPERIMENTS and parts[2] == "README.md"):
        return f"misplaced experiment record in {rel}: {name} (the Format declares it in experiments/<e>/README.md)"
    if kind == "probe" and not (len(parts) == 5 and parts[0] == EXPERIMENTS and parts[4].endswith("-Spec.md")):
        return f"misplaced probe record in {rel}: {name} (the Format declares it in a -Spec.md in experiments/<e>/<arm>/<probe>/)"
    return None


def read_documents(repo: Path, scope: set[str] | None = None) -> Extraction:
    """Every fenced `graph` block under `docs/` and `experiments/` that the ingest took.

    **`scope` is the set of repo-relative paths the graph actually holds**, and it
    is passed in rather than computed. This module used to decide for itself which
    files were in play, with a hardcoded `docs/` root and a hardcoded `process/`
    exclusion, while `hades ingest` decided from `.hadesignore`. The two agreed by
    coincidence of intent, not by construction.

    The failure that makes it matter: a document under `docs/` declaring a node,
    later added to `.hadesignore`. The ingest stops writing its `documents` row and
    this pass keeps reading its declarations, so the `declared-in` edge names a row
    that does not exist. `write_graph` checks for exactly that and would have caught
    it, but at build time, after a run, with the bad edges already computed.

    A file out of scope holding a `graph` block is reported by name, because
    "declarations nobody read" has to be visible rather than inferred from a count
    that came out low.

    **`process/` is deliberately excluded.** It holds
    `WeaverTools-Document-Format.md`, which illustrates the block grammar with
    worked `graph` examples, and a mapper that reads them files documentation
    samples as live assertions. Including it inflated the assertion count from
    355 to 413 here and moved `uncited_perturbations` from 33 to 65 against a
    baseline with a known answer. The census scopes to `docs/` for this
    reason, and the agreement is the test that caught it.

    **`experiments/` is read beside `docs/`**, with `results/` pruned, as the
    census's `docs()` reads it (#198). Walking `docs/` alone left the 21
    assertions of the first probe's Spec, its probe and experiment records and
    their `parent` edge out of the graph while the ingest held the documents
    that declare them, so a coverage query could not tell "no record" from
    "out of scope".
    """
    out = Extraction()
    seen: dict[str, str] = {}
    out_of_scope: list[str] = []

    for base, prune in DOCUMENT_BASES:
        root = repo / base
        if not root.is_dir():
            continue
        for path in _walk(root, ".md", prune):
            rel = str(path.relative_to(repo))
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if scope is not None and rel not in scope:
                # Only the ones whose absence changes the graph. Every other
                # skipped file is noise in a report someone has to read.
                if GRAPH.search(text):
                    out_of_scope.append(rel)
                continue
            # **No node is minted for the file itself.** `hades ingest` already
            # put this markdown in the graph as a `documents` row, with its text,
            # its chunks and a vector, and `declared-in` points at that row. One
            # node per file was minted here until 2026-09-13, which left
            # `wt_documents` holding 68 file nodes beside the corpus's 13
            # declared `kind: document` records -- two different things in one
            # collection, matching neither count, and the file half carrying no
            # embedding for a traversal to land on. It is the same argument the
            # module docstring already makes for source files: a node that
            # duplicates one the ingest created is a join that proves nothing.

            headings = _headings(text)
            heading_offsets = [offset for offset, _ in headings]
            line_starts = [0] + [match.end() for match in re.finditer("\n", text)]
            for block_match in GRAPH.finditer(text):
                block = block_match.group(1)
                # Either keyword starts a new record: mixed adjacent records
                # must not contribute metadata or body text to their neighbor.
                starts = [match.start() for match in re.finditer(r"^(?:node|edge): ", block, re.M)]
                stanzas = [(start, block[start:end]) for start, end in zip(starts, starts[1:] + [len(block)])]
                for stanza_offset, stanza in stanzas:
                    record = _record_head(stanza)
                    line = NODE_LINE.search(record)
                    if not line:
                        continue
                    name = line.group(1).strip()
                    kind = KIND.search(record)
                    kind = kind.group(1) if kind else "unknown"
                    tag = TAG.search(record)
                    tag = tag.group(1) if tag else None

                    # Section 5 exempts only the named system record, not every
                    # uppercase identifier or every record with kind=system.
                    system = kind == "system" and name == SYSTEM_NODE
                    if (kind == "system" and not system) or (not system and not IDENT_OK.match(name)):
                        out.notes.append(f"malformed node id in {rel}: {name!r}")
                    allowed_tags = {"ratified"} if system else (TAGS if kind == "assertion" else set())
                    if tag is not None and tag not in allowed_tags:
                        out.notes.append(f"unknown tag in {rel}: {name} ({tag})")
                    if name in seen and seen[name] != rel:
                        out.notes.append(
                            f"duplicate node id {name}: {seen[name]} and {rel}"
                        )
                    seen[name] = rel
                    misplaced = _container_note(rel, name, kind)
                    if misplaced:
                        out.notes.append(misplaced)

                    source_offset = block_match.start(1) + stanza_offset + line.start()
                    heading_index = bisect_right(heading_offsets, source_offset) - 1
                    out.nodes.append(
                        Node(ident=name, kind=kind, path=rel, tag=tag, body=stanza.strip(),
                             line=bisect_right(line_starts, source_offset),
                             section=headings[heading_index][1] if heading_index >= 0 else None)
                    )
                    out.edges.append(
                        Edge(src=name, dst=rel, relation="declared-in", basis=DECLARED)
                    )

                # Edge records within the same block.
                for _, stanza in stanzas:
                    record = _record_head(stanza)
                    rel_m = EDGE_REL.search(record)
                    src_m = EDGE_FROM.search(record)
                    dst_m = EDGE_TO.search(record)
                    if not (rel_m and src_m and dst_m):
                        continue
                    via_m = EDGE_VIA.search(record)
                    tag_m = TAG.search(record)
                    out.edges.append(
                        Edge(
                            src=src_m.group(1).strip(),
                            dst=dst_m.group(1).strip(),
                            relation=rel_m.group(1),
                            basis=DECLARED,
                            via=via_m.group(1).strip() if via_m else None,
                            tag=tag_m.group(1) if tag_m else None,
                        )
                    )

    for rel in out_of_scope:
        out.notes.append(
            f"out of scope and holds a graph block, declarations not read: {rel}"
        )
    out.notes.append(f"declaring_files_out_of_scope={len(out_of_scope)}")
    return out



def _header_citations(text: str, suffix: str) -> list[str]:
    if suffix != ".py":
        return HEADER_CITE.findall(text)
    head = []
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            break
        head.append(line)
    return HASH_HEADER.findall("\n".join(head))


def _citations(text: str, suffix: str) -> tuple[list[str], str | None]:
    if suffix != ".py":
        return ITEM_CITE.findall(text), None
    found = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type != tokenize.COMMENT or token.line[:token.start[1]].strip():
                continue
            match = COMMENT_CITE.match(token.string)
            if match:
                found.append(match.group(1))
    except (tokenize.TokenError, SyntaxError, UnicodeDecodeError) as error:
        # Never fall back to reading strings as declarations on malformed input.
        return [], f"will not tokenize, citations unread ({type(error).__name__})"
    return found, None

def read_conformance(
    repo: Path, assertions: set[str], scope: set[str] | None = None
) -> Extraction:
    """Conformance citations across workspace crates, as `cites` edges.

    `scope` is the set of repo-relative paths the graph holds, for the reason
    given on [`read_documents`]. A `cites` edge runs *from* a `codebase_files`
    node, so a citation in a file the ingest never took produces an edge with no
    source. That case was already detected by `write_graph` and is now prevented
    here, which also means the header obligation is measured over the files the
    graph contains rather than over the files on disk.

    The obligation follows supported Rust, CUDA and Python units, tests
    included, with `build.rs` and manifests excepted. Resolution is against the
    declared *assertion* identifiers, as the census resolves it: a citation of a
    crate, term or any other kind is dangling, not an edge (#198).

    Manifests are walked for citations and excused the header obligation. They
    cite with `#` rather than `//!`, which ITEM_CITE already reads, and three
    assertions in this corpus are cited from nowhere else.

    **A probe's `code/` is read like a crate's tree** (#198): the same suffixes,
    the same citation forms and the same header obligation, which is how the
    census's `sources()` treats `probes()` and what the Document Format says of
    it. Reading `crates/` alone left 71 citation pairs out of the graph.
    """
    out = Extraction()
    crates = repo / "crates"
    roots = ([crates] if crates.is_dir() else []) + _probe_code_roots(repo)
    if not crates.is_dir():
        # Still a finding, but no longer a reason to stop: probe code is read
        # whether or not the tree has a workspace.
        out.notes.append("no crates/ directory, conformance pass read probe code only")
    if not roots:
        return out

    headers_seen = 0
    files_owing = 0
    headerless: list[str] = []
    out_of_scope: list[str] = []

    paths = [p for root in roots for suffix in CONFORMANCE_SUFFIXES
             for p in _walk(root, suffix, EXPERIMENT_PRUNE if root != crates else frozenset())]
    for path in sorted(paths):
        rel = str(path.relative_to(repo))
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        citations, unreadable = _citations(text, path.suffix)
        if scope is not None and rel not in scope:
            if citations:
                out_of_scope.append(rel)
            continue
        lang = "rust" if path.suffix == ".rs" else path.suffix.lstrip(".")
        out.nodes.append(Node(ident=rel, kind="source", path=rel, lang=lang))

        headers = _header_citations(text, path.suffix)
        headers_seen += len(headers)
        if unreadable:
            out.notes.append(f"malformed citation in {rel}: {unreadable}")

        # Every citation, file-level or item-level, becomes an edge. Malformed
        # targets are reported under the identifier as written, never
        # truncated to a prefix, so grepping for the offender finds it.
        for raw in dict.fromkeys(citations):
            target = raw.strip()
            if (raw and not raw[0].isspace()) or not IDENT_OK.fullmatch(target):
                out.notes.append(f"malformed citation in {rel}: conforms: {target}")
                continue
            edge = Edge(src=rel, dst=target, relation="cites", basis=DECLARED)
            (out.edges if target in assertions else out.dangling).append(edge)

        # The obligation is separate, and it follows the unit.
        if path.name not in NO_HEADER_OWED and path.suffix not in NO_HEADER_OWED_SUFFIXES:
            files_owing += 1
            if not headers:
                headerless.append(rel)

    # The reporting rule from the handoff: a pass that sees many of something
    # and writes none of it has to say why.
    if headers_seen and not out.edges:
        out.notes.append(
            f"saw {headers_seen} conformance headers and resolved none of them, "
            f"which means the assertion set was empty or came from elsewhere"
        )

    for rel in out_of_scope:
        out.notes.append(f"out of scope and holds a citation, not read: {rel}")
    out.notes.append(f"citing_files_out_of_scope={len(out_of_scope)}")
    out.notes.append(f"headers_seen={headers_seen}")
    out.notes.append(f"sources_without_a_header={len(headerless)}")
    out.notes.append(f"sources_owing_a_header={files_owing}")
    return out


def ingest(
    repo: Path,
    doc_scope: set[str] | None = None,
    code_scope: set[str] | None = None,
) -> tuple[Extraction, Extraction]:
    """Read the corpus, bounded to what the graph holds.

    Both scopes are repo-relative path sets, supplied by the caller because the
    caller is the half with database access. `None` means unbounded, which is the
    old behaviour and is kept only for reading a tree with no graph behind it.
    """
    docs = read_documents(repo, doc_scope)
    # Assertions only, the set the census resolves citations against (#198).
    assertions = {n.ident for n in docs.nodes if n.kind == "assertion"}
    code = read_conformance(repo, assertions, code_scope)
    return docs, code
