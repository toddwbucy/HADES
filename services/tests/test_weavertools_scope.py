"""The conformance extractor reads only what the graph holds.

**The defect these tests exist for.** The extractor decided its own scope with a
hardcoded `docs/` root and a hardcoded `process/` exclusion, while `hades ingest`
decided from `.hadesignore`. Two implementations of one rule, agreeing by
coincidence of intent.

The concrete failure: a document under `docs/` that declares a node and is later
added to `.hadesignore`. The ingest stops writing its `documents` row, the
extractor keeps reading its declarations, and the `declared-in` edge names a row
that does not exist. `write_graph` checks for that and would catch it, but at
build time, after the run, with the bad edges already computed. Scope is passed in
now, so the case cannot be constructed.

No database and no GPU: the extractor is a pure function over a repository, and
these tests are what keep it one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adapters"))

from weavertools.extractor import ingest, read_documents  # noqa: E402

DECLARING = "docs/fixture-Spec.md"


@pytest.fixture
def repo(tmp_path):
    """A declared graph and citing files owned entirely by this test."""
    documents = {
        DECLARING: "```graph\nnode: fixture-assertion\nkind: assertion\n```\n",
        "docs/second-Spec.md": "```graph\nnode: fixture-second\nkind: assertion\n```\n",
        "crates/fixture/src/lib.rs": "//! conforms: fixture-assertion\npub fn first() {}\n",
        "crates/fixture/tests/check.rs": "//! conforms: fixture-second\nfn check() {}\n",
    }
    for relative, text in documents.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def _rel_md(root: Path) -> set[str]:
    return {
        str(p.relative_to(root))
        for p in (root / "docs").rglob("*.md")
        if ".git" not in p.parts and "archive" not in p.parts
    }


def test_unbounded_scope_reads_the_declaring_file(repo):
    """The baseline: with no scope, the file's declarations are read."""
    docs = read_documents(repo, None)
    assert any(n.path == DECLARING for n in docs.nodes), (
        "the baseline itself is broken: nothing was read from " + DECLARING
    )


def test_a_file_outside_the_scope_declares_nothing(repo):
    """The defect, in the shape that produced it.

    Everything is in scope except one declaring file. Nothing it declares may
    reach the graph, and no edge may name it.
    """
    scope = _rel_md(repo) - {DECLARING}
    docs = read_documents(repo, scope)

    assert not [n for n in docs.nodes if n.path == DECLARING], (
        "an out-of-scope file produced nodes, so `declared-in` would name a "
        "`documents` row the ingest never wrote"
    )
    assert not [e for e in docs.edges if e.dst == DECLARING], (
        "an out-of-scope file is the target of a `declared-in` edge"
    )


def test_an_excluded_declaring_file_is_reported_by_name(repo):
    """Silence is the other half of the defect.

    A count that came out low reads exactly like a clean run, so the file is
    named.
    """
    scope = _rel_md(repo) - {DECLARING}
    docs = read_documents(repo, scope)

    named = [n for n in docs.notes if n.startswith("out of scope") and DECLARING in n]
    assert named, f"{DECLARING} was dropped without being reported: {docs.notes[-3:]}"
    assert any(
        n.startswith("declaring_files_out_of_scope=") and not n.endswith("=0")
        for n in docs.notes
    ), "the summary count does not record the exclusion"


def test_every_declared_in_target_is_inside_the_scope(repo):
    """The invariant, over the whole corpus.

    This is the property `write_graph` used to check against the database after
    the fact. Holding it here means the bad edge is never computed.
    """
    scope = _rel_md(repo)
    docs, _ = ingest(repo, scope, None)
    targets = {e.dst for e in docs.edges if e.relation == "declared-in"}
    assert targets <= scope, f"declared-in targets outside the scope: {targets - scope}"


def test_scoping_out_a_citing_file_drops_its_cites_edges(repo):
    """The same rule on the code half.

    A `cites` edge runs from a `codebase_files` node, so a citation in a file the
    ingest never took is an edge with no source.
    """
    rs = sorted(
        str(p.relative_to(repo))
        for p in (repo / "crates").rglob("*.rs")
        if "target" not in p.parts
    )
    assert rs, "no .rs files under crates/, the corpus is not what this test assumes"

    full = set(rs)
    docs, code_full = ingest(repo, _rel_md(repo), full)
    cited_from = {e.src for e in code_full.edges if e.relation == "cites"}
    assert cited_from, "no cites edges at all, so this test proves nothing"

    victim = sorted(cited_from)[0]
    _, code_less = ingest(repo, _rel_md(repo), full - {victim})
    assert victim not in {e.src for e in code_less.edges}, (
        f"{victim} is out of scope and still produced cites edges"
    )
    assert any(
        n.startswith("out of scope") and victim in n for n in code_less.notes
    ), f"{victim} was dropped without being reported"


def test_an_inline_graph_reference_does_not_open_a_fence():
    """Prose about the grammar is not the grammar.

    Unanchored, the fence pattern matched an inline ```` ```graph ```` inside a
    sentence and ran to the next triple backtick anywhere in the file. The two
    such matches in this corpus hold no record and contributed nothing, but a
    quoted example inside the swallowed region would have injected declarations
    nobody wrote. Needs no corpus, so it runs everywhere.
    """
    from weavertools.extractor import GRAPH

    prose = (
        "A ```graph` block carries `node`/`kind` stanzas.\n"
        "\n"
        "    node: not-a-declaration\n"
        "    kind: assertion\n"
        "\n"
        "That is the grammar, and ```this``` closes nothing.\n"
    )
    assert GRAPH.findall(prose) == [], "an inline reference opened a fence"

    real = "```graph\nnode: a-real-one\nkind: assertion\n```\n"
    assert len(GRAPH.findall(real)) == 1, "a real fence stopped matching"


@pytest.mark.parametrize("separator", ["\n", "\n\n"])
def test_mixed_records_keep_their_own_metadata(tmp_path, separator):
    records = [
        "node: first\nkind: assertion",
        "edge: supports\nfrom: first\nto: second\ntag: review\nvia: contract-one",
        "node: second\nkind: term\ntag: manifest",
        "edge: depends-on\nfrom: second\nto: first",
        "node: third\nkind: axiom\ntag: perturbation",
    ]
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/spec.md").write_text(
        "```graph\n" + separator.join(records) + "\n```\n"
    )
    result = read_documents(tmp_path)
    assert [(n.ident, n.kind, n.tag) for n in result.nodes] == [
        ("first", "assertion", None),
        ("second", "term", "manifest"),
        ("third", "axiom", "perturbation"),
    ]
    declared = [e for e in result.edges if e.relation != "declared-in"]
    assert [(e.relation, e.tag, e.via) for e in declared] == [
        ("supports", "review", "contract-one"),
        ("depends-on", None, None),
    ]
    assert result.nodes[0].body == records[0]
    assert result.nodes[1].body == records[2]
    assert len([e for e in result.edges if e.relation == "declared-in"]) == 3


def test_node_metadata_stops_at_end_of_record_head(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/spec.md").write_text(
        "```graph\nnode: first\n\nDescription of the declaration.\n"
        "kind: assertion\ntag: review\n```\n"
    )
    node = read_documents(tmp_path).nodes[0]
    assert (node.kind, node.tag) == ("unknown", None)
    assert "Description of the declaration." in node.body
