"""`experiments/` is read as the census reads it, and the writer places what it reads (#198).

The adapter and WeaverTools' `process/gates/census.py` are two readers of one
rule, so on one tree they must agree. The census comparison runs the real
`census.py` against the fixture when a WeaverTools checkout is present
(`WEAVERTOOLS_CENSUS` overrides the path) and skips otherwise; the adapter's own
counts are asserted either way.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from weavertools import write_graph as w
from weavertools.extractor import ingest

CENSUS = Path(os.environ.get('WEAVERTOOLS_CENSUS', '/opt/weavertools/WeaverTools/process/gates/census.py'))
SCHEMA = Path(__file__).resolve().parents[1] / 'adapters' / 'weavertools' / 'schema.yaml'


def graph(*records):
    return '```graph\n' + '\n\n'.join(records) + '\n```\n'


FIXTURE = {
    'Cargo.toml': '[workspace]\nmembers = ["crates/fx"]\n',
    'docs/project/WeaverTools-PRD.md': '# Apex\n\n' + graph('node: WeaverTools\nkind: system\ntag: ratified'),
    'docs/crates/fx/fx-Spec.md': '# fx\n\n' + graph(
        'node: fx\nkind: crate',
        'edge: parent\nfrom: fx\nto: WeaverTools',
        'node: fx-claim\nkind: assertion\ntag: perturbation',
        'edge: asserts\nfrom: fx\nto: fx-claim'),
    'crates/fx/src/lib.rs': '//! conforms: fx-claim\n',
    # The experiment container, per Document Format section 2.
    'experiments/tuple/README.md': '# Charter\n\n' + graph('node: tuple\nkind: experiment'),
    'experiments/tuple/arm/probe-a/probe-a-Spec.md': '# Probe\n\n' + graph(
        'node: probe-a\nkind: probe',
        'edge: parent\nfrom: probe-a\nto: tuple') + '\n## Claims\n\n' + graph(
        'node: probe-a-holds\nkind: assertion\ntag: perturbation',
        'edge: asserts\nfrom: probe-a\nto: probe-a-holds',
        'node: probe-a-halts\nkind: assertion\ntag: review',
        'edge: asserts\nfrom: probe-a\nto: probe-a-halts'),
    'experiments/tuple/arm/probe-a/code/README.md': '# How to run\n',
    'experiments/tuple/arm/probe-a/code/run.py': (
        '# conforms: probe-a-holds\n# conforms: probe-a-halts\nprint("run")\n'),
    # Owes a header and has none; its item-level citation still counts.
    'experiments/tuple/arm/probe-a/code/helper.py': (
        'def f():\n    # conforms: probe-a-halts\n    return 1\n'),
    # `results/` answers to its own clock: neither reader may take anything here.
    'experiments/tuple/arm/probe-a/results/2026-09-27/notes.md': graph(
        'node: phantom-result\nkind: assertion\ntag: perturbation',
        'edge: asserts\nfrom: probe-a\nto: phantom-result'),
    'experiments/tuple/arm/probe-a/results/replay.py': '# conforms: phantom-result\n',
}

EXPECTED_KINDS = {'system': 1, 'crate': 1, 'assertion': 3, 'experiment': 1, 'probe': 1}
EXPECTED_RELATIONS = {'declared-in': 7, 'parent': 2, 'asserts': 3}
EXPECTED_CITES = {
    ('crates/fx/src/lib.rs', 'fx-claim'),
    ('experiments/tuple/arm/probe-a/code/run.py', 'probe-a-holds'),
    ('experiments/tuple/arm/probe-a/code/run.py', 'probe-a-halts'),
    ('experiments/tuple/arm/probe-a/code/helper.py', 'probe-a-halts'),
}


@pytest.fixture
def tree(tmp_path):
    for rel, text in FIXTURE.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def files(tree, *suffixes):
    return sorted(str(p.relative_to(tree)) for p in tree.rglob('*') if p.is_file() and p.suffix in suffixes)


def test_adapter_reads_experiments_and_prunes_results(tree):
    docs, code = ingest(tree)
    assert Counter(n.kind for n in docs.nodes) == EXPECTED_KINDS
    assert Counter(e.relation for e in docs.edges) == EXPECTED_RELATIONS
    assert {(e.src, e.dst) for e in code.edges} == EXPECTED_CITES
    assert code.dangling == []
    assert 'sources_without_a_header=1' in code.notes
    # Negative direction: nothing under results/ contributes.
    assert all('/results/' not in (n.path or '') for n in docs.nodes)
    assert 'phantom-result' not in {n.ident for n in docs.nodes} | {e.dst for e in docs.edges}
    assert all('/results/' not in e.src for e in code.edges + code.dangling)
    assert not any('misplaced' in note for note in docs.notes)


def test_misplaced_container_records_are_reported(tree):
    (tree / 'docs/stray.md').write_text(graph('node: loose\nkind: probe', 'node: wide\nkind: experiment'))
    docs, _ = ingest(tree)
    misplaced = [n for n in docs.notes if n.startswith('misplaced')]
    assert len(misplaced) == 2 and all('docs/stray.md' in n for n in misplaced)


def load_census(tree):
    if not CENSUS.is_file():
        pytest.skip(f'no WeaverTools census at {CENSUS}; set WEAVERTOOLS_CENSUS')
    gates = tree / 'process' / 'gates'
    gates.mkdir(parents=True)
    shutil.copy(CENSUS, gates / 'census.py')
    # The census reads the tracked set plus untracked-not-ignored files, so the
    # fixture has to be a repository, though nothing need be committed.
    subprocess.run(['git', 'init', '-q', str(tree)], check=True)
    spec = importlib.util.spec_from_file_location('fixture_census', gates / 'census.py')
    census = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(census)
    assert Path(census.ROOT) == tree
    return census


def census_reading(census):
    """Node kinds and citation pairs, built only from the census's own readers.

    `take()` returns defect lists, not totals, so the totals are composed from
    the functions it composes: `docs()`, `sources()`, `cites()`, its `GRAPH`,
    `NODE_LINE` and `field(KIND)`, with its stanza split. File selection, which
    is what #198 is about, comes entirely from the census.
    """
    kinds, declared_in = Counter(), set()
    for path in census.docs():
        rel = os.path.relpath(path, census.ROOT)
        for block in census.GRAPH.findall(census.read(path)):
            for stanza in re.split(r'(?=^\s*(?:node|edge):)', block, flags=re.M):
                line = census.NODE_LINE.search(stanza)
                if line:
                    kinds[census.field(census.KIND, stanza)[0]] += 1
                    declared_in.add(rel)
    pairs = set()
    for path, _ in census.sources():
        rel = os.path.relpath(path, census.ROOT)
        raws, unreadable = census.cites(census.read(path), rel)
        assert unreadable is None
        pairs |= {(rel, raw.strip()) for raw in raws}
    return kinds, declared_in, pairs


def test_counts_equal_the_census_on_the_same_tree(tree):
    census = load_census(tree)
    kinds, declaring_files, pairs = census_reading(census)
    reading = census.take()
    docs, code = ingest(tree)

    assert Counter(n.kind for n in docs.nodes) == kinds
    assert {n.path for n in docs.nodes} == declaring_files
    assert {(e.src, e.dst) for e in code.edges + code.dangling} == pairs
    assert sorted(e.dst for e in code.dangling) == reading['dangling_citations']
    assert f"sources_without_a_header={len(reading['sources_without_a_header'])}" in code.notes
    assert reading['sources_without_a_header'] == ['experiments/tuple/arm/probe-a/code/helper.py']
    assert reading['duplicate_node_ids'] == [] and not any(n.startswith('duplicate') for n in docs.notes)
    # And both equal the fixture as written, so agreement is not two readers
    # sharing one mistake about results/.
    assert kinds == EXPECTED_KINDS and pairs == EXPECTED_CITES


def run_writer(monkeypatch, tree, capsys):
    """The writer over the fixture, every file in scope, against a fake backend."""
    calls = []
    md, src = files(tree, '.md'), files(tree, '.rs', '.py', '.toml', '.cu')

    def backend(db, path, body=None, method='POST'):
        calls.append((path, method))
        if path == 'cursor' and 'documents' in body['query']:
            return {'result': [[f, w.file_key(f)] for f in md], 'hasMore': False}
        if path == 'cursor' and 'codebase_files' in body['query']:
            return {'result': [[f, w.file_key(f)] for f in src], 'hasMore': False}
        if path == 'collection' and method == 'GET':
            return {'result': []}
        raise AssertionError(path)
    monkeypatch.setattr(w, 'arango', backend)
    monkeypatch.setattr(w, 'resolve_source_git', lambda root: None)
    monkeypatch.setattr(sys, 'argv', ['writer', '--db', 'fixture', '--repo', str(tree), '--dry-run'])
    code = w.main()
    output = capsys.readouterr()
    counts = dict(re.findall(r'^  (wt_\w+) +(\d+)$', output.out, re.M))
    return code, output, {k: int(v) for k, v in counts.items()}, calls


def test_writer_places_experiment_records(monkeypatch, tree, capsys):
    code, output, counts, _ = run_writer(monkeypatch, tree, capsys)
    assert code == 0, output.err
    assert counts == {
        'wt_systems': 1, 'wt_crates': 1, 'wt_assertions': 3, 'wt_experiments': 1, 'wt_probes': 1,
        'wt_declared_in_edges': 7, 'wt_parent_edges': 2, 'wt_asserts_edges': 3, 'wt_cites_edges': 4,
    }


def test_unmapped_kind_fails_the_run_before_any_write(monkeypatch, tree, capsys):
    (tree / 'docs/widget.md').write_text(graph('node: gizmo\nkind: widget'))
    code, output, counts, calls = run_writer(monkeypatch, tree, capsys)
    assert code == 1
    assert "kind 'widget' has no collection" in output.err and 'gizmo' in output.err
    assert 'no write stage started' in output.err
    assert counts == {} and 'skipping' not in output.err
    # Only the two scope reads: no inventory, no collection, no import.
    assert [path for path, _ in calls] == ['cursor', 'cursor']


def test_the_old_schema_would_have_refused_the_widened_walk(monkeypatch, tree, capsys):
    """The failure #198 would have been silent without the sibling fix."""
    old = {k: v for k, v in w.NODE_COLLECTIONS.items() if k not in ('experiment', 'probe')}
    monkeypatch.setattr(w, 'NODE_COLLECTIONS', old)
    code, output, _, _ = run_writer(monkeypatch, tree, capsys)
    assert code == 1
    assert "kind 'experiment' has no collection" in output.err
    assert "kind 'probe' has no collection" in output.err
    assert 'parent edge source probe-a' in output.err


@pytest.mark.parametrize('record, finding', [
    ('edge: asserts\nfrom: fx-claim\nto: probe-a-holds', 'is outside its edge definition'),
    ('edge: parent\nfrom: tuple\nto: WeaverTools', 'is outside its edge definition'),
    ('edge: parent\nfrom: probe-a\nto: nobody', 'target nobody is not declared'),
    ('edge: haunts\nfrom: fx\nto: fx-claim', "relation 'haunts'"),
])
def test_edge_outside_its_definition_fails(monkeypatch, tree, capsys, record, finding):
    (tree / 'docs/extra.md').write_text(graph(record))
    code, output, counts, _ = run_writer(monkeypatch, tree, capsys)
    assert code == 1
    assert finding in output.err
    assert counts == {}


def schema_edge_definitions():
    """`edge_definitions` from schema.yaml, read without a YAML dependency.

    The CI Python environment carries no YAML parser, and the file's shape is
    fixed: `- name:` rows with flow-style `[a, b]` lists that may wrap.
    """
    text = SCHEMA.read_text()
    section = text.split('\nedge_definitions:\n', 1)[1].split('\nnamed_graphs:\n', 1)[0]
    section = re.sub(r'#.*', '', section)
    found = {}
    for row in re.split(r'^  - name: ', section, flags=re.M)[1:]:
        name = row.split('\n', 1)[0].strip()
        lists = [re.findall(r'[\w-]+', body) for body in re.findall(r'_collections:\s*\[([^\]]*)\]', row)]
        found[name] = (set(lists[0]), set(lists[1]))
    return found


def test_writer_endpoints_mirror_schema_yaml():
    definitions = {k: v for k, v in schema_edge_definitions().items() if k.startswith('wt_')}
    assert definitions == w.RELATION_ENDPOINTS
    declared = set(re.findall(r'^  - name: (wt_\w+)\n    type: document\n    lookup_fields: \[ident\]$',
                              SCHEMA.read_text(), re.M))
    assert set(w.NODE_COLLECTIONS.values()) == declared
