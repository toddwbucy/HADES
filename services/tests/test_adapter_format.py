"""Document Format v0.24 allowances through real extraction/writer entry points."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from weavertools import write_graph as w
from weavertools.extractor import ingest, read_documents


def make_format_fixture(root):
    files = {
        'Cargo.toml': '[workspace]\nmembers = ["crates/fixture"]\n',
        'docs/spec.md': '```graph\nnode: WeaverTools\nkind: system\ntag: ratified\n\nnode: fixture-claim\nkind: assertion\ntag: review\n```\n',
        'crates/fixture/Cargo.toml': '# conforms: fixture-claim\n[package]\nname = "fixture"\nversion = "0.0.0"\n',
        'crates/fixture/askama.toml': '# conforms: fixture-claim\n',
        'crates/fixture/build.rs': 'fn main() {}\n',
        'crates/fixture/src/lib.rs': '//! conforms: fixture-claim\npub fn fixture() {}\n',
        'crates/fixture/src/absent.rs': 'pub fn absent() {}\n',
        'crates/fixture/tests/item.rs': '/// conforms: fixture-claim\nfn check() {}\n',
        'crates/fixture/kernels/has.cu': '//! conforms: fixture-claim\n',
        'crates/fixture/kernels/missing.cu': 'void kernel() {}\n',
        'crates/fixture/loops/head.py': '#!/usr/bin/python3\n\n# conforms: fixture-claim\nvalue = 1\n',
        'crates/fixture/loops/item.py': 'value = 1\n# conforms: fixture-claim\n',
        'crates/fixture/loops/string.py': '"""\n# conforms: fixture-claim\n"""\n',
        'crates/fixture/loops/inline.py': 'value = 1 # conforms: fixture-claim\n',
        'crates/fixture/loops/broken.py': '"""unterminated\n',
        'crates/fixture/deferred.sh': 'echo no-header\n',
    }
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return files


def test_supported_sibling_languages_match_census_fixture(tmp_path):
    files = make_format_fixture(tmp_path)
    _, code = ingest(tmp_path)
    # Independently checked against census.py at WeaverTools 13dd7db; seven
    # missing file headers, including item citations which still form edges.
    assert 'sources_without_a_header=7' in code.notes
    assert 'sources_owing_a_header=10' in code.notes
    assert len(code.nodes) == 13
    assert {edge.src for edge in code.edges} == {
        'crates/fixture/' + relative for relative in [
            'Cargo.toml', 'askama.toml', 'src/lib.rs', 'tests/item.rs',
            'kernels/has.cu', 'loops/head.py', 'loops/item.py']}
    assert any('loops/broken.py' in note and 'will not tokenize' in note for note in code.notes)
    assert not code.dangling
    scope = set(files) - {'crates/fixture/loops/item.py', 'crates/fixture/kernels/missing.cu'}
    _, bounded = ingest(tmp_path, None, scope)
    assert 'sources_without_a_header=5' in bounded.notes
    assert all(edge.src in scope for edge in bounded.edges)


def test_system_record_and_ratified_tag_are_not_reported_as_defects(tmp_path, monkeypatch):
    files = make_format_fixture(tmp_path)
    reports, rows = [], []
    monkeypatch.setattr(sys, 'argv', ['writer', '--db', 'fixture', '--repo', str(tmp_path)])
    def backend(db, route, body=None, method='POST'):
        if route == 'cursor':
            names = [name for name in files if name.startswith('crates/')] if 'codebase_files' in body['query'] else ['docs/spec.md']
            return {'result': [[name, f'key-{i}'] for i, name in enumerate(names)], 'hasMore': False}
        if route == 'collection' and method == 'GET':
            return {'result': []}
        if route == 'collection':
            return {'name': body['name'], 'type': body['type']}
        if route.startswith('import?'):
            rows.extend(body)
            return dict(error=False, created=len(body), updated=0, errors=0, ignored=0, empty=0)
        if route.startswith('document/'):
            reports.append(body)
            return {'_key': 'latest', '_id': w.REPORT + '/latest'}
        raise AssertionError(route)
    monkeypatch.setattr(w, 'arango', backend)
    assert w.main() == 0
    system, = [row for row in rows if row.get('kind') == 'system']
    assert (system['ident'], system['tag']) == ('WeaverTools', 'ratified')
    assert not any('malformed node' in note or 'unknown tag' in note for note in reports[-1]['document_notes'])


@pytest.mark.parametrize('name,kind,tag', [
    ('WeaverTools', 'assertion', 'ratified'), ('OtherSystem', 'system', 'ratified'),
    ('other-system', 'system', 'ratified'), ('fixture-claim', 'assertion', 'ratified'),
    ('WeaverTools', 'system', 'review'), ('fixture-term', 'term', 'ratified'),
])
def test_system_allowance_does_not_admit_sibling_invalid_shapes(tmp_path, name, kind, tag):
    path = tmp_path / 'docs/spec.md'
    path.parent.mkdir()
    path.write_text(f'```graph\nnode: {name}\nkind: {kind}\ntag: {tag}\n```\n')
    notes = read_documents(tmp_path).notes
    assert any('malformed node' in note or 'unknown tag' in note for note in notes)


def test_malformed_and_trailing_citations_do_not_become_valid_edges(tmp_path):
    make_format_fixture(tmp_path)
    path = tmp_path / 'crates/fixture/loops/bad.py'
    path.write_text('# conforms:fixture-claim\n# conforms: fixture-claim extra\nvalue = 1 # conforms: fixture-claim\n')
    _, code = ingest(tmp_path)
    assert not any(edge.src.endswith('/bad.py') for edge in code.edges)
    assert sum('bad.py' in note and 'malformed citation' in note for note in code.notes) == 2
