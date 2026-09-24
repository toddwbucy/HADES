"""Locations follow every declared node through extraction and writer payloads."""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from weavertools import write_graph as w
from weavertools.extractor import read_documents


def test_every_node_kind_keeps_its_own_line_and_section(tmp_path, monkeypatch):
    relative = 'docs/fixture-Spec.md'
    path = tmp_path / relative
    path.parent.mkdir()
    kinds = list(w.NODE_COLLECTIONS)
    def block(selected):
        return '```graph\n' + ''.join(f'node: fixture-{kind}\nkind: {kind}\n\n' for kind in selected) + '```\n'
    first = '# First λ section ###\n\n' + block(kinds[:4])
    second = 'Second section\n--------------\n\n```text\n# Not a section\n```\n\n' + block(kinds[4:])
    monkeypatch.setattr(sys, 'argv', ['writer', '--db', 'fixture', '--repo', str(tmp_path)])
    batches = {}
    def backend(db, route, body=None, method='POST'):
        if route == 'cursor':
            return {'result': [[relative, 'document-key']], 'hasMore': False}
        if route == 'collection' and method == 'GET':
            return {'result': []}
        if route == 'collection':
            return {'name': body['name'], 'type': body['type']}
        if route.startswith('import?'):
            from urllib.parse import parse_qs
            collection = parse_qs(route.split('?', 1)[1])['collection'][0]
            batches[collection] = body
            return dict(error=False, created=len(body), updated=0, errors=0, ignored=0, empty=0)
        if route.startswith('document/'):
            return {'_key': 'latest', '_id': w.REPORT + '/latest'}
        raise AssertionError(route)
    monkeypatch.setattr(w, 'arango', backend)
    old_lines = {}
    for padding in ['', '\nInserted prose.\n\n']:
        text = first + padding + second
        path.write_text(text)
        assert w.main() == 0
        for kind in kinds:
            row, = batches[w.NODE_COLLECTIONS[kind]]
            expected_line = text.splitlines().index(f'node: fixture-{kind}') + 1
            assert row['line'] == expected_line
            assert row['path'] == relative
            assert row['section'] == ('First λ section' if kind in kinds[:4] else 'Second section')
            if padding:
                assert row['line'] == old_lines[kind] + (0 if kind in kinds[:4] else 3)
            else:
                old_lines[kind] = row['line']


def test_node_before_any_heading_has_explicit_absent_section(tmp_path):
    path = tmp_path / 'docs' / 'fixture.md'
    path.parent.mkdir()
    path.write_text('```graph\nnode: first\nkind: assertion\n\nnode: second\nkind: term\n```\n')
    nodes = read_documents(tmp_path).nodes
    assert [(node.ident, node.line, node.section) for node in nodes] == [
        ('first', 2, None), ('second', 5, None)]


def test_yaml_front_matter_does_not_assign_a_section(tmp_path):
    # Closing YAML --- was mistaken for a Setext underline (#174).
    path = tmp_path / 'docs' / 'fixture.md'
    path.parent.mkdir()
    text = ('---\ntitle: Metadata only\n---\n\n'
            '```graph\nnode: before\nkind: assertion\n```\n\n'
            'Real section\n------------\n\n'
            '```graph\nnode: after\nkind: term\n```\n')
    path.write_text(text)
    nodes = read_documents(tmp_path).nodes
    assert [(node.ident, node.section) for node in nodes] == [
        ('before', None), ('after', 'Real section')]
    for node in nodes:
        assert node.line == text.splitlines().index(f'node: {node.ident}') + 1


@pytest.mark.parametrize(('opening', 'closer'), [('--- ', '--- '), ('\ufeff---', '---'), ('---', '... ')])
def test_front_matter_bom_space_and_yaml_closer(tmp_path, opening, closer):
    path = tmp_path / 'docs' / 'fixture.md'
    path.parent.mkdir()
    text = (f'{opening}\ntitle: Metadata\n{closer}\n'
            '```graph\nnode: before\nkind: term\n```\n'
            'Heading\n---\n```graph\nnode: after\nkind: term\n```\n')
    path.write_text(text, encoding='utf-8')
    nodes = read_documents(tmp_path).nodes
    assert [(n.ident, n.section) for n in nodes] == [('before', None), ('after', 'Heading')]
    for node in nodes:
        assert node.line == text.splitlines().index(f'node: {node.ident}') + 1


def test_leading_thematic_break_preserves_setext_heading(tmp_path):
    path = tmp_path / 'docs' / 'fixture.md'
    path.parent.mkdir()
    path.write_text('---\n\nReal heading\n---\n\n```graph\nnode: here\nkind: term\n```\n')
    node, = read_documents(tmp_path).nodes
    assert node.section == 'Real heading'
