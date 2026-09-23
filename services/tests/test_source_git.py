"""Adapter report provenance uses the same Git observations as Rust (#171)."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from weavertools import write_graph as w
from weavertools.records import Node


def test_report_records_non_git_clean_and_dirty_tree(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args],
                              capture_output=True, check=True).stdout.decode().strip()
    monkeypatch.setattr(sys, 'argv', ['writer', '--db', 'fixture', '--repo', str(tmp_path)])
    monkeypatch.setattr(w, 'ingest', lambda *args: (
        SimpleNamespace(nodes=[Node('a', 'assertion')], edges=[], notes=[], dangling=[]),
        SimpleNamespace(edges=[], notes=[], dangling=[])))
    reports = []
    def backend(db, path, body=None, method='POST'):
        if path == 'cursor':
            return {'result': [['fixture.md', 'fixture']], 'hasMore': False}
        if path == 'collection' and method == 'GET':
            return {'result': []}
        if path == 'collection':
            return {'name': body['name'], 'type': body['type']}
        if path.startswith('import?'):
            return dict(error=False, created=len(body), updated=0, errors=0, ignored=0, empty=0)
        if path.startswith('document/'):
            reports.append(body)
            return {'_key': 'latest', '_id': w.REPORT + '/latest'}
        raise AssertionError(path)
    monkeypatch.setattr(w, 'arango', backend)
    assert w.main() == 0
    assert 'source_git' in reports[-1] and reports[-1]['source_git'] is None
    git('init', '-q')
    (tmp_path / 'a.md').write_text('initial')
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
    commit = git('rev-parse', 'HEAD')
    assert w.main() == 0
    assert reports[-1]['source_git'] == dict(commit=commit, dirty=False)
    (tmp_path / 'untracked').write_text('dirty')
    assert w.main() == 0
    assert reports[-1]['source_git'] == dict(commit=commit, dirty=True)


def test_non_git_observation_forces_c_locale(tmp_path, monkeypatch):
    from source_git import resolve
    git = tmp_path / 'git'
    git.write_text("#!/bin/sh\nif [ \"$LC_ALL\" = C ]; then echo 'fatal: not a git repository' >&2; else echo 'fatal: kein Git-Repository' >&2; fi\nexit 128\n")
    git.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path))
    monkeypatch.setenv('LANG', 'de_DE.UTF-8')
    monkeypatch.setenv('LC_ALL', 'de_DE.UTF-8')
    assert resolve(tmp_path) is None
