"""Failures must not turn incomplete adapter writes into successful CLI runs."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from weavertools import write_graph as w
from weavertools.records import Node


@pytest.mark.parametrize('response', [None, {}, {'error':True}, {'result':[], 'hasMore':'false'}, {'result':[['a']], 'hasMore':False}, {'result':[['a','1'],['a','2']], 'hasMore':False}])
def test_invalid_scope_rejected(monkeypatch, response):
    monkeypatch.setattr(w, 'arango', lambda *a, **k:response)
    with pytest.raises(w.AdapterError):w.code_keys_by_path('fixture')


def test_failed_continuation_does_not_return_partial_scope(monkeypatch):
    calls = []
    def backend(db, path, body=None, method='POST'):
        calls.append((path, method))
        if path == 'cursor':return {'result':[['a','1']], 'hasMore':True, 'id':'123'}
        return {'error':True}
    monkeypatch.setattr(w, 'arango', backend)
    with pytest.raises(w.AdapterError):w.document_keys_by_rel('fixture')
    assert calls == [('cursor','POST'),('cursor/123','PUT'),('cursor/123','DELETE')]


@pytest.mark.parametrize('kind', [2, 3])
def test_existing_collection_type_verified(monkeypatch, kind):
    replies = iter([{'error':True,'code':409,'errorNum':1207}, {'name':'fixture','type':kind}])
    monkeypatch.setattr(w,'arango',lambda *a,**k:next(replies))
    if kind == 2:w.ensure_collection('db','fixture',2)
    else:
        with pytest.raises(w.AdapterError):w.ensure_collection('db','fixture',2)


@pytest.mark.parametrize('failure', [None, 'scope', 'collection', 'import-api', 'import-partial', 'import-malformed', 'report'])
def test_run_failure_exit_and_truthful_count(monkeypatch, capsys, failure):
    calls=[]
    monkeypatch.setattr(sys,'argv',['writer','--db','fixture','--repo','/unused'])
    monkeypatch.setattr(w,'ingest',lambda *args:(SimpleNamespace(nodes=[Node('a','assertion')],edges=[],notes=[],dangling=[]),SimpleNamespace(edges=[],notes=[],dangling=[])))
    def backend(db, path, body=None, method='POST'):
        calls.append(path)
        if path == 'cursor':
            if failure == 'scope':return {'error':True}
            return {'result':[['fixture.md','fixture']], 'hasMore':False}
        if path == 'collection':
            if failure == 'collection':return {'error':True,'code':403}
            return {'name':body['name'],'type':body['type']}
        if path.startswith('import?'):
            if failure == 'import-api':return {'error':True, 'code':500}
            if failure == 'import-malformed':return {'error':False}
            return {'error':False,'created':0 if failure=='import-partial' else 1,'updated':0,'ignored':0,'empty':0,'errors':1 if failure=='import-partial' else 0}
        if path.startswith('document/'):
            if failure == 'report':return {'error':True}
            return {'_key':'latest','_id':w.REPORT+'/latest'}
        raise AssertionError(path)
    monkeypatch.setattr(w,'arango',backend)
    assert w.main() == (1 if failure else 0)
    output=capsys.readouterr()
    if failure:
        assert 'server acknowledged' not in output.out
        assert 'earlier writes may persist' in output.err
    else:
        assert 'server acknowledged 1 imported rows plus one report' in output.out
    if failure == 'scope':assert calls == ['cursor']
    if failure == 'collection':assert not any(p.startswith('import?') for p in calls)


@pytest.mark.parametrize('change', [{'errors':1}, {'created':0}, {'updated':True}, {'ignored':1}, {'empty':1}, {'created':-1}, {'error':True}])
def test_import_requires_complete_counters(monkeypatch, change):
    response = dict(error=False, errors=0, created=1, updated=0, ignored=0, empty=0)
    response.update(change)
    monkeypatch.setattr(w,'arango',lambda *a,**k:response)
    with pytest.raises(w.AdapterError):w.import_rows('fixture','nodes',[{'_key':'a'}])


def test_invalid_cursor_identifier_is_not_used_for_cleanup(monkeypatch):
    calls=[]
    def backend(db,path,body=None,method='POST'):
        calls.append(path)
        return {'result':[], 'hasMore':True, 'id':'../collection'}
    monkeypatch.setattr(w,'arango',backend)
    with pytest.raises(w.AdapterError):w.code_keys_by_path('fixture')
    assert calls == ['cursor']


def test_http_failure_cannot_claim_success(monkeypatch):
    import io
    import urllib.error
    response = io.BytesIO(b'{"error":false,"created":1}')
    class Opener:
        def open(self,*args,**kwargs):
            raise urllib.error.HTTPError('http://fixture.invalid',503,'failure',{},response)
    monkeypatch.setattr(w.urllib.request,'build_opener',lambda *a:Opener())
    monkeypatch.setenv('ARANGO_HOST','127.0.0.1')
    monkeypatch.delenv('ARANGO_PASSWORD',raising=False)
    result=w.arango('fixture','import')
    assert result['error'] is True and result['code']==503
    assert response.closed


@pytest.mark.parametrize('failure, acknowledged', [
    ('partial', 2), ('report', 3), ('missing-source', 4), ('missing-document', 4),
])
def test_failed_run_preserves_acknowledged_rows(monkeypatch, capsys, failure, acknowledged):
    from weavertools.records import Edge
    monkeypatch.setattr(sys, 'argv', ['writer', '--db', 'fixture', '--repo', '/unused'])
    docs = SimpleNamespace(nodes=[Node('a', 'assertion'), Node('b', 'term'), Node('c', 'term')],
                           edges=[], notes=[], dangling=[])
    code = SimpleNamespace(edges=[], notes=[], dangling=[])
    if failure == 'missing-source':
        code.edges.append(Edge('missing.rs', 'a', 'cites', 'declared'))
    if failure == 'missing-document':
        docs.edges.append(Edge('a', 'missing.md', 'declared-in', 'declared'))
    monkeypatch.setattr(w, 'ingest', lambda *args: (docs, code))
    calls = []
    def backend(db, path, body=None, method='POST'):
        calls.append(path)
        if path == 'cursor':
            return {'result': [['fixture.md', 'fixture']], 'hasMore': False}
        if path == 'collection':
            return {'name': body['name'], 'type': body['type']}
        if path.startswith('import?'):
            partial = failure == 'partial' and 'collection=wt_terms&' in path
            return dict(error=False, created=1 if partial else len(body), updated=0,
                        errors=1 if partial else 0, ignored=0, empty=0)
        if path.startswith('document/'):
            if failure == 'report':
                return {'error': True}
            return {'_key': 'latest', '_id': w.REPORT + '/latest'}
        raise AssertionError(path)
    monkeypatch.setattr(w, 'arango', backend)
    assert w.main() == 1
    output = capsys.readouterr()
    assert f'server acknowledged at least {acknowledged} imported rows' in output.err
    assert 'earlier writes may persist; no successful run is certified' in output.err
    assert 'plus one report' not in output.out
    if failure == 'partial':
        assert not any(path.startswith('document/') for path in calls)
