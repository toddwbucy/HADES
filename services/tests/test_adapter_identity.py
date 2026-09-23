"""Identity collisions and legacy refusal without a live database."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from weavertools import write_graph as w
@pytest.fixture(autouse=True)
def mock_source_git(monkeypatch):
    monkeypatch.setattr(w, 'resolve_source_git', lambda root: None)


from weavertools.records import Node, Edge


@pytest.mark.parametrize('a,b', [
    ('a/b','a_b'), ('a b','a_b'), ('a'*300+'x','a'*300+'y'),
    ('λ','_'), ('é','e'),
])
def test_node_keys_preserve_distinct_full_identities(a,b):
    first, second = w.key_for(a), w.key_for(b)
    assert first != second and first == w.key_for(a)
    assert first.isascii() and len(first) <= 254


@pytest.mark.parametrize('bad', ['', None, 42])
def test_invalid_node_identity_is_rejected(bad):
    with pytest.raises(w.IdentityError):
        w.key_for(bad)


def test_duplicate_declarations_are_deduplicated_or_rejected():
    node = Node('a','assertion',body='same')
    assert w.validate_declarations([node,node],[]) == [node]
    for conflict in [Node('a','term',body='same'), Node('a','assertion',body='different')]:
        with pytest.raises(w.IdentityError):
            w.validate_declarations([node,conflict],[])
    with pytest.raises(w.IdentityError):
        w.validate_declarations([node],[Edge('a','b','bad/relation','declared')])


@pytest.mark.parametrize('rows,more,accepted', [
    ([],False,True), (['legacy'],False,False),
    ([],True,False), (None,False,False), (['a','b'],False,False),
])
def test_existing_layout_must_be_completely_versioned(monkeypatch,rows,more,accepted):
    calls=[]
    def backend(db,path,body=None,method='POST'):
        calls.append((path,method))
        if path=='collection':
            assert method=='GET'
            return {'result':[{'name':'wt_seam_edges'},{'name':'unrelated'}]}
        assert path=='cursor' and body['bindVars']['@collection']=='wt_seam_edges'
        assert body['bindVars']['version']==2
        return {'result':rows,'hasMore':more}
    monkeypatch.setattr(w,'arango',backend)
    if accepted:
        w.require_identity_layout('fixture')
    else:
        with pytest.raises(w.IdentityError):
            w.require_identity_layout('fixture')
    assert calls==[('collection','GET'),('cursor','POST')]


@pytest.mark.parametrize('conflict', [False,True])
def test_legacy_or_conflicting_input_refuses_before_any_write(monkeypatch,capsys,conflict):
    docs=SimpleNamespace(nodes=[Node('a','assertion')],edges=[],notes=[],dangling=[])
    if conflict:
        docs.nodes.append(Node('a','term'))
    code=SimpleNamespace(edges=[],notes=[],dangling=[])
    monkeypatch.setattr(w,'ingest',lambda *args:(docs,code))
    monkeypatch.setattr(sys,'argv',['writer','--db','fixture','--repo','/unused'])
    calls=[]
    def backend(db,path,body=None,method='POST'):
        calls.append((path,method))
        if path=='collection' and method=='GET':
            return {'result':[{'name':'wt_assertions'}]}
        if path=='cursor':
            if '@collection' in body.get('bindVars',{}):
                return {'result':['legacy'],'hasMore':False}
            return {'result':[['fixture.md','paper']],'hasMore':False}
        raise AssertionError('unexpected mutation')
    monkeypatch.setattr(w,'arango',backend)
    assert w.main()==1
    assert 'no write stage started' in capsys.readouterr().err
    assert all(path=='cursor' or (path,method)==('collection','GET') for path,method in calls)
