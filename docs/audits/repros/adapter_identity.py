"""Private real extractor/writer identity probe; all database calls stubbed."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path('services/adapters').resolve()))
from weavertools import write_graph as w

def run_case(name, nodes, edges):
    with tempfile.TemporaryDirectory(prefix='hades-adapter-identity-') as directory:
        root = Path(directory)
        (root/'docs').mkdir()
        text = '\n'.join('node: '+n+'\nkind: assertion' for n in nodes)
        text += '\n'+'\n'.join('edge: seam\nfrom: '+src+'\nto: '+dst+
                                ('\nvia: '+via if via else '')+
                                ('\ntag: '+tag if tag else '')
                                for src,dst,via,tag in edges)
        (root/'docs/fixture.md').write_text('```graph\n'+text+'\n```\n')
        emitted = []
        stored = {}
        def backend(db, path, body=None, method='POST'):
            assert db == 'fixture'
            if path == 'cursor':
                return {'result':[['docs/fixture.md','paper']] if 'FOR d IN documents' in body['query'] else [],'hasMore':False}
            if path == 'collection':
                return {'name':body['name'],'type':body['type']}
            if path.startswith('import?'):
                query=parse_qs(urlsplit(path).query)
                assert query['onDuplicate']==['replace']
                collection=query['collection'][0]
                table=stored.setdefault(collection,{})
                created=updated=0
                for row in body:
                    if row['_key'] in table: updated+=1
                    else: created+=1
                    table[row['_key']]=dict(row)
                    if collection=='wt_seam_edges':emitted.append(dict(row))
                return {'created':created,'updated':updated,'errors':0,'ignored':0,'empty':0}
            if path.startswith('document/'):
                return {'_key':'latest','_id':w.REPORT+'/latest'}
            raise AssertionError(path)
        with patch.object(w,'arango',backend),patch.object(sys,'argv',['writer','--db','fixture','--repo',directory]),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            code=w.main()
        assert code==0,(name,code)
        assert len(emitted)==2 and len({e['_key'] for e in emitted})==1,name
        assert emitted[0]!=emitted[1],name
        return {'case':name,'exit_code':code,'distinct_edges_emitted':len(emitted),'unique_edge_keys':len({e['_key'] for e in emitted}),'replace_simulation_retained':len(stored['wt_seam_edges'])}

a='a'*250
cases=[
    run_case('truncated_long_identity',[a,'b','c'],[(a,'b',None,None),(a,'c',None,None)]),
    run_case('unframed_delimiter_identity',['a--seam','a','b','seam--b'],[('a--seam','b',None,None),('a','seam--b',None,None)]),
    run_case('tag_not_in_identity',['a','b'],[('a','b','same','first'),('a','b','same','second')]),
]
paths=[Path('services/adapters/weavertools')/p for p in ['write_graph.py','extractor.py','records.py']]
print(json.dumps({'scope':'Real extractor and writer over private synthetic Markdown; database API fully stubbed. Replacement persistence is simulated, not an executed database assertion.','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':cases},indent=2))
