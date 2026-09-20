#!/usr/bin/env python3
"""Verify adapter identity preservation on a newly created disposable ArangoDB server.

Uses synthetic data only; accepts a server binary, never an endpoint. Requires
permission to create private Unix and loopback sockets. Installs nothing.
"""
import argparse
import signal
import http.client,http.server,importlib.util,json,os,socket,subprocess,tempfile,threading,time,sys,types,contextlib,io
from pathlib import Path
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'services/adapters'))
from weavertools import write_graph as writer
from weavertools.records import Node
spec=importlib.util.spec_from_file_location('iso',repo/'scripts/test_isolated_database.py');iso=importlib.util.module_from_spec(spec);spec.loader.exec_module(iso)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arangod',type=Path,required=True)
args=parser.parse_args()
if not args.arangod.is_file():parser.error('arangod must name an existing binary')
iso.lower_priority()
root=Path(tempfile.mkdtemp(prefix='hades-adapter-identity-db-'));print(root,flush=True);endpoint=root/'arango.sock'
env={'PATH':os.environ['PATH'],'HOME':str(root),'LANG':'C','OMP_NUM_THREADS':'1'}
flags=['--configuration','none','--database.directory',str(root/'data'),'--server.endpoint','unix://'+str(endpoint),'--server.authentication','false','--javascript.enabled','false','--foxx.queues','false','--server.statistics','false','--server.minimal-threads','4','--server.maximal-threads','8','--server.io-threads','1','--rocksdb.block-cache-size','67108864','--rocksdb.total-write-buffer-size','67108864','--rocksdb.write-buffer-size','16777216','--rocksdb.max-background-jobs','2','--arangosearch.threads','1','--arangosearch.threads-limit','1','--log.output','-']
def unix(method,path,body=None):
    connection=http.client.HTTPConnection('localhost',timeout=5);connection.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);connection.sock.settimeout(5);connection.sock.connect(str(endpoint))
    try:
        connection.request(method,path,body,{'Content-Type':'application/json'});r=connection.getresponse();return r.status,r.read()
    finally:connection.close()
class Bridge(http.server.BaseHTTPRequestHandler):
    def run_request(self):
        size=int(self.headers.get('Content-Length',0));assert size<=1024*1024
        status,body=unix(self.command,self.path,self.rfile.read(size) if size else None)
        self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    do_GET=do_POST=do_PUT=do_DELETE=run_request
    def log_message(self,*args):pass
def interrupted(signum, frame):
    raise RuntimeError(f'private adapter probe interrupted: {signum}')
signal.signal(signal.SIGTERM,interrupted)
signal.signal(signal.SIGINT,interrupted)
log=(root/'server.log').open('w');child=None;bridge=None;thread=None
try:
    child=subprocess.Popen([str(args.arangod.resolve()),*flags],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=iso.bounded_process)
    deadline=time.monotonic()+60
    while True:
        assert child.poll() is None
        try:
            status,body=unix('GET','/_api/version');assert status==200;version=json.loads(body)['version'];break
        except OSError:
            if time.monotonic()>deadline:raise RuntimeError('startup deadline')
            time.sleep(.1)
    bridge=http.server.HTTPServer(('127.0.0.1',0),Bridge);thread=threading.Thread(target=lambda:bridge.serve_forever(poll_interval=.01));thread.start()
    os.environ.update(ARANGO_HOST='127.0.0.1',ARANGO_PORT=str(bridge.server_port),NO_PROXY='127.0.0.1,localhost');os.environ.pop('ARANGO_PASSWORD',None)
    long='a'*250
    cases=[
        ('truncation',[long,'b','c'],[(long,'b',None,None),(long,'c',None,None)]),
        ('delimiter',['a--seam','a','b','seam--b'],[('a--seam','b',None,None),('a','seam--b',None,None)]),
        ('tag',['a','b'],[('a','b','same','first'),('a','b','same','second')]),
    ]
    results=[]
    for index,(label,nodes,edges) in enumerate(cases):
        db='identity_'+str(index)
        status,body=unix('POST','/_api/database',json.dumps({'name':db}));assert status==201
        corpus=root/label
        (corpus/'docs').mkdir(parents=True)
        text='\n'.join('node: '+n+'\nkind: assertion' for n in nodes)
        text+='\n'+'\n'.join('edge: seam\nfrom: '+src+'\nto: '+dst+('\nvia: '+via if via else '')+('\ntag: '+tag if tag else '') for src,dst,via,tag in edges)
        (corpus/'docs/fixture.md').write_text('```graph\n'+text+'\n```\n')
        for name in ('documents','codebase_files'):writer.ensure_collection(db,name,2)
        writer.import_rows(db,'documents',[{'_key':'paper','source_rel':'docs/fixture.md'}])
        docs,code=writer.ingest(corpus,{'docs/fixture.md'},set())
        declared=[e for e in docs.edges if e.relation=='seam'];assert len(declared)==2
        sys.argv=['writer','--db',db,'--repo',str(corpus)]
        row_counts=[]
        for attempt in ('first','repeat'):
            stdout=io.StringIO();stderr=io.StringIO()
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):code=writer.main()
            assert code==0,(label,attempt,stdout.getvalue(),stderr.getvalue())
            response=writer.checked(writer.arango(db,'cursor',{'query':'FOR e IN wt_seam_edges RETURN KEEP(e, "_key", "_from", "_to", "tag", "via")'}),'verify')
            rows=response['result'];assert not response.get('hasMore')
            assert len(rows)==2,(label,attempt,rows)
            assert len({r['_key'] for r in rows})==2 and all(len(r['_key'].encode())<=254 for r in rows)
            logical=sorted(json.dumps(r,sort_keys=True) for r in rows)
            if attempt=='first':first_rows=logical
            else:assert logical==first_rows
            dangling=writer.checked(writer.arango(db,'cursor',{'query':'FOR e IN wt_seam_edges FILTER DOCUMENT(e._from) == null OR DOCUMENT(e._to) == null RETURN e._key'}),'endpoint verify')
            assert dangling['result']==[] and not dangling.get('hasMore')
            row_counts.append(len(rows))
        results.append({'case':label,'declared_edges':len(declared),'first_and_repeat_exit_codes':[0,0],'first_and_repeat_persisted_edge_counts':row_counts})
    # Exact document/revision snapshots prove refusal does not mutate stored rows.
    owned=list(writer.NODE_COLLECTIONS.values())+[writer.REPORT,'wt_seam_edges','wt_declared_in_edges']
    def snapshot():
        result={}
        for collection in owned:
            response=writer.checked(writer.arango(db,'cursor',{'query':'FOR d IN @@c RETURN d','bindVars':{'@c':collection}}),'snapshot')
            assert not response.get('hasMore')
            result[collection]=sorted(response['result'],key=lambda row:row['_key'])
        return result
    source=corpus/'docs/fixture.md'
    original=source.read_text()
    source.write_text(original.replace('node: a\nkind: assertion','node: a\nkind: term'))
    before=snapshot()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):kind_code=writer.main()
    assert kind_code==1 and snapshot()==before
    results.append({'case':'stored_kind_change_refused','exit_code':kind_code,'exact_rows_and_revisions_preserved':True})
    source.write_text(original)
    writer.import_rows(db,'wt_assertions',[{'_key':'legacy','ident':'legacy','kind':'assertion'}])
    before=snapshot()
    stdout=io.StringIO();stderr=io.StringIO()
    with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):legacy_code=writer.main()
    assert legacy_code==1 and 'identity preflight refused' in stderr.getvalue()
    assert snapshot()==before
    results.append({'case':'legacy_refused','exit_code':legacy_code,'exact_rows_and_revisions_preserved':True})
    # Conflict must also fail without touching a graph (including a legacy graph).
    with (corpus/'docs/fixture.md').open('a') as source:
        source.write('\n```graph\nnode: a\nkind: term\n```\n')
    before=snapshot()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):conflict_code=writer.main()
    assert conflict_code==1 and snapshot()==before
    results.append({'case':'conflicting_declaration_refused','exit_code':conflict_code,'exact_rows_and_revisions_preserved':True})
    import hashlib
    paths=[repo/'services/adapters/weavertools'/p for p in ['write_graph.py','extractor.py','records.py']]
    result={'server_version':version,'scope':'Actual extractor/writer and fresh disposable ArangoDB through an owned loopback bridge. Synthetic corpus only; no production endpoint.','source_sha256':{str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':results}

finally:
    if bridge:bridge.shutdown();bridge.server_close()
    if thread:thread.join(2)
    if child:iso.stop_group(child)
    log.close()

result['private_server_reaped']=child.poll() is not None
result['private_bridge_stopped']=not thread.is_alive()
assert result['private_server_reaped'] and result['private_bridge_stopped']
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
