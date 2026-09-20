#!/usr/bin/env python3
"""Verify adapter response handling on a newly created disposable ArangoDB server.

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
root=Path(tempfile.mkdtemp(prefix='hades-adapter-db-'));print(root,flush=True);endpoint=root/'arango.sock'
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
    status,body=unix('POST','/_api/database',json.dumps({'name':'adapter_fixture'}));assert status==201
    bridge=http.server.HTTPServer(('127.0.0.1',0),Bridge);thread=threading.Thread(target=lambda:bridge.serve_forever(poll_interval=.01));thread.start()
    os.environ.update(ARANGO_HOST='127.0.0.1',ARANGO_PORT=str(bridge.server_port),NO_PROXY='127.0.0.1,localhost');os.environ.pop('ARANGO_PASSWORD',None)
    for name in ('documents','codebase_files'):writer.ensure_collection('adapter_fixture',name,2)
    writer.import_rows('adapter_fixture','documents',[{'_key':'d','source_rel':'fixture.md'}])
    writer.import_rows('adapter_fixture','codebase_files',[{'_key':'c','path':'fixture.rs'}])
    writer.ingest=lambda *a:(types.SimpleNamespace(nodes=[Node('a','assertion')],edges=[],notes=[],dangling=[]),types.SimpleNamespace(edges=[],notes=[],dangling=[]))
    sys.argv=['writer','--db','adapter_fixture','--repo',str(root)]
    results=[]
    for label in ('first','repeat','schema-rejected'):
        if label=='schema-rejected':
            r=writer.arango('adapter_fixture','collection/wt_assertions/properties',{'schema':{'rule':{'type':'object','required':['never_supplied']},'level':'strict','message':'fixture rejection'}},method='PUT');assert not r.get('error')
        stdout=io.StringIO();stderr=io.StringIO()
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):code=writer.main()
        assert code==(1 if label=='schema-rejected' else 0),(label,stdout.getvalue(),stderr.getvalue())
        results.append({'case':label,'exit_code':code})
    try:writer.ensure_collection('adapter_fixture','documents',3)
    except writer.AdapterError:results.append({'case':'wrong-collection-type','rejected':True})
    else:raise AssertionError('wrong collection type accepted')
    result={'server_version':version,'cases':results};(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:
    if bridge:bridge.shutdown();bridge.server_close()
    if thread:thread.join(2)
    if child:iso.stop_group(child)
    log.close()
