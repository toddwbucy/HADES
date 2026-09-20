#!/usr/bin/env python3
"""Synthetic restore audit probe: private servers only, never a live endpoint.

Run with existing matching ArangoDB binaries; this script installs nothing.
Artifacts remain under a new /tmp/hades-restore-* directory. This is a small
logical-restore rehearsal, not a production backup command or RTO benchmark.
"""
import argparse
import hashlib,http.client,importlib.util,json,os,signal,socket,subprocess,tempfile,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('isolation',str(Path(__file__).resolve().parents[3] / 'scripts/test_isolated_database.py')); isolation=importlib.util.module_from_spec(spec); spec.loader.exec_module(isolation)
root=Path(tempfile.mkdtemp(prefix='hades-restore-',dir='/tmp')); os.chmod(root,0o700)
print(root,flush=True)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bin-dir',type=Path,required=True)
args=parser.parse_args()
binroot=args.bin_dir.resolve()
for name in ('arangod','arangodump','arangorestore'):
    if not (binroot/name).is_file():parser.error(f'missing existing binary: {name}')
isolation.lower_priority()
env={'PATH':os.environ['PATH'],'HOME':str(root),'LANG':'C','OMP_NUM_THREADS':'1'}
report={'scope':'Synthetic private single-server logical dump/restore; not production recovery certification','cases':[]}
child=None

def interrupted(signum, frame):
    raise RuntimeError(f'private restore probe interrupted: {signum}')

signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)

def api(method,path,body=None,expected=200):
    connection=http.client.HTTPConnection('localhost',timeout=10)
    connection.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);connection.sock.settimeout(10);connection.sock.connect(str(endpoint))
    try:
        connection.request(method,path,json.dumps(body) if body is not None else None,{'Content-Type':'application/json'})
        response=connection.getresponse(); raw=response.read(); assert response.status==expected,(response.status,raw[:1000]); return json.loads(raw) if raw else None
    finally:connection.close()

def db(method,path,body=None,expected=200):return api(method,'/_db/restore_fixture'+path,body,expected)

def inspect():
    result={}
    for col in ('nodes','links'):
        result[col]=db('POST','/_api/cursor',{'query':f'FOR d IN {col} SORT d._key RETURN UNSET(d,"_rev")'},201)['result']
    result['schema']=db('GET','/_api/collection/nodes/properties')['schema']
    result['indexes']=[{k:i[k] for k in ('type','fields','unique','sparse')} for i in db('GET','/_api/index?collection=nodes')['indexes']]
    result['graph']=db('GET','/_api/gharial/fixture_graph')['graph']['edgeDefinitions']
    result['traversal']=db('POST','/_api/cursor',{'query':'FOR v IN 1..2 OUTBOUND "nodes/a" GRAPH "fixture_graph" SORT v._key RETURN v._key'},201)['result']
    assert result['traversal']==['b','c'];return result

try:
    for phase in ('source','target'):
        work=root/phase;work.mkdir(); endpoint=work/'arango.sock'
        flags=['--configuration','none','--database.directory',str(work/'data'),'--server.endpoint','unix://'+str(endpoint),'--server.authentication','false','--javascript.enabled','false','--foxx.queues','false','--server.statistics','false','--server.minimal-threads','4','--server.maximal-threads','8','--server.io-threads','1','--rocksdb.block-cache-size','67108864','--rocksdb.total-write-buffer-size','67108864','--rocksdb.write-buffer-size','16777216','--rocksdb.max-background-jobs','2','--arangosearch.threads','1','--arangosearch.threads-limit','1','--log.output','-']
        log=(work/'server.log').open('w')
        child=subprocess.Popen([str(binroot/'arangod'),*flags],env=env,cwd=work,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=isolation.bounded_process)
        deadline=time.monotonic()+60
        while True:
            assert child.poll() is None,'private server exited'
            try:report['server_version']=api('GET','/_api/version')['version'];break
            except (OSError,http.client.HTTPException):
                if time.monotonic()>deadline:raise RuntimeError('startup deadline')
                time.sleep(.1)
        api('POST','/_api/database',{'name':'restore_fixture'},201)
        common=['--configuration','none','--server.endpoint','unix://'+str(endpoint),'--server.authentication','false','--server.database','restore_fixture','--threads','1','--include-system-collections','true']
        started=time.monotonic()
        if phase=='source':
            db('POST','/_api/gharial',{'name':'fixture_graph','edgeDefinitions':[{'collection':'links','from':['nodes'],'to':['nodes']}]},202)
            db('PUT','/_api/collection/nodes/properties',{'schema':{'rule':{'type':'object','properties':{'name':{'type':'string'}},'required':['name']},'level':'strict','message':'name required'}})
            db('POST','/_api/index?collection=nodes',{'type':'persistent','fields':['name'],'unique':True},201)
            db('POST','/_api/document/nodes',[{'_key':k,'name':k,'vector':[float(i),1.0]} for i,k in enumerate('abc')],202)
            db('POST','/_api/document/links',[{'_key':'ab','_from':'nodes/a','_to':'nodes/b'},{'_key':'bc','_from':'nodes/b','_to':'nodes/c'}],202)
            before=inspect()
            command=[str(binroot/'arangodump'),*common,'--output-directory',str(root/'dump'),'--collection','nodes','--collection','links','--collection','_graphs']
        else:
            command=[str(binroot/'arangorestore'),*common,'--input-directory',str(root/'dump')]
        with (work/'client.log').open('w') as output:
            subprocess.run(command,env=env,cwd=work,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=60,start_new_session=True,preexec_fn=isolation.bounded_process)
        report[phase+'_seconds']=time.monotonic()-started
        if phase=='target':
            after=inspect(); assert before==after,(before,after)
            db('POST','/_api/document/nodes',{'_key':'bad'},400)
            db('POST','/_api/document/nodes',{'_key':'duplicate','name':'a'},409)
            report['cases']=['documents and vectors identical','edge endpoints identical','graph traversal identical','schema identical and enforced','unique persistent index identical and enforced']
            report['restored']=after
        isolation.stop_group(child);child=None;log.close()
    report['dump_files']={str(p.relative_to(root/'dump')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'dump').rglob('*') if p.is_file()}
    report['binary_sha256']={}
    for name in ('arangod','arangodump','arangorestore'):
        with (binroot/name).open('rb') as stream:report['binary_sha256'][name]=hashlib.file_digest(stream,'sha256').hexdigest()
    report['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['isolation_helper_sha256']=hashlib.sha256(Path(isolation.__file__).read_bytes()).hexdigest()
    report['status']='passed';(root/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
finally:
    if child is not None:isolation.stop_group(child)
