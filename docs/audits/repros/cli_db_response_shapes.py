import argparse, hashlib, http.server, json, socketserver, subprocess, tempfile, threading
from pathlib import Path
parser=argparse.ArgumentParser(description='Read-only CLI probes using a private Unix peer and network namespace')
parser.add_argument('--binary',type=Path,required=True)
args=parser.parse_args()
root=Path(tempfile.mkdtemp(prefix='hades-db-shapes-'))
sock=root/'db.sock'
requests=[]
reply={}
class Server(socketserver.UnixStreamServer): pass
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self): self.respond()
 def do_POST(self): self.respond()
 def do_DELETE(self): self.respond()
 def respond(self):
  self.rfile.read(int(self.headers.get('Content-Length','0')))
  requests.append([self.command,self.path])
  data=json.dumps(reply).encode()
  self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
server=Server(str(sock),Handler)
thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
config=root/'config.json'
config.write_text(json.dumps({'database':{'name':'audit_shapes','username':'fixture','sockets':{'readonly':str(sock),'readwrite':str(sock)}}}))
env={'PATH':'/usr/bin:/bin','HOME':str(root),'HADES_CONFIG':str(config),'ARANGO_SOCKET':str(sock),'ARANGO_RO_SOCKET':str(sock),'ARANGO_RW_SOCKET':str(sock),'ARANGO_PASSWORD':'fixture-only','CUDA_VISIBLE_DEVICES':'','RUST_LOG':'error'}
binary=args.binary.resolve()
results=[]
try:
 for command,payload in [(['databases'],{}),(['databases'],{'result':[]}),(['export','fixture'],{}),(['export','fixture'],{'result':[],'hasMore':False}),(['export','fixture'],{'result':[{'_key':'a'}],'hasMore':'true','id':'fixture-cursor'})]:
  reply=payload;requests.clear()
  completed=subprocess.run(['bwrap','--ro-bind','/','/','--dev','/dev','--unshare-net','--',str(binary),'--db','audit_shapes','db',*command],env=env,cwd=root,capture_output=True,text=True,timeout=10)
  results.append({'command':command,'response':payload,'exit':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,'requests':list(requests)})
finally:
 server.shutdown();server.server_close();thread.join()
result={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'scope':'private synthetic Unix peer, unshared network, no live DB','cases':results}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(root)
for r in results:print(json.dumps(r))
