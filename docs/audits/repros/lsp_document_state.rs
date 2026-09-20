//! Private session/peer fixture; no real analyzer or database.
use hades_core::code::lsp::{
    LspError,
    session::{LanguageServer, LspSession},
};
use std::path::{Path, PathBuf};
use std::time::Duration;

const PEER: &str = r#"
import sys,json
opened={}
while True:
 headers={}
 while True:
  line=sys.stdin.buffer.readline()
  if not line: sys.exit(0)
  if line==b'\r\n': break
  k,v=line.decode().split(':',1);headers[k]=v.strip()
 message=json.loads(sys.stdin.buffer.read(int(headers['Content-Length'])))
 method=message.get('method');params=message.get('params') or {}
 if method=='exit': sys.exit(0)
 if method=='textDocument/didOpen':
  doc=params['textDocument'];opened[doc['uri']]=doc['text'];continue
 if method=='textDocument/didClose':
  opened.pop(params['textDocument']['uri'],None);continue
 if 'id' not in message: continue
 result={ 'capabilities': {} } if method=='initialize' else []
 if method=='textDocument/documentSymbol':
  uri=params['textDocument']['uri']
  if uri in opened: result=[{'name':'fixture','text':opened[uri]}]
 body=json.dumps({'jsonrpc':'2.0','id':message['id'],'result':result}).encode()
 sys.stdout.buffer.write(('Content-Length: %d\r\n\r\n'%len(body)).encode()+body);sys.stdout.buffer.flush()
"#;
struct Fixture;
impl LanguageServer for Fixture {
    const NAME: &'static str = "private-fixture";
    const LANGUAGE_ID: &'static str = "fixture";
    fn find_binary() -> Result<String, LspError> {
        Ok("/usr/bin/python3".into())
    }
    fn validate_root(root: &Path) -> Result<PathBuf, LspError> {
        Ok(root.canonicalize()?)
    }
    fn args() -> &'static [&'static str] {
        &["-c", PEER]
    }
}

#[tokio::test]
async fn failed_read_marks_document_open_without_notifying_server() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fixture.txt");
    std::fs::write(&source, [0xff]).unwrap();
    let session = LspSession::<Fixture>::start_with_options(directory.path(), None, 1)
        .await
        .unwrap();
    assert!(session.is_ready());
    assert!(session.open_file(&source).await.is_err());
    std::fs::write(&source, "now valid").unwrap();
    let retried = session.document_symbols(&source).await.unwrap();
    // Historical defect: retry returns success without sending didOpen.
    assert!(retried.is_empty());
    tokio::time::timeout(Duration::from_secs(2), session.shutdown())
        .await
        .unwrap()
        .unwrap();
    println!("AUDIT: read error left URI marked open; valid retry never sent didOpen");
}
