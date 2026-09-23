//! Private session/peer fixture; no real analyzer or database.
use hades_core::code::lsp::{
    LspError,
    session::{LanguageServer, LspSession},
};
use std::path::{Path, PathBuf};
use std::time::Duration;

const PEER: &str = r#"
import sys,json,os,time
opened={};counts={}
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
  doc=params['textDocument'];opened[doc['uri']]=doc['text'];counts[doc['uri']]=counts.get(doc['uri'],0)+1;continue
 if method=='textDocument/didClose':
  opened.pop(params['textDocument']['uri'],None);continue
 if 'id' not in message: continue
 result={ 'capabilities': {} } if method=='initialize' else []
 if method=='textDocument/documentSymbol':
  uri=params['textDocument']['uri']
  if uri in opened: result=[{'name':'fixture','text':opened[uri],'opens':counts[uri]}]
 body=json.dumps({'jsonrpc':'2.0','id':message['id'],'result':result}).encode()
 sys.stdout.buffer.write(('Content-Length: %d\r\n\r\n'%len(body)).encode()+body);sys.stdout.buffer.flush()
 if method=='workspace/symbol' and os.path.exists('block-after-ready'): time.sleep(5)
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
async fn failed_read_remains_retryable_after_content_is_fixed() {
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
    assert_eq!(retried.len(), 1);
    assert_eq!(retried[0]["text"], "now valid");
    tokio::time::timeout(Duration::from_secs(2), session.shutdown())
        .await
        .unwrap()
        .unwrap();
    println!("AUDIT: read failure leaves retry able to deliver didOpen");
}

struct ReadGate {
    path: PathBuf,
    writer: Option<std::fs::File>,
}
impl ReadGate {
    fn new(root: &Path) -> Self {
        use std::os::unix::fs::OpenOptionsExt;
        let path = root.join("gated.fifo");
        let encoded = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        // SAFETY: private temporary path, valid C string, owner-only FIFO mode.
        assert_eq!(unsafe { libc::mkfifo(encoded.as_ptr(), 0o600) }, 0);
        let writer = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&path)
            .unwrap();
        Self {
            path,
            writer: Some(writer),
        }
    }
    fn descriptors(&self) -> usize {
        std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| std::fs::read_link(entry.path()).is_ok_and(|path| path == self.path))
            .count()
    }
    async fn reader_started(&self) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while self.descriptors() < 2 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("owned FIFO reader did not open");
    }
    fn release(&mut self, text: &str) {
        use std::io::Write;
        let mut writer = self.writer.take().unwrap();
        writer.write_all(text.as_bytes()).unwrap();
        drop(writer);
    }
    async fn drained(&self) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while self.descriptors() != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("owned blocking file reader did not drain");
    }
    fn replace(&self, text: &str) {
        std::fs::remove_file(&self.path).unwrap();
        std::fs::write(&self.path, text).unwrap();
    }
}

#[tokio::test]
async fn cancelled_read_releases_uri_and_can_retry() {
    let directory = tempfile::tempdir().unwrap();
    let mut gate = ReadGate::new(directory.path());
    let session = std::sync::Arc::new(
        LspSession::<Fixture>::start_with_options(directory.path(), None, 1)
            .await
            .unwrap(),
    );
    let task_session = session.clone();
    let path = gate.path.clone();
    let task = tokio::spawn(async move { task_session.open_file(&path).await });
    gate.reader_started().await;
    task.abort();
    assert!(matches!(task.await, Err(error) if error.is_cancelled()));
    // Tokio's blocking file read still owns its fd after caller cancellation.
    gate.release("abandoned");
    gate.drained().await;
    gate.replace("retry after cancellation");
    let result = session.document_symbols(&gate.path).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0]["text"], "retry after cancellation");
    let session = std::sync::Arc::try_unwrap(session).ok().unwrap();
    tokio::time::timeout(Duration::from_secs(2), session.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn competing_opens_wait_and_other_documents_progress() {
    let directory = tempfile::tempdir().unwrap();
    let mut gate = ReadGate::new(directory.path());
    let other = directory.path().join("other.txt");
    std::fs::write(&other, "independent").unwrap();
    let session = std::sync::Arc::new(
        LspSession::<Fixture>::start_with_options(directory.path(), None, 1)
            .await
            .unwrap(),
    );
    let first_session = session.clone();
    let path = gate.path.clone();
    let first = tokio::spawn(async move { first_session.open_file(&path).await });
    gate.reader_started().await;
    let second_session = session.clone();
    let path = gate.path.clone();
    let mut second = tokio::spawn(async move { second_session.open_file(&path).await });
    let premature = tokio::time::timeout(Duration::from_millis(30), &mut second).await;
    assert!(
        premature.is_err(),
        "second open completed before didOpen could be sent"
    );
    let independent =
        tokio::time::timeout(Duration::from_secs(2), session.document_symbols(&other))
            .await
            .unwrap()
            .unwrap();
    assert_eq!(independent[0]["text"], "independent");
    gate.release("controlled");
    let first_uri = first.await.unwrap().unwrap();
    assert_eq!(second.await.unwrap().unwrap(), first_uri);
    gate.drained().await;
    assert_eq!(
        session.document_symbols(&gate.path).await.unwrap()[0]["text"],
        "controlled"
    );
    assert_eq!(
        session.document_symbols(&gate.path).await.unwrap()[0]["opens"],
        1
    );
    session.close_file(&first_uri).await.unwrap();
    session.close_file(&first_uri).await.unwrap();
    gate.replace("reopened");
    assert_eq!(
        session.document_symbols(&gate.path).await.unwrap()[0]["text"],
        "reopened"
    );
    let session = std::sync::Arc::try_unwrap(session).ok().unwrap();
    tokio::time::timeout(Duration::from_secs(2), session.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn close_waits_for_admitted_open_before_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let mut gate = ReadGate::new(directory.path());
    let session = std::sync::Arc::new(
        LspSession::<Fixture>::start_with_options(directory.path(), None, 1)
            .await
            .unwrap(),
    );
    let first_session = session.clone();
    let path = gate.path.clone();
    let first = tokio::spawn(async move { first_session.open_file(&path).await });
    gate.reader_started().await;
    let uri = url::Url::from_file_path(gate.path.canonicalize().unwrap())
        .unwrap()
        .to_string();
    let closing_session = session.clone();
    let mut closing = tokio::spawn(async move { closing_session.close_file(&uri).await });
    assert!(
        tokio::time::timeout(Duration::from_millis(30), &mut closing)
            .await
            .is_err()
    );
    gate.release("first open");
    first.await.unwrap().unwrap();
    closing.await.unwrap().unwrap();
    gate.drained().await;
    gate.replace("after queued close");
    let result = session.document_symbols(&gate.path).await.unwrap();
    assert_eq!(result[0]["text"], "after queued close");
    assert_eq!(result[0]["opens"], 2);
    let session = std::sync::Arc::try_unwrap(session).ok().unwrap();
    tokio::time::timeout(Duration::from_secs(2), session.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn cancelled_partial_did_open_does_not_publish_open_state() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("block-after-ready"), "").unwrap();
    let source = directory.path().join("large.txt");
    std::fs::write(&source, "x".repeat(1024 * 1024)).unwrap();
    let session = LspSession::<Fixture>::start_with_options(directory.path(), None, 1)
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(30), session.open_file(&source))
            .await
            .is_err()
    );
    // The partial frame invalidates transport; a stale open flag must not report success.
    assert!(session.open_file(&source).await.is_err());
    tokio::time::timeout(Duration::from_secs(2), session.shutdown())
        .await
        .unwrap()
        .unwrap();
}

mod request_outcomes {
    include!("fixtures/lsp_request_outcomes.rs");
}
