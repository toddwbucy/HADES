//! Private process groups only; no real analyzer, workspace scan or database.
use hades_core::code::lsp::{LspError, client::LspClient};
use serde_json::json;
use std::path::Path;
use std::time::Duration;

async fn pid(path: &Path) -> u32 {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(text) = std::fs::read_to_string(path)
                && let Ok(pid) = text.parse()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
fn running(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .is_ok_and(|stat| stat.split(") ").nth(1).is_some_and(|s| !s.starts_with('Z')))
}
async fn ended(pid: u32, direct: bool) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while if direct {
            Path::new(&format!("/proc/{pid}")).exists()
        } else {
            running(pid)
        } {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("owned process remained running or direct child was not reaped");
}

const PEER: &str = r#"
import os,sys,json,time
open('parent','w').write(str(os.getpid()))
if os.fork()==0:
 open('descendant','w').write(str(os.getpid()))
 time.sleep(30)
 os._exit(0)
while not os.path.exists('descendant'): time.sleep(.005)
if sys.argv[1]=='hang': time.sleep(30)
while True:
 headers={}
 while True:
  line=sys.stdin.buffer.readline()
  if not line: sys.exit(0)
  if line==b'\r\n': break
  k,v=line.decode().split(':',1);headers[k]=v.strip()
 message=json.loads(sys.stdin.buffer.read(int(headers['Content-Length'])))
 method=message.get('method')
 if method=='exit': os._exit(0)
 if method=='terminate': os._exit(0)
 if method=='malformed':
  sys.stdout.buffer.write(b'Content-Length: 1\r\n\r\n!');sys.stdout.buffer.flush();continue
 if 'id' not in message: continue
 body=json.dumps({'jsonrpc':'2.0','id':message['id'],'result':message.get('params')}).encode()
 sys.stdout.buffer.write(('Content-Length: %d\r\n\r\n'%len(body)).encode()+body)
 sys.stdout.buffer.flush()
 if method=='final': os._exit(0)
"#;

async fn fixture(mode: &str) -> (tempfile::TempDir, LspClient, u32, u32) {
    let directory = tempfile::tempdir().unwrap();
    let client = LspClient::start("/usr/bin/python3", &["-c", PEER, mode], directory.path())
        .await
        .unwrap();
    let parent = pid(&directory.path().join("parent")).await;
    let descendant = pid(&directory.path().join("descendant")).await;
    (directory, client, parent, descendant)
}
async fn cleaned(parent: u32, descendant: u32) {
    ended(parent, true).await;
    ended(descendant, false).await;
}

#[tokio::test]
async fn dropping_client_stops_descendants_and_reaps_leader() {
    let (_directory, client, parent, descendant) = fixture("hang").await;
    drop(client);
    cleaned(parent, descendant).await;
}

#[tokio::test]
async fn exited_leader_with_inherited_pipe_fails_pending_promptly() {
    let (_directory, client, parent, descendant) = fixture("normal").await;
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        client.request("terminate", json!({}), Duration::from_secs(30)),
    )
    .await
    .unwrap();
    assert!(matches!(result, Err(LspError::Process(_))), "{result:?}");
    cleaned(parent, descendant).await;
}

#[tokio::test]
async fn final_response_drains_while_owned_descendants_stop() {
    let (_directory, client, parent, descendant) = fixture("normal").await;
    assert_eq!(
        client
            .request("final", json!(42), Duration::from_secs(2))
            .await
            .unwrap(),
        json!(42)
    );
    cleaned(parent, descendant).await;
}

#[tokio::test]
async fn malformed_transport_stops_the_process_group() {
    let (_directory, client, parent, descendant) = fixture("normal").await;
    assert!(matches!(
        client
            .request("malformed", json!({}), Duration::from_secs(2))
            .await,
        Err(LspError::Process(_))
    ));
    cleaned(parent, descendant).await;
}

#[tokio::test]
async fn graceful_shutdown_stops_owned_descendants() {
    let (_directory, client, parent, descendant) = fixture("normal").await;
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
    cleaned(parent, descendant).await;
}

#[tokio::test]
async fn cancelled_shutdown_keeps_group_cleanup_owned() {
    let (_directory, client, parent, descendant) = fixture("hang").await;
    assert!(
        tokio::time::timeout(Duration::from_millis(20), client.shutdown())
            .await
            .is_err()
    );
    cleaned(parent, descendant).await;
}

#[tokio::test]
async fn unresponsive_shutdown_stops_the_entire_group() {
    let (_directory, client, parent, descendant) = fixture("hang").await;
    tokio::time::timeout(Duration::from_secs(13), client.shutdown())
        .await
        .unwrap()
        .unwrap();
    cleaned(parent, descendant).await;
}
