//! Private child reproduction; no analyzer, workspace scan or database.
use hades_core::code::lsp::client::LspClient;
use serde_json::json;
use std::time::Duration;

#[tokio::test]
async fn request_deadline_includes_a_blocked_stdin_write() {
    let directory = tempfile::tempdir().unwrap();
    let marker = directory.path().join("pid");
    // Publish readiness only after the PID is fully written and closed.
    // File creation alone raced the parent's read on loaded CI runners.
    let script = r#"
import os,sys,time
pending = sys.argv[1] + '.pending'
with open(pending, 'w') as marker:
    marker.write(str(os.getpid()))
os.replace(pending, sys.argv[1])
time.sleep(5)
"#;
    let client = LspClient::start(
        "/usr/bin/python3",
        &["-c", script, marker.to_str().unwrap()],
        directory.path(),
    )
    .await
    .unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while !marker.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let pid: u32 = std::fs::read_to_string(&marker).unwrap().parse().unwrap();
    let result = tokio::time::timeout(
        Duration::from_millis(250),
        client.request(
            "fixture",
            json!({"text":"x".repeat(1024*1024)}),
            Duration::from_millis(20),
        ),
    )
    .await;
    // The request must now return its own deadline error.
    let own_deadline = matches!(result, Ok(Err(hades_core::code::lsp::LspError::Timeout(_))));
    drop(client);
    tokio::time::timeout(Duration::from_secs(2), async {
        while std::path::Path::new(&format!("/proc/{pid}")).exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("owned fixture child was not reaped");
    assert!(
        own_deadline,
        "request deadline was not enforced: {result:?}"
    );
    println!("AUDIT: 20ms request deadline enforced during writing; owned child reaped");
}

const PEER: &str = r#"
import sys,json
while True:
    headers={}
    while True:
        line=sys.stdin.buffer.readline()
        if not line: sys.exit(0)
        if line==b'\r\n': break
        k,v=line.decode().split(':',1); headers[k]=v.strip()
    message=json.loads(sys.stdin.buffer.read(int(headers['Content-Length'])))
    method=message.get('method')
    if method=='exit': sys.exit(0)
    if method=='ignore': continue
    if method=='malformed':
        sys.stdout.buffer.write(b'Content-Length: 1\r\n\r\n!');sys.stdout.buffer.flush();continue
    if 'id' not in message: continue
    body=json.dumps({'jsonrpc':'2.0','id':message['id'],'result':message.get('params')}).encode()
    sys.stdout.buffer.write(('Content-Length: %d\r\n\r\n'%len(body)).encode()+body)
    sys.stdout.buffer.flush()
    if method=='final': sys.exit(0)
"#;

#[tokio::test]
async fn response_timeout_and_cancel_preserve_complete_frame_alignment() {
    let directory = tempfile::tempdir().unwrap();
    let mut client = LspClient::start("/usr/bin/python3", &["-c", PEER], directory.path())
        .await
        .unwrap();
    assert_eq!(
        client
            .request("echo", json!({"value":7}), Duration::from_secs(1))
            .await
            .unwrap(),
        json!({"value":7})
    );
    assert!(matches!(
        client
            .request("ignore", json!({}), Duration::from_millis(20))
            .await,
        Err(hades_core::code::lsp::LspError::Timeout(_))
    ));
    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            client.request("ignore", json!({}), Duration::from_secs(1))
        )
        .await
        .is_err()
    );
    assert!(client.is_alive());
    assert_eq!(
        client
            .request("echo", json!(9), Duration::from_secs(1))
            .await
            .unwrap(),
        json!(9)
    );
    client.notify("notice", json!({})).await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn malformed_peer_fails_pending_requests_and_closes_transport() {
    let directory = tempfile::tempdir().unwrap();
    let mut client = LspClient::start("/usr/bin/python3", &["-c", PEER], directory.path())
        .await
        .unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        client.request("malformed", json!({}), Duration::from_secs(5)),
    )
    .await
    .unwrap();
    assert!(matches!(
        result,
        Err(hades_core::code::lsp::LspError::Process(_))
    ));
    assert!(!client.is_alive());
    assert!(
        client
            .request("echo", json!({}), Duration::from_secs(1))
            .await
            .is_err()
    );
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn final_response_is_drained_before_peer_exit() {
    let directory = tempfile::tempdir().unwrap();
    let client = LspClient::start("/usr/bin/python3", &["-c", PEER], directory.path())
        .await
        .unwrap();
    assert_eq!(
        client
            .request("final", json!(42), Duration::from_secs(1))
            .await
            .unwrap(),
        json!(42)
    );
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn notification_has_its_own_write_deadline() {
    let directory = tempfile::tempdir().unwrap();
    let mut client = LspClient::start(
        "/usr/bin/python3",
        &["-c", "import time; time.sleep(30)"],
        directory.path(),
    )
    .await
    .unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(7),
        client.notify("blocked", json!({"text": "x".repeat(1024 * 1024)})),
    )
    .await
    .unwrap();
    assert!(matches!(
        result,
        Err(hades_core::code::lsp::LspError::Timeout(_))
    ));
    assert!(!client.is_alive());
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn unresponsive_peer_cannot_hold_shutdown_indefinitely() {
    let directory = tempfile::tempdir().unwrap();
    let client = LspClient::start(
        "/usr/bin/python3",
        &["-c", "import time; time.sleep(30)"],
        directory.path(),
    )
    .await
    .unwrap();
    // Five seconds for the unanswered shutdown request and five for exit grace.
    tokio::time::timeout(Duration::from_secs(13), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn peer_exit_promptly_fails_a_pending_request() {
    let directory = tempfile::tempdir().unwrap();
    let client = LspClient::start(
        "/usr/bin/python3",
        &["-c", "import sys; sys.stdin.buffer.readline()"],
        directory.path(),
    )
    .await
    .unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        client.request("exit immediately", json!({}), Duration::from_secs(30)),
    )
    .await
    .unwrap();
    assert!(result.is_err());
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn blocked_server_request_reply_fails_other_pending_requests() {
    let directory = tempfile::tempdir().unwrap();
    let script = r#"
import sys,json,time
body=json.dumps({'jsonrpc':'2.0','id':'x'*1048576,'method':'fixture'}).encode()
sys.stdout.buffer.write(('Content-Length: %d\r\n\r\n'%len(body)).encode()+body)
sys.stdout.buffer.flush()
time.sleep(30)
"#;
    let mut client = LspClient::start("/usr/bin/python3", &["-c", script], directory.path())
        .await
        .unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(7),
        client.request("waiting", json!({}), Duration::from_secs(30)),
    )
    .await
    .unwrap();
    assert!(matches!(
        result,
        Err(hades_core::code::lsp::LspError::Process(_))
    ));
    assert!(!client.is_alive());
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}
