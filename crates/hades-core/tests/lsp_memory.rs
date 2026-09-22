//! Bounded private child tests; no analyzer, workspace scan or database.
use hades_core::code::lsp::client::LspClient;
use serde_json::json;
use std::time::Duration;

async fn rejects_peer(script: &str, expected: &str) {
    let directory = tempfile::tempdir().unwrap();
    let mut client = LspClient::start("/usr/bin/python3", &["-c", script], directory.path())
        .await
        .unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        client.request("fixture", json!({}), Duration::from_secs(30)),
    )
    .await
    .unwrap();
    let error = result.unwrap_err().to_string();
    assert!(error.contains(expected), "{error}");
    assert!(!client.is_alive());
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn excessive_headers_are_rejected() {
    rejects_peer("import sys,time; sys.stdout.buffer.write(b'X: '+b'x'*65536); sys.stdout.buffer.flush(); time.sleep(5)", "header").await;
}

#[tokio::test]
async fn oversized_frame_declaration_is_rejected_without_waiting_for_body() {
    rejects_peer("import sys,time; sys.stdout.buffer.write(b'Content-Length: 16777217\\r\\n\\r\\n'); sys.stdout.buffer.flush(); time.sleep(5)", "frame exceeds").await;
}

const FLOOD: &str = r#"
import sys,json
size=int(sys.argv[1]);count=int(sys.argv[2])
body=json.dumps({'jsonrpc':'2.0','method':'window/logMessage','params':{'message':'x'*size}}).encode()
frame=('Content-Length: %d\r\n\r\n'%len(body)).encode()+body
for _ in range(count): sys.stdout.buffer.write(frame)
sys.stdout.buffer.flush()
"#;

async fn survives_flood(size: usize, count: usize, retained_count: usize) {
    let directory = tempfile::tempdir().unwrap();
    let script = format!("import sys; sys.argv=['fixture','{size}','{count}']\n{FLOOD}\n{ECHO}");
    let mut client = LspClient::start("/usr/bin/python3", &["-c", &script], directory.path())
        .await
        .unwrap();
    assert_eq!(
        client
            .request("fixture", json!(42), Duration::from_secs(3))
            .await
            .unwrap(),
        json!(42)
    );
    assert!(client.is_alive());
    let retained = client.drain_notifications(None).await;
    assert_eq!(retained.len(), retained_count);
    assert!(retained.len() <= 1024);
    assert!(
        retained
            .iter()
            .map(|v| serde_json::to_vec(v).unwrap().len())
            .sum::<usize>()
            <= 8 * 1024 * 1024
    );
    // Draining the bounded queue must leave the connection usable.
    assert_eq!(
        client
            .request("echo", json!(7), Duration::from_secs(3))
            .await
            .unwrap(),
        json!(7)
    );
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn notification_count_overflow_preserves_pending_request() {
    survives_flood(256, 2048, 1024).await;
}

#[tokio::test]
async fn notification_wire_bytes_overflow_preserves_pending_request() {
    // Each framed payload is slightly larger than 1 MiB, so only seven fit.
    survives_flood(1048576, 9, 7).await;
}

const ECHO: &str = r#"
import sys,json
while True:
 headers={}
 while True:
  line=sys.stdin.buffer.readline()
  if not line: sys.exit(0)
  if line==b'\r\n': break
  k,v=line.decode().split(':',1);headers[k]=v.strip()
 message=json.loads(sys.stdin.buffer.read(int(headers['Content-Length'])))
 if message.get('method')=='exit': sys.exit(0)
 if 'id' not in message: continue
 body=json.dumps({'jsonrpc':'2.0','id':message['id'],'result':message.get('params')}).encode()
 sys.stdout.buffer.write(('Content-Length: %d\r\n\r\n'%len(body)).encode()+body)
 sys.stdout.buffer.flush()
"#;

#[tokio::test]
async fn outbound_overflow_rejects_before_writing_and_preserves_stream() {
    let directory = tempfile::tempdir().unwrap();
    let mut client = LspClient::start("/usr/bin/python3", &["-c", ECHO], directory.path())
        .await
        .unwrap();
    let error = client
        .request(
            "oversized",
            json!("x".repeat(16 * 1024 * 1024)),
            Duration::from_secs(5),
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("outgoing LSP frame exceeds"), "{error}");
    assert!(client.is_alive());
    assert_eq!(
        client
            .request("echo", json!(42), Duration::from_secs(2))
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
async fn normal_notifications_survive_filtered_drains() {
    let directory = tempfile::tempdir().unwrap();
    let prefix = r#"
import sys,json
for method in ['$/progress','window/logMessage','$/progress']:
 body=json.dumps({'jsonrpc':'2.0','method':method,'params':{'value':'fixture'}}).encode()
 sys.stdout.buffer.write(('Content-Length: %d\r\n\r\n'%len(body)).encode()+body)
sys.stdout.buffer.flush()
"#;
    let script = format!("{prefix}\n{ECHO}");
    let client = LspClient::start("/usr/bin/python3", &["-c", &script], directory.path())
        .await
        .unwrap();
    assert_eq!(
        client
            .request("echo", json!(7), Duration::from_secs(2))
            .await
            .unwrap(),
        json!(7)
    );
    assert_eq!(
        client.drain_notifications(Some("$/progress")).await.len(),
        2
    );
    let retained = client.drain_notifications(None).await;
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0]["method"], "window/logMessage");
    assert!(client.drain_notifications(None).await.is_empty());
    tokio::time::timeout(Duration::from_secs(2), client.shutdown())
        .await
        .unwrap()
        .unwrap();
}
