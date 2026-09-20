//! Bounded baseline: oversized headers and retained notifications, no OOM probe.
use hades_core::code::lsp::client::LspClient;
use serde_json::json;
use std::path::Path;
use std::time::Duration;

#[tokio::test]
async fn oversized_header_and_unconsumed_notifications_are_accepted() {
    let directory = tempfile::tempdir().unwrap();
    let script = r#"
import sys,json,os
open("pid","w").write(str(os.getpid()))
headers={}
while True:
 line=sys.stdin.buffer.readline()
 if line==b'\r\n': break
 k,v=line.decode().split(':',1);headers[k]=v.strip()
request=json.loads(sys.stdin.buffer.read(int(headers['Content-Length'])))
def send(message,padding=False):
 body=json.dumps(message).encode()
 header=('Content-Length: %d\r\n'%len(body)).encode()
 if padding: header+=b'X-Fixture: '+b'x'*65536+b'\r\n'
 sys.stdout.buffer.write(header+b'\r\n'+body)
for i in range(2048): send({'jsonrpc':'2.0','method':'window/logMessage','params':{'message':'x'*256}})
send({'jsonrpc':'2.0','id':request['id'],'result':True},True)
sys.stdout.buffer.flush()
# Remain alive until the client closes its owned stdin.
sys.stdin.buffer.read()
"#;
    let client = LspClient::start("/usr/bin/python3", &["-c", script], directory.path())
        .await
        .unwrap();
    let result = client
        .request("fixture", json!({}), Duration::from_secs(3))
        .await
        .unwrap();
    assert_eq!(result, json!(true));
    assert!(
        client
            .drain_notifications(Some("$/progress"))
            .await
            .is_empty()
    );
    let retained = client.drain_notifications(None).await;
    assert_eq!(retained.len(), 2048);
    println!(
        "AUDIT: accepted header padding=65536 bytes, retained notifications={}",
        retained.len()
    );
    let pid: u32 = std::fs::read_to_string(directory.path().join("pid"))
        .unwrap()
        .parse()
        .unwrap();
    drop(client);
    tokio::time::timeout(Duration::from_secs(2), async {
        while Path::new(&format!("/proc/{pid}")).exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("private child was not reaped");
}
