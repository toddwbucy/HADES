//! Private child reproduction; no analyzer, workspace scan or database.
use hades_core::code::lsp::client::LspClient;
use serde_json::json;
use std::time::Duration;

#[tokio::test]
async fn request_deadline_excludes_a_blocked_stdin_write() {
    let directory = tempfile::tempdir().unwrap();
    let marker = directory.path().join("pid");
    let script = "import os,sys,time; open(sys.argv[1],'w').write(str(os.getpid())); time.sleep(5)";
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
    // Historical baseline: the outer harness deadline fires, not request's own deadline.
    let ignored_deadline = result.is_err();
    drop(client);
    tokio::time::timeout(Duration::from_secs(2), async {
        while std::path::Path::new(&format!("/proc/{pid}")).exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("owned fixture child was not reaped");
    assert!(ignored_deadline, "baseline behavior changed: {result:?}");
    println!("AUDIT: 20ms request deadline exceeded 250ms while writing; owned child reaped");
}
