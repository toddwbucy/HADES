//! Bounded historical baseline; synthetic descendants self-exit after two seconds.
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
    tokio::time::timeout(Duration::from_secs(3), async {
        while if direct {
            Path::new(&format!("/proc/{pid}")).exists()
        } else {
            running(pid)
        } {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("bounded fixture did not finish");
}

#[tokio::test]
async fn dropping_client_leaves_descendant_running_after_direct_child_reap() {
    let directory = tempfile::tempdir().unwrap();
    let script = "import os,time\nopen('parent','w').write(str(os.getpid()))\nif os.fork()==0:\n open('descendant','w').write(str(os.getpid()))\n time.sleep(2)\nelse: time.sleep(30)";
    let client = LspClient::start("/usr/bin/python3", &["-c", script], directory.path())
        .await
        .unwrap();
    let parent = pid(&directory.path().join("parent")).await;
    let child = pid(&directory.path().join("descendant")).await;
    drop(client);
    ended(parent, true).await;
    let survived = running(child);
    ended(child, false).await;
    assert!(survived, "historical descendant leak was not reproduced");
    println!(
        "AUDIT: descendant remained running after direct child reap; fixture then self-exited"
    );
}

#[tokio::test]
async fn exited_parent_with_inherited_pipe_delays_pending_failure() {
    let directory = tempfile::tempdir().unwrap();
    let script = "import os,sys,time\nopen('parent','w').write(str(os.getpid()))\nsys.stdin.buffer.readline()\nif os.fork()==0:\n open('descendant','w').write(str(os.getpid()))\n time.sleep(2)\nelse: os._exit(0)";
    let client = LspClient::start("/usr/bin/python3", &["-c", script], directory.path())
        .await
        .unwrap();
    let result = client
        .request("fixture", json!({}), Duration::from_millis(250))
        .await;
    let parent = pid(&directory.path().join("parent")).await;
    let child = pid(&directory.path().join("descendant")).await;
    ended(parent, true).await;
    let survived = running(child);
    drop(client);
    ended(child, false).await;
    assert!(matches!(result, Err(LspError::Timeout(_))), "{result:?}");
    assert!(survived);
    println!(
        "AUDIT: dead direct child did not fail pending request before its 250ms timeout while descendant held stdout"
    );
}
