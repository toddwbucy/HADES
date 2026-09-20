//! Private synthetic analyzer probes; no installed analyzer or production data.
use hades_core::code::lsp::{preflight_binary, resolve_and_probe_async};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::time::Duration;

fn fixture(directory: &Path, script: &str) -> String {
    let path = directory.join("probe");
    std::fs::write(&path, format!("#!/usr/bin/python3\n{script}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path.to_str().unwrap().to_owned()
}

async fn marker(path: &Path) -> u32 {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(contents) = std::fs::read_to_string(path)
                && let Ok(pid) = contents.parse()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("fixture did not start")
}

async fn stopped(pid: u32, reaped: bool) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
                Ok(stat)
                    if !reaped && stat.split(") ").nth(1).is_some_and(|s| s.starts_with('Z')) =>
                {
                    break;
                }
                _ => tokio::time::sleep(Duration::from_millis(5)).await,
            }
        }
    })
    .await
    .expect("fixture process remained running or direct child was not reaped");
}

#[test]
fn normal_workspace_and_error_diagnostics_are_preserved() {
    let directory = tempfile::tempdir().unwrap();
    let result = preflight_binary(
        "/usr/bin/python3",
        &["-c", "import os; print(os.getcwd())"],
        directory.path(),
    )
    .unwrap();
    assert_eq!(result, directory.path().to_str().unwrap());
    let error = preflight_binary(
        "/usr/bin/python3",
        &[
            "-c",
            "import sys; sys.stderr.write('fixture failure'); sys.exit(7)",
        ],
        directory.path(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("fixture failure"));
    assert!(error.contains(directory.path().to_str().unwrap()));
}

#[test]
fn both_output_streams_have_explicit_caps() {
    let directory = tempfile::tempdir().unwrap();
    for stream in ["stdout", "stderr"] {
        let exact = format!("import sys; sys.{stream}.write('x'*65536)");
        let result =
            preflight_binary("/usr/bin/python3", &["-c", &exact], directory.path()).unwrap();
        assert_eq!(result.len(), if stream == "stdout" { 65536 } else { 0 });
        let script = format!("import sys; sys.{stream}.write('x'*2097152)");
        let error = preflight_binary("/usr/bin/python3", &["-c", &script], directory.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains(stream), "{error}");
        assert!(error.contains("65536 bytes"), "{error}");
        assert!(error.len() < 1024);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn asynchronous_probe_keeps_executor_responsive_and_preserves_resolution() {
    let directory = tempfile::tempdir().unwrap();
    let command = fixture(
        directory.path(),
        "import os,sys,time\nopen('pid','w').write(str(os.getpid()))\nfor _ in range(500):\n if os.path.exists('release'): break\n time.sleep(.01)\nelse: sys.exit(9)\nprint(sys.argv[1])",
    );
    let root = directory.path().to_owned();
    let task =
        tokio::spawn(async move { resolve_and_probe_async("gopls", Some(&command), &root).await });
    let pid = marker(&directory.path().join("pid")).await;
    assert!(!task.is_finished());
    std::fs::write(directory.path().join("release"), "").unwrap();
    let result = task.await.unwrap();
    assert_eq!(result.source, "config/env");
    assert!(result.configured);
    assert_eq!(result.outcome.as_deref(), Ok("version"));
    stopped(pid, true).await;
}

#[tokio::test]
async fn cancellation_kills_owned_group_and_reaps_direct_child() {
    let directory = tempfile::tempdir().unwrap();
    let command = fixture(
        directory.path(),
        "import os,time\nopen('parent','w').write(str(os.getpid()))\nif os.fork()==0:\n open('descendant','w').write(str(os.getpid()))\ntime.sleep(30)",
    );
    let root = directory.path().to_owned();
    let task =
        tokio::spawn(
            async move { resolve_and_probe_async("fixture", Some(&command), &root).await },
        );
    let parent = marker(&directory.path().join("parent")).await;
    let descendant = marker(&directory.path().join("descendant")).await;
    task.abort();
    assert!(matches!(task.await, Err(error) if error.is_cancelled()));
    stopped(parent, true).await;
    stopped(descendant, false).await;
}

#[tokio::test]
async fn exited_leader_does_not_wait_for_descendant_pipe_eof() {
    let directory = tempfile::tempdir().unwrap();
    let command = fixture(
        directory.path(),
        "import os,time\nopen('parent','w').write(str(os.getpid()))\nif os.fork()==0:\n open('descendant','w').write(str(os.getpid()))\n time.sleep(30)\nelse:\n while not os.path.exists('descendant'): time.sleep(.005)\n print('fixture v1',flush=True)\n os._exit(0)",
    );
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        resolve_and_probe_async("fixture", Some(&command), directory.path()),
    )
    .await
    .unwrap();
    assert_eq!(result.outcome.as_deref(), Ok("fixture v1"));
    let parent = marker(&directory.path().join("parent")).await;
    let descendant = marker(&directory.path().join("descendant")).await;
    stopped(parent, true).await;
    stopped(descendant, false).await;
}

#[tokio::test]
async fn sleeping_probe_expires_and_is_reaped() {
    let directory = tempfile::tempdir().unwrap();
    let command = fixture(
        directory.path(),
        "import os,time\nopen('pid','w').write(str(os.getpid()))\ntime.sleep(30)",
    );
    let result = tokio::time::timeout(
        Duration::from_secs(13),
        resolve_and_probe_async("fixture", Some(&command), directory.path()),
    )
    .await
    .unwrap();
    assert!(result.outcome.unwrap_err().contains("exceeded 10s"));
    let pid = marker(&directory.path().join("pid")).await;
    stopped(pid, true).await;
}
