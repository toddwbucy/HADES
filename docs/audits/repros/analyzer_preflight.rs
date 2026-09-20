//! Bounded historical audit fixture. Synthetic Python only; no real analyzer.
use hades_core::code::lsp::preflight_binary;
use std::time::{Duration, Instant};

#[tokio::test(flavor = "current_thread")]
async fn preflight_blocks_runtime_and_captures_large_version_output() {
    let directory = tempfile::tempdir().unwrap();
    let started = Instant::now();
    let timer = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        started.elapsed()
    });
    tokio::task::yield_now().await;
    let result = preflight_binary(
        "/usr/bin/python3",
        &["-c", "import sys,time; time.sleep(0.3); sys.stdout.write('x'*2097152)"],
        directory.path(),
    ).unwrap();
    let timer_elapsed = timer.await.unwrap();
    assert_eq!(result.len(), 2 * 1024 * 1024);
    assert!(timer_elapsed >= Duration::from_millis(250));
    println!("AUDIT: captured_bytes={}, twenty_ms_timer_elapsed_ms={}", result.len(), timer_elapsed.as_millis());
}
