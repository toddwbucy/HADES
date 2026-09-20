//! Actual private viewer process/socket and synthetic child; no HADES/DB service.
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Viewer(Child);
impl Drop for Viewer {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct FixtureCleanup {
    root: PathBuf,
}
impl Drop for FixtureCleanup {
    fn drop(&mut self) {
        // Failure-only cleanup: verify the unique fixture script still owns the
        // recorded leader before signalling its private process group.
        if let Ok(bytes) = std::fs::read(self.root.join("pids"))
            && let Ok(pids) = serde_json::from_slice::<Vec<u32>>(&bytes)
            && let Some(pid) = pids.first()
            && let Ok(command) = std::fs::read(format!("/proc/{pid}/cmdline"))
            && command
                .windows(self.root.as_os_str().as_encoded_bytes().len())
                .any(|part| part == self.root.as_os_str().as_encoded_bytes())
        {
            // SAFETY: the verified synthetic backend owns this private group.
            unsafe {
                libc::kill(-(*pid as i32), libc::SIGKILL);
            }
        }
    }
}
fn running(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| s.rsplit_once(") ").map(|(_, rest)| !rest.starts_with('Z')))
        .unwrap_or(false)
}
fn pids(root: &Path) -> Vec<u32> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(bytes) = std::fs::read(root.join("pids"))
            && let Ok(pids) = serde_json::from_slice(&bytes)
        {
            return pids;
        }
        assert!(Instant::now() < deadline, "synthetic backend did not start");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn termination_reaps_discovery_dump_and_http_children() {
    for mode in ["serve", "dump", "http", "disconnect"] {
        for signal in [libc::SIGTERM, libc::SIGINT] {
            let root = tempfile::tempdir().unwrap();
            let _cleanup = FixtureCleanup {
                root: root.path().into(),
            };
            let script = root.path().join("synthetic-backend");
            std::fs::write(
                &script,
                r#"#!/usr/bin/python3
import json, os, pathlib, subprocess, time
child = subprocess.Popen(['/usr/bin/python3', '-c', 'import time; time.sleep(10)'])
pathlib.Path(__file__).with_name('pids').write_text(json.dumps([os.getpid(), child.pid]))
time.sleep(10)
"#,
            )
            .unwrap();
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
            let mut command = Command::new(env!("CARGO_BIN_EXE_hades-viewer"));
            command
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("TOKIO_WORKER_THREADS", "2")
                .arg(if matches!(mode, "http" | "disconnect") {
                    "serve"
                } else {
                    mode
                })
                .arg("--hades-bin")
                .arg(&script)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(std::fs::File::create(root.path().join("viewer.log")).unwrap());
            if mode == "dump" {
                command.args(["--db", "fixture", "--graph", "fixture"]);
            }
            // Reserve an ephemeral loopback port for the private HTTP case.
            // Discovery and dump do not bind any socket.
            let address = if matches!(mode, "http" | "disconnect") {
                let reservation = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                let address = reservation.local_addr().unwrap();
                command.args(["--db", "fixture", "--bind", &address.to_string()]);
                drop(reservation);
                Some(address)
            } else {
                None
            };
            let mut viewer = Viewer(command.spawn().unwrap());
            let mut connection = address.map(|address| {
                let deadline = Instant::now() + Duration::from_secs(5);
                loop {
                    if let Ok(mut stream) = std::net::TcpStream::connect(address) {
                        stream.write_all(format!("GET /api/graphs HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
                        break stream;
                    }
                    assert!(Instant::now() < deadline, "private viewer did not bind");
                    std::thread::sleep(Duration::from_millis(10));
                }
            });
            let children = pids(root.path());
            assert!(children.iter().all(|pid| running(*pid)));
            if mode == "disconnect" {
                let stream = connection.take().unwrap();
                stream.shutdown(std::net::Shutdown::Both).unwrap();
                drop(stream);
                let deadline = Instant::now() + Duration::from_secs(5);
                while children.iter().any(|pid| running(*pid)) {
                    assert!(
                        Instant::now() < deadline,
                        "HTTP disconnect left a running fixture child"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
            // SAFETY: this is our unreaped direct child, not a discovered service.
            assert_eq!(unsafe { libc::kill(viewer.0.id() as i32, signal) }, 0);
            let deadline = Instant::now() + Duration::from_secs(5);
            while viewer.0.try_wait().unwrap().is_none() {
                assert!(
                    Instant::now() < deadline,
                    "viewer did not finish shutdown: {}",
                    std::fs::read_to_string(root.path().join("viewer.log")).unwrap()
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            while children.iter().any(|pid| running(*pid)) {
                assert!(
                    Instant::now() < deadline,
                    "shutdown left a running fixture descendant"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
