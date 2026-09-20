//! Actual daemon signals with only private Unix sockets; no ingestion child or DB.
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

struct Daemon(Child);

#[tokio::test]
async fn sealed_configuration_bypasses_ambient_loading_only_for_ingestion() {
    let root = tempfile::tempdir().unwrap();
    let mut config = hades_core::config::HadesConfig::with_database("private_snapshot");
    config.database.password = Some("synthetic-snapshot-secret".into());
    for (subcommand, expected) in [
        ("ingest", "no inputs provided"),
        (
            "status",
            "resolved configuration is only accepted for ingestion",
        ),
    ] {
        let snapshot = hades_core::config::snapshot::Snapshot::new(&config).unwrap();
        let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_hades"));
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", root.path())
            .env("HADES_CONFIG", root.path().join("does-not-exist"))
            .env("HADES_DATABASE", "wrong")
            .env("ARANGO_PASSWORD", "wrong")
            .env("TOKIO_WORKER_THREADS", "2");
        let fd = snapshot.inherit(&mut command);
        command
            .args(["--resolved-config-fd", &fd.to_string(), subcommand])
            .kill_on_drop(true);
        assert!(!format!("{command:?}").contains("synthetic-snapshot-secret"));
        let output = tokio::time::timeout(Duration::from_secs(5), command.output())
            .await
            .unwrap()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(expected), "{error}");
        assert!(!error.contains("synthetic-snapshot-secret"));
        assert!(!String::from_utf8_lossy(&output.stdout).contains("synthetic-snapshot-secret"));
    }
}
impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn request(stream: &mut UnixStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut bytes = Vec::new();
    loop {
        let mut buf = [0; 4096];
        let n = stream.read(&mut buf).unwrap();
        assert_ne!(n, 0);
        bytes.extend_from_slice(&buf[..n]);
        assert!(bytes.len() <= 64 * 1024);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            let head = std::str::from_utf8(&bytes[..end]).unwrap();
            let length: usize = head
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|n| n.trim().parse().unwrap())
                })
                .unwrap_or(0);
            if bytes.len() >= end + 4 + length {
                return head.lines().next().unwrap().to_owned();
            }
        }
    }
}

fn accept(listener: &UnixListener) -> UnixStream {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match listener.accept() {
            Ok((stream, _)) => return stream,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("private accept: {error}"),
        }
        assert!(Instant::now() < deadline, "private request did not arrive");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn signals_drain_pending_ingest_reservations_before_daemon_exit() {
    for signal in [libc::SIGINT, libc::SIGTERM] {
        for pending_ingest in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let database_socket = root.path().join("database.sock");
            let database = UnixListener::bind(&database_socket).unwrap();
            database.set_nonblocking(true).unwrap();
            let daemon_socket = root.path().join("daemon.sock");
            let config = root.path().join("config.yaml");
            std::fs::write(&config, format!(
                "database:\n  name: fixture\n  username: fixture\n  sockets:\n    readonly: {}\n    readwrite: {}\n",
                database_socket.display(), database_socket.display()
            )).unwrap();
            let mut daemon = Daemon(
                Command::new(env!("CARGO_BIN_EXE_hades"))
                    .env_clear()
                    .env("PATH", "/usr/bin:/bin")
                    .env("HOME", root.path())
                    .env("HADES_CONFIG", &config)
                    .env("ARANGO_PASSWORD", "synthetic-fixture")
                    .env("TOKIO_WORKER_THREADS", "2")
                    .args(["--db", "fixture", "daemon", "--socket"])
                    .arg(&daemon_socket)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(std::fs::File::create(root.path().join("daemon.log")).unwrap())
                    .spawn()
                    .unwrap(),
            );
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut client = loop {
                if let Ok(stream) = UnixStream::connect(&daemon_socket) {
                    break stream;
                }
                assert!(
                    daemon.0.try_wait().unwrap().is_none(),
                    "daemon exited: {}",
                    std::fs::read_to_string(root.path().join("daemon.log")).unwrap()
                );
                assert!(Instant::now() < deadline, "private daemon did not bind");
                std::thread::sleep(Duration::from_millis(10));
            };
            let (entered, observed) = mpsc::channel();
            let (release, continue_read) = mpsc::channel();
            let fixture = if pending_ingest {
                let database = database.try_clone().unwrap();
                Some(std::thread::spawn(move || {
                    let mut connection = accept(&database);
                    assert_eq!(
                        request(&mut connection),
                        "POST /_db/fixture/_api/collection HTTP/1.1"
                    );
                    entered.send(()).unwrap();
                    continue_read.recv_timeout(Duration::from_secs(5)).unwrap();
                    connection
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
                        )
                        .unwrap();
                    drop(connection);
                    let mut connection = accept(&database);
                    assert_eq!(
                        request(&mut connection),
                        "POST /_db/fixture/_api/cursor HTTP/1.1"
                    );
                    // Fail closed before any job insertion or child spawn.
                    let body = br#"{"error":true,"errorMessage":"private fixture unavailable"}"#;
                    write!(connection, "HTTP/1.1 503 Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
                    connection.write_all(body).unwrap();
                }))
            } else {
                None
            };
            if pending_ingest {
                let body = serde_json::to_vec(
                    &serde_json::json!({"command":"ingest.start","params":{"path":root.path()}}),
                )
                .unwrap();
                client
                    .write_all(&(body.len() as u32).to_be_bytes())
                    .unwrap();
                client.write_all(&body).unwrap();
                observed.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            // SAFETY: signal only our unreaped direct child, never a discovered PID.
            assert_eq!(unsafe { libc::kill(daemon.0.id() as i32, signal) }, 0);
            if pending_ingest {
                std::thread::sleep(Duration::from_millis(100));
                assert!(
                    daemon.0.try_wait().unwrap().is_none(),
                    "daemon exited before its reservation drained"
                );
                release.send(()).unwrap();
            }
            let deadline = Instant::now() + Duration::from_secs(5);
            let status = loop {
                if let Some(status) = daemon.0.try_wait().unwrap() {
                    break status;
                }
                assert!(Instant::now() < deadline, "daemon did not finish shutdown");
                std::thread::sleep(Duration::from_millis(10));
            };
            assert!(status.success());
            if let Some(fixture) = fixture {
                fixture.join().unwrap();
            }
            assert!(!daemon_socket.exists());
            assert!(
                matches!(database.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
            );
        }
    }
}
