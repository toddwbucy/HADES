//! Real daemon/child handoff and profile drift, using only disposable fixture databases.
use hades_core::{
    config::{HadesConfig, snapshot::Snapshot},
    db::{ArangoPool, crud},
    embedding_profile::store::COLLECTION,
    ingest_jobs::Binding,
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
};

#[allow(dead_code)]
#[path = "../../hades-core/tests/common/profiled_extractor.rs"]
mod extraction;
#[allow(dead_code)]
#[path = "../../hades-core/tests/common/profiled_provider.rs"]
mod provider;
use extraction::Extractor;
use provider::{Fixture, Peer};

fn config(pool: &ArangoPool, peer: &Peer, extractor: &Extractor) -> HadesConfig {
    let mut config = HadesConfig::with_database(pool.database());
    config.apply_env_overrides().unwrap();
    config.database.name = Some(pool.database().into());
    config.database.sockets.readonly = Some(
        pool.reader()
            .socket_path()
            .unwrap()
            .to_str()
            .unwrap()
            .into(),
    );
    config.database.sockets.readwrite = Some(
        pool.writer()
            .socket_path()
            .unwrap()
            .to_str()
            .unwrap()
            .into(),
    );
    config.registered_providers = Some(peer.configuration());
    config.extraction.service.socket = extractor.path.to_str().unwrap().into();
    config.embedding.service.socket = "/absent/legacy-provider.sock".into();
    config.embedding.model.dimension = 777;
    config.batch_processing.concurrency = 1;
    config
}

async fn request(socket: &Path, command: &str, params: Value) -> Value {
    tokio::time::timeout(Duration::from_secs(15), async {
        let mut stream = tokio::net::UnixStream::connect(socket).await.unwrap();
        let bytes = serde_json::to_vec(&json!({"command":command,"params":params})).unwrap();
        stream.write_u32(bytes.len() as u32).await.unwrap();
        stream.write_all(&bytes).await.unwrap();
        let len = stream.read_u32().await.unwrap() as usize;
        assert!(len <= 1024 * 1024);
        let mut bytes = vec![0; len];
        stream.read_exact(&mut bytes).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    })
    .await
    .unwrap()
}

struct Daemon {
    child: Child,
    socket: PathBuf,
    ingests: Vec<(u32, PathBuf)>,
    _root: tempfile::TempDir,
}
impl Daemon {
    async fn start(config: &HadesConfig) -> Self {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.json");
        let socket = root.path().join("daemon.sock");
        std::fs::write(&path, serde_json::to_vec(config).unwrap()).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_hades"))
            .args([
                "--db",
                config.effective_database().unwrap(),
                "daemon",
                "--socket",
            ])
            .arg(&socket)
            .env("HADES_CONFIG", &path)
            .env(
                "ARANGO_RO_SOCKET",
                config.database.sockets.readonly.as_ref().unwrap(),
            )
            .env(
                "ARANGO_RW_SOCKET",
                config.database.sockets.readwrite.as_ref().unwrap(),
            )
            .env_remove("HADES_EXTRACTOR_SOCKET")
            .env("TOKIO_WORKER_THREADS", "2")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        tokio::time::timeout(Duration::from_secs(10), async {
            while !socket.exists() {
                assert!(child.try_wait().unwrap().is_none(), "fixture daemon exited");
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        Self {
            child,
            socket,
            ingests: Default::default(),
            _root: root,
        }
    }
    async fn finish(&mut self) {
        // Signal only this fixture's owned daemon; its normal shutdown drains workers.
        assert_eq!(
            unsafe { libc::kill(self.child.id().unwrap() as i32, libc::SIGTERM) },
            0
        );
        assert!(
            tokio::time::timeout(Duration::from_secs(15), self.child.wait())
                .await
                .unwrap()
                .unwrap()
                .success()
        );
    }
    async fn start_job(&mut self, path: &Path, force: bool) -> Value {
        let reply = request(
            &self.socket,
            "ingest.start",
            json!({"path":path,"force":force}),
        )
        .await;
        if let Some(pid) = reply["data"]["pid"].as_u64() {
            self.ingests.push((pid as u32, path.to_owned()));
        }
        reply
    }
    async fn terminal(&self, job: &str) -> Value {
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                let result = request(&self.socket, "ingest.status", json!({"job_id":job})).await;
                assert_eq!(result["success"], true, "{result}");
                if matches!(
                    result["data"]["status"].as_str(),
                    Some("completed" | "failed")
                ) {
                    return result["data"].clone();
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await
        .unwrap()
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        // Failure cleanup checks the private source path before signalling a
        // recorded child group. Historical database PIDs are never trusted.
        for (pid, path) in &self.ingests {
            if let Ok(command) = std::fs::read(format!("/proc/{pid}/cmdline"))
                && command
                    .windows(path.as_os_str().as_encoded_bytes().len())
                    .any(|part| part == path.as_os_str().as_encoded_bytes())
            {
                unsafe {
                    libc::kill(-(*pid as i32), libc::SIGKILL);
                }
            }
        }
        let _ = self.child.start_kill();
    }
}

async fn inherited(config: &HadesConfig, path: &Path, extra: &[&str]) -> std::process::Output {
    let snapshot = Snapshot::new(config).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_hades"));
    let fd = snapshot.inherit(&mut command);
    command
        .args(["--resolved-config-fd", &fd.to_string(), "ingest"])
        .arg(path)
        .args(extra)
        .env("HADES_CONFIG", "/absent/ambient-config")
        .env("HADES_DATABASE", "wrong-ambient-database")
        .env("HADES_EMBEDDER_SOCKET", "/absent/ambient-provider.sock")
        .env("ARANGO_PASSWORD", "wrong-ambient-password")
        .env("TOKIO_WORKER_THREADS", "2")
        .kill_on_drop(true);
    tokio::time::timeout(Duration::from_secs(15), command.output())
        .await
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn daemon_jobs_ingest_files_and_directories_with_persisted_profile_pins() {
    for late in [false, true] {
        with_temp_db(
            "cli_profiled_jobs",
            Fixtures::Empty,
            move |pool| async move {
                let fixture = if late {
                    Fixture::late()
                } else {
                    Fixture::new(2)
                };
                fixture.install(&pool).await;
                let peer = Peer::new(&fixture);
                let extractor = Extractor::new();
                let config = config(&pool, &peer, &extractor);
                let mut daemon = Daemon::start(&config).await;
                let root = tempfile::tempdir().unwrap();
                let file = root.path().join("notes.md");
                std::fs::write(&file, "apple café\nbanana 東京\napple tail").unwrap();
                let directory = root.path().join("documents");
                std::fs::create_dir(&directory).unwrap();
                std::fs::write(directory.join("second.md"), "banana 東京").unwrap();
                std::fs::write(directory.join("module.py"), "print('disabled')").unwrap();
                std::fs::write(directory.join("unknown.ts"), "unrouted source").unwrap();
                for path in [&file, &directory] {
                    let started = daemon.start_job(path, false).await;
                    assert_eq!(started["success"], true, "{started}");
                    let pin = &started["data"]["profile_pin"];
                    assert_eq!(pin["database"], pool.database());
                    assert_eq!(pin["profile_revision"], 2);
                    let row = daemon
                        .terminal(started["data"]["job_id"].as_str().unwrap())
                        .await;
                    assert_eq!(row["status"], "completed", "{row}");
                    assert_eq!(&row["profile_pin"], pin);
                    assert_eq!(row["result"]["success"], true);
                    if path.is_dir() {
                        assert_eq!(row["result"]["data"]["profile_pin"], *pin);
                        assert_eq!(
                            row["result"]["data"]["disabled_by_profile"]["code"],
                            json!([directory.join("module.py")])
                        );
                        assert_eq!(
                            row["result"]["data"]["unrouted"],
                            json!([{"path":directory.join("unknown.ts"),"reason":"no handler for extension"}])
                        );
                        assert!(row["result"]["data"]["code"].is_null());
                    }
                }
                assert_eq!(crud::count_collection(&pool, "documents").await.unwrap(), 2);
                assert_eq!(extractor.requests().len(), 2);
                assert!(
                    crud::list_collections(&pool, false)
                        .await
                        .unwrap()
                        .iter()
                        .all(|collection| !collection.name.starts_with("codebase_"))
                );
                let result = request(
                    &daemon.socket,
                    "db.query",
                    json!({"text":"banana","limit":1}),
                )
                .await;
                assert_eq!(result["success"], true, "{result}");
                assert_eq!(result["data"]["scanned"], 4);
                assert_eq!(
                    result["data"]["results"][0].get("late_context").is_some(),
                    late
                );
                let refused = daemon.start_job(&file, false).await;
                let row = daemon
                    .terminal(refused["data"]["job_id"].as_str().unwrap())
                    .await;
                assert_eq!(row["status"], "failed");
                let replacement = daemon.start_job(&file, true).await;
                let row = daemon
                    .terminal(replacement["data"]["job_id"].as_str().unwrap())
                    .await;
                assert_eq!(row["status"], "completed", "{row}");
                assert_eq!(crud::count_collection(&pool, "documents").await.unwrap(), 2);
                daemon.finish().await;
            },
        )
        .await;
    }
}

#[tokio::test]
async fn sealed_worker_refuses_stale_unpinned_and_overridden_admission_before_extraction() {
    with_temp_db("cli_job_stale", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let extractor = Extractor::new();
        let mut config = config(&pool, &peer, &extractor);
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("notes.md");
        std::fs::write(&file, "apple").unwrap();
        let pin = fixture.connect(&pool, &peer).await.job_pin().unwrap();
        let before = peer.observations();
        let unpinned = inherited(&config, &file, &[]).await;
        assert!(!unpinned.status.success());
        config.ingest_binding = Some(Binding::Profiled(Box::new(pin)));
        let overridden = inherited(&config, &file, &["--db", "wrong"]).await;
        assert!(!overridden.status.success());
        assert!(String::from_utf8_lossy(&overridden.stderr).contains("does not accept database"));
        let path = format!("document/{COLLECTION}/embedding_profile");
        let original = pool.writer().get(&path).await.unwrap();
        pool.writer()
            .patch(&path, &json!({"payload":original["payload"]}))
            .await
            .unwrap();
        for input in [&file, root.path()] {
            let stale = inherited(&config, input, &[]).await;
            assert!(!stale.status.success());
            assert!(
                String::from_utf8_lossy(&stale.stderr).contains("ingestion job profile changed")
            );
        }
        assert_eq!(peer.observations(), before);
        assert!(extractor.requests().is_empty());
        assert_eq!(crud::count_collection(&pool, "documents").await.unwrap(), 0);
    })
    .await;
}

#[tokio::test]
async fn profile_drift_provider_reload_and_shutdown_do_not_publish_partial_jobs() {
    for failure in ["profile", "provider", "shutdown"] {
        with_temp_db("cli_job_drift", Fixtures::Empty, move |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let extractor = Extractor::new();
            let mut daemon = Daemon::start(&config(&pool, &peer, &extractor)).await;
            let root = tempfile::tempdir().unwrap();
            let file = root.path().join("notes.md");
            std::fs::write(&file, "apple").unwrap();
            let (entered, release) = extractor.hold();
            let started = daemon.start_job(&file, false).await;
            assert_eq!(started["success"], true, "{started}");
            tokio::time::timeout(Duration::from_secs(5), entered)
                .await
                .unwrap()
                .unwrap();
            let job = started["data"]["job_id"].as_str().unwrap();
            let row = if failure == "shutdown" {
                daemon.finish().await;
                let _ = release.send(());
                crud::get_document(&pool, "hades_ingest_jobs", job)
                    .await
                    .unwrap()
            } else {
                if failure == "profile" {
                    let path = format!("document/{COLLECTION}/embedding_profile");
                    let original = pool.writer().get(&path).await.unwrap();
                    pool.writer()
                        .patch(&path, &json!({"payload":original["payload"]}))
                        .await
                        .unwrap();
                } else {
                    let mut receipt = peer.receipt();
                    receipt.serving_generation = "reloaded-during-job".into();
                    peer.set_receipt(receipt);
                }
                release.send(()).unwrap();
                let row = daemon.terminal(job).await;
                daemon.finish().await;
                row
            };
            assert_eq!(row["status"], "failed", "{row}");
            assert_eq!(row["profile_pin"], started["data"]["profile_pin"]);
            assert!(peer.requests().is_empty());
            for collection in ["documents", "chunks", "embeddings"] {
                assert_eq!(crud::count_collection(&pool, collection).await.unwrap(), 0);
            }
        })
        .await;
    }
}
