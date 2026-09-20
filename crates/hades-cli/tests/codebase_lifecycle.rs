//! Full CLI graph lifecycle on disposable data with deterministic embedding RPCs.
//! This validates pipeline contracts, not production-model retrieval quality.
use axum::{
    Json, Router,
    routing::{get, post},
};
use hades_core::db::{ArangoPool, keys};
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::process::Command;
use tokio::task::JoinHandle;

const MODEL: &str = "jinaai/jina-embeddings-v4";

struct Embedder {
    fail: Arc<AtomicBool>,
    hold: Arc<AtomicBool>,
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
    socket: PathBuf,
    task: JoinHandle<()>,
    _directory: tempfile::TempDir,
}
impl Drop for Embedder {
    fn drop(&mut self) {
        self.release.notify_one();
        self.task.abort();
    }
}
impl Embedder {
    async fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("embedder.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let fail = Arc::new(AtomicBool::new(false));
        let fault = fail.clone();
        let hold = Arc::new(AtomicBool::new(false));
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let hold_request = hold.clone();
        let entered_request = entered.clone();
        let release_request = release.clone();
        let app = Router::new()
            .route("/v1/models", get(|| async {
                Json(json!({"data":[{"id":MODEL,"dimension":2048,"max_seq_length":8192,"device":"cpu"}]}))
            }))
            .route("/v1/embeddings", post(move |Json(body): Json<Value>| {
                let fault = fault.clone();
                let hold = hold_request.clone();
                let entered = entered_request.clone();
                let release = release_request.clone();
                async move {
                if hold.load(Ordering::SeqCst) {
                    entered.notify_one();
                    release.notified().await;
                }
                if fault.load(Ordering::SeqCst) { return Json(json!({"error":"injected embedding failure"})); }
                assert_eq!(body["task"], "code", "code ingest/query must agree on the adapter");
                let inputs = body["input"].as_array().unwrap();
                let mut data = Vec::new();
                if let Some(bounds) = body["late_chunk"]["boundaries"].as_array() {
                    assert_eq!(inputs.len(), 1);
                    let text = inputs[0].as_str().unwrap();
                    for (i, boundary) in bounds.iter().enumerate() {
                        let start = boundary[0].as_u64().unwrap() as usize;
                        let end = boundary[1].as_u64().unwrap() as usize;
                        data.push(json!({"index":0,"embedding":vector(&text[start..end]),
                            "chunk_index":i,"char_start":start,"char_end":end}));
                    }
                } else {
                    for (i, input) in inputs.iter().enumerate() {
                        data.push(json!({"index":i,"embedding":vector(input.as_str().unwrap())}));
                    }
                }
                Json(json!({"model":MODEL,"data":data}))
            }}));
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            fail,
            hold,
            entered,
            release,
            socket,
            task,
            _directory: directory,
        }
    }
}
fn vector(text: &str) -> Vec<f32> {
    let mut vector = vec![0.; 2048];
    let index = if text.contains("sapphire") {
        2
    } else if text.contains("quartz") {
        0
    } else {
        1
    };
    vector[index] = 1.;
    vector
}

fn cli_command(pool: &ArangoPool, embedder: &Embedder, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_hades"));
    command
        .args(["--db", pool.database()])
        .args(args)
        .env("HADES_EMBEDDER_SOCKET", &embedder.socket)
        .env(
            "HADES_EXTRACTOR_SOCKET",
            embedder._directory.path().join("absent-extractor.sock"),
        )
        .env_remove("HADES_DISABLE_LATE_CHUNKING")
        .env_remove("HADES_DEFAULT_COLLECTION")
        .kill_on_drop(true);
    command
}

async fn cli(pool: &ArangoPool, embedder: &Embedder, args: &[&str], success: bool) -> Value {
    let output = cli_command(pool, embedder, args).output().await.unwrap();
    assert_eq!(
        output.status.success(),
        success,
        "{args:?}\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["success"], success, "{report}");
    report["data"].clone()
}
async fn ingest(pool: &ArangoPool, embedder: &Embedder, root: &Path) -> Value {
    cli(pool, embedder, &["ingest", root.to_str().unwrap()], true).await
}
async fn validate(pool: &ArangoPool, embedder: &Embedder) {
    let result = cli(pool, embedder, &["codebase", "validate"], true).await;
    assert_eq!(result["summary"]["failed"], 0, "{result}");
}
async fn search(pool: &ArangoPool, embedder: &Embedder, text: &str) -> Value {
    cli(
        pool,
        embedder,
        &[
            "db",
            "query",
            text,
            "--collection",
            "codebase",
            "--limit",
            "10",
        ],
        true,
    )
    .await
}
fn assert_hit(result: &Value, key: &str, marker: &str) {
    let hit = &result["results"][0];
    assert_eq!(hit["file_key"], key, "{result}");
    assert!(hit["text"].as_str().unwrap().contains(marker), "{result}");
    assert_eq!(hit["score"], 1.0);
}

struct PrivateDaemon {
    child: tokio::process::Child,
    ingests: Vec<(u32, PathBuf)>,
}
impl Drop for PrivateDaemon {
    fn drop(&mut self) {
        // Failure cleanup verifies the unique private source path before
        // signalling a recorded group. Never signal a historical database PID.
        for (pid, root) in &self.ingests {
            if let Ok(command) = std::fs::read(format!("/proc/{pid}/cmdline"))
                && command
                    .windows(root.as_os_str().as_encoded_bytes().len())
                    .any(|part| part == root.as_os_str().as_encoded_bytes())
            {
                // SAFETY: fixture identity checked; this ingest owns a fresh group.
                unsafe {
                    libc::kill(-(*pid as i32), libc::SIGKILL);
                }
            }
        }
        let _ = self.child.start_kill();
    }
}

async fn daemon_request(socket: &Path, request: Value) -> Value {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let mut stream = tokio::net::UnixStream::connect(socket).await.unwrap();
        let bytes = serde_json::to_vec(&request).unwrap();
        stream.write_u32(bytes.len() as u32).await.unwrap();
        stream.write_all(&bytes).await.unwrap();
        let size = stream.read_u32().await.unwrap() as usize;
        assert!(size <= 1024 * 1024);
        let mut body = vec![0; size];
        stream.read_exact(&mut body).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    })
    .await
    .expect("private daemon request timed out")
}

struct PrivateMcp {
    client: reqwest::Client,
    url: String,
    session: String,
    next_id: std::sync::atomic::AtomicU64,
}

impl PrivateMcp {
    async fn connect(address: std::net::SocketAddr) -> Self {
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();
        let mut mcp = Self {
            client,
            url: format!("http://{address}/mcp"),
            session: String::new(),
            next_id: std::sync::atomic::AtomicU64::new(2),
        };
        let response = mcp
            .post(json!({"jsonrpc":"2.0","id":1,"method":"initialize",
            "params":{"protocolVersion":"2025-03-26","capabilities":{},
                "clientInfo":{"name":"isolated-ingest-test","version":"1"}}}))
            .await;
        mcp.session = response.headers()["mcp-session-id"]
            .to_str()
            .unwrap()
            .to_owned();
        assert!(Self::message(response, 1).await["result"].is_object());
        let response = mcp
            .post(json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
            .await;
        assert_eq!(response.status(), reqwest::StatusCode::ACCEPTED);
        mcp
    }

    async fn post(&self, body: Value) -> reqwest::Response {
        let mut request = self
            .client
            .post(&self.url)
            .bearer_auth("private-fixture-token")
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", "2025-03-26")
            .json(&body);
        if !self.session.is_empty() {
            request = request.header("mcp-session-id", &self.session);
        }
        let response = request.send().await.unwrap();
        assert!(
            response.status().is_success(),
            "MCP HTTP {}",
            response.status()
        );
        response
    }

    async fn message(mut response: reqwest::Response, id: u64) -> Value {
        // Private replies can be JSON or SSE; bound bytes before accumulation.
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.unwrap() {
            assert!(bytes.len() + chunk.len() <= 1024 * 1024);
            bytes.extend_from_slice(&chunk);
        }
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
            assert_eq!(value["id"], id);
            return value;
        }
        let body = std::str::from_utf8(&bytes).unwrap();
        body.lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
            .find(|value| value["id"] == id)
            .unwrap_or_else(|| panic!("missing private MCP response {id}: {body}"))
    }

    async fn start(&self, db: &str, path: &Path) -> Value {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let response = self
            .post(json!({"jsonrpc":"2.0","id":id,"method":"tools/call",
            "params":{"name":"ingest_start","arguments":{"db":db,"path":path}}}))
            .await;
        let message = Self::message(response, id).await;
        let text = message["result"]["content"][0]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("unexpected MCP result: {message}"));
        serde_json::from_str(text).unwrap_or_else(|_| panic!("unexpected tool envelope: {text}"))
    }
}

#[tokio::test]
async fn mcp_ingestion_limit_and_tree_ownership_span_databases() {
    with_temp_db("mcp_ingest_a", Fixtures::Codebase, |a| async move {
        with_temp_db("mcp_ingest_b", Fixtures::Codebase, |b| async move {
            let embedder = Embedder::new().await;
            embedder.hold.store(true, Ordering::SeqCst);
            let tree = tempfile::tempdir().unwrap();
            let roots: Vec<_> = (0..4)
                .map(|i| {
                    let root = tree.path().join(format!("source{i}"));
                    std::fs::create_dir(&root).unwrap();
                    std::fs::write(
                        root.join("fixture.py"),
                        "def target():\n    return 'quartz_mcp'\n",
                    )
                    .unwrap();
                    root
                })
                .collect();
            let token = tree.path().join("token");
            std::fs::write(&token, "private-fixture-token\n").unwrap();
            let socket = tree.path().join("daemon.sock");
            let log = tree.path().join("daemon.log");
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            drop(listener);
            let prefixes = format!("{},{}", a.database(), b.database());
            let child = cli_command(
                &a,
                &embedder,
                &[
                    "daemon",
                    "--socket",
                    socket.to_str().unwrap(),
                    "--mcp-bind",
                    &address.to_string(),
                    "--mcp-token-file",
                    token.to_str().unwrap(),
                    "--mcp-dbs",
                    b.database(),
                    "--mcp-db-prefix",
                    &prefixes,
                    "--mcp-ingest-root",
                    tree.path().to_str().unwrap(),
                ],
            )
            .env("HADES_USE_GPU", "false")
            .env("TOKIO_WORKER_THREADS", "2")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::fs::File::create(&log).unwrap())
            .spawn()
            .unwrap();
            let mut daemon = PrivateDaemon {
                child,
                ingests: Vec::new(),
            };
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                while tokio::net::UnixStream::connect(&socket).await.is_err() {
                    assert!(
                        daemon.child.try_wait().unwrap().is_none(),
                        "{}",
                        std::fs::read_to_string(&log).unwrap()
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            })
            .await
            .unwrap();
            let mcp = PrivateMcp::connect(address).await;
            let first = mcp.start(a.database(), &roots[0]).await;
            if let Some(pid) = first["data"]["pid"].as_u64() {
                daemon.ingests.push((pid as u32, roots[0].clone()));
            }
            assert_eq!(first["success"], true, "{first}");
            // Both contenders target B while A already owns one slot. A
            // per-database counter or check-before-reserve race admits three.
            let (candidate_one, candidate_two) = tokio::join!(
                mcp.start(b.database(), &roots[1]),
                mcp.start(b.database(), &roots[2])
            );
            for (result, root) in [(&candidate_one, &roots[1]), (&candidate_two, &roots[2])] {
                if let Some(pid) = result["data"]["pid"].as_u64() {
                    daemon.ingests.push((pid as u32, root.clone()));
                }
            }
            let (second, refused) = if candidate_one["success"] == true {
                (&candidate_one, &candidate_two)
            } else {
                (&candidate_two, &candidate_one)
            };
            assert_eq!(second["success"], true, "{second}");
            assert_eq!(refused["success"], false, "{refused}");
            assert!(
                refused
                    .to_string()
                    .contains("already admitted across this service"),
                "{refused}"
            );
            let mut jobs = Vec::new();
            for (result, pool) in [(&first, &a), (second, &b)] {
                assert_eq!(result["success"], true, "{result}");
                assert_eq!(result["data"]["database"], pool.database());
                let pid = result["data"]["pid"].as_u64().unwrap() as u32;
                jobs.push((
                    pool,
                    result["data"]["job_id"].as_str().unwrap().to_owned(),
                    pid,
                ));
            }
            assert_ne!(jobs[0].2, jobs[1].2);
            // Held embedder requests keep both children alive throughout all
            // competing starts. Count the actual direct children, not job rows.
            let tasks_path = format!("/proc/{}/task", daemon.child.id().unwrap());
            let children = || -> Vec<u32> {
                let mut ids = Vec::new();
                for task in std::fs::read_dir(&tasks_path).unwrap() {
                    let path = task.unwrap().path().join("children");
                    match std::fs::read_to_string(path) {
                        Ok(text) => {
                            ids.extend(text.split_whitespace().map(|id| id.parse::<u32>().unwrap()))
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => panic!("cannot count private daemon children: {error}"),
                    }
                }
                ids.sort_unstable();
                ids.dedup();
                ids
            };
            let mut expected = vec![jobs[0].2, jobs[1].2];
            expected.sort_unstable();
            assert_eq!(children(), expected);
            let (over_a, over_b, overlap) = tokio::join!(
                mcp.start(a.database(), &roots[3]),
                mcp.start(b.database(), &roots[3]),
                mcp.start(b.database(), &roots[0])
            );
            for denied in [&over_a, &over_b] {
                assert_eq!(denied["success"], false, "{denied}");
                assert!(
                    denied
                        .to_string()
                        .contains("already admitted across this service"),
                    "{denied}"
                );
            }
            assert_eq!(overlap["success"], false, "{overlap}");
            assert!(
                overlap.to_string().contains("overlapping tree"),
                "{overlap}"
            );
            assert_eq!(children(), expected);
            for (pool, job, pid) in &jobs {
                let row = hades_core::db::crud::get_document(pool, "hades_ingest_jobs", job)
                    .await
                    .unwrap();
                assert_eq!(row["status"], "running");
                assert_eq!(row["pid"], *pid);
                assert_eq!(
                    hades_core::db::crud::count_collection(pool, "hades_ingest_jobs")
                        .await
                        .unwrap(),
                    1
                );
            }
            assert_eq!(
                unsafe { libc::kill(daemon.child.id().unwrap() as i32, libc::SIGTERM) },
                0
            );
            assert!(
                tokio::time::timeout(std::time::Duration::from_secs(10), daemon.child.wait())
                    .await
                    .unwrap()
                    .unwrap()
                    .success()
            );
            for (pool, job, pid) in &jobs {
                assert_eq!(unsafe { libc::kill(*pid as i32, 0) }, -1);
                assert_eq!(
                    std::io::Error::last_os_error().raw_os_error(),
                    Some(libc::ESRCH)
                );
                let row = hades_core::db::crud::get_document(pool, "hades_ingest_jobs", job)
                    .await
                    .unwrap();
                assert_eq!(row["status"], "failed", "{row}");
                assert!(row["detail"].as_str().unwrap().contains("shutdown"));
            }
            assert!(!socket.exists());
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn daemon_shutdown_reaps_running_ingestion_and_persists_failure() {
    with_temp_db(
        "daemon_ingest_shutdown",
        Fixtures::Codebase,
        |pool| async move {
            for signal in [libc::SIGTERM, libc::SIGINT] {
                let embedder = Embedder::new().await;
                embedder.hold.store(true, Ordering::SeqCst);
                let tree = tempfile::tempdir().unwrap();
                let source = tree.path().join("source");
                std::fs::create_dir(&source).unwrap();
                std::fs::write(
                    source.join("fixture.py"),
                    "def target():\n    return 'quartz_shutdown'\n",
                )
                .unwrap();
                let socket = tree.path().join("daemon.sock");
                let log = tree.path().join("daemon.log");
                let child = cli_command(
                    &pool,
                    &embedder,
                    &["daemon", "--socket", socket.to_str().unwrap()],
                )
                .env("HADES_USE_GPU", "false")
                .env("TOKIO_WORKER_THREADS", "2")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::fs::File::create(&log).unwrap())
                .spawn()
                .unwrap();
                let mut daemon = PrivateDaemon {
                    child,
                    ingests: Vec::new(),
                };
                tokio::time::timeout(std::time::Duration::from_secs(10), async {
                    loop {
                        if tokio::net::UnixStream::connect(&socket).await.is_ok() {
                            break;
                        }
                        assert!(
                            daemon.child.try_wait().unwrap().is_none(),
                            "private daemon exited: {}",
                            std::fs::read_to_string(&log).unwrap()
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                })
                .await
                .unwrap();
                let before = snapshot_graph(&pool).await;
                let started = daemon_request(
                    &socket,
                    json!({"command":"ingest.start","params":{"path":source}}),
                )
                .await;
                assert_eq!(started["success"], true, "{started}");
                let job = started["data"]["job_id"].as_str().unwrap();
                let pid = started["data"]["pid"].as_u64().unwrap() as u32;
                daemon.ingests.push((pid, source.clone()));
                // The request connection has already closed. Detached ownership
                // must keep this actual ingestion alive while it awaits the mock.
                tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    embedder.entered.notified(),
                )
                .await
                .expect("private ingestion did not reach synthetic embedding");
                let status = daemon_request(
                    &socket,
                    json!({"command":"ingest.status","params":{"job_id":job}}),
                )
                .await;
                assert_eq!(status["data"]["status"], "running", "{status}");
                assert_eq!(status["data"]["owned_by_service"], true);
                assert!(std::path::Path::new(&format!("/proc/{pid}")).exists());
                let duplicate = daemon_request(
                    &socket,
                    json!({"command":"ingest.start","params":{"path":source}}),
                )
                .await;
                assert_eq!(duplicate["success"], false, "{duplicate}");
                // SAFETY: signal only the private daemon's unreaped direct child.
                assert_eq!(
                    unsafe { libc::kill(daemon.child.id().unwrap() as i32, signal) },
                    0
                );
                let exited =
                    tokio::time::timeout(std::time::Duration::from_secs(10), daemon.child.wait())
                        .await
                        .expect("private daemon did not drain ingestion")
                        .unwrap();
                assert!(
                    exited.success(),
                    "{}",
                    std::fs::read_to_string(&log).unwrap()
                );
                // Direct ingestion was reaped by the daemon before the daemon exit.
                assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
                assert_eq!(
                    std::io::Error::last_os_error().raw_os_error(),
                    Some(libc::ESRCH)
                );
                let recorded = hades_core::db::crud::get_document(&pool, "hades_ingest_jobs", job)
                    .await
                    .unwrap();
                assert_eq!(recorded["status"], "failed", "{recorded}");
                assert!(recorded["detail"].as_str().unwrap().contains("shutdown"));
                assert_eq!(recorded["pid"], pid);
                assert!(recorded["finished_at"].is_string());
                assert_eq!(recorded["output_storage"], "bounded_job_record");
                assert_eq!(recorded["log_path"], Value::Null);
                assert_eq!(snapshot_graph(&pool).await, before);
                assert!(!socket.exists());
                embedder.hold.store(false, Ordering::SeqCst);
                embedder.release.notify_one();
                // A new daemon instance must admit a retry after the previous
                // owner durably recorded failure, and persist normal completion.
                let child = cli_command(
                    &pool,
                    &embedder,
                    &["daemon", "--socket", socket.to_str().unwrap()],
                )
                .env("HADES_USE_GPU", "false")
                .env("TOKIO_WORKER_THREADS", "2")
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::fs::File::create(&log).unwrap())
                .spawn()
                .unwrap();
                let mut restarted = PrivateDaemon {
                    child,
                    ingests: Vec::new(),
                };
                tokio::time::timeout(std::time::Duration::from_secs(10), async {
                    while tokio::net::UnixStream::connect(&socket).await.is_err() {
                        assert!(restarted.child.try_wait().unwrap().is_none());
                        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    }
                })
                .await
                .unwrap();
                // Simulate an interrupted previous owner. Its saved PID is a
                // live, unrelated fixture process: neither PID existence nor
                // reuse establishes ownership for this new daemon instance.
                let stale_job = "0123456789abcdef0123456789abcdef";
                hades_core::db::crud::insert_document(
                    &pool,
                    "hades_ingest_jobs",
                    &json!({"_key":stale_job,"status":"running",
                        "owner_instance":recorded["owner_instance"],
                        "pid":std::process::id(),"path":source}),
                )
                .await
                .unwrap();
                let stale_before =
                    hades_core::db::crud::get_document(&pool, "hades_ingest_jobs", stale_job)
                        .await
                        .unwrap();
                let stale_status = daemon_request(
                    &socket,
                    json!({"command":"ingest.status","params":{"job_id":stale_job}}),
                )
                .await;
                assert_eq!(stale_status["success"], true, "{stale_status}");
                assert_eq!(stale_status["data"]["status"], "recovery_required");
                assert_eq!(stale_status["data"]["recorded_status"], "running");
                assert_eq!(stale_status["data"]["owned_by_service"], false);
                let refused = daemon_request(
                    &socket,
                    json!({"command":"ingest.start","params":{"path":source}}),
                )
                .await;
                assert_eq!(refused["success"], false, "{refused}");
                assert_eq!(
                    hades_core::db::crud::get_document(&pool, "hades_ingest_jobs", stale_job)
                        .await
                        .unwrap(),
                    stale_before,
                    "reading or refusing an unowned job must not rewrite its record"
                );
                assert_eq!(snapshot_graph(&pool).await, before);
                // Only this disposable fixture is reconciled explicitly. The
                // daemon must never infer a terminal outcome from a saved PID.
                hades_core::db::crud::update_document(
                    &pool,
                    "hades_ingest_jobs",
                    stale_job,
                    &json!({"status":"failed","detail":"fixture-only reconciliation"}),
                )
                .await
                .unwrap();
                let retry = daemon_request(
                    &socket,
                    json!({"command":"ingest.start","params":{"path":source}}),
                )
                .await;
                assert_eq!(retry["success"], true, "{retry}");
                let retry_job = retry["data"]["job_id"].as_str().unwrap();
                let retry_pid = retry["data"]["pid"].as_u64().unwrap() as u32;
                restarted.ingests.push((retry_pid, source.clone()));
                let completed = tokio::time::timeout(std::time::Duration::from_secs(20), async {
                    loop {
                        let status = daemon_request(
                            &socket,
                            json!({"command":"ingest.status","params":{"job_id":retry_job}}),
                        )
                        .await;
                        assert_eq!(status["success"], true, "{status}");
                        match status["data"]["status"].as_str() {
                            Some("running" | "starting") => {
                                tokio::time::sleep(std::time::Duration::from_millis(20)).await
                            }
                            _ => break status["data"].clone(),
                        }
                    }
                })
                .await
                .unwrap();
                assert_eq!(completed["status"], "completed", "{completed}");
                assert_eq!(completed["result"]["success"], true);
                assert_ne!(completed["owner_instance"], recorded["owner_instance"]);
                assert_eq!(unsafe { libc::kill(retry_pid as i32, 0) }, -1);
                assert_eq!(
                    std::io::Error::last_os_error().raw_os_error(),
                    Some(libc::ESRCH)
                );
                validate(&pool, &embedder).await;
                assert_eq!(
                    unsafe { libc::kill(restarted.child.id().unwrap() as i32, signal) },
                    0
                );
                assert!(
                    tokio::time::timeout(
                        std::time::Duration::from_secs(10),
                        restarted.child.wait()
                    )
                    .await
                    .unwrap()
                    .unwrap()
                    .success()
                );
                assert!(!socket.exists());
                hades_core::db::crud::delete_document(&pool, "hades_ingest_jobs", stale_job)
                    .await
                    .unwrap();
            }
        },
    )
    .await;
}

#[tokio::test]
async fn ingest_query_modify_move_delete_and_partial_failure_recover() {
    with_temp_db("cli_lifecycle", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        let provider = root.join("provider.py");
        let consumer = root.join("consumer.py");
        std::fs::write(&provider, "def target():\n    return 'quartz_original'\n").unwrap();
        std::fs::write(&consumer, "from provider import target\n\ndef caller():\n    return target()\n").unwrap();
        let original_key = keys::scoped_file_key(root.to_str().unwrap(), "provider.py");
        let first = ingest(&pool, &embedder, root).await;
        assert_eq!(first["code"]["completed"], 2);
        assert_eq!(first["code"]["embedding"]["files_with_embedding_failures"], 0);
        assert!(first["code"]["embedding"]["total_embeddings"].as_u64().unwrap() > 0);
        validate(&pool, &embedder).await;
        assert_hit(&search(&pool, &embedder, "quartz").await, &original_key, "quartz_original");
        let repeated = ingest(&pool, &embedder, root).await;
        assert_eq!(repeated["code"]["skipped"], 2);

        // Content changes and definition-line shifts must update text/vectors
        // and re-point the unchanged consumer's inbound symbol edges.
        std::fs::write(&provider, "# moved definition\n\ndef target():\n    return 'sapphire_revised'\n").unwrap();
        ingest(&pool, &embedder, root).await;
        validate(&pool, &embedder).await;
        let changed = search(&pool, &embedder, "sapphire").await;
        assert_hit(&changed, &original_key, "sapphire_revised");
        assert!(!changed.to_string().contains("quartz_original"));

        let renamed = root.join("renamed.py");
        std::fs::rename(&provider, &renamed).unwrap();
        std::fs::write(&consumer, "from renamed import target\n\ndef caller():\n    return target()\n").unwrap();
        ingest(&pool, &embedder, root).await;
        let drift = cli(&pool, &embedder, &["codebase", "drift", root.to_str().unwrap(), "--full"], true).await;
        assert_eq!(drift["stale"]["keys"], json!([original_key]));
        cli(&pool, &embedder, &["codebase", "retire", "--file", &original_key, "--yes"], true).await;
        validate(&pool, &embedder).await;
        let moved_key = keys::scoped_file_key(root.to_str().unwrap(), "renamed.py");
        assert_hit(&search(&pool, &embedder, "sapphire").await, &moved_key, "sapphire_revised");

        // Extraction setup fails, but the code phase must remain durable and
        // the top-level batch must report failure. No real extractor is used.
        std::fs::write(root.join("notes.md"), "A document needing extraction.\n").unwrap();
        std::fs::write(&consumer, "from renamed import target\n\ndef caller():\n    # recovered_code_phase\n    return target()\n").unwrap();
        let partial = cli(&pool, &embedder, &["ingest", root.to_str().unwrap()], false).await;
        assert!(partial["document_phase_error"].as_str().is_some(), "{partial}");
        assert_eq!(partial["code"]["failed"], 0);
        validate(&pool, &embedder).await;
        let recovered = search(&pool, &embedder, "consumer").await;
        assert!(recovered.to_string().contains("recovered_code_phase"));
        std::fs::remove_file(root.join("notes.md")).unwrap();
        let retry = ingest(&pool, &embedder, root).await;
        assert_eq!(retry["code"]["skipped"], 2);

        std::fs::remove_file(renamed).unwrap();
        std::fs::remove_file(consumer).unwrap();
        let drift = cli(&pool, &embedder, &["codebase", "drift", root.to_str().unwrap(), "--full"], true).await;
        assert_eq!(drift["stale"]["count"], 2);
        for key in drift["stale"]["keys"].as_array().unwrap() {
            cli(&pool, &embedder, &["codebase", "retire", "--file", key.as_str().unwrap(), "--yes"], true).await;
        }
        validate(&pool, &embedder).await;
        assert_eq!(search(&pool, &embedder, "sapphire").await["result_count"], 0);
        for collection in ["codebase_files", "codebase_chunks", "codebase_symbols", "codebase_embeddings"] {
            assert_eq!(hades_core::db::crud::count_collection(&pool, collection).await.unwrap(), 0);
        }
    }).await;
}

#[tokio::test]
async fn failed_chunk_replacement_preserves_previous_file_graph() {
    with_temp_db("failed_replace", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        let file = tree.path().join("provider.py");
        std::fs::write(&file, "def target():\n    return 'quartz_original'\n").unwrap();
        std::fs::write(tree.path().join("consumer.py"), "from provider import target\n\ndef caller():\n    return target()\n").unwrap();
        ingest(&pool, &embedder, tree.path()).await;
        let before = snapshot_graph(&pool).await;
        let before_chunks = hades_core::db::crud::count_collection(&pool, "codebase_chunks").await.unwrap();
        let before_symbols = hades_core::db::crud::count_collection(&pool, "codebase_symbols").await.unwrap();
        assert!(before_chunks > 0 && before_symbols > 0);
        pool.writer().put("collection/codebase_chunks/properties", &json!({
            "schema": {"level":"strict", "message":"isolated replacement rejection", "rule": {
                "type":"object", "required":["audit_required_marker"]
            }}
        })).await.unwrap();
        std::fs::write(&file, "# shifted\n\ndef target():\n    return 'sapphire_revised'\n").unwrap();
        let failed = cli(&pool, &embedder, &["ingest", tree.path().to_str().unwrap()], false).await;
        let after_chunks = hades_core::db::crud::count_collection(&pool, "codebase_chunks").await.unwrap();
        let after_symbols = hades_core::db::crud::count_collection(&pool, "codebase_symbols").await.unwrap();
        println!("FAILED_REPLACEMENT_EVIDENCE {}", json!({"before_chunks":before_chunks,"before_symbols":before_symbols,"after_chunks":after_chunks,"after_symbols":after_symbols,"report":failed}));
        assert_eq!(after_chunks, before_chunks, "failed replacement must preserve committed chunks");
        assert_eq!(after_symbols, before_symbols, "failed replacement must preserve committed symbols");
        assert_eq!(snapshot_graph(&pool).await, before, "failed replacement must preserve full graph contents");
        pool.writer().put("collection/codebase_chunks/properties", &json!({"schema": null})).await.unwrap();
        ingest(&pool, &embedder, tree.path()).await;
        validate(&pool, &embedder).await;
        let key = keys::scoped_file_key(tree.path().to_str().unwrap(), "provider.py");
        assert_hit(&search(&pool, &embedder, "sapphire").await, &key, "sapphire_revised");
    }).await;
}

async fn snapshot_graph(pool: &ArangoPool) -> Value {
    let mut snapshot = serde_json::Map::new();
    for (collection, _) in hades_core::db::collections::CODEBASE.all_collections() {
        let result = hades_core::db::query::query(
            pool,
            "FOR d IN @@collection SORT d._key RETURN UNSET(d, '_rev')",
            Some(&json!({"@collection":collection})),
            None,
            false,
            hades_core::db::query::ExecutionTarget::Writer,
        )
        .await
        .unwrap();
        snapshot.insert(collection.into(), json!(result.results));
    }
    Value::Object(snapshot)
}

#[tokio::test]
async fn embedding_failure_preserves_committed_graph_before_transaction() {
    with_temp_db(
        "embedding_rollback",
        Fixtures::Codebase,
        |pool| async move {
            let embedder = Embedder::new().await;
            let tree = tempfile::tempdir().unwrap();
            let file = tree.path().join("provider.py");
            std::fs::write(&file, "def target():\n    return 'quartz_original'\n").unwrap();
            ingest(&pool, &embedder, tree.path()).await;
            let before = snapshot_graph(&pool).await;
            std::fs::write(&file, "def target():\n    return 'sapphire_revised'\n").unwrap();
            embedder.fail.store(true, Ordering::SeqCst);
            let failed = cli(
                &pool,
                &embedder,
                &["ingest", tree.path().to_str().unwrap()],
                false,
            )
            .await;
            assert_eq!(
                failed["code"]["embedding"]["files_with_embedding_failures"],
                1
            );
            assert_eq!(snapshot_graph(&pool).await, before);
            embedder.fail.store(false, Ordering::SeqCst);
            ingest(&pool, &embedder, tree.path()).await;
            validate(&pool, &embedder).await;
        },
    )
    .await;
}

#[tokio::test]
async fn relationship_failure_is_retried_without_force() {
    with_temp_db(
        "relationship_retry",
        Fixtures::Codebase,
        |pool| async move {
            let embedder = Embedder::new().await;
            let tree = tempfile::tempdir().unwrap();
            std::fs::write(
                tree.path().join("provider.py"),
                "def target():\n    return 'quartz'\n",
            )
            .unwrap();
            std::fs::write(
                tree.path().join("consumer.py"),
                "from provider import target\n\ndef caller():\n    return target()\n",
            )
            .unwrap();
            pool.writer().put("collection/codebase_calls_edges/properties", &json!({
            "schema":{"level":"strict","rule":{"type":"object","required":["fault_marker"]}}
        })).await.unwrap();
            let failure = cli(
                &pool,
                &embedder,
                &["ingest", tree.path().to_str().unwrap()],
                false,
            )
            .await;
            assert_eq!(failure["code"]["import_edges"], 0);
            assert_eq!(failure["code"]["python_call_edges"], 0);
            assert!(
                failure["code"]["relationship_error"]
                    .as_str()
                    .unwrap()
                    .contains("failed to atomically store")
            );

            for name in ["provider.py", "consumer.py"] {
                let key = keys::scoped_file_key(tree.path().to_str().unwrap(), name);
                assert_eq!(
                    pool.reader()
                        .get(&format!("document/codebase_files/{key}"))
                        .await
                        .unwrap()["relationships_pending"],
                    true
                );
            }
            for collection in ["codebase_imports_edges", "codebase_calls_edges"] {
                assert_eq!(
                    hades_core::db::crud::count_collection(&pool, collection)
                        .await
                        .unwrap(),
                    0
                );
            }
            pool.writer()
                .put(
                    "collection/codebase_calls_edges/properties",
                    &json!({"schema":null}),
                )
                .await
                .unwrap();
            let retry = ingest(&pool, &embedder, tree.path()).await;
            assert_eq!(retry["code"]["skipped"], 0);
            assert!(
                retry["code"]["python_call_edges"].as_u64().unwrap() > 0,
                "{retry}"
            );
            for name in ["provider.py", "consumer.py"] {
                let key = keys::scoped_file_key(tree.path().to_str().unwrap(), name);
                assert_eq!(
                    pool.reader()
                        .get(&format!("document/codebase_files/{key}"))
                        .await
                        .unwrap()["relationships_pending"],
                    false
                );
            }
            validate(&pool, &embedder).await;
            assert_eq!(
                ingest(&pool, &embedder, tree.path()).await["code"]["skipped"],
                2
            );
        },
    )
    .await;
}

#[tokio::test]
async fn changed_consumer_keeps_relationships_to_unchanged_provider() {
    with_temp_db("unchanged_provider", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        std::fs::write(tree.path().join("provider.py"), "def target():\n    return 'quartz'\n").unwrap();
        let consumer = tree.path().join("consumer.py");
        std::fs::write(&consumer, "from provider import target\n\ndef caller():\n    return target()\n").unwrap();
        ingest(&pool, &embedder, tree.path()).await;
        let before = snapshot_graph(&pool).await;
        assert!(!before["codebase_calls_edges"].as_array().unwrap().is_empty());
        std::fs::write(&consumer, "from provider import target\n\ndef caller():\n    # revised consumer body\n    return target()\n").unwrap();
        let updated = ingest(&pool, &embedder, tree.path()).await;
        assert_eq!(updated["code"]["skipped"], 1);
        let after = snapshot_graph(&pool).await;
        assert_eq!(after["codebase_calls_edges"], before["codebase_calls_edges"], "unchanged provider must remain a resolution target");
        assert_eq!(after["codebase_imports_edges"], before["codebase_imports_edges"]);
        validate(&pool, &embedder).await;
    }).await;
}

#[tokio::test]
async fn killed_cli_during_embedding_preserves_graph_and_retries() {
    with_temp_db(
        "killed_preparation",
        Fixtures::Codebase,
        |pool| async move {
            let embedder = Embedder::new().await;
            let tree = tempfile::tempdir().unwrap();
            let file = tree.path().join("provider.py");
            std::fs::write(&file, "def target():\n    return 'quartz'\n").unwrap();
            ingest(&pool, &embedder, tree.path()).await;
            let before = snapshot_graph(&pool).await;
            std::fs::write(&file, "# move\ndef target():\n    return 'sapphire'\n").unwrap();
            embedder.hold.store(true, Ordering::SeqCst);
            let mut child =
                cli_command(&pool, &embedder, &["ingest", tree.path().to_str().unwrap()])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .unwrap();
            tokio::time::timeout(
                std::time::Duration::from_secs(10),
                embedder.entered.notified(),
            )
            .await
            .expect("CLI never reached the controlled embedding request");
            child.start_kill().unwrap();
            let status = tokio::time::timeout(std::time::Duration::from_secs(10), child.wait())
                .await
                .expect("test CLI did not exit")
                .unwrap();
            assert!(!status.success());
            assert_eq!(
                snapshot_graph(&pool).await,
                before,
                "process interruption during preparation must preserve all collections"
            );
            embedder.hold.store(false, Ordering::SeqCst);
            embedder.release.notify_one();
            let retry = ingest(&pool, &embedder, tree.path()).await;
            assert_eq!(retry["code"]["failed"], 0);
            validate(&pool, &embedder).await;
            let key = keys::scoped_file_key(tree.path().to_str().unwrap(), "provider.py");
            assert_hit(
                &search(&pool, &embedder, "sapphire").await,
                &key,
                "sapphire",
            );
        },
    )
    .await;
}

#[tokio::test]
async fn explicitly_unparsed_provider_preserves_registered_language_targets() {
    with_temp_db("unparsed_target", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path().to_str().unwrap();
        std::fs::write(
            tree.path().join("provider.legacy"),
            "def target():\n    return 'quartz'\n",
        )
        .unwrap();
        let consumer = tree.path().join("consumer.py");
        std::fs::write(
            &consumer,
            "from provider import target\n\ndef caller():\n    return target()\n",
        )
        .unwrap();
        cli(
            &pool,
            &embedder,
            &[
                "codebase",
                "ingest",
                tree.path().join("provider.legacy").to_str().unwrap(),
                "--language",
                "python",
            ],
            true,
        )
        .await;
        cli(
            &pool,
            &embedder,
            &["codebase", "ingest", root, "--unparsed-ext", "legacy"],
            true,
        )
        .await;
        let before = snapshot_graph(&pool).await;
        assert!(
            !before["codebase_calls_edges"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        std::fs::write(
            &consumer,
            "from provider import target\n\ndef caller():\n    # changed\n    return target()\n",
        )
        .unwrap();
        let result = cli(
            &pool,
            &embedder,
            &["codebase", "ingest", root, "--unparsed-ext", "legacy"],
            true,
        )
        .await;
        assert_eq!(result["skipped"], 1);
        let after = snapshot_graph(&pool).await;
        assert_eq!(
            after["codebase_calls_edges"],
            before["codebase_calls_edges"]
        );
        assert_eq!(
            after["codebase_imports_edges"],
            before["codebase_imports_edges"]
        );
        let key = keys::scoped_file_key(root, "provider.legacy");
        let provider = pool
            .reader()
            .get(&format!("document/codebase_files/{key}"))
            .await
            .unwrap();
        assert_eq!(provider["analysis_tier"], "semantic");
        assert_eq!(provider["language"], "Python");
        validate(&pool, &embedder).await;
    })
    .await;
}

#[tokio::test]
async fn analyzer_failure_retains_summary_and_failure_envelope() {
    use std::os::unix::fs::PermissionsExt;
    with_temp_db("analyzer_failure", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        std::fs::create_dir(tree.path().join("src")).unwrap();
        std::fs::write(tree.path().join("Cargo.toml"), "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n").unwrap();
        std::fs::write(tree.path().join("src/lib.rs"), "pub fn target() -> u32 { 7 }\n").unwrap();
        let tools = tempfile::tempdir().unwrap();
        let analyzer = tools.path().join("failing-analyzer");
        std::fs::write(&analyzer, "#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'rust-analyzer fixture'; exit 0; fi\nexit 1\n").unwrap();
        std::fs::set_permissions(&analyzer, std::fs::Permissions::from_mode(0o700)).unwrap();
        for direct in [true, false] {
            let root = tree.path().to_str().unwrap();
            let args = if direct { vec!["codebase", "ingest", root] } else { vec!["ingest", root] };
            let output = tokio::time::timeout(std::time::Duration::from_secs(30),
                cli_command(&pool, &embedder, &args).env("HADES_RUST_ANALYZER_PATH", &analyzer).output())
                .await.expect("failing analyzer did not terminate promptly").unwrap();
            assert!(!output.status.success());
            let report: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| panic!("missing JSON summary: {error}; stderr={}", String::from_utf8_lossy(&output.stderr)));
            assert_eq!(report["success"], false, "{report}");
            let code = if direct { &report["data"] } else { &report["data"]["code"] };
            assert_eq!(code["completed"], 1, "file commit must remain visible: {report}");
            assert!(code["enrichment_error"].as_str().unwrap().contains("rust-analyzer"), "{report}");
            assert_eq!(code["rust_analyzer"]["crates_analyzed"], 0);
        }
        assert!(hades_core::db::crud::count_collection(&pool, "codebase_chunks").await.unwrap() > 0);
        validate(&pool, &embedder).await;
    }).await;
}

#[tokio::test]
async fn partial_extraction_failure_is_not_marked_as_empty_success() {
    use std::os::unix::fs::PermissionsExt;
    with_temp_db(
        "partial_extraction",
        Fixtures::Codebase,
        |pool| async move {
            let embedder = Embedder::new().await;
            let tree = tempfile::tempdir().unwrap();
            std::fs::create_dir(tree.path().join("src")).unwrap();
            std::fs::write(
                tree.path().join("Cargo.toml"),
                "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
            )
            .unwrap();
            std::fs::write(tree.path().join("src/lib.rs"), "mod broken;\n").unwrap();
            std::fs::write(
                tree.path().join("src/broken.rs"),
                "pub fn target() -> u32 { 7 }\n",
            )
            .unwrap();
            let scripts = tempfile::tempdir().unwrap();
            let analyzer = scripts.path().join("partial-analyzer");
            std::fs::write(&analyzer, include_str!("fixtures/partial_analyzer.py")).unwrap();
            std::fs::set_permissions(&analyzer, std::fs::Permissions::from_mode(0o700)).unwrap();
            let root = tree.path().to_str().unwrap();
            let output = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                cli_command(&pool, &embedder, &["codebase", "ingest", root])
                    .env("HADES_RUST_ANALYZER_PATH", &analyzer)
                    .output(),
            )
            .await
            .expect("partial analyzer fixture timed out")
            .unwrap();
            assert!(
                !output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
            let report: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(report["success"], false);
            let data = &report["data"];
            assert_eq!(data["completed"], 2);
            assert_eq!(data["rust_analyzer"]["crates_analyzed"], 1);
            assert_eq!(
                data["rust_analyzer"]["failed_files"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
            assert!(
                data["enrichment_error"]
                    .as_str()
                    .unwrap()
                    .contains("src/broken.rs")
            );
            for (path, expected) in [("src/lib.rs", json!(true)), ("src/broken.rs", Value::Null)] {
                let key = keys::scoped_file_key(root, path);
                assert_eq!(
                    pool.reader()
                        .get(&format!("document/codebase_files/{key}"))
                        .await
                        .unwrap()["ra_analyzed"],
                    expected
                );
            }
            validate(&pool, &embedder).await;
        },
    )
    .await;
}

#[tokio::test]
async fn failed_workspace_is_not_hidden_by_another_successful_workspace() {
    use std::os::unix::fs::PermissionsExt;
    with_temp_db("partial_workspace", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        for name in ["successful", "failed"] {
            let root = tree.path().join(name);
            std::fs::create_dir_all(root.join("src")).unwrap();
            std::fs::write(
                root.join("Cargo.toml"),
                format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n"),
            )
            .unwrap();
            std::fs::write(root.join("src/lib.rs"), "pub fn target() -> u32 { 7 }\n").unwrap();
        }
        let scripts = tempfile::tempdir().unwrap();
        let analyzer = scripts.path().join("partial-analyzer");
        std::fs::write(&analyzer, include_str!("fixtures/partial_analyzer.py")).unwrap();
        std::fs::set_permissions(&analyzer, std::fs::Permissions::from_mode(0o700)).unwrap();
        let root = tree.path().to_str().unwrap();
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            cli_command(&pool, &embedder, &["codebase", "ingest", root])
                .env("HADES_RUST_ANALYZER_PATH", &analyzer)
                .output(),
        )
        .await
        .expect("workspace fixture timed out")
        .unwrap();
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(!output.status.success(), "{report}");
        assert_eq!(report["success"], false);
        let data = &report["data"];
        assert_eq!(data["completed"], 2);
        assert_eq!(data["rust_analyzer"]["crates_analyzed"], 1);
        assert_eq!(
            data["rust_analyzer"]["failed_workspaces"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert!(
            data["enrichment_error"]
                .as_str()
                .unwrap()
                .contains("failed/src/lib.rs")
        );
        validate(&pool, &embedder).await;
    })
    .await;
}

#[tokio::test]
async fn killed_cli_after_transactional_chunk_write_rolls_back_and_retries() {
    with_temp_db(
        "cli_transaction_death",
        Fixtures::Codebase,
        |pool| async move {
            let embedder = Embedder::new().await;
            let tree = tempfile::tempdir().unwrap();
            let file = tree.path().join("provider.py");
            std::fs::write(&file, "def target():\n    return 'quartz'\n").unwrap();
            std::fs::write(
                tree.path().join("consumer.py"),
                "from provider import target\n\ndef caller():\n    return target()\n",
            )
            .unwrap();
            ingest(&pool, &embedder, tree.path()).await;
            let before = snapshot_graph(&pool).await;
            std::fs::write(&file, "# moved\ndef target():\n    return 'sapphire'\n").unwrap();

            // Transparent private UDS proxy: forward transaction headers unchanged,
            // but withhold one response after ArangoDB acknowledges replacement chunks.
            let directory = tempfile::tempdir().unwrap();
            let socket = directory.path().join("database.sock");
            let listener = tokio::net::UnixListener::bind(&socket).unwrap();
            let client = reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .unix_socket(pool.writer().socket_path().unwrap().to_path_buf())
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap();
            let entered = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let paused = Arc::new(AtomicBool::new(false));
            let signal = entered.clone();
            let gate = release.clone();
            let app = Router::new().fallback(move |request: axum::extract::Request| {
                let client = client.clone();
                let signal = signal.clone();
                let gate = gate.clone();
                let paused = paused.clone();
                async move {
                    let (parts, body) = request.into_parts();
                    let intercept = parts.method == axum::http::Method::POST
                        && parts.uri.path().ends_with("/document/codebase_chunks")
                        && parts.headers.contains_key("x-arango-trx-id");
                    let body = axum::body::to_bytes(body, 32 * 1024 * 1024).await.unwrap();
                    let response = client
                        .request(
                            parts.method,
                            format!("http://localhost{}", parts.uri.path_and_query().unwrap()),
                        )
                        .headers(parts.headers)
                        .body(body)
                        .send()
                        .await
                        .unwrap();
                    let status = response.status();
                    let headers = response.headers().clone();
                    let bytes = response.bytes().await.unwrap();
                    if intercept && !paused.swap(true, Ordering::SeqCst) {
                        assert!(status.is_success());
                        let rows: Value = serde_json::from_slice(&bytes).unwrap();
                        assert!(
                            rows.as_array()
                                .unwrap()
                                .iter()
                                .all(|row| row["error"] != true)
                        );
                        signal.notify_one();
                        gate.notified().await;
                    }
                    let mut response = axum::http::Response::new(axum::body::Body::from(bytes));
                    *response.status_mut() = status;
                    *response.headers_mut() = headers;
                    response
                }
            });
            let proxy = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let mut child =
                cli_command(&pool, &embedder, &["ingest", tree.path().to_str().unwrap()])
                    .env("ARANGO_RO_SOCKET", &socket)
                    .env("ARANGO_RW_SOCKET", &socket)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(15), entered.notified())
                .await
                .expect("CLI never reached an acknowledged transactional chunk write");
            child.start_kill().unwrap();
            assert!(
                !tokio::time::timeout(std::time::Duration::from_secs(10), child.wait())
                    .await
                    .unwrap()
                    .unwrap()
                    .success()
            );
            release.notify_one();
            proxy.abort();
            let _ = proxy.await;

            // No ingestion cleanup task survives the process. Acquisition of all
            // collection locks proves the abandoned transaction has released them.
            tokio::time::timeout(std::time::Duration::from_secs(100), async {
                loop {
                    let collections = hades_core::db::collections::CODEBASE
                        .all_collections()
                        .iter()
                        .map(|(name, _)| name.to_string())
                        .collect();
                    if hades_core::db::transaction::run(&pool, collections, |_| async { Ok(()) })
                        .await
                        .is_ok()
                    {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                }
            })
            .await
            .expect("abandoned CLI transaction did not release its locks");
            assert_eq!(
                snapshot_graph(&pool).await,
                before,
                "all collections must survive process death during replacement"
            );
            ingest(&pool, &embedder, tree.path()).await;
            validate(&pool, &embedder).await;
            let key = keys::scoped_file_key(tree.path().to_str().unwrap(), "provider.py");
            assert_hit(
                &search(&pool, &embedder, "sapphire").await,
                &key,
                "sapphire",
            );
        },
    )
    .await;
}
