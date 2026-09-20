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
use tokio::process::Command;
use tokio::task::JoinHandle;

const MODEL: &str = "jinaai/jina-embeddings-v4";

struct Embedder {
    socket: PathBuf,
    task: JoinHandle<()>,
    _directory: tempfile::TempDir,
}
impl Drop for Embedder {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Embedder {
    async fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("embedder.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let app = Router::new()
            .route("/v1/models", get(|| async {
                Json(json!({"data":[{"id":MODEL,"dimension":2048,"max_seq_length":8192,"device":"cpu"}]}))
            }))
            .route("/v1/embeddings", post(|Json(body): Json<Value>| async move {
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
            }));
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
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

async fn cli(pool: &ArangoPool, embedder: &Embedder, args: &[&str], success: bool) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_hades"))
        .args(["--db", pool.database()])
        .args(args)
        .env("HADES_EMBEDDER_SOCKET", &embedder.socket)
        .env(
            "HADES_EXTRACTOR_SOCKET",
            embedder._directory.path().join("absent-extractor.sock"),
        )
        .env_remove("HADES_DISABLE_LATE_CHUNKING")
        .env_remove("HADES_DEFAULT_COLLECTION")
        .kill_on_drop(true)
        .output()
        .await
        .unwrap();
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
