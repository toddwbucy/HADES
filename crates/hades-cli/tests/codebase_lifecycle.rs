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
        let fail = Arc::new(AtomicBool::new(false));
        let fault = fail.clone();
        let app = Router::new()
            .route("/v1/models", get(|| async {
                Json(json!({"data":[{"id":MODEL,"dimension":2048,"max_seq_length":8192,"device":"cpu"}]}))
            }))
            .route("/v1/embeddings", post(move |Json(body): Json<Value>| { let fault = fault.clone(); async move {
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
