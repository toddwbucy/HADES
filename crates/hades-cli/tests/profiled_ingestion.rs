//! Exercise the real foreground CLI with a disposable database and private service fixtures.
use hades_core::{
    config::HadesConfig,
    db::{ArangoPool, query},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::Value;
use std::path::Path;
use tokio::process::Command;

#[allow(dead_code)]
#[path = "../../hades-core/tests/common/profiled_extractor.rs"]
mod extraction;
#[allow(dead_code)]
#[path = "../../hades-core/tests/common/profiled_provider.rs"]
mod provider;
use extraction::Extractor;
use provider::{Fixture, Peer};

/// Freeze fixture routes while deliberately making the legacy embedder unreachable.
fn config(
    root: &Path,
    pool: &ArangoPool,
    peer: &Peer,
    extractor: &Extractor,
) -> std::path::PathBuf {
    let path = root.join("hades.yaml");
    let mut config = HadesConfig::with_database(pool.database());
    config.registered_providers = Some(peer.configuration());
    config.extraction.service.socket = extractor.path.to_str().unwrap().into();
    config.embedding.model.name = "wrong/legacy-model".into();
    config.embedding.model.dimension = 777;
    config.chunking.document_late_chunking = true;
    config.batch_processing.concurrency = 1;
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    path
}
/// Each subprocess receives only the current harness-owned database endpoints.
fn command(pool: &ArangoPool, config: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_hades"));
    command
        .args(["--db", pool.database()])
        .args(args)
        .current_dir(config.parent().unwrap())
        .env("HADES_CONFIG", config)
        .env("ARANGO_RO_SOCKET", pool.reader().socket_path().unwrap())
        .env("ARANGO_RW_SOCKET", pool.writer().socket_path().unwrap())
        .env("HADES_EMBEDDER_SOCKET", "/absent/legacy-provider.sock")
        .env_remove("HADES_EXTRACTOR_SOCKET")
        .env_remove("HADES_DEFAULT_COLLECTION")
        .kill_on_drop(true);
    command
}
/// Inspect committed sources, including their Arango revisions.
async fn sources(pool: &ArangoPool) -> Vec<Value> {
    query::query(
        pool,
        "FOR d IN documents SORT d._key RETURN d",
        None,
        None,
        false,
        query::ExecutionTarget::Writer,
    )
    .await
    .unwrap()
    .results
}

/// The CLI preserves saved model policy, returns binding evidence, and never consumes a legacy checkpoint.
#[tokio::test]
async fn named_files_use_bound_policy_and_atomic_namespaced_metadata() {
    with_temp_db("cli_profiled_ingest", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let extractor = Extractor::new();
        let root = tempfile::tempdir().unwrap();
        let config = config(root.path(), &pool, &peer, &extractor);
        let first = root.path().join("notes.md");
        let second = root.path().join("notes.txt");
        std::fs::write(&first, "apple café\nbanana").unwrap();
        std::fs::write(&second, "banana 東京").unwrap();
        let checkpoint = root.path().join(".hades-batch-state.json");
        std::fs::write(&checkpoint, "legacy checkpoint must remain untouched").unwrap();
        let output = command(
            &pool,
            &config,
            &[
                "ingest",
                "notes.md",
                "notes.txt",
                "--root",
                root.path().to_str().unwrap(),
                "--metadata",
                r#"{"_key":"foreign","embedding_binding":{"dimension":777},"title":"Notes"}"#,
            ],
        )
        .output()
        .await
        .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["data"]["completed"], 2);
        assert_eq!(result["data"]["embedding_profile_state"], "ready");
        assert_eq!(
            std::fs::read_to_string(&checkpoint).unwrap(),
            "legacy checkpoint must remain untouched"
        );
        let saved = sources(&pool).await;
        assert_eq!(saved.len(), 2);
        assert_ne!(saved[0]["_key"], saved[1]["_key"]);
        for row in &saved {
            assert_eq!(row["embedding_binding"]["dimension"], 2);
            assert_eq!(row["file_ingestion"]["metadata"]["title"], "Notes");
            assert_ne!(row["_key"], "foreign");
        }
        let output = command(&pool, &config, &["db", "query", "apple"])
            .output()
            .await
            .unwrap();
        assert!(output.status.success());
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["data"]["results"][0]["text"], "apple café\n");
        std::fs::write(&first, "apple shorter").unwrap();
        let output = command(
            &pool,
            &config,
            &[
                "ingest",
                "notes.md",
                "--root",
                root.path().to_str().unwrap(),
                "--force",
            ],
        )
        .output()
        .await
        .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(sources(&pool).await.len(), 2);
        for row in sources(&pool).await {
            assert_eq!(row["file_ingestion"]["metadata"]["title"], "Notes");
        }
    })
    .await;
}

/// Named batches reject policy overrides and code before any extractor or model observation.
#[tokio::test]
async fn unsupported_overrides_and_unpinned_checkpoints_are_refused_before_services() {
    with_temp_db(
        "cli_profiled_ingest_policy",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let extractor = Extractor::new();
            let root = tempfile::tempdir().unwrap();
            let config = config(root.path(), &pool, &peer, &extractor);
            std::fs::write(root.path().join("notes.md"), "apple").unwrap();
            std::fs::write(root.path().join("code.py"), "print('banana')").unwrap();
            for args in [
                vec!["ingest", "notes.md", "--task", "retrieval.passage"],
                vec!["ingest", "notes.md", "--claims", "CS-01"],
                vec!["ingest", "notes.md", "--batch"],
                vec!["ingest", "notes.md", "--resume"],
                vec!["ingest", "notes.md", "--reset"],
                vec!["ingest", "notes.md", "--concurrency", "5"],
                vec!["ingest", "notes.md", "code.py"],
            ] {
                let output = command(&pool, &config, &args).output().await.unwrap();
                assert!(!output.status.success(), "{args:?}");
                let message = String::from_utf8_lossy(&output.stderr);
                assert!(
                    message.contains("model-bound") || message.contains("disabled_by_profile"),
                    "{args:?}: {message}"
                );
            }
            assert_eq!(peer.observations(), 0);
            assert!(extractor.requests().is_empty());
            assert!(sources(&pool).await.is_empty());
        },
    )
    .await;
}

/// Documents-only directories share the new path without creating any code schema.
#[tokio::test]
async fn document_only_tree_preserves_relative_identities() {
    with_temp_db(
        "cli_profiled_ingest_tree",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let extractor = Extractor::new();
            let root = tempfile::tempdir().unwrap();
            let config = config(root.path(), &pool, &peer, &extractor);
            let tree = root.path().join("corpus");
            std::fs::create_dir_all(tree.join("nested")).unwrap();
            std::fs::write(tree.join("README.md"), "apple").unwrap();
            std::fs::write(tree.join("nested/README.md"), "banana").unwrap();
            let output = command(&pool, &config, &["ingest", tree.to_str().unwrap()])
                .output()
                .await
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["data"]["documents"]["completed"], 2);
            assert_eq!(
                result["data"]["documents"]["embedding_profile_state"],
                "ready"
            );
            let saved = sources(&pool).await;
            assert_eq!(saved.len(), 2);
            let mut ids: Vec<_> = saved
                .iter()
                .map(|d| d["source_id"].as_str().unwrap())
                .collect();
            ids.sort();
            assert_eq!(
                ids,
                ["relative-file:README.md", "relative-file:nested/README.md"]
            );
        },
    )
    .await;
}

/// An extraction failure is a failed item and nonzero process exit; another source may succeed.
#[tokio::test]
async fn partial_file_failure_has_honest_results_and_no_partial_source() {
    with_temp_db(
        "cli_profiled_ingest_partial",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let extractor = Extractor::new();
            let root = tempfile::tempdir().unwrap();
            let config = config(root.path(), &pool, &peer, &extractor);
            std::fs::write(root.path().join("first.md"), "apple").unwrap();
            std::fs::write(root.path().join("second.md"), "banana").unwrap();
            extractor.respond(Err(tonic::Status::unavailable("fixture extraction failed")));
            let output = command(&pool, &config, &["ingest", "first.md", "second.md"])
                .output()
                .await
                .unwrap();
            assert!(!output.status.success());
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["success"], false);
            assert_eq!(result["data"]["completed"], 1);
            assert_eq!(result["data"]["failed"], 1);
            let saved = sources(&pool).await;
            assert_eq!(saved.len(), 1);
            assert_eq!(saved[0]["full_text"], "banana");
            assert!(!root.path().join(".hades-batch-state.json").exists());
        },
    )
    .await;
}
