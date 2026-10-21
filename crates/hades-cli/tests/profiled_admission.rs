//! Exercise real CLI entry points against disposable, profile-bound workspaces.
use hades_core::{
    config::HadesConfig,
    db::{ArangoPool, crud, query},
    pipeline::profiled_documents::DocumentInput,
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
use std::path::Path;
use tokio::process::Command;

#[allow(dead_code)]
#[path = "../../hades-core/tests/common/profiled_provider.rs"]
mod provider;
use provider::{Fixture, Peer};

fn command(pool: &ArangoPool, config: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_hades"));
    command
        .args(["--db", pool.database()])
        .args(args)
        .env("HADES_CONFIG", config)
        .env("ARANGO_RO_SOCKET", pool.reader().socket_path().unwrap())
        .env("ARANGO_RW_SOCKET", pool.writer().socket_path().unwrap())
        .env("HADES_EMBEDDER_SOCKET", "/absent/legacy-provider.sock")
        .env("HADES_EXTRACTOR_SOCKET", "/absent/extractor.sock")
        .env_remove("HADES_DEFAULT_COLLECTION")
        .kill_on_drop(true);
    command
}

#[tokio::test]
async fn cli_query_uses_saved_provider_despite_conflicting_legacy_settings() {
    with_temp_db("cli_profiled_query", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        let prepared = documents
            .prepare(
                DocumentInput {
                    source_id: "source".into(),
                    text: "apple café".into(),
                },
                false,
            )
            .await
            .unwrap();
        documents.commit(prepared).await.unwrap();
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("hades.yaml");
        let mut config = HadesConfig::with_database(pool.database());
        config.registered_providers = Some(peer.configuration());
        config.embedding.model.name = "wrong/legacy-model".into();
        config.embedding.model.dimension = 777;
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let output = command(&pool, &config_path, &["db", "query", "apple"])
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["data"]["dimension"], 2);
        assert_eq!(result["data"]["embedding_profile_state"], "ready");
        assert_eq!(result["data"]["results"][0]["text"], "apple café");
    })
    .await;
}

#[tokio::test]
async fn missing_provider_and_legacy_commands_stop_before_services_or_writes() {
    with_temp_db(
        "cli_profiled_refusals",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let directory = tempfile::tempdir().unwrap();
            let config = directory.path().join("hades.yaml");
            std::fs::write(&config, "{}\n").unwrap();
            let prose = directory.path().join("notes.md");
            std::fs::write(&prose, "apple café").unwrap();
            std::fs::write(directory.path().join("module.py"), "print('banana')\n").unwrap();
            let before = crud::list_collections(&pool, false)
                .await
                .unwrap()
                .iter()
                .map(|c| c.name.clone())
                .collect::<Vec<_>>();
            let root = directory.path().to_str().unwrap();
            let absent_checkpoint = directory.path().join("absent-checkpoint");
            // Satisfy the CLI's explicit GPU declaration without reaching a
            // trainer. No graph schema exists, and the update checkpoint is a
            // private missing path, so even a guard regression cannot train.
            for (args, expected) in [
                (
                    vec!["ingest", prose.to_str().unwrap()],
                    "requires registered_providers for document ingestion",
                ),
                (
                    vec!["codebase", "ingest", root],
                    "is unavailable for a model-bound or initializing workspace",
                ),
                (
                    vec!["--gpu", "0", "graph-embed", "train"],
                    "is unavailable for a model-bound or initializing workspace",
                ),
                (
                    vec![
                        "--gpu",
                        "0",
                        "graph-embed",
                        "update",
                        "--checkpoint-dir",
                        absent_checkpoint.to_str().unwrap(),
                    ],
                    "is unavailable for a model-bound or initializing workspace",
                ),
            ] {
                let output = command(&pool, &config, &args).output().await.unwrap();
                assert!(!output.status.success(), "{args:?}");
                let error = format!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(error.contains(expected), "{args:?}: {error}");
            }
            assert_eq!(
                crud::list_collections(&pool, false)
                    .await
                    .unwrap()
                    .iter()
                    .map(|c| c.name.clone())
                    .collect::<Vec<_>>(),
                before
            );
            for collection in ["documents", "chunks", "embeddings"] {
                let result = query::query(
                    &pool,
                    "FOR r IN @@c RETURN r",
                    Some(&json!({"@c":collection})),
                    None,
                    false,
                    query::ExecutionTarget::Writer,
                )
                .await
                .unwrap();
                assert!(result.results.is_empty());
            }
        },
    )
    .await;
}
