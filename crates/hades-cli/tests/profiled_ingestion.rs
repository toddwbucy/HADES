//! Exercise the real foreground CLI with a disposable database and private service fixtures.
use hades_core::{
    config::HadesConfig,
    db::{ArangoPool, query},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
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

/// Include every collection so a skipped code phase cannot quietly initialize its schema.
async fn collection_names(pool: &ArangoPool) -> std::collections::BTreeSet<String> {
    hades_core::db::crud::list_collections(pool, false)
        .await
        .unwrap()
        .into_iter()
        .map(|row| row.name)
        .collect()
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

/// A documents-only profile consumes prose from a mixed tree and names every disabled code file.
#[tokio::test]
async fn mixed_trees_report_disabled_code_without_starting_its_pipeline() {
    for late in [false, true] {
        with_temp_db(
            "cli_profiled_mixed_tree",
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
                let root = tempfile::tempdir().unwrap();
                let config = config(root.path(), &pool, &peer, &extractor);
                let tree = root.path().join("corpus");
                std::fs::create_dir_all(tree.join("nested")).unwrap();
                for (path, source) in [
                    ("README.md", "apple example: `print('apple')`"),
                    ("nested/README.md", "banana 東京"),
                    ("module.py", "print('disabled Python')"),
                    ("nested/lib.rs", "fn main() {}"),
                    (
                        "script",
                        "#!/usr/bin/env python3\nprint('disabled script')\n",
                    ),
                    ("data.custom", "explicit parser-free code input"),
                    ("unknown.ts", "unrouted TypeScript input"),
                ] {
                    std::fs::write(tree.join(path), source).unwrap();
                }
                let before = collection_names(&pool).await;
                let output = command(
                    &pool,
                    &config,
                    &["ingest", tree.to_str().unwrap(), "--unparsed-ext", "custom"],
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
                let data = &result["data"];
                assert_eq!(data["routed"], json!({"code":4,"documents":2,"unrouted":1}));
                assert!(data["code"].is_null());
                assert_eq!(data["documents"]["completed"], 2);
                assert_eq!(data["documents"]["embedding_profile_state"], "ready");
                assert_eq!(
                    data["disabled_by_profile"]["code"],
                    json!([
                        tree.join("data.custom"),
                        tree.join("module.py"),
                        tree.join("nested/lib.rs"),
                        tree.join("script")
                    ])
                );
                assert_eq!(
                    data["unrouted"],
                    json!([{
                        "path":tree.join("unknown.ts"), "reason":"no handler for extension"
                    }])
                );
                assert_eq!(extractor.requests().len(), 2);
                let saved = sources(&pool).await;
                assert_eq!(saved.len(), 2);
                assert!(
                    saved
                        .iter()
                        .any(|row| row["full_text"] == "apple example: `print('apple')`")
                );
                assert_eq!(collection_names(&pool).await, before);
                // A document failure still reports disabled paths and the other document's result.
                extractor.respond(Err(tonic::Status::unavailable("fixture extraction failed")));
                let output = command(
                    &pool,
                    &config,
                    &[
                        "ingest",
                        tree.to_str().unwrap(),
                        "--unparsed-ext",
                        "custom",
                        "--force",
                    ],
                )
                .output()
                .await
                .unwrap();
                assert!(!output.status.success());
                let partial: Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(partial["success"], false);
                assert_eq!(partial["data"]["documents"]["completed"], 1);
                assert_eq!(partial["data"]["documents"]["failed"], 1);
                assert_eq!(
                    partial["data"]["disabled_by_profile"],
                    data["disabled_by_profile"]
                );
                assert_eq!(sources(&pool).await.len(), 2);
                assert_eq!(collection_names(&pool).await, before);
            },
        )
        .await;
    }
}

/// Even a no-op tree needs a ready profile and valid options, but no provider or extractor.
#[tokio::test]
async fn disabled_only_and_empty_trees_validate_admission_without_services() {
    with_temp_db(
        "cli_profiled_empty_tree",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let extractor = Extractor::new();
            let root = tempfile::tempdir().unwrap();
            let config = config(root.path(), &pool, &peer, &extractor);
            let empty = root.path().join("empty");
            let code = root.path().join("code");
            std::fs::create_dir(&empty).unwrap();
            std::fs::create_dir(&code).unwrap();
            std::fs::write(code.join("module.py"), "print('disabled')").unwrap();
            use std::os::unix::ffi::OsStringExt;
            let unusual = code.join(std::ffi::OsString::from_vec(b"bad\xff.py".to_vec()));
            std::fs::write(&unusual, "print('also disabled')").unwrap();
            let before = collection_names(&pool).await;
            for path in [&empty, &code] {
                let output = command(&pool, &config, &["ingest", path.to_str().unwrap()])
                    .output()
                    .await
                    .unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let result: Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(result["success"], true);
                assert_eq!(result["data"]["embedding_profile_state"], "ready");
                assert!(result["data"]["documents"].is_null());
                assert!(result["data"]["code"].is_null());
                assert_eq!(
                    result["data"]["disabled_by_profile"]["code"],
                    if path == &code {
                        json!([
                            unusual.display().to_string(),
                            code.join("module.py").display().to_string()
                        ])
                    } else {
                        json!([])
                    }
                );
                for (extra, expected) in [
                    (
                        vec!["--task", "retrieval.passage"],
                        "does not accept --task or --claims",
                    ),
                    (
                        vec!["--allow-degraded-enrichment"],
                        "does not support degraded code enrichment",
                    ),
                    (vec!["--concurrency", "5"], "requires concurrency 1..=4"),
                    (
                        vec!["--collection", "codebase"],
                        "supports the documents collection profile only",
                    ),
                    (vec!["--metadata", "[]"], "--metadata must be a JSON object"),
                    (vec!["--metadata", "{"], "--metadata must be valid JSON"),
                ] {
                    let mut args = vec!["ingest", path.to_str().unwrap()];
                    args.extend(extra);
                    let output = command(&pool, &config, &args).output().await.unwrap();
                    assert!(!output.status.success(), "{args:?}");
                    let error = String::from_utf8_lossy(&output.stderr);
                    assert!(error.contains(expected), "{args:?}: {error}");
                }
            }
            // Removing the ready pointer cannot turn an empty tree into a successful no-op.
            pool.writer()
                .delete("document/hades_workspace/embedding_profile")
                .await
                .unwrap();
            for path in [&empty, &code] {
                let output = command(&pool, &config, &["ingest", path.to_str().unwrap()])
                    .output()
                    .await
                    .unwrap();
                assert!(!output.status.success());
            }
            assert_eq!(peer.observations(), 0);
            assert!(extractor.requests().is_empty());
            assert!(sources(&pool).await.is_empty());
            assert_eq!(collection_names(&pool).await, before);
        },
    )
    .await;
}

/// No-op admission must reject the same database policy/storage as a real document.
#[tokio::test]
async fn empty_trees_reject_unsupported_document_policy_and_storage() {
    for fault in ["edge-collection", "distance", "implicit-late"] {
        with_temp_db("cli_noop_policy", Fixtures::Empty, move |pool| async move {
            let mut fixture = if fault == "implicit-late" {
                Fixture::late()
            } else {
                Fixture::new(2)
            };
            let workload = fixture.profile.workloads.documents.as_mut().unwrap();
            if fault == "distance" {
                workload.space.representation.distance =
                    hades_core::embedding_profile::Distance::Dot;
            } else if fault == "implicit-late"
                && let hades_core::embedding_profile::ChunkPolicy::Late { passages, .. } =
                    &mut workload.chunking
            {
                *passages = None;
            }
            fixture.receipt.space_id = workload.space.id().unwrap();
            fixture.install(&pool).await;
            if fault == "edge-collection" {
                pool.writer().delete("collection/chunks").await.unwrap();
                pool.writer()
                    .post("collection", &json!({"name":"chunks","type":3}))
                    .await
                    .unwrap();
            }
            let peer = Peer::new(&fixture);
            let extractor = Extractor::new();
            let root = tempfile::tempdir().unwrap();
            let config = config(root.path(), &pool, &peer, &extractor);
            let tree = root.path().join("corpus");
            std::fs::create_dir(&tree).unwrap();
            let expected = match fault {
                "edge-collection" => "document storage requires three document collections",
                "distance" => "supports cosine distance only",
                _ => "late document chunking requires an explicit passage policy",
            };
            for file in [
                None,
                Some("module.py"),
                Some("unknown.ts"),
                Some("notes.md"),
            ] {
                if let Some(file) = file {
                    std::fs::write(tree.join(file), "apple").unwrap();
                }
                let output = command(&pool, &config, &["ingest", tree.to_str().unwrap()])
                    .output()
                    .await
                    .unwrap();
                let error = String::from_utf8_lossy(&output.stderr);
                assert!(
                    !output.status.success(),
                    "{fault}/{file:?} reported success"
                );
                assert!(error.contains(expected), "{fault}/{file:?}: {error}");
            }
            assert_eq!(peer.observations(), 0);
            assert!(extractor.requests().is_empty());
            assert!(sources(&pool).await.is_empty());
        })
        .await;
    }
}

/// A pin records historical admission; failed revalidation must not advertise readiness.
#[tokio::test]
async fn tree_profile_drift_during_document_startup_does_not_report_ready() {
    for removed in [false, true] {
        with_temp_db("cli_tree_drift", Fixtures::Empty, move |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let extractor = Extractor::new();
            let root = tempfile::tempdir().unwrap();
            let config = config(root.path(), &pool, &peer, &extractor);
            let tree = root.path().join("corpus");
            std::fs::create_dir(&tree).unwrap();
            std::fs::write(tree.join("notes.md"), "apple").unwrap();
            let (entered, release) = peer.hold_next_verification();
            let child = command(&pool, &config, &["ingest", tree.to_str().unwrap()])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(5), entered)
                .await
                .unwrap()
                .unwrap();
            let path = "document/hades_workspace/embedding_profile";
            if removed {
                pool.writer().delete(path).await.unwrap();
            } else {
                let original = pool.writer().get(path).await.unwrap();
                pool.writer()
                    .patch(path, &json!({"payload":original["payload"]}))
                    .await
                    .unwrap();
            }
            release.send(()).unwrap();
            let output = child.wait_with_output().await.unwrap();
            assert!(!output.status.success());
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["success"], false);
            assert!(
                result["data"]["document_phase_error"].is_string(),
                "{result}"
            );
            assert!(
                result["data"].get("embedding_profile_state").is_none(),
                "{result}"
            );
            assert!(result["data"]["profile_pin"].is_object());
            assert!(extractor.requests().is_empty());
            assert!(sources(&pool).await.is_empty());
        })
        .await;
    }
}

/// An enabled code workload is unsupported, never mislabeled as disabled and skipped.
#[tokio::test]
async fn enabled_code_and_initializing_profiles_refuse_trees_before_services() {
    for state in ["enabled-code", "documents-disabled", "initializing"] {
        with_temp_db(
            "cli_profiled_tree_policy",
            Fixtures::Empty,
            move |pool| async move {
                let mut fixture = Fixture::new(2);
                if state == "enabled-code" {
                    fixture.profile.workloads.code = fixture.profile.workloads.documents.clone();
                    fixture.install(&pool).await;
                } else if state == "documents-disabled" {
                    fixture.profile.workloads.code = fixture.profile.workloads.documents.take();
                    fixture.install(&pool).await;
                } else {
                    hades_core::embedding_profile::store::ProfileStore::new(pool.clone())
                        .unwrap()
                        .initialize(fixture.profile.clone(), fixture.manifest.clone())
                        .await
                        .unwrap();
                }
                let peer = Peer::new(&fixture);
                let extractor = Extractor::new();
                let root = tempfile::tempdir().unwrap();
                let config = config(root.path(), &pool, &peer, &extractor);
                let tree = root.path().join("corpus");
                std::fs::create_dir(&tree).unwrap();
                std::fs::write(tree.join("README.md"), "apple").unwrap();
                if state != "documents-disabled" {
                    std::fs::write(tree.join("module.py"), "print('banana')").unwrap();
                }
                let before = collection_names(&pool).await;
                let output = command(&pool, &config, &["ingest", tree.to_str().unwrap()])
                    .output()
                    .await
                    .unwrap();
                assert!(!output.status.success());
                let error = String::from_utf8_lossy(&output.stderr);
                assert!(
                    error.contains(match state {
                        "enabled-code" => "enabled code workload cannot be skipped",
                        "documents-disabled" => "documents are disabled_by_profile",
                        _ => "not ready",
                    }),
                    "{error}"
                );
                assert_eq!(peer.observations(), 0);
                assert!(extractor.requests().is_empty());
                assert_eq!(collection_names(&pool).await, before);
                if state != "initializing" {
                    assert!(sources(&pool).await.is_empty());
                }
            },
        )
        .await;
    }
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
