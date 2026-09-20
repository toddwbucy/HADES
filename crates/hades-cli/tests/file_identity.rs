//! Run only against the explicitly isolated test database harness.
use hades_core::db::query::{ExecutionTarget, query};
use hades_core::db::{ArangoPool, crud, keys};
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

fn cli(pool: &ArangoPool, args: &[&str]) -> Output {
    let isolation = tempfile::tempdir().unwrap();
    Command::new(env!("CARGO_BIN_EXE_hades"))
        .args(["--db", pool.database()])
        .args(args)
        .env(
            "HADES_EMBEDDER_SOCKET",
            isolation.path().join("absent-embedder.sock"),
        )
        .env(
            "HADES_EXTRACTOR_SOCKET",
            isolation.path().join("absent-extractor.sock"),
        )
        .output()
        .unwrap()
}

fn ingest(pool: &ArangoPool, root: &Path) {
    let result = cli(pool, &["codebase", "ingest", root.to_str().unwrap()]);
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stderr),
        String::from_utf8_lossy(&result.stdout)
    );
}

async fn rows(pool: &ArangoPool, aql: &str, bind: Value) -> Vec<Value> {
    query(pool, aql, Some(&bind), None, false, ExecutionTarget::Reader)
        .await
        .unwrap()
        .results
}

async fn assert_file(pool: &ArangoPool, root: &Path, relative: &str) -> String {
    let key = keys::scoped_file_key(root.to_str().unwrap(), relative);
    let file = crud::get_document(pool, "codebase_files", &key)
        .await
        .unwrap();
    assert_eq!(file["path"], relative);
    assert_eq!(file["ingest_root"], root.to_str().unwrap());
    assert_eq!(file["file_key_version"], 2);
    for collection in ["codebase_symbols", "codebase_chunks"] {
        let owned = rows(
            pool,
            "FOR d IN @@col FILTER d.file_key == @key RETURN d._key",
            json!({"@col":collection,"key":key}),
        )
        .await;
        assert!(!owned.is_empty(), "{relative} lost {collection}");
    }
    key
}

#[tokio::test]
async fn identities_survive_reingest_modify_and_retire_across_roots() {
    with_temp_db("file_identity", Fixtures::Codebase, |pool| async move {
        let tree = tempfile::tempdir().unwrap();
        let first = tree.path().join("first");
        let second = tree.path().join("second");
        std::fs::create_dir_all(&second).unwrap();
        let paths = vec![
            "a/b.py".to_string(),
            "a_b.py".into(),
            "a.b.py".into(),
            "λ.py".into(),
            format!(
                "{}/{}/{}/long.py",
                "x".repeat(100),
                "y".repeat(100),
                "z".repeat(100)
            ),
        ];
        for relative in &paths {
            let path = first.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "def retained():\n    return 1\n").unwrap();
        }
        std::fs::write(second.join("a_b.py"), "def second_root():\n    return 2\n").unwrap();
        for _ in 0..2 {
            ingest(&pool, &first);
            ingest(&pool, &second);
            let count = rows(&pool, "RETURN LENGTH(codebase_files)", json!({})).await;
            assert_eq!(count, vec![json!(6)]);
            for relative in &paths {
                assert_file(&pool, &first, relative).await;
            }
            assert_file(&pool, &second, "a_b.py").await;
        }
        std::fs::write(
            first.join("a/b.py"),
            "def retained():\n    return 99\n\ndef added():\n    return 3\n",
        )
        .unwrap();
        ingest(&pool, &first);
        for relative in &paths {
            assert_file(&pool, &first, relative).await;
        }
        assert_file(&pool, &second, "a_b.py").await;
        let key = keys::scoped_file_key(first.to_str().unwrap(), "a/b.py");
        let symbols = rows(
            &pool,
            "FOR s IN codebase_symbols FILTER s.file_key == @key RETURN s.name",
            json!({"key":key}),
        )
        .await;
        assert!(symbols.contains(&json!("added")));
        let drift = cli(
            &pool,
            &["codebase", "drift", first.to_str().unwrap(), "--full"],
        );
        assert!(
            drift.status.success(),
            "{}",
            String::from_utf8_lossy(&drift.stderr)
        );
        let report: Value = serde_json::from_slice(&drift.stdout).unwrap();
        assert_eq!(report["data"]["matched"], 5);
        assert_eq!(report["data"]["stale"]["count"], 0);
        assert_eq!(report["data"]["other_roots"]["count"], 1);
        std::fs::remove_file(first.join("a/b.py")).unwrap();
        let drift = cli(
            &pool,
            &["codebase", "drift", first.to_str().unwrap(), "--full"],
        );
        assert!(drift.status.success());
        let report: Value = serde_json::from_slice(&drift.stdout).unwrap();
        assert_eq!(report["data"]["stale"]["keys"], json!([key]));
        let result = cli(&pool, &["codebase", "retire", "--file", &key, "--yes"]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        for collection in ["codebase_symbols", "codebase_chunks", "codebase_embeddings"] {
            assert!(
                rows(
                    &pool,
                    "FOR d IN @@col FILTER d.file_key == @key RETURN d._key",
                    json!({"@col":collection,"key":key})
                )
                .await
                .is_empty()
            );
        }
        for relative in &paths[1..] {
            assert_file(&pool, &first, relative).await;
        }
        assert_file(&pool, &second, "a_b.py").await;
    })
    .await;
}

#[tokio::test]
async fn conflicting_or_legacy_identity_is_rejected_before_purge() {
    with_temp_db("identity_guard", Fixtures::Codebase, |pool| async move {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("a.py"), "def kept():\n    return 1\n").unwrap();
        ingest(&pool, root.path());
        let key = keys::scoped_file_key(root.path().to_str().unwrap(), "a.py");
        let symbols = rows(&pool, "FOR s IN codebase_symbols RETURN s", json!({})).await;
        let mut file = crud::get_document(&pool, "codebase_files", &key)
            .await
            .unwrap();
        file["path"] = json!("different.py");
        file.as_object_mut().unwrap().remove("_rev");
        file.as_object_mut().unwrap().remove("_id");
        crud::insert_documents(&pool, "codebase_files", &[file.clone()], true)
            .await
            .unwrap();
        let output = cli(
            &pool,
            &[
                "codebase",
                "ingest",
                root.path().to_str().unwrap(),
                "--force",
            ],
        );
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("identity conflict"));
        assert_eq!(
            symbols,
            rows(&pool, "FOR s IN codebase_symbols RETURN s", json!({})).await
        );
        file.as_object_mut().unwrap().remove("file_key_version");
        crud::insert_documents(&pool, "codebase_files", &[file], true)
            .await
            .unwrap();
        let output = cli(
            &pool,
            &[
                "codebase",
                "ingest",
                root.path().to_str().unwrap(),
                "--force",
            ],
        );
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("explicit migration"));
        let drift = cli(
            &pool,
            &["codebase", "drift", root.path().to_str().unwrap(), "--full"],
        );
        assert!(!drift.status.success());
        assert!(String::from_utf8_lossy(&drift.stderr).contains("explicit migration"));
        assert!(
            drift.stdout.is_empty(),
            "legacy drift must emit no retirement candidates"
        );
        assert_eq!(
            symbols,
            rows(&pool, "FOR s IN codebase_symbols RETURN s", json!({})).await
        );
    })
    .await;
}

#[tokio::test]
async fn call_and_import_edges_stay_inside_their_root_namespace() {
    with_temp_db("identity_edges", Fixtures::Codebase, |pool| async move {
        let trees = tempfile::tempdir().unwrap();
        let roots = [trees.path().join("one"), trees.path().join("two")];
        for root in &roots {
            std::fs::create_dir(root).unwrap();
            std::fs::write(root.join("provider.py"), "def target():\n    return 1\n").unwrap();
            std::fs::write(root.join("consumer.py"), "from provider import target\n\ndef caller():\n    return target()\n").unwrap();
            ingest(&pool, root);
        }
        for iteration in 0..2 {
            if iteration == 1 {
                // Move the definition line while the dependent remains unchanged.
                std::fs::write(roots[0].join("provider.py"), "# shifted\n\ndef target():\n    return 2\n").unwrap();
                ingest(&pool, &roots[0]);
                ingest(&pool, &roots[1]);
            }
            for collection in ["codebase_calls_edges", "codebase_imports_edges"] {
                let edges = rows(&pool,
                    "FOR e IN @@edges LET a = DOCUMENT(e._from) LET b = DOCUMENT(e._to) RETURN { a, b }",
                    json!({"@edges": collection})).await;
                assert!(!edges.is_empty(), "fixture must exercise {collection}");
                let mut seen = std::collections::BTreeSet::new();
                for edge in edges {
                    let mut owners = Vec::new();
                    for endpoint in ["a", "b"] {
                        let node = &edge[endpoint];
                        assert!(!node.is_null(), "dangling {collection} endpoint: {edge}");
                        let file_key = node["file_key"].as_str().or_else(|| node["_key"].as_str()).unwrap();
                        let file = crud::get_document(&pool, "codebase_files", file_key).await.unwrap();
                        owners.push(file["ingest_root"].as_str().unwrap().to_owned());
                    }
                    assert_eq!(owners[0], owners[1], "cross-root edge in {collection}");
                    seen.insert(owners[0].clone());
                }
                assert_eq!(seen.len(), 2, "both roots must retain {collection}");
            }
        }
    }).await;
}
