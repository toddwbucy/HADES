//! Real disposable storage, a verified synthetic provider, and private gRPC extraction.
#![cfg(unix)]
use hades_core::{
    db::{ArangoPool, crud, query},
    pipeline::profiled_documents::{
        DocumentInput,
        files::{FileInput, MAX_FILE_BYTES},
    },
    test_support::{Fixtures, with_temp_db},
};
use hades_proto::extraction::{ExtractResponse, SourceType};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc, time::Duration};

#[path = "common/profiled_extractor.rs"]
mod extraction;
#[allow(dead_code)]
#[path = "common/profiled_provider.rs"]
mod provider;
#[allow(dead_code)]
#[path = "common/setup_proxy.rs"]
mod proxy;
use extraction::Extractor;
use provider::{Fixture, Peer};

/// Default file intent, with all embedding settings owned by the database.
fn input(path: &Path) -> FileInput {
    FileInput {
        path: path.into(),
        root: None,
        id: None,
        source_git: None,
        metadata: json!({}),
    }
}
/// Include revisions to catch writes even when replacement content happens to match.
async fn rows(pool: &ArangoPool) -> Value {
    let mut saved = json!({});
    for collection in ["documents", "chunks", "embeddings"] {
        saved[collection] = json!(
            query::query(
                pool,
                "FOR r IN @@c SORT r._key RETURN r",
                Some(&json!({"@c":collection})),
                None,
                false,
                query::ExecutionTarget::Writer
            )
            .await
            .unwrap()
            .results
        );
    }
    saved
}
/// Create a small document inside the test's temporary directory.
fn write(root: &Path, name: &str, content: &str) -> std::path::PathBuf {
    let path = root.join(name);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, content).unwrap();
    path
}

/// Path edits after request admission cannot alter the bytes sent for extraction.
#[tokio::test]
async fn file_snapshot_provenance_and_metadata_commit_with_verified_vectors() {
    with_temp_db(
        "profiled_file_snapshot",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let documents = Arc::new(fixture.connect(&pool, &peer).await);
            let extractor = Extractor::new();
            let client = extractor.client().await;
            let root = tempfile::tempdir().unwrap();
            let text = "apple café\nbanana 東京";
            let path = write(root.path(), "日本語.md", text);
            let mut request = input(&path);
            request.metadata =
                json!({"embedding_binding":{"provider_id":"forged"}, "_key":"forged"});
            request.source_git = Some(hades_core::source_git::SourceGit {
                commit: Some("a".repeat(40)),
                dirty: true,
            });
            let (entered, release) = extractor.hold();
            let worker = documents.clone();
            let task =
                tokio::spawn(async move { worker.ingest_file(request, &client, false).await });
            entered.await.unwrap();
            std::fs::write(&path, "modified while extracting").unwrap();
            release.send(()).unwrap();
            let receipt = task.await.unwrap().unwrap();
            let saved = rows(&pool).await;
            let document = &saved["documents"][0];
            assert_eq!(document["full_text"], text);
            assert_eq!(document["_key"], receipt.document_key);
            assert_eq!(
                document["file_ingestion"]["content_sha256"],
                Sha256::digest(text.as_bytes())
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            );
            assert_eq!(document["file_ingestion"]["content_bytes"], text.len());
            assert_eq!(
                document["file_ingestion"]["source_git"]["commit"],
                "a".repeat(40)
            );
            assert_eq!(document["file_ingestion"]["metadata"]["_key"], "forged");
            assert_eq!(
                document["embedding_binding"]["provider_id"],
                fixture.profile.runtime_binding.provider_id
            );
            let requests = extractor.requests();
            assert_eq!(requests[0].content, text.as_bytes());
            assert_eq!(requests[0].file_path, "日本語.md");
            assert_eq!(peer.requests()[0]["task"], "retrieval.passage");
            assert_eq!(
                documents.search("banana", 1, 10).await.unwrap().items[0]["text"],
                "banana 東京"
            );
        },
    )
    .await;
}

/// Explicit replacement updates metadata with content; a failed final write rolls all of it back.
#[tokio::test]
async fn replacement_is_explicit_atomic_and_cannot_discard_file_provenance() {
    with_temp_db("profiled_file_replace", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2); fixture.install(&pool).await;
        let peer = Peer::new(&fixture); let documents = fixture.connect(&pool, &peer).await;
        let extractor = Extractor::new(); let client = extractor.client().await;
        let root = tempfile::tempdir().unwrap();
        let path = write(root.path(), "notes.md", "apple\nbanana");
        let mut original = input(&path);
        original.metadata = json!({"title":"old", "curation":"retain"});
        let receipt = documents.ingest_file(original, &client, false).await.unwrap();
        let other = write(root.path(), "other.md", "banana");
        documents.ingest_file(input(&other), &client, false).await.unwrap();
        std::fs::write(&path, "apple shorter").unwrap();
        assert!(documents.ingest_file(input(&path), &client, false).await.unwrap_err().to_string().contains("replacement must be explicit"));
        assert_eq!(extractor.requests().len(), 2);
        let mut replacement = input(&path); replacement.metadata = json!({"title":"new"});
        documents.ingest_file(replacement, &client, true).await.unwrap();
        let before = rows(&pool).await;
        assert_eq!(before["chunks"].as_array().unwrap().len(), 2);
        let source = before["documents"].as_array().unwrap().iter().find(|d| d["_key"] == receipt.document_key).unwrap();
        assert_eq!(source["file_ingestion"]["metadata"]["title"], "new");
        assert_eq!(source["file_ingestion"]["metadata"]["curation"], "retain");
        assert!(documents.prepare(DocumentInput { source_id: source["source_id"].as_str().unwrap().into(), text:"discard provenance".into() }, true)
            .await.err().unwrap().to_string().contains("cannot adopt or discard"));
        pool.writer().put("collection/embeddings/properties", &json!({"schema":{
            "level":"strict", "rule":{"type":"object","required":["reject_fixture"]}, "message":"late write failure"
        }})).await.unwrap();
        assert!(documents.ingest_file(input(&path), &client, true).await.is_err());
        assert_eq!(rows(&pool).await, before);
    }).await;
}

/// Exact relative names avoid normalization collisions and permit moving an unchanged tree.
#[tokio::test]
async fn identities_keep_extensions_and_refuse_force_collisions() {
    with_temp_db(
        "profiled_file_identity",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let documents = fixture.connect(&pool, &peer).await;
            let extractor = Extractor::new();
            let client = extractor.client().await;
            let root = tempfile::tempdir().unwrap();
            let first = write(root.path(), "a/README.md", "apple");
            let second = write(root.path(), "a/README.txt", "banana");
            let mut keys = Vec::new();
            for path in [&first, &second] {
                let mut request = input(path);
                request.root = Some(root.path().into());
                keys.push(
                    documents
                        .ingest_file(request, &client, false)
                        .await
                        .unwrap()
                        .document_key,
                );
            }
            assert_ne!(keys[0], keys[1]);
            let mut named = input(&first);
            named.id = Some("explicit".into());
            documents.ingest_file(named, &client, false).await.unwrap();
            let mut collision = input(&second);
            collision.id = Some("explicit".into());
            assert!(
                documents
                    .ingest_file(collision, &client, true)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("different file")
            );
            assert_eq!(extractor.requests().len(), 3);
            let relocated = tempfile::tempdir().unwrap();
            let path = write(relocated.path(), "a/README.md", "apple moved");
            let mut moved = input(&path);
            moved.root = Some(relocated.path().into());
            assert_eq!(
                documents
                    .ingest_file(moved, &client, true)
                    .await
                    .unwrap()
                    .document_key,
                keys[0]
            );
            assert_eq!(rows(&pool).await["documents"].as_array().unwrap().len(), 3);
        },
    )
    .await;
}

/// Every rejected extraction ends before token planning, embedding, or database mutation.
#[tokio::test]
async fn invalid_extractions_do_not_embed_or_publish() {
    with_temp_db(
        "profiled_file_extract_errors",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let documents = fixture.connect(&pool, &peer).await;
            let extractor = Extractor::new();
            let client = extractor.client().await;
            let root = tempfile::tempdir().unwrap();
            let path = write(root.path(), "notes.md", "apple");
            let before = rows(&pool).await;
            for response in [
                Err(tonic::Status::unavailable("fixture failure")),
                Ok(ExtractResponse::default()),
                Ok(ExtractResponse {
                    full_text: "apple".into(),
                    metadata: [("extractor".into(), "pymupdf_fallback".into())].into(),
                    ..Default::default()
                }),
                Ok(ExtractResponse {
                    full_text: "apple".into(),
                    source_type: SourceType::Code.into(),
                    ..Default::default()
                }),
                Ok(ExtractResponse {
                    full_text: "a".repeat(2 * 1024 * 1024 + 1),
                    ..Default::default()
                }),
                Ok(ExtractResponse {
                    full_text: "apple".into(),
                    metadata: [("oversized".into(), "x".repeat(64 * 1024))].into(),
                    ..Default::default()
                }),
            ] {
                extractor.respond(response);
                assert!(
                    documents
                        .ingest_file(input(&path), &client, false)
                        .await
                        .is_err()
                );
                assert_eq!(rows(&pool).await, before);
                assert!(peer.plans().is_empty());
                assert!(peer.requests().is_empty());
            }
        },
    )
    .await;
}

/// File size, format, identity and root errors are refused before extraction.
#[tokio::test]
async fn input_limits_and_unsupported_files_fail_before_extraction() {
    with_temp_db("profiled_file_limits", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        let extractor = Extractor::new();
        let client = extractor.client().await;
        let root = tempfile::tempdir().unwrap();
        let path = write(root.path(), "notes.md", "apple");
        let empty = write(root.path(), "empty.md", "");
        let large = write(root.path(), "large.md", "");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&large)
            .unwrap()
            .set_len((MAX_FILE_BYTES + 1) as u64)
            .unwrap();
        let code = write(root.path(), "code.py", "print('apple')");
        let mut invalid = input(&path);
        invalid.id = Some("bad\nID".into());
        let mut outside = input(&path);
        outside.root = Some(root.path().join("elsewhere"));
        std::fs::create_dir(root.path().join("elsewhere")).unwrap();
        let mut metadata = input(&path);
        metadata.metadata = json!({"too_large":"x".repeat(64 * 1024)});
        for request in [
            input(&empty),
            input(&large),
            input(&code),
            invalid,
            outside,
            metadata,
        ] {
            assert!(
                documents
                    .ingest_file(request, &client, false)
                    .await
                    .is_err()
            );
        }
        assert!(extractor.requests().is_empty());
        assert!(peer.plans().is_empty());
        assert!(
            rows(&pool).await["documents"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    })
    .await;
}

/// The profile snapshot and source revision are pinned before the extractor can stall.
#[tokio::test]
async fn changes_during_extraction_cannot_replace_newer_data() {
    with_temp_db("profiled_file_drift", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = Arc::new(fixture.connect(&pool, &peer).await);
        let extractor = Extractor::new();
        let client = extractor.client().await;
        let root = tempfile::tempdir().unwrap();
        let path = write(root.path(), "notes.md", "apple");
        let receipt = documents
            .ingest_file(input(&path), &client, false)
            .await
            .unwrap();
        for profile_drift in [false, true] {
            let (entered, release) = extractor.hold();
            let worker = documents.clone();
            let request = input(&path);
            let client = client.clone();
            let task =
                tokio::spawn(async move { worker.ingest_file(request, &client, true).await });
            entered.await.unwrap();
            let endpoint = if profile_drift {
                "document/hades_workspace/embedding_profile".into()
            } else {
                format!("document/documents/{}", receipt.document_key)
            };
            let current = pool.writer().get(&endpoint).await.unwrap();
            let patch = if profile_drift {
                json!({"payload":current["payload"]})
            } else {
                json!({"source_id":current["source_id"]})
            };
            pool.writer().patch(&endpoint, &patch).await.unwrap();
            let before = rows(&pool).await;
            release.send(()).unwrap();
            let error = task.await.unwrap().unwrap_err().to_string();
            assert!(
                error.contains(if profile_drift {
                    "stale database profile"
                } else {
                    "source changed after preparation"
                }),
                "{error}"
            );
            assert_eq!(rows(&pool).await, before);
        }
    })
    .await;
}

/// An otherwise compatible reload still invalidates the in-flight provider observation.
#[tokio::test]
async fn provider_reload_during_extraction_cannot_publish_under_old_admission() {
    with_temp_db(
        "profiled_file_provider_drift",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let documents = Arc::new(fixture.connect(&pool, &peer).await);
            let extractor = Extractor::new();
            let client = extractor.client().await;
            let root = tempfile::tempdir().unwrap();
            let path = write(root.path(), "notes.md", "apple");
            documents
                .ingest_file(input(&path), &client, false)
                .await
                .unwrap();
            let before = rows(&pool).await;
            let (entered, release) = extractor.hold();
            let worker = documents.clone();
            let task =
                tokio::spawn(async move { worker.ingest_file(input(&path), &client, true).await });
            entered.await.unwrap();
            let mut receipt = peer.receipt();
            receipt.serving_generation = "reloaded-during-extraction".into();
            peer.set_receipt(receipt);
            release.send(()).unwrap();
            assert!(task.await.unwrap().is_err());
            assert_eq!(rows(&pool).await, before);
        },
    )
    .await;
}

/// A cancelled transaction cannot leave source metadata committed without its passages.
#[tokio::test]
async fn cancelled_file_commit_rolls_back_the_whole_document() {
    with_temp_db("profiled_file_cancel", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let proxy = proxy::Proxy::new(
            pool.writer(),
            proxy::Rule {
                method: "POST",
                path: "document/chunks",
                name: None,
                nth: 1,
                before: false,
                hold: true,
                fail: false,
                callback: None,
            },
        );
        let client = hades_core::db::ArangoClient::with_socket(
            proxy.admin.socket_path().unwrap().into(),
            pool.database(),
            "fixture",
            "fixture",
        );
        let documents = fixture
            .connect(&ArangoPool::new(client.clone(), client), &peer)
            .await;
        let extractor = Extractor::new();
        let client = extractor.client().await;
        let root = tempfile::tempdir().unwrap();
        let path = write(root.path(), "notes.md", "apple\nbanana");
        let before = rows(&pool).await;
        let task =
            tokio::spawn(async move { documents.ingest_file(input(&path), &client, false).await });
        tokio::time::timeout(Duration::from_secs(5), proxy.entered.notified())
            .await
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        proxy.release.notify_one();
        // An exclusive profile transaction waits behind cleanup and proves the
        // rollback has finished before inspecting nontransactional record reads.
        tokio::time::timeout(
            Duration::from_secs(5),
            hades_core::embedding_profile::store::ProfileStore::new(pool.clone())
                .unwrap()
                .load(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(rows(&pool).await, before);
    })
    .await;
}

/// Selected source details reject malformed persisted file provenance as well as vector drift.
#[tokio::test]
async fn corrupt_file_provenance_blocks_search_and_replacement() {
    with_temp_db(
        "profiled_file_corruption",
        Fixtures::Empty,
        |pool| async move {
            let fixture = Fixture::new(2);
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            let documents = fixture.connect(&pool, &peer).await;
            let extractor = Extractor::new();
            let client = extractor.client().await;
            let root = tempfile::tempdir().unwrap();
            let path = write(root.path(), "notes.md", "apple");
            let receipt = documents
                .ingest_file(input(&path), &client, false)
                .await
                .unwrap();
            let original = crud::get_document(&pool, "documents", &receipt.document_key)
                .await
                .unwrap();
            let mut corrupt = original["file_ingestion"].clone();
            corrupt["source_git"] = json!({"commit":null,"dirty":false,"forged":"unknown"});
            crud::update_document(
                &pool,
                "documents",
                &receipt.document_key,
                &json!({"file_ingestion":corrupt}),
            )
            .await
            .unwrap();
            let before = rows(&pool).await;
            assert!(documents.search("apple", 1, 10).await.is_err());
            assert!(
                documents
                    .ingest_file(input(&path), &client, true)
                    .await
                    .is_err()
            );
            assert_eq!(extractor.requests().len(), 1);
            assert_eq!(rows(&pool).await, before);
        },
    )
    .await;
}
