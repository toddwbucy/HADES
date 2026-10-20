//! Stored profile -> verified HTTP provider -> atomic documents -> exact search.
//! All database writes use the disposable harness; no model, GPU, or live corpus.
#![cfg(unix)]

use hades_core::{
    db::{ArangoPool, crud, query},
    embedding_profile::{ChunkPolicy, Distance, store::COLLECTION},
    pipeline::profiled_documents::{DocumentInput, PassageSpan, ProfiledDocuments},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};

#[path = "common/profiled_provider.rs"]
mod provider;
use provider::{Fault, Fixture, Peer};

fn input(id: &str, passages: &[&str]) -> DocumentInput {
    let mut text = String::new();
    let mut spans = Vec::new();
    for passage in passages {
        let start = text.len();
        text.push_str(passage);
        spans.push(PassageSpan {
            start_byte: start,
            end_byte: text.len(),
        });
    }
    DocumentInput {
        source_id: id.into(),
        text,
        passages: spans,
    }
}

/// Include Arango revisions so a failed operation cannot silently rewrite old data.
async fn records(pool: &ArangoPool) -> Value {
    let mut saved = json!({});
    for collection in ["documents", "chunks", "embeddings"] {
        let result = query::query(
            pool,
            "FOR r IN @@c SORT r._key RETURN r",
            Some(&json!({"@c":collection})),
            None,
            false,
            query::ExecutionTarget::Writer,
        )
        .await
        .unwrap();
        saved[collection] = json!(result.results);
    }
    saved
}

async fn insert(documents: &ProfiledDocuments, id: &str, text: &[&str]) -> String {
    let prepared = documents.prepare(input(id, text), false).await.unwrap();
    documents.commit(prepared).await.unwrap().document_key
}

/// A change-and-restore is still a different admission snapshot because _rev changes.
async fn touch_profile(pool: &ArangoPool) {
    let path = format!("document/{COLLECTION}/embedding_profile");
    let original = pool.writer().get(&path).await.unwrap();
    pool.writer()
        .patch(&path, &json!({"payload":original["payload"]}))
        .await
        .unwrap();
}

#[tokio::test]
async fn roundtrip_records_identity_and_uses_asymmetric_tasks() {
    with_temp_db("bound_roundtrip", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        let source = "notes/日本語: \"a\" / ? @source";
        let key = insert(&documents, source, &["apple café\n", "banana 東京"]).await;
        let rows = records(&pool).await;
        assert_eq!(rows["documents"].as_array().unwrap().len(), 1);
        assert_eq!(rows["documents"][0]["source_id"], source);
        assert_eq!(rows["documents"][0]["full_text"], "apple café\nbanana 東京");
        for collection in ["chunks", "embeddings"] {
            assert_eq!(rows[collection].as_array().unwrap().len(), 2);
            for row in rows[collection].as_array().unwrap() {
                assert_eq!(row["parent_key"], key);
                assert_eq!(
                    row["embedding_binding"],
                    rows["documents"][0]["embedding_binding"]
                );
                assert_eq!(row["embedding_binding"]["profile_revision"], 2);
                assert_eq!(row["source_sha256"], rows["documents"][0]["source_sha256"]);
            }
        }
        let apple = documents.search("apple", 1, 2).await.unwrap();
        let banana = documents.search("banana", 2, 2).await.unwrap();
        assert_eq!(apple.scanned, 2);
        assert_eq!(apple.items.len(), 1);
        assert_eq!(apple.items[0]["parent_key"], key);
        assert_eq!(apple.items[0]["score"], 1.0);
        assert_eq!(banana.items[0]["score"], 1.0);
        assert_eq!(banana.items[1]["score"], 0.0);
        assert_ne!(apple.items[0]["chunk_key"], banana.items[0]["chunk_key"]);
        let requests = peer.requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0]["input"], json!(["apple café\n", "banana 東京"]));
        assert_eq!(requests[0]["task"], "retrieval.passage");
        for request in &requests[1..] {
            assert_eq!(request["task"], "retrieval.query");
        }
        for request in requests {
            assert_eq!(request["model"], fixture.profile.model.id);
            assert_eq!(request["expected_receipt"], json!(fixture.receipt));
        }
    })
    .await;
}

#[tokio::test]
async fn replacement_is_explicit_atomic_and_rejects_stale_preparations() {
    with_temp_db("bound_replace", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        let key = insert(&documents, "source", &["apple", "banana"]).await;
        let unrelated = insert(&documents, "other", &["apple"]).await;
        let before = records(&pool).await;
        assert!(
            documents
                .prepare(input("source", &["new"]), false)
                .await
                .is_err()
        );
        assert_eq!(peer.requests().len(), 2);
        let first = documents
            .prepare(input("source", &["apple replaced"]), true)
            .await
            .unwrap();
        let second = documents
            .prepare(input("source", &["stale"]), true)
            .await
            .unwrap();
        documents.commit(first).await.unwrap();
        let after = records(&pool).await;
        assert!(
            documents
                .commit(second)
                .await
                .unwrap_err()
                .to_string()
                .contains("source changed")
        );
        assert_eq!(records(&pool).await, after);
        assert_eq!(after["chunks"].as_array().unwrap().len(), 2);
        assert_eq!(after["embeddings"].as_array().unwrap().len(), 2);
        for collection in ["documents", "chunks", "embeddings"] {
            let unrelated_row = |r: &&Value| r["_key"] == unrelated || r["parent_key"] == unrelated;
            assert_eq!(
                before[collection]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(unrelated_row),
                after[collection]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(unrelated_row)
            );
        }
        assert_eq!(
            crud::get_document(&pool, "documents", &key).await.unwrap()["full_text"],
            "apple replaced"
        );
        let prepared = documents
            .prepare(input("source", &["will roll back"]), true)
            .await
            .unwrap();
        pool.writer().put("collection/embeddings/properties", &json!({
            "schema":{"level":"strict","rule":{"type":"object","required":["fixture_reject"]},
                "message":"reject after metadata and chunk writes"}
        })).await.unwrap();
        assert!(documents.commit(prepared).await.is_err());
        assert_eq!(records(&pool).await, after);
    })
    .await;
}

#[tokio::test]
async fn failed_provider_batches_never_publish_partial_documents() {
    with_temp_db("bound_provider_fail", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        insert(&documents, "source", &["apple"]).await;
        let before = records(&pool).await;
        for fault in [
            Fault::MissingReceipt,
            Fault::WrongModel,
            Fault::Short,
            Fault::Zero,
            Fault::NotNormalized,
            Fault::Reject,
        ] {
            peer.fault_after(1, fault);
            assert!(
                documents
                    .prepare(input("source", &["replacement"]), true)
                    .await
                    .is_err()
            );
            assert_eq!(records(&pool).await, before);
        }
        let count = peer.requests().len();
        peer.fault_after(2, Fault::MissingReceipt);
        assert!(
            documents
                .prepare(input("source", &["apple"; 17]), true)
                .await
                .is_err()
        );
        assert_eq!(peer.requests().len(), count + 2);
        assert_eq!(records(&pool).await, before);
        // Receipt identity is owned by the peer, so an old pinned client detects drift.
        for field in ["artifact_md5", "space_id", "serving_generation"] {
            let mut receipt = json!(fixture.receipt);
            receipt[field] = json!(match field {
                "artifact_md5" => "0".repeat(32),
                "space_id" => "0".repeat(64),
                _ => "load-other".into(),
            });
            peer.set_receipt(serde_json::from_value(receipt).unwrap());
            assert!(
                documents
                    .prepare(input("source", &["replacement"]), true)
                    .await
                    .is_err()
            );
            assert!(documents.search("apple", 1, 10).await.is_err());
            assert_eq!(records(&pool).await, before);
        }
    })
    .await;
}

#[tokio::test]
async fn drift_during_inference_refuses_commit_and_query_results() {
    with_temp_db("bound_drift", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = Arc::new(fixture.connect(&pool, &peer).await);
        insert(&documents, "source", &["apple"]).await;
        let before = records(&pool).await;
        let (started, release) = peer.hold_next();
        let writer = documents.clone();
        let pending = tokio::spawn(async move {
            writer
                .prepare(input("source", &["replacement"]), true)
                .await
        });
        tokio::time::timeout(Duration::from_secs(5), started)
            .await
            .unwrap()
            .unwrap();
        touch_profile(&pool).await;
        release.send(()).unwrap();
        let prepared = pending.await.unwrap().unwrap();
        assert!(documents.commit(prepared).await.is_err());
        assert_eq!(records(&pool).await, before);
        let count = peer.requests().len();
        assert!(documents.search("apple", 1, 10).await.is_err());
        assert_eq!(
            peer.requests().len(),
            count,
            "stale policy contacted the provider"
        );
        let fresh = Arc::new(fixture.connect(&pool, &peer).await);
        let (started, release) = peer.hold_next();
        let query = fresh.clone();
        let pending = tokio::spawn(async move { query.search("apple", 1, 10).await });
        tokio::time::timeout(Duration::from_secs(5), started)
            .await
            .unwrap()
            .unwrap();
        touch_profile(&pool).await;
        release.send(()).unwrap();
        assert!(pending.await.unwrap().is_err());
        assert_eq!(records(&pool).await, before);
    })
    .await;
}

#[tokio::test]
async fn a_verified_reload_can_read_compatible_old_vectors() {
    with_temp_db("bound_reload", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        insert(&documents, "source", &["apple"]).await;
        let before = records(&pool).await;
        let mut receipt = peer.receipt();
        receipt.serving_generation = "load-restarted".into();
        peer.set_receipt(receipt);
        assert!(documents.search("apple", 1, 10).await.is_err());
        let refreshed = fixture.connect(&pool, &peer).await;
        let found = refreshed.search("apple", 1, 10).await.unwrap();
        assert_eq!(found.items[0]["score"], 1.0);
        assert_eq!(found.binding.expected.serving_generation, "load-restarted");
        assert_eq!(records(&pool).await, before);
        let replacement = refreshed
            .prepare(input("source", &["banana"]), true)
            .await
            .unwrap();
        refreshed.commit(replacement).await.unwrap();
        assert_eq!(
            refreshed.search("banana", 1, 10).await.unwrap().items[0]["score"],
            1.0
        );
    })
    .await;
}

#[tokio::test]
async fn two_database_dimensions_and_providers_stay_isolated() {
    with_temp_db("bound_two", Fixtures::Empty, |one| async move {
        with_temp_db("bound_three", Fixtures::Empty, move |two| async move {
            let a = Fixture::new(2);
            let b = Fixture::new(3);
            a.install(&one).await;
            b.install(&two).await;
            let peer_a = Peer::new(&a);
            let peer_b = Peer::new(&b);
            let da = a.connect(&one, &peer_a).await;
            let db = b.connect(&two, &peer_b).await;
            let (pa, pb) = tokio::join!(
                da.prepare(input("same", &["apple"]), false),
                db.prepare(input("same", &["banana"]), false)
            );
            da.commit(pa.unwrap()).await.unwrap();
            db.commit(pb.unwrap()).await.unwrap();
            let misplaced = da.prepare(input("new", &["apple"]), false).await.unwrap();
            assert!(db.commit(misplaced).await.is_err());
            let (ra, rb) = tokio::join!(da.search("apple", 1, 10), db.search("banana", 1, 10));
            assert_eq!(ra.unwrap().binding.dimension, 2);
            assert_eq!(rb.unwrap().binding.dimension, 3);
            for (pool, dimension) in [(&one, 2), (&two, 3)] {
                let rows = records(pool).await;
                assert_eq!(rows["embeddings"].as_array().unwrap().len(), 1);
                assert_eq!(rows["embeddings"][0]["dimension"], dimension);
                assert_eq!(
                    rows["embeddings"][0]["embedding"].as_array().unwrap().len(),
                    dimension
                );
            }
            for (peer, fixture) in [(&peer_a, &a), (&peer_b, &b)] {
                for request in peer.requests() {
                    assert_eq!(request["model"], fixture.profile.model.id);
                    assert_eq!(
                        request["expected_receipt"]["space_id"],
                        json!(fixture.receipt.space_id)
                    );
                }
            }
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn incompatible_or_orphan_vectors_refuse_the_whole_search() {
    with_temp_db("bound_rows", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        insert(&documents, "source", &["apple", "banana"]).await;
        let rows = records(&pool).await;
        let original = rows["embeddings"][0].clone();
        let path = format!("document/embeddings/{}", original["_key"].as_str().unwrap());
        for (pointer, value) in [
            ("/format", json!("legacy")),
            ("/model", json!("wrong")),
            ("/dimension", json!(3)),
            ("/embedding_binding/profile_revision", json!(3)),
            ("/embedding_binding/generation", json!("other-corpus")),
            ("/embedding_binding/provider_id", json!("other-provider")),
            ("/embedding_binding/task", json!("retrieval.query")),
            (
                "/embedding_binding/expected/artifact_md5",
                json!("0".repeat(32)),
            ),
            (
                "/embedding_binding/expected/space_id",
                json!("0".repeat(64)),
            ),
            ("/embedding_binding/expected/serving_generation", json!("")),
            ("/embedding", json!([0, 0])),
            ("/embedding", json!([2, 0])),
            ("/embedding", json!([1])),
            ("/embedding", json!([1e100, 0])),
            ("/chunk_key", json!("missing")),
            ("/parent_key", json!("missing")),
            ("/chunk_index", json!(99)),
            ("/source_sha256", json!("0".repeat(64))),
        ] {
            let mut corrupted = original.clone();
            *corrupted.pointer_mut(pointer).unwrap() = value;
            pool.writer().put(&path, &corrupted).await.unwrap();
            assert!(documents.search("apple", 1, 10).await.is_err(), "{pointer}");
        }
        pool.writer().put(&path, &original).await.unwrap();
        assert!(documents.search("apple", 1, 10).await.is_ok());
        let chunk = original["chunk_key"].as_str().unwrap();
        pool.writer()
            .delete(&format!("document/chunks/{chunk}"))
            .await
            .unwrap();
        assert!(documents.search("apple", 1, 10).await.is_err());
    })
    .await;
}

#[tokio::test]
async fn scan_limits_discard_partial_rankings_and_pages_remain_complete() {
    with_temp_db("bound_scan", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        insert(&documents, "source", &["apple"; 17]).await;
        assert!(
            documents
                .search("apple", 1, 16)
                .await
                .unwrap_err()
                .to_string()
                .contains("row budget")
        );
        let result = documents.search("apple", 3, 17).await.unwrap();
        assert_eq!(result.scanned, 17);
        assert_eq!(result.items.len(), 3);
        assert!(
            result
                .items
                .windows(2)
                .all(|w| w[0]["chunk_key"].as_str() < w[1]["chunk_key"].as_str())
        );
    })
    .await;
}

#[tokio::test]
async fn invalid_plans_and_unsupported_policy_never_contact_the_provider() {
    with_temp_db("bound_admission", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        for (text, spans) in [
            ("é apple", vec![(0, 1), (1, 8)]),
            ("apple", vec![(1, 5)]),
            ("apple", vec![(0, 2), (3, 5)]),
            ("apple", vec![(0, 3)]),
            ("apple", vec![(0, 3), (0, 5)]),
            ("apple", vec![(0, 5), (1, 3)]),
            ("apple ", vec![(0, 5), (5, 6)]),
            ("", vec![]),
        ] {
            let bad = DocumentInput {
                source_id: "source".into(),
                text: text.into(),
                passages: spans
                    .into_iter()
                    .map(|(start_byte, end_byte)| PassageSpan {
                        start_byte,
                        end_byte,
                    })
                    .collect(),
            };
            assert!(documents.prepare(bad, false).await.is_err());
        }
        assert!(
            documents
                .prepare(input("bad\nid", &["apple"]), false)
                .await
                .is_err()
        );
        assert!(documents.search("", 1, 10).await.is_err());
        assert!(documents.search("apple", 0, 10).await.is_err());
        assert!(peer.requests().is_empty());
        assert!(
            records(&pool).await["documents"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    })
    .await;
    for mode in ["late", "dot", "disabled"] {
        with_temp_db("bound_policy", Fixtures::Empty, move |pool| async move {
            let mut fixture = Fixture::new(2);
            match mode {
                "late" => {
                    fixture.profile.capabilities.late_chunking = true;
                    fixture.profile.capabilities.tokenizer_offsets = true;
                    fixture
                        .profile
                        .workloads
                        .documents
                        .as_mut()
                        .unwrap()
                        .chunking = ChunkPolicy::Late {
                        context_cap_tokens: 1000,
                        minimum_window_overlap_tokens: 100,
                    };
                }
                "dot" => {
                    let space = &mut fixture.profile.workloads.documents.as_mut().unwrap().space;
                    space.representation.distance = Distance::Dot;
                    fixture.receipt.space_id = space.id().unwrap();
                }
                _ => {
                    fixture.profile.workloads.code = fixture.profile.workloads.documents.take();
                }
            }
            fixture.install(&pool).await;
            let peer = Peer::new(&fixture);
            assert!(
                ProfiledDocuments::connect(pool.clone(), peer.connection(), &fixture.receipt)
                    .await
                    .is_err()
            );
            assert!(peer.requests().is_empty());
        })
        .await;
    }
}

#[tokio::test]
async fn replacement_preserves_unknown_metadata_and_foreign_children() {
    with_temp_db("bound_foreign", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        let key = insert(&documents, "source", &["apple"]).await;
        pool.writer()
            .patch(
                &format!("document/documents/{key}"),
                &json!({"authored_note":"keep"}),
            )
            .await
            .unwrap();
        let before = records(&pool).await;
        assert!(
            documents
                .prepare(input("source", &["replacement"]), true)
                .await
                .is_err()
        );
        assert_eq!(records(&pool).await, before);
        let mut original = before["documents"][0].clone();
        original.as_object_mut().unwrap().remove("authored_note");
        pool.writer()
            .put(&format!("document/documents/{key}"), &original)
            .await
            .unwrap();
        let prepared = documents
            .prepare(input("source", &["replacement"]), true)
            .await
            .unwrap();
        crud::insert_document(
            &pool,
            "chunks",
            &json!({"_key":"foreign","parent_key":key,"note":"keep"}),
        )
        .await
        .unwrap();
        let before = records(&pool).await;
        assert!(documents.commit(prepared).await.is_err());
        assert_eq!(records(&pool).await, before);
    })
    .await;
}

#[tokio::test]
async fn cancelled_inference_has_no_database_side_effects() {
    with_temp_db("bound_cancel", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = Arc::new(fixture.connect(&pool, &peer).await);
        let before = records(&pool).await;
        let (started, release) = peer.hold_next();
        let worker = documents.clone();
        let pending =
            tokio::spawn(async move { worker.prepare(input("source", &["apple"]), false).await });
        tokio::time::timeout(Duration::from_secs(5), started)
            .await
            .unwrap()
            .unwrap();
        pending.abort();
        assert!(pending.await.err().unwrap().is_cancelled());
        release.send(()).unwrap();
        assert_eq!(records(&pool).await, before);
    })
    .await;
}
