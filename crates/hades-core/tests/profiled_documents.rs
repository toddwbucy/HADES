//! Stored profile -> verified HTTP provider -> atomic documents -> exact search.
//! All database writes use the disposable harness; no model, GPU, or live corpus.
#![cfg(unix)]

use hades_core::{
    db::{ArangoPool, crud, query},
    embedding_profile::{ChunkPolicy, Distance, store::COLLECTION},
    pipeline::profiled_documents::{DocumentInput, ProfiledDocuments},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};

#[path = "common/profiled_provider.rs"]
mod provider;
use provider::{Fault, Fixture, Peer};

/// Supply extraction text only; the private peer plans conservative newline passages.
fn input(id: &str, passages: &[&str]) -> DocumentInput {
    let mut text = String::new();
    for passage in passages {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(passage);
    }
    DocumentInput {
        source_id: id.into(),
        text,
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

/// Exercise preparation and commit together, returning the generated source identity.
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

/// Verify actual wire roles and source/chunk/vector identity through a complete search.
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
        let plan = &rows["documents"][0]["passage_plan"];
        assert_eq!(plan["format"], "hades-token-passages-v1");
        assert_eq!(plan["source_sha256"], rows["documents"][0]["source_sha256"]);
        assert_eq!(plan["receipt"], json!(fixture.receipt));
        assert_eq!(plan["spans"][0]["end_byte"], "apple café\n".len());
        assert_eq!(peer.plans().len(), 1);
        assert_eq!(peer.plans()[0]["passage_tokens"], 500);
        assert_eq!(peer.plans()[0]["overlap_tokens"], 100);
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

/// Preserve old and unrelated records on stale preparation or a late schema rejection.
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

/// Discard all batches when representation, receipt, or provider admission fails.
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

/// Change policy while inference is held, then prove publication and ranking refuse it.
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

/// A slow provider load cannot admit a database policy that changed during observation.
#[tokio::test]
async fn profile_drift_during_provider_verification_refuses_admission() {
    with_temp_db("bound_verify_drift", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let catalog = peer.catalog();
        let (started, release) = peer.hold_next_verification();
        let reading = pool.clone();
        let pending =
            tokio::spawn(async move { ProfiledDocuments::connect(reading, &catalog).await });
        tokio::time::timeout(Duration::from_secs(5), started)
            .await
            .unwrap()
            .unwrap();
        touch_profile(&pool).await;
        release.send(()).unwrap();
        assert!(pending.await.unwrap().is_err());
        assert_eq!(peer.observations(), 1);
        assert!(peer.plans().is_empty() && peer.requests().is_empty());
        assert!(
            records(&pool).await["documents"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    })
    .await;
}

/// Retain old passage observations while requiring a fresh client after provider reload.
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

/// Alternate and concurrently query different widths without leaking requests or prepared work.
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

/// Corrupt each identity dimension and row reference; no partial ranking may escape.
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

/// Verify stored token evidence agrees across source/chunk rows before any ranking escapes.
#[tokio::test]
async fn stored_plan_corruption_and_unproven_v1_rows_refuse_search() {
    with_temp_db("bound_plan_rows", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        insert(&documents, "source", &["apple", "banana"]).await;
        let rows = records(&pool).await;
        for (collection, pointer, value) in [
            ("documents", "/format", json!("hades-bound-document-v1")),
            (
                "documents",
                "/passage_plan/format",
                json!("hades-supplied-byte-spans-v1"),
            ),
            (
                "documents",
                "/passage_plan/source_sha256",
                json!("0".repeat(64)),
            ),
            ("documents", "/passage_plan/overlap_tokens", json!(99)),
            (
                "documents",
                "/passage_plan/receipt/serving_generation",
                json!("different-load"),
            ),
            ("documents", "/passage_plan/max_input_tokens", json!(1)),
            ("documents", "/passage_plan/spans/0/body_tokens", json!(1)),
            ("chunks", "/span/body_tokens", json!(1)),
            ("chunks", "/span/start_byte", json!(1)),
        ] {
            let original = &rows[collection][0];
            let path = format!(
                "document/{collection}/{}",
                original["_key"].as_str().unwrap()
            );
            let mut wrong = original.clone();
            *wrong.pointer_mut(pointer).unwrap() = value;
            pool.writer().put(&path, &wrong).await.unwrap();
            assert!(
                documents.search("apple", 1, 10).await.is_err(),
                "{collection}{pointer}"
            );
            pool.writer().put(&path, original).await.unwrap();
        }
        assert!(documents.search("apple", 1, 10).await.is_ok());
    })
    .await;
}

/// Cross multiple scan pages and distinguish a complete result from a truncated ranking.
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

/// Reject malformed sources and disabled or unsupported policy before external work.
#[tokio::test]
async fn invalid_plans_and_unsupported_policy_never_contact_the_provider() {
    with_temp_db("bound_admission", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        for text in [
            "".to_owned(),
            " \n\t".to_owned(),
            "x".repeat(2 * 1024 * 1024 + 1),
        ] {
            assert!(
                documents
                    .prepare(
                        DocumentInput {
                            source_id: "source".into(),
                            text
                        },
                        false
                    )
                    .await
                    .is_err()
            );
        }
        assert!(peer.plans().is_empty());
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
                        passages: None,
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
                ProfiledDocuments::connect(pool.clone(), &peer.catalog())
                    .await
                    .is_err()
            );
            assert!(peer.requests().is_empty());
            assert_eq!(peer.observations(), 0);
        })
        .await;
    }
}

/// Incompatible or malformed plans cannot trigger inference or replace existing data.
#[tokio::test]
async fn rejected_token_plans_preserve_old_documents_before_inference() {
    with_temp_db("bound_plan_refusal", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        insert(&documents, "source", &["apple"]).await;
        let before = records(&pool).await;
        let good = json!({"start_byte":0,"end_byte":8,"body_tokens":7,"input_tokens":10,"overlap_tokens":0});
        let mut patches = vec![
            json!({"format":"unknown"}), json!({"model":"other"}), json!({"task":"retrieval.query"}),
            json!({"source_sha256":"0".repeat(64)}), json!({"passage_tokens":499}),
            json!({"overlap_tokens":49}), json!({"max_input_tokens":0}),
            json!({"max_input_tokens":2049}), json!({"spans":[]}), json!({"unknown":true}),
            json!({"spans":vec![good.clone();129]}),
        ];
        for (key,value) in [("start_byte",json!(1)),("end_byte",json!(1)),("end_byte",json!(9)),
            ("end_byte",json!(7)),("body_tokens",json!(501)),("input_tokens",json!(2049)),
            ("input_tokens",json!(0)),("overlap_tokens",json!(1)),("body_tokens",json!(true))] {
            let mut span = good.clone(); span[key] = value;
            patches.push(json!({"spans":[span]}));
        }
        let mut receipt = json!(fixture.receipt);
        receipt["serving_generation"] = json!("other-load");
        patches.push(json!({"receipt":receipt}));
        for patch in patches {
            peer.patch_next_plan(patch);
            assert!(documents.prepare(input("source", &["é apple"]), true).await.is_err());
            assert_eq!(peer.requests().len(), 1);
            assert_eq!(records(&pool).await, before);
        }
    }).await;
}

/// A policy revision during tokenization must fail before spending model inference.
#[tokio::test]
async fn policy_drift_during_planning_stops_before_embedding() {
    with_temp_db("bound_plan_drift", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::new(2);
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = Arc::new(fixture.connect(&pool, &peer).await);
        let (entered, release) = peer.hold_next_plan();
        let job = {
            let documents = documents.clone();
            tokio::spawn(async move { documents.prepare(input("source", &["apple"]), false).await })
        };
        entered.await.unwrap();
        touch_profile(&pool).await;
        release.send(()).unwrap();
        assert!(job.await.unwrap().is_err());
        assert!(peer.requests().is_empty());
        assert!(
            records(&pool).await["documents"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    })
    .await;
}

/// Refuse overwrites that would erase additional authored fields or unmanaged children.
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

/// Cancel a request held inside the provider and verify all database records remain unchanged.
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
