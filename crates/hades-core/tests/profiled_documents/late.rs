//! Contextual provenance through the same guarded document workflow as conventional passages.
use super::*;

#[tokio::test]
async fn contextual_roundtrip_retains_windows_and_supports_reload_and_replacement() {
    with_temp_db("late_roundtrip", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::late();
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        let key = insert(
            &documents,
            "late/source",
            &["apple café", "banana 東京", "apple 🚀"],
        )
        .await;
        let rows = records(&pool).await;
        let source = &rows["documents"][0];
        assert_eq!(source["format"], "hades-bound-document-v3");
        assert_eq!(
            source["passage_plan"]["windows"].as_array().unwrap().len(),
            2
        );
        assert_eq!(source["late_token_ranges"].as_array().unwrap().len(), 3);
        assert_eq!(
            peer.requests().len(),
            2,
            "each assigned window encoded once"
        );
        for family in ["chunks", "embeddings"] {
            for row in rows[family].as_array().unwrap() {
                let index = row["chunk_index"].as_u64().unwrap() as usize;
                assert_eq!(row["late_token_range"], source["late_token_ranges"][index]);
                assert_eq!(row["embedding_binding"], source["embedding_binding"]);
            }
        }
        let result = documents.search("banana", 1, 10).await.unwrap();
        assert_eq!(result.scanned, 3);
        assert_eq!(result.items[0]["text"], "banana 東京\n");
        assert_eq!(result.items[0]["late_context"]["window_index"], 0);
        assert_eq!(
            result.items[0]["late_context"]["token_range"],
            source["late_token_ranges"][1]
        );
        let mut receipt = peer.receipt();
        receipt.serving_generation = "contextual-reloaded".into();
        peer.set_receipt(receipt);
        assert!(documents.search("banana", 1, 10).await.is_err());
        let fresh = fixture.connect(&pool, &peer).await;
        assert_eq!(
            fresh.search("banana", 1, 10).await.unwrap().items[0]["parent_key"],
            key
        );
        let prepared = fresh
            .prepare(input("late/source", &["apple replacement"]), true)
            .await
            .unwrap();
        fresh.commit(prepared).await.unwrap();
        let replaced = records(&pool).await;
        for family in ["documents", "chunks", "embeddings"] {
            assert_eq!(replaced[family].as_array().unwrap().len(), 1);
        }
        assert_eq!(replaced["documents"][0]["_key"], key);
        assert_eq!(fresh.search("apple", 1, 10).await.unwrap().scanned, 1);
    })
    .await;
}

#[tokio::test]
async fn incomplete_contextual_inference_and_plan_failures_preserve_prior_records() {
    with_temp_db("late_failures", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::late();
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        insert(&documents, "source", &["apple original"]).await;
        let before = records(&pool).await;
        for fault in [
            Fault::Reject,
            Fault::MissingReceipt,
            Fault::Short,
            Fault::NotNormalized,
        ] {
            peer.fault_after(2, fault);
            assert!(
                documents
                    .prepare(
                        input("source", &["apple new", "banana new", "apple tail"]),
                        true
                    )
                    .await
                    .is_err()
            );
            assert_eq!(records(&pool).await, before);
        }
        let calls = peer.requests().len();
        peer.patch_next_plan(json!({"assignments":[99]}));
        assert!(
            documents
                .prepare(input("source", &["apple new"]), true)
                .await
                .is_err()
        );
        assert_eq!(
            peer.requests().len(),
            calls,
            "invalid plan reached inference"
        );
        assert_eq!(records(&pool).await, before);
    })
    .await;
}

#[tokio::test]
async fn damaged_contextual_provenance_refuses_search_and_replacement_without_writes() {
    with_temp_db("late_corruption", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::late();
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = fixture.connect(&pool, &peer).await;
        insert(
            &documents,
            "source",
            &["apple café", "banana 東京", "apple tail"],
        )
        .await;
        let original = records(&pool).await;
        for (family, pointer, value) in [
            ("documents", "/format", json!("hades-bound-document-v2")),
            ("documents", "/late_token_ranges", Value::Null),
            ("documents", "/late_token_ranges/0/token_end", json!(99999)),
            ("documents", "/passage_plan/assignments/0", json!(99)),
            (
                "documents",
                "/passage_plan/windows/0/input_tokens",
                json!(0),
            ),
            (
                "documents",
                "/passage_plan/windows/1/overlap_tokens",
                json!(0),
            ),
            (
                "documents",
                "/passage_plan/windows/1/end_byte",
                json!(99999),
            ),
            (
                "documents",
                "/passage_plan/receipt/space_id",
                json!("0".repeat(64)),
            ),
            ("chunks", "/late_token_range/token_end", json!(99999)),
            ("embeddings", "/late_token_range/token_start", json!(99999)),
            ("embeddings", "/format", json!("hades-bound-document-v2")),
        ] {
            let row = &original[family][0];
            let path = format!("document/{family}/{}", row["_key"].as_str().unwrap());
            let mut bad = row.clone();
            *bad.pointer_mut(pointer).unwrap() = value;
            pool.writer().put(&path, &bad).await.unwrap();
            let before = records(&pool).await;
            assert!(
                documents.search("banana", 1, 10).await.is_err(),
                "{family}{pointer}"
            );
            // Document corruption fails before inference; child corruption is
            // validated under the replacement transaction before removing rows.
            if let Ok(prepared) = documents
                .prepare(input("source", &["replacement"]), true)
                .await
            {
                assert!(
                    documents.commit(prepared).await.is_err(),
                    "{family}{pointer}"
                );
            }
            assert_eq!(records(&pool).await, before, "{family}{pointer}");
            pool.writer().put(&path, row).await.unwrap();
        }
        assert!(documents.search("banana", 1, 10).await.is_ok());
    })
    .await;
}

#[tokio::test]
async fn policy_change_during_contextual_inference_prevents_commit() {
    with_temp_db("late_drift", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::late();
        fixture.install(&pool).await;
        let peer = Peer::new(&fixture);
        let documents = Arc::new(fixture.connect(&pool, &peer).await);
        let before = records(&pool).await;
        let (started, release) = peer.hold_next();
        let worker = documents.clone();
        let pending = tokio::spawn(async move {
            worker
                .prepare(input("source", &["apple", "banana", "tail"]), false)
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
    })
    .await;
}
