//! Admission must distinguish workloads and keep replacement lookups indexed.
use super::*;
use hades_core::{
    db::index,
    pipeline::profiled_documents::{DocumentInput, ProfiledDocuments},
};

#[tokio::test]
async fn one_load_selects_distinct_document_and_code_spaces() {
    with_temp_db("code_mixed", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::mixed(2, 3);
        fixture.install_mixed(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let documents = ProfiledDocuments::connect(pool.clone(), &peer.catalog())
            .await
            .unwrap();
        let code = connect(&pool, &peer).await;
        let document = documents
            .commit(
                documents
                    .prepare(
                        DocumentInput {
                            source_id: "notes".into(),
                            text: "apple".into(),
                        },
                        false,
                    )
                    .await
                    .unwrap(),
            )
            .await
            .unwrap();
        let code = Arc::new(code);
        let receipt = code
            .commit(code.prepare(input("one.rs", ""), false).await.unwrap())
            .await
            .unwrap();
        assert_eq!(document.binding.dimension, 2);
        assert_eq!(receipt.binding.dimension, 3);
        assert_eq!(document.binding.task, "retrieval.passage");
        assert_eq!(receipt.binding.task, "fixture.code.passage");
        assert_eq!(
            document.binding.expected.artifact_md5,
            receipt.binding.expected.artifact_md5
        );
        assert_eq!(
            document.binding.expected.serving_generation,
            receipt.binding.expected.serving_generation
        );
        assert_ne!(
            document.binding.expected.space_id,
            receipt.binding.expected.space_id
        );
        assert_eq!(
            peer.requests()[0]["expected_receipt"],
            json!(document.binding.expected)
        );
        assert_eq!(
            peer.requests()[1]["expected_receipt"],
            json!(receipt.binding.expected)
        );
        let before = rows(&pool).await;
        let requests = peer.requests().len();
        // The peer owns the observation; a document receipt must not satisfy a code request.
        peer.set_code_receipt(fixture.receipt.clone());
        assert!(code.prepare(input("two.rs", ""), false).await.is_err());
        assert_eq!(peer.requests().len(), requests);
        assert_eq!(rows(&pool).await, before);
    })
    .await;
}

async fn owner_index(pool: &ArangoPool, collection: &str, field: &str) -> Value {
    pool.writer()
        .get(&format!("index?collection={collection}"))
        .await
        .unwrap()["indexes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|index| index["type"] == "persistent" && index["fields"] == json!([field]))
        .unwrap()
        .clone()
}

#[tokio::test]
async fn missing_or_incompatible_owner_indexes_refuse_admission_and_publication() {
    with_temp_db("code_owner_indexes", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::mixed(2, 3);
        fixture.install_mixed(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        code.commit(code.prepare(input("one.rs", ""), false).await.unwrap()).await.unwrap();
        for (collection, _) in CODEBASE.all_collections() {
            if collection == CODEBASE.files { continue; }
            let code = connect(&pool, &peer).await;
            let prepared = code.prepare(input("one.rs", "let changed = 1;"), true).await.unwrap();
            let index = owner_index(&pool, collection, "file_key").await;
            let plan = pool.writer().post("explain", &json!({
                "query":"FOR r IN @@c OPTIONS {indexHint:@index,forceIndexHint:true} FILTER r.file_key == @owner RETURN r",
                "bindVars":{"@c":collection,"index":index["name"],"owner":prepared.file_key()}
            })).await.unwrap();
            assert!(plan["plan"]["nodes"].as_array().unwrap().iter().any(|node| node["type"] == "IndexNode"));
            index::drop_index(&pool, index["id"].as_str().unwrap()).await.unwrap();
            let before = rows(&pool).await;
            let observations = peer.observations();
            assert!(ProfiledCode::connect(pool.clone(), &peer.catalog()).await.is_err());
            assert_eq!(peer.observations(), observations);
            assert!(code.commit(prepared).await.is_err());
            assert_eq!(rows(&pool).await, before);
            if collection == CODEBASE.symbols {
                let plans = peer.plans().len();
                assert!(code.prepare(symbols::shifted_input("one.rs", ""), true).await.is_err());
                assert_eq!(peer.plans().len(), plans, "reconciliation requires the admitted index before inference");
            }
            // A sparse or compound index is not a complete owner lookup contract.
            let wrong = pool.writer().post(&format!("index?collection={collection}"), &json!({
                "type":"persistent","fields":["file_key","chunk_index"],"unique":false,"sparse":true
            })).await.unwrap();
            assert!(ProfiledCode::connect(pool.clone(), &peer.catalog()).await.is_err());
            index::drop_index(&pool, wrong["id"].as_str().unwrap()).await.unwrap();
            index::ensure_lookup_index(&pool, collection, "file_key").await.unwrap();
        }
        let documents = ProfiledDocuments::connect(pool.clone(), &peer.catalog()).await.unwrap();
        for collection in ["chunks", "embeddings"] {
            let prepared = documents.prepare(DocumentInput {source_id:"notes".into(), text:"apple".into()}, false).await.unwrap();
            let index = owner_index(&pool, collection, "parent_key").await;
            index::drop_index(&pool, index["id"].as_str().unwrap()).await.unwrap();
            ProfiledDocuments::connect(pool.clone(), &peer.catalog()).await.unwrap();
            let plans = peer.plans().len();
            assert!(documents.prepare(DocumentInput {source_id:"notes".into(), text:"apple".into()}, false)
                .await.err().unwrap().to_string().contains("parent_key index"));
            assert_eq!(peer.plans().len(), plans);
            assert!(documents.commit(prepared).await.is_err());
            assert_eq!(hades_core::db::crud::count_collection(&pool,"documents").await.unwrap(),0);
            index::ensure_lookup_index(&pool, collection, "parent_key").await.unwrap();
        }
    }).await;
}
