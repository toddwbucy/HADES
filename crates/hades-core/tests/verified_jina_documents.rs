//! Opt-in qualification: actual registered Jina HTTP provider and disposable DB.
//! The Python qualification owner supplies a private socket and checked observation.
#![cfg(unix)]

use hades_core::{
    db::{ArangoPool, crud},
    embedding_profile::{ChunkPolicy, store::ProfileStore},
    persephone::{
        bound_embedding::ProviderConnection, embedding::EmbeddingEndpoint,
        provider_registry::ProviderCatalog,
    },
    pipeline::profiled_documents::{DocumentInput, ProfiledDocuments},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
use std::{
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

/// Resolve only the explicitly supplied private qualification context.
fn context() -> Value {
    let path = PathBuf::from(
        std::env::var("HADES_VERIFIED_JINA_CONTEXT")
            .expect("run through qualify_verified_jina.py and the isolated database runner"),
    );
    let bytes = std::fs::read(path).unwrap();
    assert!(
        bytes.len() <= 2 * 1024 * 1024,
        "qualification context exceeds bound"
    );
    serde_json::from_slice(&bytes).unwrap()
}

/// Initialize a ready synthetic corpus using the actual observed model and representation.
async fn install(pool: &ArangoPool, context: &Value) -> ProfiledDocuments {
    let catalog = ProviderCatalog::new(
        PathBuf::from(context["registry"].as_str().unwrap()),
        vec![ProviderConnection {
            provider_id: context["provider_id"].as_str().unwrap().into(),
            endpoint: EmbeddingEndpoint::Unix(PathBuf::from(context["socket"].as_str().unwrap())),
            timeout: Duration::from_secs(120),
            max_response_bytes: 8 * 1024 * 1024,
        }],
    )
    .unwrap();
    let observed = catalog
        .verify(context["provider_id"].as_str().unwrap())
        .await
        .unwrap();
    // The independent Rust registry/HTTP observation must agree with the Python qualification.
    assert_eq!(json!(observed.description()), context["description"]);
    assert_eq!(json!(observed.manifest()), context["manifest"]);
    let passage_tokens = 500.min(observed.description().capabilities.max_input_tokens);
    let profile = observed
        .document_profile(
            "jina-qualification-corpus-v1".into(),
            ChunkPolicy::Conventional {
                passage_tokens,
                overlap_tokens: 50.min(passage_tokens / 4),
            },
        )
        .unwrap();
    let store = ProfileStore::new(pool.clone()).unwrap();
    let initial = store
        .initialize(profile, observed.manifest().clone())
        .await
        .unwrap();
    for collection in ["documents", "chunks", "embeddings"] {
        crud::create_collection(pool, collection, Some(2))
            .await
            .unwrap();
    }
    store.mark_ready(&initial).await.unwrap();
    ProfiledDocuments::connect(pool.clone(), &catalog)
        .await
        .unwrap()
}

/// Prove real vectors survive the wire, storage checks, and asymmetric query ranking.
#[tokio::test]
#[ignore = "requires explicit registered-model qualification and a private database"]
async fn registered_jina_document_roundtrip() {
    let context = context();
    with_temp_db("real_verified_jina", Fixtures::Empty, move |pool| async move {
        let documents = install(&pool, &context).await;
        let inputs = context["cases"]["documents"].as_array().unwrap();
        let queries = context["cases"]["queries"].as_array().unwrap();
        assert!(!inputs.is_empty() && !queries.is_empty());
        let mut keys = std::collections::BTreeMap::new();
        let started = Instant::now();
        for input in inputs {
            let source = input["id"].as_str().unwrap();
            let text = input["text"].as_str().unwrap();
            let prepared = documents.prepare(DocumentInput {
                source_id:source.into(), text:text.into(),
            }, false).await.unwrap();
            let receipt = documents.commit(prepared).await.unwrap();
            assert_eq!(receipt.binding.dimension, 2048);
            assert_eq!(receipt.binding.task, "retrieval.passage");
            assert_eq!(json!(receipt.binding.expected), context["description"]["receipt"]);
            let stored = crud::get_document(&pool, "documents", &receipt.document_key).await.unwrap();
            assert_eq!(stored["full_text"], text);
            assert_eq!(stored["embedding_binding"], json!(receipt.binding));
            keys.insert(source.to_owned(), receipt.document_key);
        }
        let ingest_seconds = started.elapsed().as_secs_f64();
        for collection in ["documents","chunks","embeddings"] {
            let count = pool.writer().get(&format!("collection/{collection}/count")).await.unwrap();
            assert_eq!(count["count"], inputs.len());
        }
        let mut observed = Vec::new();
        for query in queries {
            let started = Instant::now();
            let result = documents.search(query["text"].as_str().unwrap(), 1, inputs.len() as u64).await.unwrap();
            assert_eq!(result.scanned, inputs.len() as u64);
            assert_eq!(result.binding.task, "retrieval.query");
            assert_eq!(result.items.len(), 1);
            assert_eq!(result.items[0]["parent_key"],
                keys[query["expected_document"].as_str().unwrap()]);
            observed.push(json!({"query":query["id"],"expected":query["expected_document"],
                "rank":1,"score":result.items[0]["score"],"seconds":started.elapsed().as_secs_f64()}));
        }
        // Keep retrieval ranking unchanged: add the independent long fixture only afterward.
        let long_text = context["planning_text"].as_str().unwrap();
        let started = Instant::now();
        let prepared = documents.prepare(DocumentInput {
            source_id:"unicode-token-planning".into(), text:long_text.into(),
        }, false).await.unwrap();
        let receipt = documents.commit(prepared).await.unwrap();
        let stored = crud::get_document(&pool, "documents", &receipt.document_key).await.unwrap();
        assert_eq!(stored["full_text"], long_text);
        assert_eq!(stored["embedding_binding"], json!(receipt.binding));
        let plan = &stored["passage_plan"];
        assert!(plan["spans"].as_array().unwrap().len() > 1);
        assert_eq!(plan["source_sha256"], stored["source_sha256"]);
        for collection in ["chunks", "embeddings"] {
            let count = pool.writer().get(&format!("collection/{collection}/count")).await.unwrap();
            assert_eq!(count["count"].as_u64().unwrap(),
                inputs.len() as u64 + plan["spans"].as_array().unwrap().len() as u64);
        }
        // Search validates every persisted triplet, including all long-document chunks.
        let total = inputs.len() + plan["spans"].as_array().unwrap().len();
        assert_eq!(documents.search("Unicode café", 1, total as u64).await.unwrap().scanned, total as u64);
        let report = json!({"passed":true,"profile_revision":2,"documents":inputs.len(),
            "dimension":2048,"receipt":context["description"]["receipt"],
            "protected_registration_and_live_description_checked":true,
            "suite_sha256":context["suite_sha256"],"ingest_seconds":ingest_seconds,"queries":observed,
            "planning":{"plan":plan,"source_bytes":long_text.len(),"seconds":started.elapsed().as_secs_f64()}});
        let path = std::env::var("HADES_VERIFIED_JINA_REPORT").expect("explicit new evidence file");
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();
        file.write_all(&serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        file.sync_all().unwrap();
    }).await;
}
