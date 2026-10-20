//! Opt-in qualification: actual registered Jina HTTP provider and disposable DB.
//! The Python qualification owner supplies a private socket and checked observation.
#![cfg(unix)]

use hades_core::{
    db::{ArangoPool, crud},
    embedding_profile::{ArtifactManifest, EmbeddingProfile, ProviderReceipt, store::ProfileStore},
    persephone::{bound_embedding::ProviderConnection, embedding::EmbeddingEndpoint},
    pipeline::profiled_documents::{DocumentInput, PassageSpan, ProfiledDocuments},
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
    let descriptor = &context["description"];
    let manifest =
        ArtifactManifest::from_json(&serde_json::to_vec(&context["manifest"]).unwrap()).unwrap();
    let profile = EmbeddingProfile::from_json(&serde_json::to_vec(&json!({
        "format":"hades-embedding-profile-v1", "revision":1, "state":"initializing",
        "generation":"jina-qualification-corpus-v1", "model":descriptor["model"],
        "capabilities":descriptor["capabilities"], "runtime_binding":{"provider_id":context["provider_id"]},
        "workloads":{"code":null,"documents":{"space":descriptor["space"],"specialized_code":false,
            "chunking":{"mode":"conventional",
                "passage_tokens":500.min(descriptor["capabilities"]["max_input_tokens"].as_u64().unwrap()),
                "overlap_tokens":0}}}
    })).unwrap(), &manifest).unwrap();
    let store = ProfileStore::new(pool.clone()).unwrap();
    let initial = store.initialize(profile, manifest).await.unwrap();
    for collection in ["documents", "chunks", "embeddings"] {
        crud::create_collection(pool, collection, Some(2))
            .await
            .unwrap();
    }
    store.mark_ready(&initial).await.unwrap();
    let receipt: ProviderReceipt = serde_json::from_value(descriptor["receipt"].clone()).unwrap();
    ProfiledDocuments::connect(
        pool.clone(),
        ProviderConnection {
            provider_id: context["provider_id"].as_str().unwrap().into(),
            endpoint: EmbeddingEndpoint::Unix(PathBuf::from(context["socket"].as_str().unwrap())),
            timeout: Duration::from_secs(120),
            max_response_bytes: 8 * 1024 * 1024,
        },
        &receipt,
    )
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
                passages:vec![PassageSpan{start_byte:0,end_byte:text.len()}],
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
        let report = json!({"passed":true,"profile_revision":2,"documents":inputs.len(),
            "dimension":2048,"receipt":context["description"]["receipt"],
            "suite_sha256":context["suite_sha256"],"ingest_seconds":ingest_seconds,"queries":observed});
        let path = std::env::var("HADES_VERIFIED_JINA_REPORT").expect("explicit new evidence file");
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();
        file.write_all(&serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        file.sync_all().unwrap();
    }).await;
}
