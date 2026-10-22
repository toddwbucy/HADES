//! Opt-in qualification: actual registered Jina HTTP provider and disposable DB.
//! The Python qualification owner supplies a private socket and checked observation.
#![cfg(unix)]

use hades_core::{
    config::{HadesConfig, RegisteredProviderConfig, RegisteredProvidersConfig},
    db::{ArangoClient, ArangoPool, crud},
    embedding_profile::{ChunkPolicy, PassagePolicy, WorkloadKind, store::ProfileStore},
    pipeline::profiled_documents::{ProfiledDocuments, files::FileInput},
    service::{ConnectionPolicy, handle_request},
    test_support::{Fixtures, with_temp_db},
    workspace_setup::{SetupReceipt, SetupRequest, SetupStatus, WorkspaceSetup},
};
use serde_json::{Value, json};
use std::{
    io::Write,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    time::{Duration, Instant},
};

#[allow(dead_code)]
#[path = "common/profiled_extractor.rs"]
mod extraction;

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

fn providers(context: &Value) -> RegisteredProvidersConfig {
    RegisteredProvidersConfig {
        registry_root: context["registry"].as_str().unwrap().into(),
        providers: vec![RegisteredProviderConfig {
            id: context["provider_id"].as_str().unwrap().into(),
            socket: context["socket"].as_str().unwrap().into(),
            timeout_seconds: 120,
            max_response_bytes: 8 * 1024 * 1024,
        }],
    }
}

async fn public_search(pool: &ArangoPool, config: &HadesConfig, text: &str) -> Value {
    let response = handle_request(
        pool,
        config,
        ConnectionPolicy::agent_only(),
        &serde_json::to_vec(&json!({"command":"db.query","params":{"text":text,"limit":1}}))
            .unwrap(),
        Duration::from_secs(120),
    )
    .await;
    assert!(response.success, "{response:?}");
    response.data.unwrap()
}

/// Initialize a ready synthetic corpus using the actual observed model and representation.
async fn install(pool: &ArangoPool, context: &Value) -> (ProfiledDocuments, SetupReceipt) {
    let catalog = providers(context).catalog().unwrap();
    let observed = catalog
        .verify(context["provider_id"].as_str().unwrap())
        .await
        .unwrap();
    // The independent Rust registry/HTTP observation must agree with the Python qualification.
    assert_eq!(json!(observed.description()), context["description"]);
    assert_eq!(json!(observed.manifest()), context["manifest"]);
    let passage_tokens = 500.min(
        observed
            .description()
            .advertised_workload(WorkloadKind::Documents)
            .unwrap()
            .capabilities
            .max_input_tokens,
    );
    let policy = if context["chunk_policy"] == "late" {
        ChunkPolicy::Late {
            context_cap_tokens: 1024,
            minimum_window_overlap_tokens: 128,
            passages: Some(PassagePolicy {
                passage_tokens: 256,
                overlap_tokens: 32,
            }),
        }
    } else {
        ChunkPolicy::Conventional {
            passage_tokens,
            overlap_tokens: 50.min(passage_tokens / 4),
        }
    };
    let profile = observed
        .document_profile("jina-qualification-corpus-v1".into(), policy)
        .unwrap();
    // with_temp_db owns this exact name and cleanup. Let the coordinator create it afresh.
    let admin = ArangoClient::with_socket(
        pool.writer().socket_path().unwrap().into(),
        "_system",
        &std::env::var("HADES_TEST_USER").unwrap_or_else(|_| "root".into()),
        &std::env::var("ARANGO_PASSWORD").unwrap(),
    );
    admin
        .delete(&format!("database/{}", pool.database()))
        .await
        .unwrap();
    let ledger = tempfile::tempdir().unwrap();
    std::fs::set_permissions(ledger.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let setup = WorkspaceSetup::new(
        ledger.path().into(),
        "jina-qualification".into(),
        "qualification-owner".into(),
        admin,
        catalog.clone(),
    )
    .unwrap();
    let request = SetupRequest {
        operation_id: "jina-qualification-setup".into(),
        database: pool.database().into(),
        profile,
        manifest: observed.manifest().clone(),
    };
    let receipt = setup.run(request.clone()).await.unwrap();
    assert_eq!(receipt.status, SetupStatus::Ready);
    let store = ProfileStore::new(pool.clone()).unwrap();
    let snapshot = store.load().await.unwrap();
    assert_eq!(setup.run(request).await.unwrap(), receipt);
    assert_eq!(store.load().await.unwrap(), snapshot);
    (
        ProfiledDocuments::connect(pool.clone(), &catalog)
            .await
            .unwrap(),
        receipt,
    )
}

/// Prove real vectors survive the wire, storage checks, and asymmetric query ranking.
#[tokio::test]
#[ignore = "requires explicit registered-model qualification and a private database"]
async fn registered_jina_document_roundtrip() {
    let context = context();
    with_temp_db("real_verified_jina", Fixtures::Empty, move |pool| async move {
        let (documents, setup_receipt) = install(&pool, &context).await;
        let mut config = HadesConfig::with_database(pool.database());
        config.registered_providers = Some(providers(&context));
        config.embedding.service.socket = "/absent/legacy-provider.sock".into();
        let inputs = context["cases"]["documents"].as_array().unwrap();
        let queries = context["cases"]["queries"].as_array().unwrap();
        assert!(!inputs.is_empty() && !queries.is_empty());
        let mut keys = std::collections::BTreeMap::new();
        // Isolate file transport from extraction quality. Actual Jina inference
        // receives fixed corpus text through the production file-ingestion path.
        let source_directory = tempfile::tempdir().unwrap();
        let extractor = extraction::Extractor::new();
        let extraction_client = extractor.client().await;
        let started = Instant::now();
        for input in inputs {
            let source = input["id"].as_str().unwrap();
            let text = input["text"].as_str().unwrap();
            let path = source_directory.path().join(format!("{}.txt", keys.len()));
            std::fs::write(&path, text).unwrap();
            let receipt = documents.ingest_file(FileInput {
                path, root:None, id:Some(source.into()), source_git:None,
                metadata:json!({"qualification_case":source}),
            }, &extraction_client, false).await.unwrap();
            assert_eq!(receipt.binding.dimension, 2048);
            assert_eq!(receipt.binding.task, "retrieval.passage");
            assert_eq!(json!(receipt.binding.expected), context["description"]["receipt"]);
            let stored = crud::get_document(&pool, "documents", &receipt.document_key).await.unwrap();
            assert_eq!(stored["full_text"], text);
            assert_eq!(stored["embedding_binding"], json!(receipt.binding));
            assert_eq!(stored["file_ingestion"]["content_bytes"], text.len());
            assert_eq!(stored["file_ingestion"]["metadata"]["qualification_case"], source);
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
            let result = public_search(&pool, &config, query["text"].as_str().unwrap()).await;
            assert_eq!(result["scanned"], inputs.len());
            assert_eq!(result["embedding_profile_state"], "ready");
            assert_eq!(result["embedding_binding"]["task"], "retrieval.query");
            assert_eq!(result["results"].as_array().unwrap().len(), 1);
            assert_eq!(result["results"][0]["parent_key"],
                keys[query["expected_document"].as_str().unwrap()]);
            let expected = inputs.iter().find(|input| input["id"] == query["expected_document"]).unwrap();
            assert_eq!(result["results"][0]["text"], expected["text"]);
            observed.push(json!({"query":query["id"],"expected":query["expected_document"],
                "rank":1,"score":result["results"][0]["score"],"seconds":started.elapsed().as_secs_f64()}));
        }
        // Keep retrieval ranking unchanged: add the independent long fixture only afterward.
        let long_text = context["planning_text"].as_str().unwrap();
        let started = Instant::now();
        let path = source_directory.path().join("unicode-token-planning.txt");
        std::fs::write(&path, long_text).unwrap();
        let receipt = documents.ingest_file(FileInput {
            path, root:None, id:Some("unicode-token-planning".into()), source_git:None,
            metadata:json!({"qualification_case":"unicode-token-planning"}),
        }, &extraction_client, false).await.unwrap();
        let stored = crud::get_document(&pool, "documents", &receipt.document_key).await.unwrap();
        assert_eq!(stored["full_text"], long_text);
        assert_eq!(stored["embedding_binding"], json!(receipt.binding));
        let plan = &stored["passage_plan"];
        let span_field = if context["chunk_policy"] == "late" { "passages" } else { "spans" };
        assert!(plan[span_field].as_array().unwrap().len() > 1);
        assert_eq!(plan["source_sha256"], stored["source_sha256"]);
        for collection in ["chunks", "embeddings"] {
            let count = pool.writer().get(&format!("collection/{collection}/count")).await.unwrap();
            assert_eq!(count["count"].as_u64().unwrap(),
                inputs.len() as u64 + plan[span_field].as_array().unwrap().len() as u64);
        }
        // Search validates every persisted triplet, including all long-document chunks.
        let total = inputs.len() + plan[span_field].as_array().unwrap().len();
        assert_eq!(public_search(&pool, &config, "Unicode café").await["scanned"], total);
        let mut report = json!({"passed":true,"profile_revision":2,"documents":inputs.len(),
            "dimension":2048,"receipt":context["description"]["receipt"],
            "protected_registration_and_live_description_checked":true,
            "recoverable_setup_and_identical_retry_checked":true,"setup":setup_receipt,
            "public_search_and_source_text_checked":true,
            "file_ingestion_and_snapshot_provenance_checked":true,
            "extractor_implementation":"private text passthrough fixture; not a Docling quality qualification",
            "suite_sha256":context["suite_sha256"],"ingest_seconds":ingest_seconds,"queries":observed,
            "planning":{"plan":plan,"source_bytes":long_text.len(),"seconds":started.elapsed().as_secs_f64()}});
        if context["chunk_policy"] == "late" {
            assert_eq!(stored["format"], "hades-bound-document-v3");
            assert_eq!(stored["late_token_ranges"].as_array().unwrap().len(), receipt.chunks);
            report["planning"]["token_ranges"] = stored["late_token_ranges"].clone();
            report["chunk_policy"] = json!("late");
        }
        let path = std::env::var("HADES_VERIFIED_JINA_REPORT").expect("explicit new evidence file");
        let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();
        file.write_all(&serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        file.sync_all().unwrap();
    }).await;
}
