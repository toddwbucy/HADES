//! Opt-in actual registered Jina qualification through the production Rust transport.
#![cfg(unix)]
use hades_core::{
    config::{RegisteredProviderConfig, RegisteredProvidersConfig},
    embedding_profile::{ChunkPolicy, EmbeddingRole, PassagePolicy, ProfileState, WorkloadKind},
};
use serde_json::{Value, json};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|&n| f64::from(n).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

#[tokio::test]
#[ignore = "run through qualify_verified_late.py with an explicitly approved local registration"]
async fn registered_contextual_transport() {
    let bytes = std::fs::read(std::env::var("HADES_VERIFIED_LATE_CONTEXT").unwrap()).unwrap();
    assert!(bytes.len() <= 2 * 1024 * 1024);
    let context: Value = serde_json::from_slice(&bytes).unwrap();
    let id = context["provider_id"].as_str().unwrap();
    let catalog = RegisteredProvidersConfig {
        registry_root: context["registry"].as_str().unwrap().into(),
        providers: vec![RegisteredProviderConfig {
            id: id.into(),
            socket: context["socket"].as_str().unwrap().into(),
            timeout_seconds: 120,
            max_response_bytes: 8 * 1024 * 1024,
        }],
    }
    .catalog()
    .unwrap();
    let observed = catalog.verify(id).await.unwrap();
    assert_eq!(json!(observed.description()), context["description"]);
    assert_eq!(json!(observed.manifest()), context["manifest"]);
    let mut profile = observed
        .document_profile(
            "contextual-qualification-v1".into(),
            ChunkPolicy::Late {
                context_cap_tokens: 1024,
                minimum_window_overlap_tokens: 128,
                passages: Some(PassagePolicy {
                    passage_tokens: 256,
                    overlap_tokens: 32,
                }),
            },
        )
        .unwrap();
    // No database is created by this provider qualification.
    profile.state = ProfileState::Ready;
    let connect =
        |role| observed.client(&profile, observed.manifest(), WorkloadKind::Documents, role);
    let passages = connect(EmbeddingRole::Passage).await.unwrap();
    let queries = connect(EmbeddingRole::Query).await.unwrap();
    let documents = context["cases"]["documents"].as_array().unwrap();
    let mut vectors = Vec::new();
    for document in documents {
        let text = document["text"].as_str().unwrap();
        let plan = passages.plan_late_passages(text).await.unwrap();
        assert_eq!(plan.output.passages.len(), 1);
        let result = passages
            .embed_planned_late(text, &plan.output)
            .await
            .unwrap();
        vectors.push(result.output.embeddings[0].clone());
    }
    let mut rankings = Vec::new();
    for query in context["cases"]["queries"].as_array().unwrap() {
        let text = query["text"].as_str().unwrap().to_owned();
        let result = queries
            .embed(std::slice::from_ref(&text), None)
            .await
            .unwrap();
        let vector = &result.output.embeddings[0];
        let best = vectors
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| cosine(vector, a).total_cmp(&cosine(vector, b)))
            .unwrap()
            .0;
        assert_eq!(documents[best]["id"], query["expected_document"]);
        let repeat = queries.embed(&[text], None).await.unwrap();
        let repeat_cosine = cosine(vector, &repeat.output.embeddings[0]);
        assert!(repeat_cosine >= 0.99999);
        rankings.push(json!({"query":query["id"],"expected":query["expected_document"],
            "first":documents[best]["id"],"rank":1,"score":cosine(vector,&vectors[best]),"repeat_cosine":repeat_cosine}));
    }
    let text = context["planning_text"].as_str().unwrap();
    let plan = passages.plan_late_passages(text).await.unwrap();
    assert!(
        plan.output.windows.len() > 1 && plan.output.passages.len() > plan.output.windows.len()
    );
    let result = passages
        .embed_planned_late(text, &plan.output)
        .await
        .unwrap();
    assert_eq!(result.output.embeddings.len(), plan.output.passages.len());
    let report = json!({"passed":true,"profile":profile,"retrieval":rankings,
        "planning":{"plan":plan.output,"token_ranges":result.output.token_ranges},
        "document_count":vectors.len(),"dimension":result.output.embeddings[0].len()});
    std::fs::write(
        std::env::var("HADES_VERIFIED_LATE_REPORT").unwrap(),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
