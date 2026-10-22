//! Opt-in actual-model qualification; all profiles are in memory and no DB is created.
#![cfg(unix)]
use hades_core::{
    config::{RegisteredProviderConfig, RegisteredProvidersConfig},
    embedding_profile::{ChunkPolicy, EmbeddingRole, ProfileState, Workload, WorkloadKind},
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
#[ignore = "run through qualify_verified_code.py with an explicitly approved local registration"]
async fn registered_conventional_workloads() {
    let bytes = std::fs::read(std::env::var("HADES_VERIFIED_CODE_CONTEXT").unwrap()).unwrap();
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
    let documents = observed
        .description()
        .advertised_workload(WorkloadKind::Documents)
        .unwrap();
    let code = observed
        .description()
        .advertised_workload(WorkloadKind::Code)
        .unwrap();
    assert_eq!(documents.receipt.artifact_md5, code.receipt.artifact_md5);
    assert_eq!(
        documents.receipt.serving_generation,
        code.receipt.serving_generation
    );
    assert_ne!(documents.receipt.space_id, code.receipt.space_id);
    assert_ne!(
        documents.capabilities.max_input_tokens,
        code.capabilities.max_input_tokens
    );
    let ceiling = documents
        .capabilities
        .max_input_tokens
        .max(code.capabilities.max_input_tokens);
    let chunking = ChunkPolicy::Conventional {
        passage_tokens: 256.min(ceiling / 2),
        overlap_tokens: 8.min(ceiling / 8),
    };
    let mut profile = observed
        .document_profile(
            "conventional-code-qualification-v1".into(),
            chunking.clone(),
        )
        .unwrap();
    // Explicit test intent, not inferred database policy or persisted setup state.
    profile.state = ProfileState::Ready;
    profile.capabilities.max_input_tokens = ceiling;
    profile.capabilities.specialized_code = true;
    profile.capabilities.tasks.push("code".into());
    profile.workloads.code = Some(Workload {
        space: code.space.clone(),
        chunking,
        specialized_code: true,
    });
    profile.validate(observed.manifest()).unwrap();
    let mut report = json!({"passed":false,"profile":profile});
    for (name, kind) in [
        ("documents", WorkloadKind::Documents),
        ("code", WorkloadKind::Code),
    ] {
        let selected = observed.description().advertised_workload(kind).unwrap();
        let passages = observed
            .client(&profile, observed.manifest(), kind, EmbeddingRole::Passage)
            .await
            .unwrap();
        let queries = observed
            .client(&profile, observed.manifest(), kind, EmbeddingRole::Query)
            .await
            .unwrap();
        assert_eq!(&passages.binding().expected, selected.receipt);
        assert_eq!(&queries.binding().expected, selected.receipt);
        let cases = &context["cases"][name];
        let documents = cases["documents"].as_array().unwrap();
        let texts: Vec<_> = documents
            .iter()
            .map(|row| row["text"].as_str().unwrap().to_owned())
            .collect();
        let vectors = passages.embed(&texts, None).await.unwrap();
        assert_eq!(vectors.output.embeddings.len(), documents.len());
        assert_eq!(&vectors.binding.expected, selected.receipt);
        let mut rankings = Vec::new();
        for query in cases["queries"].as_array().unwrap() {
            let text = query["text"].as_str().unwrap().to_owned();
            let result = queries
                .embed(std::slice::from_ref(&text), None)
                .await
                .unwrap();
            assert_eq!(&result.binding.expected, selected.receipt);
            let vector = &result.output.embeddings[0];
            let best = vectors
                .output
                .embeddings
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
                "first":documents[best]["id"],"rank":1,
                "score":cosine(vector,&vectors.output.embeddings[best]),"repeat_cosine":repeat_cosine}));
        }
        let text = context["planning_texts"][name].as_str().unwrap();
        let plan = passages.plan_passages(text).await.unwrap();
        assert_eq!(
            plan.output.max_input_tokens,
            selected.capabilities.max_input_tokens
        );
        assert!(plan.output.spans.len() > 1);
        let texts: Vec<_> = plan
            .output
            .spans
            .iter()
            .map(|span| text[span.start_byte..span.end_byte].to_owned())
            .collect();
        let result = passages.embed(&texts, None).await.unwrap();
        assert_eq!(result.output.embeddings.len(), plan.output.spans.len());
        assert_eq!(&result.binding.expected, selected.receipt);
        report[name] = json!({"retrieval":rankings,"planning":plan.output,
            "document_count":documents.len(),"planned_embeddings":result.output.embeddings.len(),
            "dimension":result.output.embeddings[0].len()});
    }
    report["passed"] = json!(true);
    std::fs::write(
        std::env::var("HADES_VERIFIED_CODE_REPORT").unwrap(),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
