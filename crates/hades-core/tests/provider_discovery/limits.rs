//! Planning must respect the selected observation while retaining saved policy.

use super::{Fixture, Peer, Reply, workloads::mixed};
use hades_core::embedding_profile::{ChunkPolicy, EmbeddingRole, PassagePolicy, WorkloadKind};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const TEXT: &str = "café 東京";

fn fixture(late: bool, document_limit: u32, code_limit: u32) -> Fixture {
    let mut fixture = mixed();
    fixture.profile.capabilities.max_input_tokens = 8192;
    fixture.profile.capabilities.late_chunking = late;
    fixture.profile.capabilities.tokenizer_offsets = late;
    for (name, workload, limit) in [
        (
            "documents",
            &mut fixture.profile.workloads.documents,
            document_limit,
        ),
        ("code", &mut fixture.profile.workloads.code, code_limit),
    ] {
        workload.as_mut().unwrap().chunking = if late {
            ChunkPolicy::Late {
                context_cap_tokens: 4096,
                minimum_window_overlap_tokens: 4,
                passages: Some(PassagePolicy {
                    passage_tokens: 1000,
                    overlap_tokens: 0,
                }),
            }
        } else {
            ChunkPolicy::Conventional {
                passage_tokens: 1000,
                overlap_tokens: 0,
            }
        };
        let capabilities = &mut fixture.description["workloads"][name]["capabilities"];
        capabilities["max_input_tokens"] = json!(limit);
        capabilities["late_chunking"] = json!(late);
        capabilities["tokenizer_offsets"] = json!(late);
    }
    fixture
}

/// Independent provider output with valid source coverage, including prompted token counts.
fn plan(fixture: &Fixture, kind: WorkloadKind, limit: u32) -> Value {
    let (name, workload) = match kind {
        WorkloadKind::Documents => ("documents", fixture.profile.workloads.documents.as_ref()),
        WorkloadKind::Code => ("code", fixture.profile.workloads.code.as_ref()),
    };
    let workload = workload.unwrap();
    let receipt = if fixture.description.get("workloads").is_some() {
        &fixture.description["workloads"][name]["receipt"]
    } else {
        &fixture.description["receipt"]
    };
    let spans = json!([{"start_byte":0,"end_byte":TEXT.len(),
        "body_tokens":3,"input_tokens":6,"overlap_tokens":0}]);
    let mut result = json!({"model":fixture.profile.model.id,"task":workload.space.passage.task,
        "receipt":receipt,"max_input_tokens":limit,
        "source_sha256":Sha256::digest(TEXT.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>()});
    match workload.chunking {
        ChunkPolicy::Conventional {
            passage_tokens,
            overlap_tokens,
        } => {
            result["format"] = json!("hades-token-passages-v1");
            result["passage_tokens"] = json!(passage_tokens);
            result["overlap_tokens"] = json!(overlap_tokens);
            result["spans"] = spans;
        }
        ChunkPolicy::Late {
            context_cap_tokens,
            minimum_window_overlap_tokens,
            passages: Some(passages),
        } => {
            result["format"] = json!("hades-late-passages-v1");
            result["passage_tokens"] = json!(passages.passage_tokens);
            result["overlap_tokens"] = json!(passages.overlap_tokens);
            result["context_cap_tokens"] = json!(context_cap_tokens);
            result["minimum_window_overlap_tokens"] = json!(minimum_window_overlap_tokens);
            result["passages"] = spans.clone();
            result["windows"] = spans;
            result["assignments"] = json!([0]);
        }
        _ => unreachable!(),
    }
    result
}

#[tokio::test]
async fn conventional_plans_obey_each_workloads_observed_ceiling() {
    // Either workload may have the smaller serving limit; larger providers cannot raise policy.
    for (document_limit, code_limit) in [(8192, 512), (512, 8192), (16384, 16384)] {
        let fixture = fixture(false, document_limit, code_limit);
        let cases = [
            (WorkloadKind::Documents, document_limit.min(8192)),
            (WorkloadKind::Code, code_limit.min(8192)),
        ];
        let mut replies = vec![Reply::json(fixture.description.clone())];
        for (kind, limit) in cases {
            replies.push(Reply::json(plan(&fixture, kind, limit)));
            replies.push(Reply::json(plan(&fixture, kind, limit + 1)));
            replies.push(Reply::json(plan(&fixture, kind, limit)));
        }
        let peer = Peer::new(replies);
        let observed = fixture
            .catalog(&peer)
            .verify_profile(&fixture.profile, &fixture.manifest)
            .await
            .unwrap();
        for (kind, limit) in cases {
            let client = observed
                .client(
                    &fixture.profile,
                    &fixture.manifest,
                    kind,
                    EmbeddingRole::Passage,
                )
                .await
                .unwrap();
            let clone = client.clone();
            assert_eq!(
                client
                    .plan_passages(TEXT)
                    .await
                    .unwrap()
                    .output
                    .max_input_tokens,
                limit
            );
            assert!(
                client.plan_passages(TEXT).await.is_err(),
                "{kind:?}: plan exceeded {limit}"
            );
            assert_eq!(
                clone
                    .plan_passages(TEXT)
                    .await
                    .unwrap()
                    .output
                    .max_input_tokens,
                limit
            );
        }
        assert_eq!(fixture.profile.capabilities.max_input_tokens, 8192);
        assert_eq!(peer.requests().len(), 7);
        for (_, body) in peer.requests().into_iter().skip(1) {
            let request: Value = serde_json::from_slice(&body).unwrap();
            // These fields describe saved policy; the provider clamps it using its own tokenizer.
            assert_eq!(request["max_input_tokens"], 8192);
            assert_eq!(request["passage_tokens"], 1000);
        }
    }
}

#[tokio::test]
async fn late_plans_and_reused_plans_obey_the_selected_live_ceiling() {
    let fixture = fixture(true, 8192, 512);
    let peer = Peer::new(vec![
        Reply::json(fixture.description.clone()),
        Reply::json(plan(&fixture, WorkloadKind::Documents, 4096)),
        Reply::json(plan(&fixture, WorkloadKind::Code, 512)),
        Reply::json(plan(&fixture, WorkloadKind::Code, 513)),
    ]);
    let observed = fixture
        .catalog(&peer)
        .verify_profile(&fixture.profile, &fixture.manifest)
        .await
        .unwrap();
    let documents = observed
        .client(
            &fixture.profile,
            &fixture.manifest,
            WorkloadKind::Documents,
            EmbeddingRole::Passage,
        )
        .await
        .unwrap();
    assert_eq!(
        documents
            .plan_late_passages(TEXT)
            .await
            .unwrap()
            .output
            .max_input_tokens,
        4096
    );
    let code = observed
        .client(
            &fixture.profile,
            &fixture.manifest,
            WorkloadKind::Code,
            EmbeddingRole::Passage,
        )
        .await
        .unwrap();
    let mut planned = code.plan_late_passages(TEXT).await.unwrap().output;
    assert_eq!(planned.max_input_tokens, 512);
    assert!(code.plan_late_passages(TEXT).await.is_err());
    // Public plans can be modified after planning: recheck before sending any inference.
    planned.max_input_tokens = 513;
    assert!(
        code.clone()
            .embed_planned_late(TEXT, &planned)
            .await
            .is_err()
    );
    assert_eq!(peer.requests().len(), 4);
    for (_, body) in peer.requests().into_iter().skip(1) {
        let request: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(request["context_cap_tokens"], 4096);
        assert_eq!(request["max_input_tokens"], 8192);
    }
}

#[tokio::test]
async fn original_text_provider_keeps_policy_and_checks_its_smaller_live_limit() {
    let mut fixture = Fixture::new();
    fixture.profile.workloads.code = fixture.profile.workloads.documents.clone();
    fixture.description["capabilities"]["max_input_tokens"] = json!(128);
    let peer = Peer::new(vec![
        Reply::json(fixture.description.clone()),
        Reply::json(plan(&fixture, WorkloadKind::Code, 128)),
        Reply::json(plan(&fixture, WorkloadKind::Code, 129)),
    ]);
    let observed = fixture
        .catalog(&peer)
        .verify_profile(&fixture.profile, &fixture.manifest)
        .await
        .unwrap();
    let client = observed
        .client(
            &fixture.profile,
            &fixture.manifest,
            WorkloadKind::Code,
            EmbeddingRole::Passage,
        )
        .await
        .unwrap();
    assert_eq!(
        client
            .plan_passages(TEXT)
            .await
            .unwrap()
            .output
            .max_input_tokens,
        128
    );
    assert!(client.plan_passages(TEXT).await.is_err());
    assert_eq!(fixture.profile.capabilities.max_input_tokens, 2048);
    assert_eq!(peer.requests().len(), 3);
    for (_, body) in peer.requests().into_iter().skip(1) {
        let request: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(request["max_input_tokens"], 2048);
        assert_eq!(request["passage_tokens"], 500);
    }
}
