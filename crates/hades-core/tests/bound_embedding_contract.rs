//! Profile-bound HTTP contracts over private sockets; no model, GPU, or database.
#![cfg(unix)]

use std::time::Duration;

use hades_core::embedding_profile::{
    ArtifactManifest, ChunkPolicy, EmbeddingProfile, EmbeddingRole, ProfileState, ProviderReceipt,
    WorkloadKind,
};
use hades_core::persephone::bound_embedding::{BoundEmbeddingClient, ProviderConnection};
use hades_core::persephone::embedding::{EmbeddingEndpoint, EmbeddingError};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[path = "bound_embedding_contract/late.rs"]
mod late;

struct Fixture {
    manifest: ArtifactManifest,
    profile: EmbeddingProfile,
    receipt: ProviderReceipt,
}

impl Fixture {
    /// Use the shared portable contract, including its non-Jina vector width.
    fn new(late: bool) -> Self {
        let wire: Value =
            serde_json::from_str(include_str!("fixtures/embedding_profile.json")).unwrap();
        let manifest =
            ArtifactManifest::from_json(&serde_json::to_vec(&wire["manifest"]).unwrap()).unwrap();
        let mut profile =
            EmbeddingProfile::from_json(&serde_json::to_vec(&wire["profile"]).unwrap(), &manifest)
                .unwrap();
        if late {
            profile.capabilities.late_chunking = true;
            profile.capabilities.tokenizer_offsets = true;
            profile.workloads.documents.as_mut().unwrap().chunking = ChunkPolicy::Late {
                context_cap_tokens: 1000,
                minimum_window_overlap_tokens: 100,
                passages: None,
            };
        }
        Self {
            manifest,
            profile,
            receipt: serde_json::from_value(wire["receipt"].clone()).unwrap(),
        }
    }

    /// Build a normal or boundary-pooled response carrying the observed receipt.
    fn response(&self, count: usize, late: bool) -> Value {
        let data: Vec<_> = (0..count)
            .map(|i| {
                let mut row = json!({"index":i,"embedding":vec![0.5; 384]});
                if late {
                    row["index"] = json!(0);
                    row["chunk_index"] = json!(i);
                    row["char_start"] = json!(i * 2);
                    row["char_end"] = json!((i + 1) * 2);
                }
                row
            })
            .collect();
        json!({"model":self.profile.model.id,"receipt":self.receipt,"data":data})
    }

    /// Connect one fixed role through the trusted synthetic provider mapping.
    async fn connect(&self, mock: &Mock, role: EmbeddingRole) -> BoundEmbeddingClient {
        BoundEmbeddingClient::connect(
            &self.profile,
            &self.manifest,
            WorkloadKind::Documents,
            role,
            mock.connection(),
            &self.receipt,
        )
        .await
        .unwrap()
    }
}

struct Mock {
    path: std::path::PathBuf,
    server: Option<tokio::task::JoinHandle<Vec<Value>>>,
    _directory: tempfile::TempDir,
}

impl Mock {
    /// Serve exact raw response bodies so malformed/duplicate identity fields are testable.
    fn new(replies: Vec<(u16, String)>) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("provider.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let server = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(10), async move {
                let mut requests = Vec::new();
                for (status, response) in replies {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    let mut headers = Vec::new();
                    while !headers.ends_with(b"\r\n\r\n") {
                        headers.push(stream.read_u8().await.unwrap());
                        assert!(headers.len() < 8192);
                    }
                    let headers = String::from_utf8(headers).unwrap();
                    assert!(headers.starts_with("POST /v1/embeddings HTTP/1.1\r\n")
                        || headers.starts_with("POST /v1/passages HTTP/1.1\r\n")
                        || headers.starts_with("POST /v1/late-passages HTTP/1.1\r\n")
                        || headers.starts_with("POST /v1/late-embeddings HTTP/1.1\r\n"));
                    let length: usize = headers.lines().find_map(|line| {
                        line.to_lowercase().strip_prefix("content-length:").map(|n| n.trim().parse().unwrap())
                    }).unwrap();
                    assert!(length < 3 * 1024 * 1024);
                    let mut body = vec![0; length];
                    stream.read_exact(&mut body).await.unwrap();
                    let mut request: Value = serde_json::from_slice(&body).unwrap();
                    request["_fixture_path"] = json!(headers.lines().next().unwrap());
                    requests.push(request);
                    let wire = format!("HTTP/1.1 {status} Response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len());
                    // Rejection by the client's byte limit can close the socket early.
                    let _ = stream.write_all(wire.as_bytes()).await;
                }
                requests
            }).await.expect("mock requests must complete before deadline")
        });
        Self {
            path,
            server: Some(server),
            _directory: directory,
        }
    }

    /// Default successful JSON responses for ordinary contract tests.
    fn json(replies: Vec<Value>) -> Self {
        Self::new(
            replies
                .into_iter()
                .map(|body| (200, body.to_string()))
                .collect(),
        )
    }

    /// Resolve only operational settings, with no implicit model or task defaults.
    fn connection(&self) -> ProviderConnection {
        ProviderConnection {
            provider_id: "fixture-local-provider".into(),
            endpoint: EmbeddingEndpoint::Unix(self.path.clone()),
            timeout: Duration::from_secs(5),
            max_response_bytes: 1024 * 1024,
        }
    }

    /// Join the provider and return the actual wire requests for assertions.
    async fn finish(mut self) -> Vec<Value> {
        self.server.take().unwrap().await.unwrap()
    }
}

impl Drop for Mock {
    /// Cancel outstanding fixture work if a test assertion fails.
    fn drop(&mut self) {
        if let Some(server) = &self.server {
            server.abort();
        }
    }
}

/// Describe a tokenizer-owned complete plan independently of the client's expected receipt.
fn passage_plan(fixture: &Fixture, text: &str) -> Value {
    json!({"format":"hades-token-passages-v1", "model":fixture.profile.model.id,
        "task":"retrieval.passage", "receipt":fixture.receipt,
        "source_sha256":Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>(),
        "passage_tokens":500,"overlap_tokens":100,"max_input_tokens":2048,
        "spans":[{"start_byte":0,"end_byte":text.len(),"body_tokens":3,"input_tokens":6,"overlap_tokens":0}]})
}

/// The database owns policy/task/identity; the response preserves source-byte coordinates.
#[tokio::test]
async fn plans_with_explicit_policy_and_same_provider_pin() {
    let fixture = Fixture::new(false);
    let text = "café e\u{301} 東京";
    let mock = Mock::json(vec![passage_plan(&fixture, text)]);
    let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
    let result = client.plan_passages(text).await.unwrap();
    assert_eq!(result.binding, *client.binding());
    assert_eq!(result.output.spans[0].end_byte, text.len());
    let requests = mock.finish().await;
    assert_eq!(requests[0]["_fixture_path"], "POST /v1/passages HTTP/1.1");
    assert_eq!(requests[0]["input"], text);
    assert_eq!(requests[0]["passage_tokens"], 500);
    assert_eq!(requests[0]["overlap_tokens"], 100);
    assert_eq!(requests[0]["max_input_tokens"], 2048);
    assert_eq!(requests[0]["expected_receipt"], json!(fixture.receipt));
}

/// Overlap is measured independently; zero-overlap profiles require adjacent source spans.
#[tokio::test]
async fn planning_checks_overlap_and_effective_serving_ceiling() {
    let text = "café e\u{301} 東京";
    for variant in [
        "valid",
        "adjacent",
        "overlap_count",
        "serving_ceiling",
        "zero_policy",
    ] {
        let mut fixture = Fixture::new(false);
        let mut response = passage_plan(&fixture, text);
        response["spans"] = json!([
            {"start_byte":0,"end_byte":6,"body_tokens":3,"input_tokens":6,"overlap_tokens":0},
            {"start_byte":3,"end_byte":text.len(),"body_tokens":6,"input_tokens":9,"overlap_tokens":2}
        ]);
        match variant {
            "adjacent" => {
                response["spans"][1]["start_byte"] = json!(6);
                response["spans"][1]["overlap_tokens"] = json!(0);
            }
            "overlap_count" => response["spans"][1]["overlap_tokens"] = json!(101),
            "serving_ceiling" => response["max_input_tokens"] = json!(8),
            "zero_policy" => {
                fixture
                    .profile
                    .workloads
                    .documents
                    .as_mut()
                    .unwrap()
                    .chunking = ChunkPolicy::Conventional {
                    passage_tokens: 500,
                    overlap_tokens: 0,
                };
                response["overlap_tokens"] = json!(0);
                response["spans"][1]["overlap_tokens"] = json!(0);
            }
            _ => {}
        }
        let mock = Mock::json(vec![response]);
        let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
        assert_eq!(
            client.plan_passages(text).await.is_ok(),
            matches!(variant, "valid" | "adjacent"),
            "{variant}"
        );
        mock.finish().await;
    }
}

/// Small plan metadata must not authorize more than 4 MiB of repeated source allocation.
#[tokio::test]
async fn planning_bounds_aggregate_overlap_bytes() {
    let fixture = Fixture::new(false);
    let text = "x".repeat(1024 * 1024);
    let mut response = passage_plan(&fixture, &text);
    response["spans"] = json!(
        (0..5)
            .map(|i| json!({
                "start_byte":i,"end_byte":text.len()-4+i,"body_tokens":3,"input_tokens":6,
                "overlap_tokens":if i == 0 { 0 } else { 1 }
            }))
            .collect::<Vec<_>>()
    );
    let mock = Mock::json(vec![response]);
    let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
    let error = client
        .plan_passages(&text)
        .await
        .err()
        .expect("overlap amplification must fail");
    assert!(
        error
            .to_string()
            .contains("overlapping passages exceed 4 MiB"),
        "{error}"
    );
    mock.finish().await;
}

/// Reject duplicate/unknown/coerced fields before a plan can authorize slicing or inference.
#[tokio::test]
async fn planning_wire_retains_strict_duplicate_and_type_checks() {
    let fixture = Fixture::new(false);
    let original = passage_plan(&fixture, "é apple").to_string();
    for raw in [
        original.replacen("\"spans\":", "\"spans\":[],\"spans\":", 1),
        original.replacen("\"end_byte\":", "\"end_byte\":8,\"end_byte\":", 1),
        original.replacen(
            "\"passage_tokens\":",
            "\"passage_tokens\":500,\"passage_tokens\":",
            1,
        ),
        original.replacen("\"body_tokens\":3", "\"body_tokens\":3.0", 1),
        original.replacen("\"body_tokens\":3", "\"body_tokens\":\"3\"", 1),
        original.replacen("\"body_tokens\":3", "\"body_tokens\":3,\"unknown\":1", 1),
        original.replacen("\"spans\":", "\"unknown\":1,\"spans\":", 1),
    ] {
        let mock = Mock::new(vec![(200, raw)]);
        let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
        assert!(client.plan_passages("é apple").await.is_err());
        assert_eq!(mock.finish().await.len(), 1);
    }
}

/// Guard whole-source coverage, Unicode boundaries, metadata, and finite request fan-out.
#[tokio::test]
async fn plans_refuse_invalid_source_coverage_and_resource_amplification() {
    let fixture = Fixture::new(false);
    let original = passage_plan(&fixture, "é apple");
    let mut responses = Vec::new();
    for (field, value) in [
        ("start_byte", json!(1)),
        ("end_byte", json!(1)),
        ("end_byte", json!(7)),
        ("end_byte", json!(9)),
        ("body_tokens", json!(501)),
        ("input_tokens", json!(0)),
        ("input_tokens", json!(2049)),
        ("overlap_tokens", json!(1)),
    ] {
        let mut response = original.clone();
        response["spans"][0][field] = value;
        responses.push(response);
    }
    for (field, value) in [
        ("format", json!("other")),
        ("source_sha256", json!("0".repeat(64))),
        ("model", json!("other")),
        ("task", json!("retrieval.query")),
        ("passage_tokens", json!(499)),
        ("overlap_tokens", json!(99)),
        ("max_input_tokens", json!(0)),
        ("max_input_tokens", json!(2049)),
        ("spans", json!([])),
        ("spans", json!(vec![original["spans"][0].clone(); 129])),
    ] {
        let mut response = original.clone();
        response[field] = value;
        responses.push(response);
    }
    for ranges in [
        vec![(0, 2), (3, 8)],
        vec![(0, 4), (0, 8)],
        vec![(0, 8), (2, 6)],
    ] {
        let mut response = original.clone();
        response["spans"]=json!(ranges.into_iter().map(|(start,end)| {
            json!({"start_byte":start,"end_byte":end,"body_tokens":2,"input_tokens":5,"overlap_tokens":0})
        }).collect::<Vec<_>>());
        responses.push(response);
    }
    for response in responses {
        let mock = Mock::json(vec![response]);
        let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
        assert!(client.plan_passages("é apple").await.is_err());
        mock.finish().await;
    }
}

/// Queries, late policies, blank text and oversized sources are refused without transport.
#[tokio::test]
async fn unsupported_planning_is_rejected_before_contact() {
    for (late, role, text) in [
        (false, EmbeddingRole::Query, "one".into()),
        (true, EmbeddingRole::Passage, "one".into()),
        (false, EmbeddingRole::Passage, " \n\t".into()),
        (
            false,
            EmbeddingRole::Passage,
            "x".repeat(2 * 1024 * 1024 + 1),
        ),
    ] {
        let fixture = Fixture::new(late);
        let mock = Mock::json(vec![]);
        let client = fixture.connect(&mock, role).await;
        assert!(client.plan_passages(&text).await.is_err());
        assert!(mock.finish().await.is_empty());
    }
}

/// Resolve both task roles and a non-default width without caller overrides.
#[tokio::test]
async fn derives_explicit_model_dimension_and_asymmetric_tasks_from_profile() {
    for role in [EmbeddingRole::Passage, EmbeddingRole::Query] {
        let fixture = Fixture::new(false);
        let mock = Mock::json(vec![fixture.response(2, false)]);
        let client = fixture.connect(&mock, role).await;
        let result = client
            .embed(&["one".into(), "two".into()], None)
            .await
            .unwrap();
        assert_eq!(result.output.dimension, 384);
        assert_eq!(result.output.embeddings.len(), 2);
        assert_eq!(result.binding, *client.binding());
        assert_eq!(result.binding.expected, fixture.receipt);
        let requests = mock.finish().await;
        assert_eq!(requests[0]["model"], fixture.profile.model.id);
        assert_eq!(
            requests[0]["task"],
            if role == EmbeddingRole::Passage {
                "retrieval.passage"
            } else {
                "retrieval.query"
            }
        );
        assert_eq!(
            requests[0]["expected_receipt"],
            serde_json::to_value(fixture.receipt).unwrap()
        );
    }
}

/// Missing identity or drift in any receipt component must discard vectors.
#[tokio::test]
async fn rejects_missing_malformed_and_changed_receipts() {
    let fixture = Fixture::new(false);
    let original = fixture.response(1, false);
    let mut responses = Vec::new();
    let mut missing = original.clone();
    missing.as_object_mut().unwrap().remove("receipt");
    responses.push(missing);
    for (field, value) in [
        ("format", json!("hades-provider-receipt-v2")),
        ("artifact_md5", json!("0".repeat(32))),
        ("space_id", json!("0".repeat(64))),
        ("serving_generation", json!("load-fixture-2")),
        ("serving_generation", json!("")),
        ("serving_generation", json!("bad\nlabel")),
        ("space_id", json!("malformed")),
        ("extra", json!(true)),
    ] {
        let mut response = original.clone();
        response["receipt"][field] = value;
        responses.push(response);
    }
    let mut null = original.clone();
    null["receipt"] = Value::Null;
    responses.push(null);
    for response in responses {
        let mock = Mock::json(vec![response]);
        let client = fixture.connect(&mock, EmbeddingRole::Query).await;
        assert!(matches!(
            client.embed(&["one".into()], None).await,
            Err(EmbeddingError::InvalidResponse(_))
        ));
        assert_eq!(mock.finish().await.len(), 1);
    }
}

/// Even equal duplicate keys are ambiguous and cannot certify a response.
#[tokio::test]
async fn duplicate_identity_keys_are_rejected_before_value_parsing() {
    let fixture = Fixture::new(false);
    let response = fixture.response(1, false).to_string();
    for (old, replacement) in [
        (
            "\"receipt\":",
            format!(
                "\"receipt\":{},\"receipt\":",
                serde_json::to_string(&fixture.receipt).unwrap()
            ),
        ),
        (
            "\"model\":",
            format!("\"model\":{},\"model\":", json!(fixture.profile.model.id)),
        ),
        (
            "\"serving_generation\":",
            "\"serving_generation\":\"load-fixture-1\",\"serving_generation\":".into(),
        ),
    ] {
        let mock = Mock::new(vec![(200, response.replacen(old, &replacement, 1))]);
        let client = fixture.connect(&mock, EmbeddingRole::Query).await;
        let error = client.embed(&["one".into()], None).await.err().unwrap();
        assert!(error.to_string().contains("duplicate field"), "{error}");
        mock.finish().await;
    }
}

/// Receipt presence cannot excuse bad vectors or legacy model-path aliases.
#[tokio::test]
async fn receipt_checks_preserve_vector_validation_and_require_exact_model_names() {
    let mut fixture = Fixture::new(false);
    fixture.profile.model.id = "jinaai/jina-embeddings-v4".into();
    let original = fixture.response(1, false);
    let mut responses = Vec::new();
    for model in ["other-model", "/models/jinaai--jina-embeddings-v4"] {
        let mut response = original.clone();
        response["model"] = json!(model);
        responses.push(response);
    }
    for vector in [json!(vec![0.5; 2]), json!(vec![1e100; 384])] {
        let mut response = original.clone();
        response["data"][0]["embedding"] = vector;
        responses.push(response);
    }
    for response in responses {
        let mock = Mock::json(vec![response]);
        let client = fixture.connect(&mock, EmbeddingRole::Query).await;
        assert!(client.embed(&["one".into()], None).await.is_err());
        mock.finish().await;
    }
}

/// Reject a later generation without exposing earlier batches or sending more.
#[tokio::test]
async fn receipt_drift_discards_completed_batches_without_rebinding() {
    let fixture = Fixture::new(false);
    let first = fixture.response(1, false);
    let mut reloaded = first.clone();
    reloaded["receipt"]["serving_generation"] = json!("reloaded");
    let mock = Mock::json(vec![first, reloaded]);
    let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
    let result = client
        .embed(&["one".into(), "two".into(), "three".into()], Some(1))
        .await;
    assert!(
        result
            .err()
            .unwrap()
            .to_string()
            .contains("receipt changed")
    );
    let requests = mock.finish().await;
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(request["expected_receipt"], json!(fixture.receipt));
    }
}

/// Adaptive OOM retries change batch sizes while preserving the identity pin.
#[tokio::test]
async fn oom_splits_keep_the_original_pin_on_every_retry() {
    let fixture = Fixture::new(false);
    let response = fixture.response(1, false).to_string();
    let mock = Mock::new(vec![
        (500, "CUDA out of memory".into()),
        (200, response.clone()),
        (200, response),
    ]);
    let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
    let result = client
        .embed(&["one".into(), "two".into()], None)
        .await
        .unwrap();
    assert_eq!(result.output.embeddings.len(), 2);
    let requests = mock.finish().await;
    assert_eq!(
        requests
            .iter()
            .map(|r| r["input"].as_array().unwrap().len())
            .collect::<Vec<_>>(),
        [2, 1, 1]
    );
    for request in requests {
        assert_eq!(request["expected_receipt"], json!(fixture.receipt));
    }
}

/// Every late input must report the same generation before returning any result.
#[tokio::test]
async fn late_inputs_pin_receipts_and_discard_results_on_reload() {
    for reload in [false, true] {
        let fixture = Fixture::new(true);
        let first = fixture.response(2, true);
        let mut second = first.clone();
        if reload {
            second["receipt"]["serving_generation"] = json!("reloaded");
        }
        let mock = Mock::json(vec![first, second]);
        let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
        let result = client
            .embed_late_chunked(
                &["abcd".into(), "efgh".into()],
                &[vec![(0, 2), (2, 4)], vec![(0, 2), (2, 4)]],
            )
            .await;
        assert_eq!(result.is_err(), reload);
        if let Ok(result) = result {
            assert_eq!(result.output.per_input.len(), 2);
            assert_eq!(result.binding.expected, fixture.receipt);
        }
        for request in mock.finish().await {
            assert_eq!(request["task"], "retrieval.passage");
            assert_eq!(request["expected_receipt"], json!(fixture.receipt));
        }
    }
}

/// Refuse invalid profile/provider combinations before attempting inference.
#[tokio::test]
async fn invalid_bindings_fail_before_transport() {
    for case in 0..8 {
        let mut fixture = Fixture::new(false);
        let mock = Mock::json(vec![]);
        let mut provider = mock.connection();
        let mut workload = WorkloadKind::Documents;
        match case {
            0 => fixture.profile.state = ProfileState::Initializing,
            1 => workload = WorkloadKind::Code,
            2 => provider.provider_id = "another-provider".into(),
            3 => fixture.receipt.artifact_md5 = "0".repeat(32).try_into().unwrap(),
            4 => fixture.receipt.space_id = "0".repeat(64).try_into().unwrap(),
            5 => {
                fixture
                    .profile
                    .workloads
                    .documents
                    .as_mut()
                    .unwrap()
                    .space
                    .representation
                    .dimension = 0
            }
            6 => provider.max_response_bytes = 0,
            7 => provider.timeout = Duration::ZERO,
            _ => unreachable!(),
        }
        let result = BoundEmbeddingClient::connect(
            &fixture.profile,
            &fixture.manifest,
            workload,
            EmbeddingRole::Query,
            provider,
            &fixture.receipt,
        )
        .await;
        assert!(
            matches!(result, Err(EmbeddingError::Contract(_))),
            "case {case}"
        );
        assert!(mock.finish().await.is_empty());
    }
}

/// Empty work and unsupported modes must not produce unchecked success results.
#[tokio::test]
async fn request_policy_rejects_empty_work_and_chunking_fallback_before_transport() {
    for late in [false, true] {
        let fixture = Fixture::new(late);
        let mock = Mock::json(vec![]);
        let query = fixture.connect(&mock, EmbeddingRole::Query).await;
        let passage = fixture.connect(&mock, EmbeddingRole::Passage).await;
        assert!(query.embed(&[], None).await.is_err());
        assert!(
            query
                .embed_late_chunked(&["ab".into()], &[vec![(0, 2)]])
                .await
                .is_err()
        );
        if late {
            assert!(passage.embed(&["ab".into()], None).await.is_err());
        }
        for bounds in [vec![], vec![vec![]], vec![vec![(0, 3)]], vec![vec![(1, 1)]]] {
            assert!(matches!(
                passage.embed_late_chunked(&["é😀".into()], &bounds).await,
                Err(EmbeddingError::Contract(_))
            ));
        }
        if !late {
            assert!(
                passage
                    .embed_late_chunked(&["ab".into()], &[vec![(0, 2)]])
                    .await
                    .is_err()
            );
        }
        assert!(mock.finish().await.is_empty());
    }
}

/// Enforce the caller's byte budget before parsing a response or its receipt.
#[tokio::test]
async fn bound_transport_limits_response_bytes() {
    let fixture = Fixture::new(false);
    let response = fixture.response(1, false).to_string();
    let mock = Mock::new(vec![(200, response.clone())]);
    let mut provider = mock.connection();
    provider.max_response_bytes = response.len() - 1;
    let client = BoundEmbeddingClient::connect(
        &fixture.profile,
        &fixture.manifest,
        WorkloadKind::Documents,
        EmbeddingRole::Query,
        provider,
        &fixture.receipt,
    )
    .await
    .unwrap();
    assert!(
        client
            .embed(&["one".into()], None)
            .await
            .err()
            .unwrap()
            .to_string()
            .contains("length limit exceeded")
    );
    mock.finish().await;
}

/// Concurrent clients and clones cannot inherit mutations from another binding.
#[tokio::test]
async fn independent_clients_and_clones_retain_their_original_profile_snapshot() {
    let mut first = Fixture::new(false);
    let mut second = Fixture::new(false);
    second.profile.generation = "second-corpus".into();
    let space = &mut second.profile.workloads.documents.as_mut().unwrap().space;
    space.query.prompt = "Different query: ".into();
    second.receipt.space_id = space.id().unwrap();
    let a = Mock::json(vec![first.response(1, false), first.response(1, false)]);
    let b = Mock::json(vec![second.response(1, false)]);
    let client_a = first.connect(&a, EmbeddingRole::Query).await;
    let clone_a = client_a.clone();
    let client_b = second.connect(&b, EmbeddingRole::Query).await;
    // A caller mutating its profile object does not retarget an admitted client.
    first.profile.revision += 1;
    first.profile.generation = "changed-after-binding".into();
    first.receipt.serving_generation = "changed-after-binding".into();
    let texts = ["query".into()];
    let (result_a, result_b) =
        tokio::join!(client_a.embed(&texts, None), client_b.embed(&texts, None));
    let result_a = result_a.unwrap();
    let result_b = result_b.unwrap();
    assert_eq!(result_a.binding.profile_revision, 1);
    assert_eq!(result_a.binding.generation, "corpus-fixture-1");
    assert_eq!(result_b.binding.generation, "second-corpus");
    assert_ne!(
        result_a.binding.expected.space_id,
        result_b.binding.expected.space_id
    );
    assert_eq!(
        clone_a.embed(&texts, None).await.unwrap().binding,
        result_a.binding
    );
    for request in a.finish().await {
        assert_eq!(
            request["expected_receipt"],
            json!(result_a.binding.expected)
        );
    }
    assert_eq!(
        b.finish().await[0]["expected_receipt"],
        json!(result_b.binding.expected)
    );
}

/// A late corpus still embeds each query as one conventional query vector.
#[tokio::test]
async fn late_passage_profiles_still_use_conventional_query_embedding() {
    let fixture = Fixture::new(true);
    let mock = Mock::json(vec![fixture.response(1, false)]);
    let client = fixture.connect(&mock, EmbeddingRole::Query).await;
    assert_eq!(
        client
            .embed(&["query".into()], None)
            .await
            .unwrap()
            .output
            .dimension,
        384
    );
    let requests = mock.finish().await;
    assert_eq!(requests[0]["task"], "retrieval.query");
    assert!(requests[0].get("late_chunk").is_none());
}

/// Provider admission refusals must not trigger fallback or automatic rebinding.
#[tokio::test]
async fn provider_refusal_is_not_retried_or_downgraded() {
    let fixture = Fixture::new(false);
    let mock = Mock::new(vec![(
        409,
        json!({"error":{"code":"PE_RECEIPT_MISMATCH"}}).to_string(),
    )]);
    let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
    assert!(matches!(
        client.embed(&["one".into(), "two".into()], None).await,
        Err(EmbeddingError::Http { status: 409, .. })
    ));
    assert_eq!(mock.finish().await.len(), 1);
}
