//! Strict contextual planning/embedding contracts over a private HTTP peer.
use super::*;
use hades_core::embedding_profile::PassagePolicy;

const TEXT: &str = "café 東京";

fn digest(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn fixture() -> Fixture {
    let mut fixture = Fixture::new(true);
    fixture
        .profile
        .workloads
        .documents
        .as_mut()
        .unwrap()
        .chunking = ChunkPolicy::Late {
        context_cap_tokens: 32,
        minimum_window_overlap_tokens: 4,
        passages: Some(PassagePolicy {
            passage_tokens: 3,
            overlap_tokens: 0,
        }),
    };
    fixture
}

fn plan(fixture: &Fixture) -> Value {
    json!({"format":"hades-late-passages-v1","source_sha256":digest(TEXT),
        "model":fixture.profile.model.id,"task":"retrieval.passage","receipt":fixture.receipt,
        "passage_tokens":3,"overlap_tokens":0,"context_cap_tokens":32,
        "minimum_window_overlap_tokens":4,"max_input_tokens":16,
        "passages":[
            {"start_byte":0,"end_byte":6,"body_tokens":3,"input_tokens":6,"overlap_tokens":0},
            {"start_byte":6,"end_byte":12,"body_tokens":2,"input_tokens":5,"overlap_tokens":0}],
        "windows":[{"start_byte":0,"end_byte":12,"body_tokens":6,"input_tokens":9,"overlap_tokens":0}],
        "assignments":[0,0]})
}

fn response(fixture: &Fixture) -> Value {
    let vector = vec![1.0 / (384.0_f32).sqrt(); 384];
    json!({"format":"hades-late-embedding-v1","model":fixture.profile.model.id,
        "source_sha256":digest(TEXT),"receipt":fixture.receipt,"input_tokens":9,
        "data":[
            {"index":0,"embedding":vector,"char_start":0,"char_end":5,"token_start":2,"token_end":5},
            {"index":1,"embedding":vector,"char_start":5,"char_end":7,"token_start":6,"token_end":8}]})
}

#[tokio::test]
async fn complete_contextual_work_preserves_byte_character_and_token_coordinates() {
    let fixture = fixture();
    let mock = Mock::json(vec![plan(&fixture), response(&fixture)]);
    let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
    let planned = client.plan_late_passages(TEXT).await.unwrap();
    let result = client
        .embed_planned_late(TEXT, &planned.output)
        .await
        .unwrap();
    assert_eq!(result.binding, planned.binding);
    assert_eq!(result.output.embeddings.len(), 2);
    assert_eq!(result.output.token_ranges[1].token_start, 6);
    let requests = mock.finish().await;
    assert_eq!(requests.len(), 2, "one inference request per used window");
    assert_eq!(
        requests[0]["_fixture_path"],
        "POST /v1/late-passages HTTP/1.1"
    );
    assert_eq!(requests[0]["context_cap_tokens"], 32);
    assert_eq!(requests[0]["passage_tokens"], 3);
    assert_eq!(requests[1]["boundaries"], json!([[0, 5], [5, 7]]));
    for request in requests {
        assert_eq!(request["expected_receipt"], json!(fixture.receipt));
    }
}

#[tokio::test]
async fn malformed_plans_and_unconfigured_roles_never_start_inference() {
    for invalid in [
        "gap",
        "utf8",
        "assignment",
        "containment",
        "minimum",
        "policy",
        "live",
        "receipt",
    ] {
        let fixture = fixture();
        let mut bad = plan(&fixture);
        match invalid {
            "gap" => bad["passages"][1]["start_byte"] = json!(9),
            "utf8" => bad["passages"][0]["end_byte"] = json!(4),
            "assignment" => bad["assignments"][1] = json!(99),
            "containment" => bad["windows"][0]["end_byte"] = json!(9),
            "minimum" => {
                bad["windows"] = json!([
                    {"start_byte":0,"end_byte":6,"body_tokens":3,"input_tokens":6,"overlap_tokens":0},
                    {"start_byte":6,"end_byte":12,"body_tokens":2,"input_tokens":5,"overlap_tokens":0}]);
                bad["assignments"] = json!([0, 1]);
            }
            "policy" => bad["context_cap_tokens"] = json!(2048),
            "live" => bad["max_input_tokens"] = json!(0),
            "receipt" => bad["receipt"]["serving_generation"] = json!("reloaded"),
            _ => unreachable!(),
        }
        let mock = Mock::json(vec![bad]);
        let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
        assert!(client.plan_late_passages(TEXT).await.is_err(), "{invalid}");
        assert_eq!(mock.finish().await.len(), 1);
    }
    let fixture = Fixture::new(true); // Existing late contracts have no document passage policy.
    let mock = Mock::json(vec![]);
    let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
    assert!(client.plan_late_passages(TEXT).await.is_err());
    assert!(mock.finish().await.is_empty());
}

#[tokio::test]
async fn malformed_contextual_vectors_are_never_returned_as_complete_results() {
    for invalid in [
        "chars",
        "tokens",
        "missing_receipt",
        "extra",
        "zero",
        "norm",
        "hash",
        "count",
        "width",
        "partial",
    ] {
        let fixture = fixture();
        let mut bad = response(&fixture);
        match invalid {
            "chars" => bad["data"][0]["char_end"] = json!(6),
            "tokens" => bad["data"][0]["token_end"] = json!(100),
            "missing_receipt" => {
                bad.as_object_mut().unwrap().remove("receipt");
            }
            "extra" => bad["untrusted"] = json!(true),
            "zero" => bad["data"][0]["embedding"] = json!(vec![0.0; 384]),
            "norm" => bad["data"][0]["embedding"] = json!(vec![1.0; 384]),
            "hash" => bad["source_sha256"] = json!(digest("other")),
            "count" => bad["input_tokens"] = json!(10),
            "width" => bad["data"][0]["embedding"] = json!([1.0]),
            "partial" => {
                bad["data"].as_array_mut().unwrap().pop();
            }
            _ => unreachable!(),
        }
        let mock = Mock::json(vec![plan(&fixture), bad]);
        let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
        let planned = client.plan_late_passages(TEXT).await.unwrap();
        assert!(
            client
                .embed_planned_late(TEXT, &planned.output)
                .await
                .is_err(),
            "{invalid}"
        );
        assert_eq!(mock.finish().await.len(), 2);
    }
}

#[tokio::test]
async fn a_later_window_failure_discards_earlier_contextual_vectors() {
    let fixture = fixture();
    let mut planned = plan(&fixture);
    planned["windows"] = json!([
        {"start_byte":0,"end_byte":9,"body_tokens":6,"input_tokens":9,"overlap_tokens":0},
        {"start_byte":3,"end_byte":12,"body_tokens":6,"input_tokens":9,"overlap_tokens":4}]);
    planned["assignments"] = json!([0, 1]);
    let mut first = response(&fixture);
    first["source_sha256"] = json!(digest(&TEXT[..9]));
    first["data"].as_array_mut().unwrap().truncate(1);
    let mock = Mock::new(vec![
        (200, planned.to_string()),
        (200, first.to_string()),
        (
            409,
            json!({"error":{"message":"provider reloaded","code":"PE_RECEIPT_MISMATCH"}})
                .to_string(),
        ),
    ]);
    let client = fixture.connect(&mock, EmbeddingRole::Passage).await;
    let plan = client.plan_late_passages(TEXT).await.unwrap();
    assert!(client.embed_planned_late(TEXT, &plan.output).await.is_err());
    assert_eq!(mock.finish().await.len(), 3);
}
