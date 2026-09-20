//! Frozen real-model vectors: ranking parity without model loading or services.
use hades_core::retrieval::TopK;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[test]
fn streamed_ranking_preserves_frozen_jina_and_legacy_rankings() {
    let dataset_bytes = include_bytes!("../../../evaluation/retrieval/repository-v1.json");
    let bytes = include_bytes!("../../../evaluation/retrieval/results-v1/vectors.f32");
    let provenance: Value = serde_json::from_slice(include_bytes!(
        "../../../evaluation/retrieval/results-v1/provenance.json"
    ))
    .unwrap();
    assert_eq!(
        Sha256::digest(dataset_bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        provenance["dataset_sha256"].as_str().unwrap()
    );
    assert_eq!(
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        provenance["vectors_sha256"].as_str().unwrap()
    );
    let dataset: Value = serde_json::from_slice(dataset_bytes).unwrap();
    let metrics: Value = serde_json::from_slice(include_bytes!(
        "../../../evaluation/retrieval/results-v1/metrics.json"
    ))
    .unwrap();
    let docs = dataset["documents"].as_array().unwrap();
    let queries = dataset["queries"].as_array().unwrap();
    let vectors: Vec<Vec<f32>> = bytes
        .as_chunks::<{ 2048 * 4 }>()
        .0
        .iter()
        .map(|row| {
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect()
        })
        .collect();
    assert_eq!(bytes.len(), (docs.len() + queries.len()) * 2048 * 4);
    for (qi, query) in queries.iter().enumerate() {
        let q = &vectors[docs.len() + qi];
        let mut top = TopK::new(q.clone(), "frozen-jina-v4".into(), "parent_key", 10).unwrap();
        let mut legacy = Vec::new();
        for (i, doc) in docs.iter().enumerate() {
            let v = &vectors[i];
            top.insert(json!({"chunk_key":doc["id"],"parent_key":"fixture","model":"frozen-jina-v4","dimension":2048,"embedding":v})).unwrap();
            // Independent reference: the pre-remediation f32 full-scan formula.
            let dot: f32 = q.iter().zip(v).map(|(a, b)| a * b).sum();
            let nq = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nv = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            legacy.push((dot / (nq * nv), doc["id"].as_str().unwrap()));
        }
        legacy.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
        let actual = top.into_items();
        let actual: Vec<_> = actual
            .iter()
            .map(|v| v["chunk_key"].as_str().unwrap())
            .collect();
        let reference = metrics["per_query"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["query"] == query["id"] && r["method"] == "vector")
            .unwrap();
        let expected: Vec<_> = reference["ranking"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(actual, expected, "frozen Python reference: {}", query["id"]);
        assert_eq!(
            actual,
            legacy
                .iter()
                .take(10)
                .map(|(_, id)| *id)
                .collect::<Vec<_>>(),
            "legacy f32 reference: {}",
            query["id"]
        );
    }
}
