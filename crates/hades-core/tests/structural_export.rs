//! Private Unix-socket structural export failure contracts.
use hades_core::db::{ArangoClient, ArangoPool};
use hades_core::graph::export::{ExportConfig, export_embeddings};
use hades_core::graph::types::IDMap;
use serde_json::json;

#[allow(dead_code)]
#[path = "common/cursor_mock.rs"]
mod cursor_mock;
use cursor_mock::{Mock, Reply};

fn ids() -> IDMap {
    let mut map = IDMap::new();
    map.get_or_create("nodes/a");
    map.get_or_create("nodes/b");
    map
}

#[tokio::test]
async fn failed_export_batch_must_not_report_success() {
    let mock = Mock::new(vec![Reply {
        status: 500,
        body: json!({"error":true,"code":500,"errorNum":1203,"errorMessage":"synthetic rejected update"}),
        gate: None,
    }]).await;
    let result =
        export_embeddings(&mock.pool, &ids(), &[0.1, 0.2], 1, &ExportConfig::default()).await;
    assert!(result.is_err(), "failed write reported success: {result:?}");
}

#[tokio::test]
async fn short_export_batch_must_not_report_success() {
    let mock = Mock::new(vec![Reply::page(json!({
        "error":false,"result":[1],"hasMore":false,
        "extra":{"stats":{"writesExecuted":1,"writesIgnored":1}}
    }))])
    .await;
    let result =
        export_embeddings(&mock.pool, &ids(), &[0.1, 0.2], 1, &ExportConfig::default()).await;
    assert!(
        result.is_err(),
        "partial write reported success: {result:?}"
    );
}

#[tokio::test]
async fn zero_batch_size_must_return_error_instead_of_panicking() {
    let mock = Mock::new(vec![]).await;
    let pool = mock.pool.clone();
    let result = tokio::spawn(async move {
        export_embeddings(
            &pool,
            &ids(),
            &[0.1, 0.2],
            1,
            &ExportConfig { chunk_size: 0 },
        )
        .await
    })
    .await;
    assert!(result.is_ok(), "invalid configuration panicked: {result:?}");
    assert!(result.unwrap().is_err());
}

#[tokio::test]
async fn invalid_vectors_fail_before_requests() {
    for (values, dim) in [
        (vec![f32::NAN, 0.], 1),
        (vec![f32::INFINITY, 0.], 1),
        (vec![], 0),
        (vec![], usize::MAX),
    ] {
        let mut mock = Mock::new(vec![]).await;
        assert!(
            export_embeddings(&mock.pool, &ids(), &values, dim, &ExportConfig::default())
                .await
                .is_err()
        );
        assert!(mock.events.try_recv().is_err());
    }
}

#[tokio::test]
async fn malformed_acknowledgments_fail() {
    for body in [
        json!({"result":[1,2],"hasMore":false}),
        json!({"result":[1,1],"hasMore":false,"extra":{"stats":{"writesIgnored":1}}}),
        json!({"result":[1,1],"hasMore":false,"extra":{"stats":{"writesIgnored":"invalid"}}}),
    ] {
        let mock = Mock::new(vec![Reply::page(body)]).await;
        assert!(
            export_embeddings(&mock.pool, &ids(), &[0.1, 0.2], 1, &ExportConfig::default())
                .await
                .is_err()
        );
    }
}

#[tokio::test]
async fn later_failure_reports_acknowledged_progress() {
    use hades_core::graph::export::ExportError;
    let mock = Mock::new(vec![
        Reply::page(json!({"result":[1],"hasMore":false})),
        Reply {
            status: 500,
            body: json!({"error":true,"errorNum":1203,"errorMessage":"synthetic"}),
            gate: None,
        },
    ])
    .await;
    let error = export_embeddings(
        &mock.pool,
        &ids(),
        &[0.1, 0.2],
        1,
        &ExportConfig { chunk_size: 1 },
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        ExportError::BatchFailed {
            acknowledged: 1,
            attempted: 1,
            ..
        }
    ));
}

#[tokio::test]
async fn compact_rows_and_collection_bindings_are_preserved() {
    use hades_core::graph::export::export_embeddings_subset;
    let mut mock = Mock::new(vec![Reply::page(json!({"result":[1,1],"hasMore":false}))]).await;
    let result = export_embeddings_subset(
        &mock.pool,
        &ids(),
        &[1, 0],
        &[0.9, 0.8],
        1,
        &ExportConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(result.total_exported, 2);
    let body = mock.event("POST cursor").await;
    assert_eq!(body["bindVars"]["@collection"], "nodes");
    assert!(body["query"].as_str().unwrap().contains("@@collection"));
    assert!(!body["query"].as_str().unwrap().contains("ignoreErrors"));
    assert_eq!(body["bindVars"]["updates"][0]["_key"], "b");
    assert_eq!(body["bindVars"]["updates"][1]["_key"], "a");
    assert!(
        (body["bindVars"]["updates"][0]["structural_embedding"][0]
            .as_f64()
            .unwrap()
            - 0.9)
            .abs()
            < 1e-6
    );
    assert!(
        (body["bindVars"]["updates"][1]["structural_embedding"][0]
            .as_f64()
            .unwrap()
            - 0.8)
            .abs()
            < 1e-6
    );
}

#[tokio::test]
async fn compact_subset_validation_never_sends_invalid_rows() {
    use hades_core::graph::export::export_embeddings_subset;
    for (indices, values, dim, chunk) in [
        (vec![1], vec![f32::NAN], 1, 1),
        (vec![0, 1], vec![], usize::MAX, 1),
        (vec![1], vec![], 0, 1),
        (vec![1], vec![0.5], 1, 0),
        (vec![2], vec![0.5], 1, 1),
        (vec![1, 1], vec![0.5, 0.5], 1, 1),
    ] {
        let mut mock = Mock::new(vec![]).await;
        let result = export_embeddings_subset(
            &mock.pool,
            &ids(),
            &indices,
            &values,
            dim,
            &ExportConfig { chunk_size: chunk },
        )
        .await;
        assert!(result.is_err());
        assert!(mock.events.try_recv().is_err());
    }
}
