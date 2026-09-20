//! Private mock-socket contracts; never connect to an installed database.
use super::*;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::task::JoinHandle;
#[path = "../../tests/common/cursor_mock.rs"]
mod cursor_mock;
use cursor_mock::{Mock, Reply};
fn first() -> Value {
    json!({"id":"123", "hasMore":true, "result":[1],"extra":{"stats":{"fullCount":2}}})
}
fn last() -> Value {
    json!({"hasMore":false,"result":[2]})
}
fn run(pool: ArangoPool, limits: QueryLimits) -> JoinHandle<Result<QueryResult, ArangoError>> {
    tokio::spawn(async move {
        query_with_limits(
            &pool,
            "RETURN 1",
            None,
            Some(1),
            true,
            ExecutionTarget::Reader,
            limits,
        )
        .await
    })
}

#[tokio::test]
async fn completes_and_deletes_cursor_on_its_creating_endpoint() {
    let mut mock = Mock::new(vec![Reply::page(first()), Reply::page(last())]).await;
    // The reader must never be contacted when split sockets cannot handle cursor state.
    let absent = ArangoClient::with_socket(
        mock._dir.path().join("absent.sock"),
        "fixture",
        "fixture",
        "fixture",
    );
    mock.pool = ArangoPool::new(absent, mock.pool.writer().clone());
    let query = run(mock.pool.clone(), QueryLimits::default());
    let body = mock.event("POST cursor").await;
    assert_eq!(body["options"]["fullCount"], true);
    assert_eq!(body["options"]["maxRuntime"], 60.0);
    assert_eq!(body["ttl"], 60.0);
    mock.event("POST cursor/123").await;
    mock.released().await;
    let result = query.await.unwrap().unwrap();
    assert_eq!(result.results, vec![json!(1), json!(2)]);
    assert_eq!(result.full_count, Some(2));
}

#[tokio::test]
async fn malformed_first_page_still_releases_known_cursor() {
    for body in [
        json!({"id":"123","hasMore":true}),
        json!({"id":"123","hasMore":true,"result":{}}),
        json!({"id":"123","result":[]}),
    ] {
        let mut mock = Mock::new(vec![Reply::page(body)]).await;
        let query = run(mock.pool.clone(), QueryLimits::default());
        mock.event("POST cursor").await;
        mock.released().await;
        assert!(query.await.unwrap().is_err());
    }
}

#[tokio::test]
async fn continuation_errors_and_malformed_pages_release_cursor() {
    for reply in [
        Reply::page(json!({"id":"123","hasMore":false,"result":null})),
        Reply {
            status: 500,
            body: json!({"error":true,"errorNum":1500,"errorMessage":"fixture"}),
            gate: None,
        },
    ] {
        let mut mock = Mock::new(vec![Reply::page(first()), reply]).await;
        let query = run(mock.pool.clone(), QueryLimits::default());
        mock.event("POST cursor").await;
        mock.event("POST cursor/123").await;
        mock.released().await;
        assert!(query.await.unwrap().is_err());
    }
}

#[tokio::test]
async fn caller_cancellation_during_creation_keeps_response_ownership() {
    let gate = Arc::new(Notify::new());
    let mut mock = Mock::new(vec![Reply::blocked(first(), gate.clone())]).await;
    let query = run(mock.pool.clone(), QueryLimits::default());
    mock.event("POST cursor").await;
    query.abort();
    assert!(query.await.unwrap_err().is_cancelled());
    gate.notify_one();
    mock.released().await; // no continuation request after the caller disappears
}

#[tokio::test]
async fn caller_cancellation_during_pagination_deletes_without_waiting_for_page() {
    let mut mock = Mock::new(vec![
        Reply::page(first()),
        Reply::blocked(last(), Arc::new(Notify::new())),
    ])
    .await;
    let query = run(mock.pool.clone(), QueryLimits::default());
    mock.event("POST cursor").await;
    mock.event("POST cursor/123").await;
    query.abort();
    assert!(query.await.unwrap_err().is_cancelled());
    mock.released().await;
}

#[tokio::test]
async fn total_query_timeout_releases_cursor() {
    let mut mock = Mock::new(vec![
        Reply::page(first()),
        Reply::blocked(last(), Arc::new(Notify::new())),
    ])
    .await;
    let limits = QueryLimits {
        lifetime: Duration::from_millis(100),
        ..QueryLimits::default()
    };
    let query = run(mock.pool.clone(), limits);
    mock.event("POST cursor").await;
    mock.event("POST cursor/123").await;
    mock.released().await;
    assert!(
        query
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("lifetime exceeded")
    );
}

#[tokio::test]
async fn deletion_response_cannot_hold_caller_past_cleanup_budget() {
    let mut mock = Mock::new(vec![
        Reply::page(json!({"id":"123", "hasMore":false,"result":[1]})),
        Reply::blocked(json!({"error":false}), Arc::new(Notify::new())),
    ])
    .await;
    let limits = QueryLimits {
        cleanup: Duration::from_millis(50),
        ..QueryLimits::default()
    };
    let query = run(mock.pool.clone(), limits);
    mock.event("POST cursor").await;
    mock.released().await;
    let result = tokio::time::timeout(Duration::from_secs(1), query)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(result.results, vec![json!(1)]);
}
