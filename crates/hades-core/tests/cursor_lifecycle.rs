//! Requires an explicitly isolated disposable ArangoDB fixture.
use hades_core::db::query::{ExecutionTarget, query};
use hades_core::db::{ArangoClient, ArangoPool};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Notify;
// The shared fixture also serves service-free unit contracts.
#[allow(dead_code)]
#[path = "common/cursor_mock.rs"]
mod cursor_mock;
use cursor_mock::{Mock, Reply};
fn first() -> serde_json::Value {
    json!({"id":"placeholder","hasMore":true,"result":[]})
}
fn last() -> serde_json::Value {
    json!({"hasMore":false,"result":[]})
}
#[tokio::test]
async fn disposable_database_confirms_cursor_is_gone_after_cancellation() {
    use hades_core::test_support::{Fixtures, with_temp_db};
    with_temp_db("cursor_cancel", Fixtures::Empty, |database| async move {
        for during_creation in [true, false] {
            let gate = Arc::new(Notify::new());
            let replies = if during_creation {
                vec![Reply::blocked(first(), gate.clone())]
            } else {
                vec![Reply::page(first()), Reply::blocked(last(), gate.clone())]
            };
            let mut proxy = Mock::with_backend(replies, Some(database.writer().clone())).await;
            let pool = proxy.pool.clone();
            let caller = tokio::spawn(async move {
                query(
                    &pool,
                    "FOR n IN 1..5 RETURN n",
                    None,
                    Some(1),
                    false,
                    ExecutionTarget::Reader,
                )
                .await
            });
            let initial = proxy.event("POST cursor").await;
            let id = initial["_cursor_id"]
                .as_str()
                .expect("real database cursor ID");
            if !during_creation {
                proxy.event(&format!("POST cursor/{id}")).await;
            }
            caller.abort();
            assert!(caller.await.unwrap_err().is_cancelled());
            if during_creation {
                gate.notify_one();
            }
            proxy.event(&format!("DELETE cursor/{id}")).await;
            let error = database
                .writer()
                .post(&format!("cursor/{id}"), &json!({}))
                .await
                .unwrap_err();
            assert!(error.is_not_found(), "server cursor still exists: {error}");
        }
    })
    .await;
}
