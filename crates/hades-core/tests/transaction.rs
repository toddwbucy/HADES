//! Transaction ownership contracts on private Unix mocks and disposable data.
use hades_core::db::{ArangoClient, ArangoError, ArangoPool, transaction};
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Notify;
#[allow(dead_code)]
#[path = "common/cursor_mock.rs"]
mod cursor_mock;
use cursor_mock::{Mock, Reply};

#[tokio::test]
async fn cancellation_during_creation_waits_for_id_then_aborts() {
    let gate = Arc::new(Notify::new());
    let mut mock = Mock::new(vec![
        Reply::blocked(
            json!({"result":{"id":"123","status":"running"}}),
            gate.clone(),
        ),
        Reply::page(json!({"result":{"id":"123","status":"aborted"}})),
    ])
    .await;
    let pool = mock.pool.clone();
    let caller = tokio::spawn(async move {
        transaction::run(&pool, vec!["fixture".into()], |_| async {
            panic!("cancelled caller must not start transaction work");
            #[allow(unreachable_code)]
            Ok::<_, ArangoError>(())
        })
        .await
    });
    let body = mock.event("POST transaction/begin").await;
    assert_eq!(body["collections"]["exclusive"], json!(["fixture"]));
    assert_eq!(body["collections"]["allowImplicit"], false);
    caller.abort();
    assert!(caller.await.unwrap_err().is_cancelled());
    gate.notify_one();
    mock.event("DELETE transaction/123").await;
}

#[tokio::test]
async fn panic_aborts_before_returning_error() {
    let mut mock = Mock::new(vec![
        Reply::page(json!({"result":{"id":"123","status":"running"}})),
        Reply::page(json!({"result":{"id":"123","status":"aborted"}})),
    ])
    .await;
    let result = transaction::run(&mock.pool, vec!["fixture".into()], |_| async {
        panic!("fixture panic");
        #[allow(unreachable_code)]
        Ok::<_, ArangoError>(())
    })
    .await;
    assert!(result.unwrap_err().to_string().contains("panicked"));
    mock.event("POST transaction/begin").await;
    mock.event("DELETE transaction/123").await;
}

#[tokio::test]
async fn commit_requires_positive_acknowledgment() {
    let mut mock = Mock::new(vec![
        Reply::page(json!({"result":{"id":"123","status":"running"}})),
        Reply::page(json!({"result":{"id":"123","status":"aborted"}})),
        Reply::page(json!({"result":{"id":"123","status":"aborted"}})),
    ])
    .await;
    let result = transaction::run(&mock.pool, vec!["fixture".into()], |_| async { Ok(7) }).await;
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("outcome uncertain")
    );
    mock.event("POST transaction/begin").await;
    mock.event("PUT transaction/123").await;
    mock.event("DELETE transaction/123").await;
}

#[tokio::test]
async fn committed_writes_persist_and_failed_writes_rollback() {
    with_temp_db("transaction", Fixtures::Codebase, |pool| async move {
        transaction::run(&pool, vec!["codebase_files".into()], |client| async move {
            client
                .post(
                    "document/codebase_files",
                    &json!({"_key":"retained","value":1}),
                )
                .await?;
            Ok(())
        })
        .await
        .unwrap();
        let error = transaction::run(&pool, vec!["codebase_files".into()], |client| async move {
            client.delete("document/codebase_files/retained").await?;
            client
                .post("document/codebase_files", &json!({"_key":"partial"}))
                .await?;
            Err::<(), _>(ArangoError::Request("injected write failure".into()))
        })
        .await
        .unwrap_err();
        assert!(error.to_string().contains("injected write failure"));
        assert_eq!(
            pool.reader()
                .get("document/codebase_files/retained")
                .await
                .unwrap()["value"],
            1
        );
        assert!(
            pool.reader()
                .get("document/codebase_files/partial")
                .await
                .unwrap_err()
                .is_not_found()
        );
    })
    .await;
}

#[tokio::test]
async fn cancelled_operation_rolls_back_and_releases_exclusive_lock() {
    with_temp_db(
        "transaction_cancel",
        Fixtures::Codebase,
        |pool| async move {
            let written = Arc::new(Notify::new());
            let signal = written.clone();
            let other = pool.clone();
            let caller = tokio::spawn(async move {
                transaction::run(&other, vec!["codebase_files".into()], |client| async move {
                    client
                        .post("document/codebase_files", &json!({"_key":"cancelled"}))
                        .await?;
                    signal.notify_one();
                    std::future::pending::<Result<(), ArangoError>>().await
                })
                .await
            });
            tokio::time::timeout(std::time::Duration::from_secs(5), written.notified())
                .await
                .unwrap();
            caller.abort();
            assert!(caller.await.unwrap_err().is_cancelled());
            // Acquiring the next exclusive transaction proves cleanup released the lock.
            transaction::run(&pool, vec!["codebase_files".into()], |client| async move {
                assert!(
                    client
                        .get("document/codebase_files/cancelled")
                        .await
                        .unwrap_err()
                        .is_not_found()
                );
                client
                    .post("document/codebase_files", &json!({"_key":"next"}))
                    .await?;
                Ok(())
            })
            .await
            .unwrap();
        },
    )
    .await;
}
