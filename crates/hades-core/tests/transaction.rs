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

/// Subprocess entry point; ordinary test runs do nothing here. The parent owns
/// the disposable database and passes its identity explicitly.
#[tokio::test]
async fn abandoned_transaction_child() {
    let Ok(database) = std::env::var("HADES_TRANSACTION_CHILD_DB") else {
        return;
    };
    assert!(database.starts_with("hades_test_"));
    assert_eq!(std::env::var("ARANGO_TESTS").unwrap(), "1");
    let socket = std::env::var("ARANGO_SOCKET").unwrap();
    let client =
        ArangoClient::with_socket(socket.into(), &database, "root", "isolated-fixture-only");
    let pool = ArangoPool::new(client.clone(), client);
    transaction::run(&pool, vec!["codebase_files".into()], |client| async move {
        client.delete("document/codebase_files/retained").await?;
        client
            .post("document/codebase_files", &json!({"_key":"partial"}))
            .await?;
        std::fs::write(
            std::env::var("HADES_TRANSACTION_CHILD_READY").unwrap(),
            "writes completed",
        )
        .unwrap();
        std::future::pending::<Result<(), ArangoError>>().await
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn killed_writer_rolls_back_on_server_expiry_and_releases_lock() {
    with_temp_db("process_death", Fixtures::Codebase, |pool| async move {
        pool.writer()
            .post(
                "document/codebase_files",
                &json!({"_key":"retained","value":7}),
            )
            .await
            .unwrap();
        let before = pool
            .reader()
            .get("document/codebase_files/retained")
            .await
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let ready = directory.path().join("ready");
        let mut child = tokio::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "abandoned_transaction_child", "--nocapture"])
            .env("HADES_TRANSACTION_CHILD_DB", pool.database())
            .env("HADES_TRANSACTION_CHILD_READY", &ready)
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while !ready.exists() {
                assert!(
                    child.try_wait().unwrap().is_none(),
                    "writer exited before writing"
                );
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await
        .expect("writer never reached the transaction checkpoint");
        child.start_kill().unwrap();
        let status = tokio::time::timeout(std::time::Duration::from_secs(10), child.wait())
            .await
            .unwrap()
            .unwrap();
        assert!(!status.success());
        // There is no client cleanup task left alive. The private server uses
        // its normal 60-second stream idle timeout; allow its cleanup sweep.
        tokio::time::timeout(std::time::Duration::from_secs(100), async {
            loop {
                let expected = before.clone();
                let recovered = transaction::run(
                    &pool,
                    vec!["codebase_files".into()],
                    move |client| async move {
                        assert_eq!(
                            client.get("document/codebase_files/retained").await?,
                            expected
                        );
                        assert!(
                            client
                                .get("document/codebase_files/partial")
                                .await
                                .unwrap_err()
                                .is_not_found()
                        );
                        client
                            .post("document/codebase_files", &json!({"_key":"after_expiry"}))
                            .await?;
                        Ok(())
                    },
                )
                .await;
                if recovered.is_ok() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
        })
        .await
        .expect("abandoned transaction did not roll back and release its lock");
        assert!(
            pool.reader()
                .get("document/codebase_files/after_expiry")
                .await
                .is_ok()
        );
    })
    .await;
}

#[tokio::test]
async fn operation_deadline_rolls_back_writes_and_releases_lock() {
    with_temp_db(
        "operation_deadline",
        Fixtures::Codebase,
        |pool| async move {
            pool.writer()
                .post(
                    "document/codebase_files",
                    &json!({"_key":"retained","value":9}),
                )
                .await
                .unwrap();
            let before = pool
                .reader()
                .get("document/codebase_files/retained")
                .await
                .unwrap();
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(75),
                transaction::run(&pool, vec!["codebase_files".into()], |client| async move {
                    client.delete("document/codebase_files/retained").await?;
                    client
                        .post("document/codebase_files", &json!({"_key":"partial"}))
                        .await?;
                    std::future::pending::<Result<(), ArangoError>>().await
                }),
            )
            .await
            .expect("transaction operation deadline did not return");
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("operation timed out")
            );
            transaction::run(
                &pool,
                vec!["codebase_files".into()],
                move |client| async move {
                    assert_eq!(
                        client.get("document/codebase_files/retained").await?,
                        before
                    );
                    assert!(
                        client
                            .get("document/codebase_files/partial")
                            .await
                            .unwrap_err()
                            .is_not_found()
                    );
                    client
                        .post("document/codebase_files", &json!({"_key":"after_timeout"}))
                        .await?;
                    Ok(())
                },
            )
            .await
            .unwrap();
        },
    )
    .await;
}
