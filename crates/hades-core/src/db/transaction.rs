//! Serial stream transactions with an independent cancellation/abort owner.
//!
//! Only use supported transactional APIs (Document API and AQL, not Import API).
//! Prepare external/model work before entering the transaction. A commit whose
//! response is lost has an unknown outcome; callers must reconcile durable state.
use std::{future::Future, panic::AssertUnwindSafe, time::Duration};

use futures::FutureExt;
use serde_json::json;
use tokio::sync::oneshot;
use tracing::warn;

use super::{ArangoClient, ArangoError, ArangoPool};

/// Run serial database operations in a transaction. All declared collections
/// receive exclusive write locks; lock acquisition is bounded to five seconds.
/// The operation has a 60-second deadline and the server enforces a 32 MiB size
/// budget. Cancellation before commit triggers abort rather than publishing work.
/// Once commit has been sent, cancellation cannot promise rollback.
///
/// The owner waits for transaction creation even if the caller disappears so it
/// can abort the returned ID. A lost creation response or unavailable abort leaves
/// server idle expiry as the fallback; no transaction is deliberately committed
/// in that case. Callback panics are converted to errors and also trigger abort.
pub async fn run<T, F, Fut>(
    pool: &ArangoPool,
    collections: Vec<String>,
    operation: F,
) -> Result<T, ArangoError>
where
    T: Send + 'static,
    F: FnOnce(ArangoClient) -> Fut + Send + 'static,
    Fut: Future<Output = Result<T, ArangoError>> + Send + 'static,
{
    if collections.is_empty() {
        return Err(ArangoError::Request(
            "transaction requires collections".into(),
        ));
    }
    let client = pool.writer().clone();
    let (mut sender, receiver) = oneshot::channel();
    tokio::spawn(async move {
        let created = client
            .post(
                "transaction/begin",
                &json!({
                    "collections": {"exclusive": collections, "allowImplicit": false},
                    "lockTimeout": 5, "maxTransactionSize": 32 * 1024 * 1024
                }),
            )
            .await;
        let result = async {
            let created = created?;
            let id = created["result"]["id"].as_str()
                .ok_or_else(|| ArangoError::Request("missing transaction identifier".into()))?;
            let scoped = client.in_transaction(id)?;
            let outcome = tokio::select! {
                biased;
                _ = sender.closed() => Err(ArangoError::Request("transaction caller cancelled".into())),
                outcome = tokio::time::timeout(Duration::from_secs(60), AssertUnwindSafe(async move {
                    operation(scoped).await
                }).catch_unwind()) => match outcome {
                    Ok(Ok(result)) => result,
                    Ok(Err(_)) => Err(ArangoError::Request("transaction callback panicked".into())),
                    Err(_) => Err(ArangoError::Request("transaction operation timed out".into())),
                }
            };
            match outcome {
                Ok(value) if !sender.is_closed() => {
                    match client.put(&format!("transaction/{id}"), &json!({})).await {
                        Ok(response) if response["result"]["status"] == "committed" => Ok(value),
                        result => {
                            abort(&client, id).await;
                            Err(ArangoError::Request(format!("transaction commit outcome uncertain: {result:?}")))
                        }
                    }
                }
                result => {
                    abort(&client, id).await;
                    result.and_then(|_| Err(ArangoError::Request("transaction caller cancelled".into())))
                }
            }
        }.await;
        let _ = sender.send(result);
    });
    receiver
        .await
        .map_err(|_| ArangoError::Request("transaction owner stopped".into()))?
}

async fn abort(client: &ArangoClient, id: &str) {
    match tokio::time::timeout(
        Duration::from_secs(2),
        client.delete(&format!("transaction/{id}")),
    )
    .await
    {
        Ok(Ok(response)) if response["result"]["status"] == "aborted" => {}
        result => warn!(
            transaction_id = id,
            ?result,
            "transaction abort not acknowledged; server expiry remains fallback"
        ),
    }
}
