//! AQL query execution with cursor-based pagination.
//!
//! Use [`ExecutionTarget::Writer`] for mutating AQL. When reader and writer
//! endpoints differ, the entire cursor lifecycle uses the writer because the
//! read-only proxy cannot continue or delete cursor state. A separate bounded
//! task retains cursor ownership across caller cancellation.

use std::time::Duration;

use serde_json::Value;
use tokio::sync::oneshot;
use tracing::{debug, instrument, trace};

use super::error::ArangoError;
use super::pool::ArangoPool;
use super::transport::ArangoClient;

/// Default batch size for AQL queries.
const DEFAULT_BATCH_SIZE: u32 = 1000;

/// Controls which pool endpoint receives the initial cursor request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExecutionTarget {
    /// Read-only AQL; use the shared reader or the writer for split endpoints.
    #[default]
    Reader,
    /// Route through `pool.writer()` — required for mutating AQL
    /// (INSERT, UPDATE, REPLACE, REMOVE, UPSERT).
    Writer,
}

/// Result of an AQL query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// All result documents accumulated across pages.
    pub results: Vec<Value>,
    /// Total matching documents (only set when `full_count` was requested).
    pub full_count: Option<u64>,
    /// Extra metadata from ArangoDB (stats, profile, warnings).
    pub extra: Option<Value>,
}

/// Execute an AQL query with optional bind variables and pagination.
///
/// Creation, continuation, and cleanup use the same client. Read queries use
/// the reader only when its endpoint is shared with the writer; otherwise all
/// cursor operations use the writer to avoid the read-only proxy limitation.
///
/// All pages are accumulated into a single `Vec<Value>`. A task owns the cursor
/// across caller cancellation, with a 60-second query lifetime and two-second
/// cleanup budget. Server `maxRuntime` and idle cursor TTL provide fallback
/// bounds if transport fails or the Tokio runtime shuts down before deletion.
#[instrument(skip(pool, bind_vars), fields(db = %pool.database()))]
pub async fn query(
    pool: &ArangoPool,
    aql: &str,
    bind_vars: Option<&Value>,
    batch_size: Option<u32>,
    full_count: bool,
    target: ExecutionTarget,
) -> Result<QueryResult, ArangoError> {
    query_with_limits(
        pool,
        aql,
        bind_vars,
        batch_size,
        full_count,
        target,
        QueryLimits::default(),
    )
    .await
}

#[derive(Clone, Copy)]
struct QueryLimits {
    lifetime: Duration,
    cleanup: Duration,
    cursor_ttl: Duration,
}

impl Default for QueryLimits {
    fn default() -> Self {
        Self {
            lifetime: Duration::from_secs(60),
            cleanup: Duration::from_secs(2),
            cursor_ttl: Duration::from_secs(60),
        }
    }
}

/// Own cursor state outside the caller's cancellable future. Creation is allowed
/// to finish so cancellation before the first response does not discard its ID.
/// Pagination observes receiver closure promptly; cleanup runs independently.
#[allow(clippy::too_many_arguments)]
async fn query_with_limits(
    pool: &ArangoPool,
    aql: &str,
    bind_vars: Option<&Value>,
    batch_size: Option<u32>,
    full_count: bool,
    target: ExecutionTarget,
    limits: QueryLimits,
) -> Result<QueryResult, ArangoError> {
    let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE);
    if batch_size == 0 {
        return Err(ArangoError::Request(
            "cursor batch size must be positive".into(),
        ));
    }
    let mut body = serde_json::json!({
        "query": aql,
        "batchSize": batch_size,
        "ttl": limits.cursor_ttl.as_secs_f64(),
        "options": {
            "fullCount": full_count,
            "maxRuntime": limits.lifetime.as_secs_f64(),
        },
    });
    if let Some(vars) = bind_vars {
        body["bindVars"] = vars.clone();
    }
    debug!(batch_size, full_count, ?target, "executing AQL query");
    trace!(aql, "query text");
    // The read-only proxy cannot continue/delete cursor state. Keep the entire
    // lifecycle on the writer when endpoints differ, as before.
    let client = match target {
        ExecutionTarget::Reader if pool.is_shared() => pool.reader().clone(),
        _ => pool.writer().clone(),
    };
    let (mut send, receive) = oneshot::channel();
    tokio::spawn(async move {
        let mut cursor = CursorOwner {
            client,
            ids: Vec::new(),
        };
        let result = tokio::time::timeout(
            limits.lifetime,
            execute_cursor(&mut cursor, &body, &mut send),
        )
        .await
        .unwrap_or_else(|_| Err(ArangoError::Request("AQL cursor lifetime exceeded".into())));
        cursor.cleanup(limits.cleanup).await;
        let _ = send.send(result);
    });
    receive
        .await
        .map_err(|_| ArangoError::Request("cursor owner stopped before completion".into()))?
}

struct CursorOwner {
    client: ArangoClient,
    // Normally one ID; retain a second unexpected ID for cleanup before rejecting
    // a malformed continuation. An error stops pagination, bounding this at two.
    ids: Vec<String>,
}

impl CursorOwner {
    fn observe(&mut self, response: &Value) -> Result<(), ArangoError> {
        let Some(value) = response.get("id") else {
            return Ok(());
        };
        let id = value
            .as_str()
            .filter(|id| {
                !id.is_empty()
                    && id.len() <= 128
                    && id
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
            })
            .ok_or_else(|| ArangoError::Request("invalid cursor ID".into()))?;
        if self.ids.first().is_some_and(|known| known != id) {
            self.ids.push(id.to_string());
            return Err(ArangoError::Request(
                "cursor ID changed during pagination".into(),
            ));
        }
        if self.ids.is_empty() {
            self.ids.push(id.to_string());
        }
        Ok(())
    }

    async fn cleanup(self, budget: Duration) {
        let result = tokio::time::timeout(budget, async {
            for id in self.ids {
                if let Err(error) = self.client.delete(&format!("cursor/{id}")).await
                    && !error.is_not_found()
                {
                    tracing::warn!(%error, "cursor deletion failed; server TTL remains the fallback");
                }
            }
        }).await;
        if result.is_err() {
            tracing::warn!("cursor cleanup timed out; server TTL remains the fallback");
        }
    }
}

fn has_more(response: &Value) -> Result<bool, ArangoError> {
    response
        .get("hasMore")
        .and_then(Value::as_bool)
        .ok_or_else(|| ArangoError::Request("cursor response missing boolean hasMore".into()))
}

async fn execute_cursor(
    cursor: &mut CursorOwner,
    body: &Value,
    send: &mut oneshot::Sender<Result<QueryResult, ArangoError>>,
) -> Result<QueryResult, ArangoError> {
    if send.is_closed() {
        return Err(ArangoError::Request("query caller cancelled".into()));
    }
    // Shield only creation. If the caller disappears during this request, the
    // owner still receives and deletes the ID, bounded by the query lifetime.
    let response = cursor.client.post("cursor", body).await?;
    cursor.observe(&response)?; // capture ID BEFORE validating the first page
    let mut results = extract_results(&response)?;
    let extra = response.get("extra").cloned();
    let full_count = extra
        .as_ref()
        .and_then(|e| e["stats"]["fullCount"].as_u64());
    let mut more = has_more(&response)?;
    while more {
        let id = cursor.ids.first().ok_or_else(|| {
            ArangoError::Request("hasMore=true but no cursor ID in response".into())
        })?;
        let path = format!("cursor/{id}");
        let empty = serde_json::json!({});
        let response = tokio::select! {
            biased;
            _ = send.closed() => return Err(ArangoError::Request("query caller cancelled".into())),
            response = cursor.client.post(&path, &empty) => response?,
        };
        cursor.observe(&response)?;
        results.extend(extract_results(&response)?);
        more = has_more(&response)?;
    }
    debug!(total_results = results.len(), "query complete");
    Ok(QueryResult {
        results,
        full_count,
        extra,
    })
}

/// Execute an AQL query and return the first result, or `None` if empty.
///
/// Uses the default batch size so the server can satisfy the request in
/// a single round-trip without extra cursor management overhead.
#[instrument(skip(pool, bind_vars), fields(db = %pool.database()))]
pub async fn query_single(
    pool: &ArangoPool,
    aql: &str,
    bind_vars: Option<&Value>,
    target: ExecutionTarget,
) -> Result<Option<Value>, ArangoError> {
    let result = query(pool, aql, bind_vars, None, false, target).await?;
    Ok(result.results.into_iter().next())
}

/// Extract the `result` array from a cursor response.
///
/// Errors if the response has no `result` key or if it is not an array,
/// which indicates a malformed cursor payload from ArangoDB.
fn extract_results(resp: &Value) -> Result<Vec<Value>, ArangoError> {
    match resp.get("result") {
        Some(v) if v.is_array() => Ok(v.as_array().unwrap().clone()),
        Some(_) => Err(ArangoError::Request(
            "cursor response 'result' field is not an array".to_string(),
        )),
        None => Err(ArangoError::Request(
            "cursor response missing 'result' field".to_string(),
        )),
    }
}

/// Remove every document in `collection` where ANY of `fields` equals `value`.
/// Returns the number of documents removed.
///
/// This is the single supported shape for field-match deletes. It exists
/// because the inlined versions kept reproducing the same two defects:
///
/// - **ArangoDB error 1552.** One bind map shared across two single-collection
///   queries declares a parameter the first query never uses, and ArangoDB
///   rejects declared-but-unused binds. That aborted every `--force` document
///   refresh (#169). Here each call builds exactly its own binds, so the
///   defect class is unrepresentable.
/// - **Field drift between writer and reader.** Field names arrive as data
///   (attribute-name binds, `d.@f`), so callers pass the same constants the
///   write path uses instead of re-typing them into query strings.
///   Attribute-name binds are resolved at parse time, before planning:
///   verified via `_api/explain` that `FILTER d.@f0 == @value` selects the
///   same persistent index (`IndexNode`, `fields=file_key`) as the literal
///   `FILTER d.file_key == @value`.
pub async fn remove_docs_by_fields(
    pool: &ArangoPool,
    collection: &str,
    fields: &[&str],
    value: &str,
) -> Result<u64, ArangoError> {
    if fields.is_empty() {
        return Err(ArangoError::Request(
            "remove_docs_by_fields: no fields given".to_string(),
        ));
    }
    // FILTER d.@f0 == @value OR d.@f1 == @value ...
    let filter = fields
        .iter()
        .enumerate()
        .map(|(i, _)| format!("d.@f{i} == @value"))
        .collect::<Vec<_>>()
        .join(" OR ");
    let aql = format!(
        "LET removed = (FOR d IN @@col FILTER {filter} REMOVE d IN @@col RETURN 1) \
         RETURN LENGTH(removed)"
    );
    let mut bind = serde_json::json!({ "@col": collection, "value": value });
    for (i, f) in fields.iter().enumerate() {
        bind[format!("f{i}")] = serde_json::json!(f);
    }
    let result = query_single(pool, &aql, Some(&bind), ExecutionTarget::Writer).await?;
    Ok(result.and_then(|v| v.as_u64()).unwrap_or(0))
}

#[cfg(test)]
#[path = "query_tests.rs"]
mod tests;
