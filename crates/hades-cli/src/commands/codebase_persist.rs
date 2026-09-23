//! Atomic persistence of already-prepared file replacements.
use hades_core::db::{
    ArangoClient, ArangoError, ArangoPool, collections::CODEBASE, keys, transaction,
};
use serde_json::{Value, json};

pub(super) async fn revision(
    client: &ArangoClient,
    key: &str,
) -> Result<Option<String>, ArangoError> {
    match client
        .get(&format!("document/{}/{key}", CODEBASE.files))
        .await
    {
        Ok(doc) => doc["_rev"]
            .as_str()
            .map(|s| Some(s.to_owned()))
            .ok_or_else(|| ArangoError::Request("stored file has no revision".into())),
        Err(error) if error.is_not_found() => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) struct Replacement {
    pub key: String,
    pub expected_revision: Option<String>,
    pub chunks: Vec<Value>,
    pub symbols: Vec<Value>,
    pub embeddings: Vec<Value>,
    pub defines: Vec<Value>,
    pub file: Value,
    pub purge_symbols: bool,
    pub merge_file: bool,
    pub symbol_remap: Vec<(String, String)>,
}

#[derive(Debug)]
pub(super) struct StoredFile {
    pub moved_edges: u64,
    pub revision: String,
}

impl Replacement {
    pub async fn store(mut self, pool: &ArangoPool) -> Result<StoredFile, ArangoError> {
        let collections = CODEBASE
            .all_collections()
            .iter()
            .map(|(name, _)| name.to_string())
            .collect();
        transaction::run(pool, collections, move |client| async move {
            if revision(&client, &self.key).await? != self.expected_revision {
                return Err(ArangoError::Request("file changed during preparation; retry ingestion".into()));
            }
            if self.purge_symbols {
                let aql = "LET ids = APPEND((FOR s IN @@symbols FILTER s.file_key == @key RETURN s._id), [CONCAT(@files_name, '/', @key)]) \
                    LET syms = (FOR d IN @@symbols FILTER d.file_key == @key REMOVE d IN @@symbols RETURN 1) \
                    LET defs = (FOR e IN @@defines FILTER e._from IN ids REMOVE e IN @@defines RETURN 1) \
                    LET calls = (FOR e IN @@calls FILTER e._from IN ids REMOVE e IN @@calls RETURN 1) \
                    LET impls = (FOR e IN @@implements FILTER e._from IN ids REMOVE e IN @@implements RETURN 1) \
                    LET imps = (FOR e IN @@imports FILTER e._from IN ids REMOVE e IN @@imports RETURN 1) RETURN 1";
                query(&client, aql, json!({"@symbols":CODEBASE.symbols,"@defines":CODEBASE.defines_edges,
                    "@calls":CODEBASE.calls_edges,"@implements":CODEBASE.implements_edges,
                    "@imports":CODEBASE.imports_edges,"files_name":CODEBASE.files,"key":self.key}), true).await?;
            }
            for collection in [CODEBASE.chunks, CODEBASE.embeddings] {
                query(&client, "FOR d IN @@collection FILTER d.file_key == @key REMOVE d IN @@collection",
                    json!({"@collection":collection,"key":self.key}), false).await?;
            }
            for (collection, docs) in [
                (CODEBASE.chunks, self.chunks), (CODEBASE.symbols, self.symbols),
                (CODEBASE.embeddings, self.embeddings), (CODEBASE.defines_edges, self.defines),
            ] {
                if !docs.is_empty() {
                    // The Import API does not participate in stream transactions.
                    let response = client.post(&format!("document/{collection}?overwriteMode=replace"), &json!(docs)).await?;
                    let rows = response.as_array().ok_or_else(|| ArangoError::Request("invalid batch document response".into()))?;
                    if rows.len() != docs.len() || rows.iter().any(|row| row["error"] == true) {
                        return Err(ArangoError::Request(format!("failed to store {collection} documents")));
                    }
                }
            }
            let moved = remap_inbound(&client, &self.symbol_remap).await?;
            self.file["_key"] = json!(self.key);
            let mode = if self.merge_file { "update" } else { "replace" };
            // POST update defaults to removing nulls; preserve explicit non-Git
            // provenance through repeated raw-file replacement (#171).
            let response = client.post(&format!("document/{}?overwriteMode={mode}&keepNull=true", CODEBASE.files), &self.file).await?;
            let revision = response["_rev"].as_str()
                .ok_or_else(|| ArangoError::Request("file write returned no revision".into()))?.to_owned();
            Ok(StoredFile { moved_edges: moved, revision })
        }).await
    }
}

/// Commit the complete relationship stage only if every contributing file still
/// has the exact revision returned by its acknowledged replacement transaction.
/// Earlier file commits remain durable if this separate stage fails.
pub(super) async fn store_relationships(
    pool: &ArangoPool,
    revisions: std::collections::HashMap<String, String>,
    batches: Vec<(&'static str, Vec<Value>)>,
) -> Result<(), ArangoError> {
    if revisions.is_empty() && batches.iter().all(|(_, docs)| docs.is_empty()) {
        return Ok(());
    }
    let collections = CODEBASE
        .all_collections()
        .iter()
        .map(|(name, _)| name.to_string())
        .collect();
    transaction::run(pool, collections, move |client| async move {
        for (key, expected) in &revisions {
            if revision(&client, key).await?.as_deref() != Some(expected.as_str()) {
                return Err(ArangoError::Request(
                    "file changed during relationship preparation; retry ingestion".into(),
                ));
            }
        }
        for (collection, docs) in batches {
            if ![CODEBASE.calls_edges, CODEBASE.imports_edges].contains(&collection) {
                return Err(ArangoError::Request(
                    "unsupported relationship collection".into(),
                ));
            }
            for batch in docs.chunks(2_000) {
                let mut ids = std::collections::HashSet::new();
                for doc in batch {
                    for field in ["_from", "_to"] {
                        let id = doc[field].as_str().ok_or_else(|| {
                            ArangoError::Request("missing relationship endpoint".into())
                        })?;
                        if !id.starts_with(&format!("{}/", CODEBASE.files))
                            && !id.starts_with(&format!("{}/", CODEBASE.symbols))
                        {
                            return Err(ArangoError::Request(
                                "invalid relationship endpoint".into(),
                            ));
                        }
                        ids.insert(id);
                    }
                }
                let found = client.post("cursor", &json!({
                    "query":"RETURN (FOR id IN @ids FILTER DOCUMENT(id) == null RETURN id)",
                    "bindVars":{"ids":ids},"batchSize":1,"ttl":30,"memoryLimit":33554432,
                    "options":{"maxRuntime":30,"failOnWarning":true}
                })).await?;
                let missing = Some(hades_core::db::query::completed_rows(&found).map_err(|_| ArangoError::Request(format!("invalid relationship endpoint check response for {collection}")))?)
                    .filter(|rows| rows.len() == 1)
                    .and_then(|rows| rows[0].as_array())
                    .filter(|ids| ids.iter().all(Value::is_string));
                let missing = missing
                    .ok_or_else(|| ArangoError::Request(format!(
                        "invalid relationship endpoint check response for {collection}")))?;
                if !missing.is_empty() {
                    let mut missing_ids: Vec<_> = missing.iter().map(|id| id.as_str().unwrap()).collect();
                    missing_ids.sort_unstable();
                    return Err(ArangoError::Request(format!(
                        "relationship endpoint no longer exists in batch for {collection}: {} missing endpoints; first {} ids: {:?}; unchanged files are skipped by content hash, so --force or a fresh database is required to regenerate missing endpoints",
                        missing_ids.len(), missing_ids.len().min(10), &missing_ids[..missing_ids.len().min(10)]
                    )));
                }
                let response = client
                    .post(
                        &format!("document/{collection}?overwriteMode=replace"),
                        &json!(batch),
                    )
                    .await?;
                let rows = response.as_array().ok_or_else(|| {
                    ArangoError::Request("invalid relationship write response".into())
                })?;
                if rows.len() != batch.len() || rows.iter().any(|row| row["error"] == true) {
                    return Err(ArangoError::Request(format!(
                        "failed to store {collection} relationships"
                    )));
                }
            }
        }
        // A failed or cancelled stage leaves this marker true. A later ingest
        // must rebuild relationships even when the source digest is unchanged.
        for key in revisions.keys() {
            client
                .patch(
                    &format!("document/{}/{key}", CODEBASE.files),
                    &json!({"relationships_pending":false}),
                )
                .await?;
        }
        Ok(())
    })
    .await
}

/// Remap each collection from its original snapshot, including overlapping key
/// chains. The caller must hold the codebase collections in one transaction.
pub(super) async fn remap_inbound(
    client: &ArangoClient,
    remap: &[(String, String)],
) -> Result<u64, ArangoError> {
    use std::collections::{HashMap, HashSet};
    if remap.is_empty() {
        return Ok(0);
    }
    let client = client.clone().with_response_limit(32 * 1024 * 1024)?;
    let by_old: HashMap<&str, &str> = remap
        .iter()
        .map(|(old, new)| (old.as_str(), new.as_str()))
        .collect();
    let olds: Vec<String> = by_old
        .keys()
        .map(|key| format!("{}/{key}", CODEBASE.symbols))
        .collect();
    let mut moved = 0;
    for (edges, kind) in [
        (CODEBASE.imports_edges, "imports"),
        (CODEBASE.calls_edges, "calls"),
        (CODEBASE.implements_edges, "implements"),
    ] {
        // One bounded aggregate row prevents a pagination task from outliving
        // the transaction. Exhausting either budget aborts the replacement.
        let response = client
            .post(
                "cursor",
                &json!({
                    "query":"RETURN (FOR e IN @@edges FILTER e._to IN @olds RETURN e)",
                    "bindVars":{"@edges":edges,"olds":olds}, "batchSize":1, "ttl":30,
                    "memoryLimit":33554432,
                    "options":{"maxRuntime":30,"failOnWarning":true}
                }),
            )
            .await?;
        let rows = hades_core::db::query::completed_rows(&response)?;
        let found = (rows.len() == 1)
            .then(|| rows[0].as_array())
            .flatten()
            .ok_or_else(|| ArangoError::Request("invalid remap snapshot response".into()))?;
        let mut rewritten = Vec::with_capacity(found.len());
        let mut superseded = Vec::new();
        for doc in found {
            let (Some(old_id), Some(from_id), Some(old_key)) = (
                doc["_to"].as_str(),
                doc["_from"].as_str(),
                doc["_key"].as_str(),
            ) else {
                return Err(ArangoError::Request("invalid inbound edge identity".into()));
            };
            let old_suffix = old_id.rsplit('/').next().unwrap_or(old_id);
            let new_suffix = by_old
                .get(old_suffix)
                .ok_or_else(|| ArangoError::Request("unexpected inbound edge target".into()))?;
            let from_suffix = from_id.rsplit('/').next().unwrap_or(from_id);
            let new_key = keys::edge_key(from_suffix, kind, new_suffix);
            let mut next = doc.clone();
            let object = next
                .as_object_mut()
                .ok_or_else(|| ArangoError::Request("invalid inbound edge".into()))?;
            object.remove("_id");
            object.remove("_rev");
            object.insert("_key".into(), json!(new_key));
            object.insert(
                "_to".into(),
                json!(format!("{}/{}", CODEBASE.symbols, new_suffix)),
            );
            if new_key != old_key {
                superseded.push(old_key.to_owned());
            }
            rewritten.push(next);
        }
        let written_keys: HashSet<&str> = rewritten
            .iter()
            .filter_map(|doc| doc["_key"].as_str())
            .collect();
        superseded.retain(|key| !written_keys.contains(key.as_str()));
        for batch in rewritten.chunks(2_000) {
            let response = client
                .post(
                    &format!("document/{edges}?overwriteMode=replace"),
                    &json!(batch),
                )
                .await?;
            let rows = response
                .as_array()
                .ok_or_else(|| ArangoError::Request("invalid remap write response".into()))?;
            if rows.len() != batch.len() || rows.iter().any(|row| row["error"] == true) {
                return Err(ArangoError::Request(format!(
                    "failed to remap {edges} documents"
                )));
            }
            moved += batch.len() as u64;
        }
        for batch in superseded.chunks(2_000) {
            query(
                &client,
                "FOR k IN @keys REMOVE k IN @@edges",
                json!({"@edges":edges,"keys":batch}),
                false,
            )
            .await?;
        }
    }
    Ok(moved)
}

pub(super) async fn query(
    client: &ArangoClient,
    aql: &str,
    binds: Value,
    returns_marker: bool,
) -> Result<(), ArangoError> {
    let result = client
        .post(
            "cursor",
            &json!({"query":aql,"bindVars":binds,"batchSize":1,
        "ttl":30,"options":{"maxRuntime":30,"failOnWarning":true}}),
        )
        .await?;
    let rows = hades_core::db::query::completed_rows(&result)?;
    // The symbol purge's outer RETURN 1 differs from rowless mutations (#164).
    if (returns_marker && rows.as_slice() != [json!(1)]) || (!returns_marker && !rows.is_empty()) {
        return Err(ArangoError::Request(
            "invalid persistence result rows".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn remap_snapshot_requires_explicit_complete_array() {
        use axum::{Json, Router};
        let valid_rows = json!([[]]);
        for response in [
            json!({}),
            json!({"result":valid_rows}),
            json!({"hasMore":null,"result":valid_rows}),
            json!({"hasMore":"false","result":valid_rows}),
            json!({"hasMore":0,"result":valid_rows}),
            json!({"hasMore":true,"result":valid_rows}),
            json!({"hasMore":false,"result":null}),
            json!({"hasMore":false,"result":{}}),
            json!({"hasMore":false,"result":[]}),
            json!({"hasMore":false,"result":valid_rows}),
        ] {
            let valid = response == json!({"hasMore":false,"result":valid_rows});
            let root = tempfile::tempdir().unwrap();
            let socket = root.path().join("db.sock");
            let listener = tokio::net::UnixListener::bind(&socket).unwrap();
            let served = response.clone();
            let app = Router::new().fallback(move || {
                let response = served.clone();
                async move { Json(response) }
            });
            let peer = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let config = serde_json::from_value(json!({"database":{"name":"fixture", "sockets":{"readonly":socket,"readwrite":socket}}})).unwrap();
            let pool = ArangoPool::from_config(&config).unwrap();
            assert_eq!(
                (remap_inbound(pool.writer(), &[("old".into(), "new".into())])
                    .await
                    .map(|_| ()))
                .is_ok(),
                valid,
                "{response}"
            );
            peer.abort();
            let _ = peer.await;
        }
    }

    use super::*;
    use hades_core::test_support::{Fixtures, with_temp_db};

    #[tokio::test]
    async fn relationship_endpoint_cursor_requires_explicit_completion() {
        use axum::{Json, Router, extract::Request};
        use std::sync::{Arc, Mutex};
        for has_more in [
            None,
            Some(Value::Null),
            Some(json!(true)),
            Some(json!(0)),
            Some(json!("false")),
            Some(json!([])),
            Some(json!({})),
            Some(json!(false)),
        ] {
            let valid = has_more == Some(json!(false));
            let mut response = json!({"result":[[]]});
            if let Some(value) = has_more {
                response["hasMore"] = value;
            }
            let root = tempfile::tempdir().unwrap();
            let socket = root.path().join("db.sock");
            let listener = tokio::net::UnixListener::bind(&socket).unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let captured = requests.clone();
            let app = Router::new().fallback(move |request: Request| {
                let requests = captured.clone();
                let response = response.clone();
                async move {
                    let method = request.method().as_str();
                    let path = request.uri().path().split("/_api/").nth(1).unwrap();
                    requests.lock().unwrap().push(format!("{method} {path}"));
                    Json(match (method, path) {
                        ("POST", "transaction/begin") => json!({"result":{"id":"1"}}),
                        ("POST", "cursor") => response,
                        ("POST", "document/codebase_imports_edges") => json!([{"error":false}]),
                        ("PUT", "transaction/1") => json!({"result":{"status":"committed"}}),
                        ("DELETE", "transaction/1") => json!({"result":{"status":"aborted"}}),
                        _ => panic!("unexpected request {method} {path}"),
                    })
                }
            });
            let peer = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let config = serde_json::from_value(json!({"database":{
                "name":"private_cursor_fixture", "sockets":{"readonly":socket,"readwrite":socket}
            }}))
            .unwrap();
            let pool = ArangoPool::from_config(&config).unwrap();
            let result = store_relationships(
                &pool,
                Default::default(),
                vec![(
                    CODEBASE.imports_edges,
                    vec![json!({"_key":"edge",
                    "_from":"codebase_files/source", "_to":"codebase_symbols/target"})],
                )],
            )
            .await;
            let calls = requests.lock().unwrap().clone();
            if valid {
                result.unwrap();
                assert_eq!(
                    calls,
                    [
                        "POST transaction/begin",
                        "POST cursor",
                        "POST document/codebase_imports_edges",
                        "PUT transaction/1"
                    ]
                );
            } else {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("invalid relationship endpoint check response")
                );
                assert_eq!(
                    calls,
                    [
                        "POST transaction/begin",
                        "POST cursor",
                        "DELETE transaction/1"
                    ],
                    "malformed cursor must abort before writing edges"
                );
            }
            peer.abort();
            let _ = peer.await;
        }
    }

    #[tokio::test]
    async fn mutation_query_requires_explicit_completion_and_empty_rows() {
        use axum::{Json, Router};
        for response in [
            json!({}),
            json!({"result":[]}),
            json!({"hasMore":null,"result":[]}),
            json!({"hasMore":"false","result":[]}),
            json!({"hasMore":0,"result":[]}),
            json!({"hasMore":true,"result":[]}),
            json!({"hasMore":false,"result":null}),
            json!({"hasMore":false,"result":[[]]}),
            json!({"hasMore":false,"result":[]}),
        ] {
            let valid = response == json!({"hasMore":false,"result":[]});
            let root = tempfile::tempdir().unwrap();
            let socket = root.path().join("db.sock");
            let listener = tokio::net::UnixListener::bind(&socket).unwrap();
            let app = Router::new().fallback(move || {
                let response = response.clone();
                async move { Json(response) }
            });
            let peer = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let config = serde_json::from_value(json!({"database":{"name":"fixture", "sockets":{"readonly":socket,"readwrite":socket}}})).unwrap();
            let pool = ArangoPool::from_config(&config).unwrap();
            assert_eq!(
                query(
                    pool.writer(),
                    "FOR d IN fixture REMOVE d IN fixture",
                    json!({}),
                    false
                )
                .await
                .is_ok(),
                valid
            );
            peer.abort();
            let _ = peer.await;
        }
    }

    fn prepared(expected_revision: Option<String>, text: &str) -> Replacement {
        Replacement {
            key: "file".into(),
            expected_revision,
            chunks: vec![json!({"_key":"chunk","file_key":"file","text":text})],
            symbols: Vec::new(),
            embeddings: Vec::new(),
            defines: Vec::new(),
            file: json!({"path":"file.sh","content_hash":text,"chunk_count":1}),
            purge_symbols: false,
            merge_file: true,
            symbol_remap: Vec::new(),
        }
    }

    #[tokio::test]
    async fn purge_request_failure_aborts_without_attempting_replacement_writes() {
        use axum::{
            Json, Router,
            http::{Method, StatusCode},
        };
        use std::sync::Arc;
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("arango.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let calls = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let recorded = calls.clone();
        let app = Router::new().fallback(move |request: axum::extract::Request| {
            let recorded = recorded.clone();
            async move {
                let method = request.method().clone();
                let path = request.uri().path().to_owned();
                let body = axum::body::to_bytes(request.into_body(), 1024 * 1024).await.unwrap();
                recorded.lock().await.push((method.clone(), path.clone()));
                match (method, path.as_str()) {
                    (Method::POST, path) if path.ends_with("/transaction/begin") =>
                        (StatusCode::OK, Json(json!({"result":{"id":"123","status":"running"}}))),
                    (Method::GET, path) if path.ends_with("/document/codebase_files/file") =>
                        (StatusCode::OK, Json(json!({"_rev":"before"}))),
                    (Method::POST, path) if path.ends_with("/cursor") => {
                        let query: Value = serde_json::from_slice(&body).unwrap();
                        assert!(query["query"].as_str().unwrap().contains("REMOVE d IN @@symbols"));
                        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":true,"errorNum":1,"errorMessage":"injected purge failure"})))
                    }
                    (Method::DELETE, path) if path.ends_with("/transaction/123") =>
                        (StatusCode::OK, Json(json!({"result":{"status":"aborted"}}))),
                    _ => (StatusCode::BAD_REQUEST, Json(json!({"error":true,"errorMessage":"unexpected request"}))),
                }
            }
        });
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = ArangoClient::with_socket(socket, "fixture", "fixture", "fixture");
        let pool = ArangoPool::new(client.clone(), client);
        let mut replacement = prepared(Some("before".into()), "new");
        replacement.purge_symbols = true;
        let error = replacement.store(&pool).await.unwrap_err();
        server.abort();
        assert!(error.to_string().contains("injected purge failure"));
        let calls = calls.lock().await;
        assert_eq!(
            calls.len(),
            4,
            "replacement writes or commit followed a failed purge: {calls:?}"
        );
        assert_eq!(calls[3].0, Method::DELETE);
        assert!(calls[3].1.ends_with("/transaction/123"));
    }

    #[tokio::test]
    async fn replacement_and_relationship_commit_cannot_both_use_one_revision() {
        with_temp_db("stage_race", Fixtures::Codebase, |pool| async move {
            let mut initial = prepared(None, "original");
            initial.file["relationships_pending"] = json!(true);
            let committed = initial.store(&pool).await.unwrap();
            pool.writer().post("document/codebase_files", &json!({"_key":"target"})).await.unwrap();
            let revisions = std::collections::HashMap::from([
                ("file".to_owned(), committed.revision.clone()),
                ("target".to_owned(), revision(pool.writer(), "target").await.unwrap().unwrap()),
            ]);
            let mut replacement = prepared(Some(committed.revision), "replacement");
            replacement.purge_symbols = true;
            replacement.file["relationships_pending"] = json!(true);
            let edge = json!({"_key":"relationship", "_from":"codebase_files/file", "_to":"codebase_files/target"});
            let (replaced, related) = tokio::join!(
                replacement.store(&pool),
                store_relationships(&pool, revisions, vec![(CODEBASE.calls_edges, vec![edge])])
            );
            assert_ne!(replaced.is_ok(), related.is_ok(), "a shared preparation revision admits only one stage");
            let file = pool.reader().get("document/codebase_files/file").await.unwrap();
            let chunk = pool.reader().get("document/codebase_chunks/chunk").await.unwrap();
            let edge = pool.reader().get("document/codebase_calls_edges/relationship").await;
            match replaced {
                Ok(_) => {
                assert_eq!(chunk["text"], "replacement");
                assert_eq!(file["relationships_pending"], true);
                assert!(edge.unwrap_err().is_not_found());
                assert!(related.unwrap_err().to_string().contains("changed during relationship preparation"));
                }
                Err(error) => {
                assert_eq!(chunk["text"], "original");
                assert_eq!(file["relationships_pending"], false);
                assert_eq!(edge.unwrap()["_to"], "codebase_files/target");
                assert!(error.to_string().contains("changed during preparation"));
            }
            }
        }).await;
    }

    #[tokio::test]
    async fn relationship_stage_rolls_back_retries_and_rejects_stale_inputs() {
        with_temp_db("relationship_atomic", Fixtures::Codebase, |pool| async move {
            let stored = prepared(None, "original").store(&pool).await.unwrap();
            let revisions = std::collections::HashMap::from([("file".to_owned(), stored.revision)]);
            pool.writer().post("document/codebase_files", &json!({"_key":"target"})).await.unwrap();
            let edge = json!({"_key":"relationship", "_from":"codebase_files/file", "_to":"codebase_files/target"});
            let batches = vec![(CODEBASE.imports_edges, vec![edge.clone()]), (CODEBASE.calls_edges, vec![edge.clone()])];
            pool.writer().put("collection/codebase_calls_edges/properties", &json!({
                "schema":{"level":"strict","rule":{"type":"object","required":["fault_marker"]}}
            })).await.unwrap();
            let error = store_relationships(&pool, revisions.clone(), batches.clone()).await.unwrap_err();
            assert!(error.to_string().contains("failed to store codebase_calls_edges"), "{error}");
            for collection in [CODEBASE.calls_edges, CODEBASE.imports_edges] {
                assert!(pool.reader().get(&format!("document/{collection}/relationship")).await.unwrap_err().is_not_found());
            }
            pool.writer().put("collection/codebase_calls_edges/properties", &json!({"schema":null})).await.unwrap();
            store_relationships(&pool, revisions.clone(), batches.clone()).await.unwrap();
            let before = pool.reader().get("document/codebase_imports_edges/relationship").await.unwrap();
            prepared(revision(pool.writer(), "file").await.unwrap(), "newer").store(&pool).await.unwrap();
            let error = store_relationships(&pool, revisions, batches).await.unwrap_err();
            assert!(error.to_string().contains("changed during relationship preparation"));
            assert_eq!(pool.reader().get("document/codebase_imports_edges/relationship").await.unwrap(), before);
            let current = std::collections::HashMap::from([("file".to_owned(), revision(pool.writer(), "file").await.unwrap().unwrap())]);
            pool.writer().delete("document/codebase_files/target").await.unwrap();
            let error = store_relationships(&pool, current, vec![(CODEBASE.calls_edges, vec![edge])]).await.unwrap_err();
            assert!(error.to_string().contains("endpoint no longer exists"));
            assert!(error.to_string().contains(CODEBASE.calls_edges));
            assert!(error.to_string().contains("1 missing endpoints"));
            assert!(error.to_string().contains("codebase_files/target"));
        }).await;
    }

    #[tokio::test]
    async fn rejected_inbound_remap_rolls_back_file_replacement() {
        with_temp_db("remap_rollback", Fixtures::Codebase, |pool| async move {
            prepared(None, "original").store(&pool).await.unwrap();
            pool.writer()
                .post(
                    "document/codebase_symbols",
                    &json!({
                        "_key":"old", "file_key":"file", "qualified_name":"target", "start_line":1
                    }),
                )
                .await
                .unwrap();
            let old_edge = keys::edge_key("consumer", "calls", "old");
            pool.writer().post("document/codebase_calls_edges", &json!({
                "_key":old_edge, "_from":"codebase_files/consumer", "_to":"codebase_symbols/old"
            })).await.unwrap();
            let paths = [
                "document/codebase_files/file".to_owned(),
                "document/codebase_chunks/chunk".to_owned(),
                "document/codebase_symbols/old".to_owned(),
                format!("document/codebase_calls_edges/{old_edge}"),
            ];
            let mut before = Vec::new();
            for path in &paths {
                before.push(pool.reader().get(path).await.unwrap());
            }
            pool.writer().put("collection/codebase_calls_edges/properties", &json!({
                "schema":{"level":"strict","rule":{"type":"object","required":["fault_marker"]}}
            })).await.unwrap();
            let observed = revision(pool.writer(), "file").await.unwrap();
            let replacement = || {
                let mut next = prepared(observed.clone(), "replacement");
                next.purge_symbols = true;
                next.symbols = vec![json!({"_key":"new", "file_key":"file",
                    "qualified_name":"target", "start_line":2})];
                next.symbol_remap = vec![("old".into(), "new".into())];
                next
            };
            let error = replacement().store(&pool).await.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("failed to remap codebase_calls_edges"),
                "{error}"
            );
            for (path, expected) in paths.iter().zip(before) {
                assert_eq!(pool.reader().get(path).await.unwrap(), expected);
            }
            assert!(
                pool.reader()
                    .get("document/codebase_symbols/new")
                    .await
                    .unwrap_err()
                    .is_not_found()
            );
            let new_path = format!(
                "document/codebase_calls_edges/{}",
                keys::edge_key("consumer", "calls", "new")
            );
            assert!(
                pool.reader()
                    .get(&new_path)
                    .await
                    .unwrap_err()
                    .is_not_found()
            );
            pool.writer()
                .put(
                    "collection/codebase_calls_edges/properties",
                    &json!({"schema":null}),
                )
                .await
                .unwrap();
            assert_eq!(replacement().store(&pool).await.unwrap().moved_edges, 1);
            assert_eq!(
                pool.reader().get(&new_path).await.unwrap()["_to"],
                "codebase_symbols/new"
            );
            assert!(
                pool.reader()
                    .get(&paths[3])
                    .await
                    .unwrap_err()
                    .is_not_found()
            );
            assert_eq!(
                pool.reader().get(&paths[1]).await.unwrap()["text"],
                "replacement"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn concurrent_prepared_replacements_only_commit_once() {
        with_temp_db("replacement_race", Fixtures::Codebase, |pool| async move {
            prepared(None, "original").store(&pool).await.unwrap();
            let observed = revision(pool.writer(), "file").await.unwrap();
            let (first, second) = tokio::join!(
                prepared(observed.clone(), "first").store(&pool),
                prepared(observed, "second").store(&pool)
            );
            assert_ne!(
                first.is_ok(),
                second.is_ok(),
                "exactly one stale preparation may commit"
            );
            let winner = if first.is_ok() { "first" } else { "second" };
            assert_eq!(
                pool.reader()
                    .get("document/codebase_files/file")
                    .await
                    .unwrap()["content_hash"],
                winner
            );
            assert_eq!(
                pool.reader()
                    .get("document/codebase_chunks/chunk")
                    .await
                    .unwrap()["text"],
                winner
            );
            let error = first.err().or_else(|| second.err()).unwrap();
            assert!(error.to_string().contains("changed during preparation"));
        })
        .await;
    }

    #[tokio::test]
    async fn rejected_fallback_write_preserves_merged_metadata_and_chunks() {
        with_temp_db("fallback_rollback", Fixtures::Codebase, |pool| async move {
            pool.writer()
                .post(
                    "document/codebase_files",
                    &json!({"_key":"file","custom":"keep"}),
                )
                .await
                .unwrap();
            prepared(revision(pool.writer(), "file").await.unwrap(), "original")
                .store(&pool)
                .await
                .unwrap();
            let before = pool
                .reader()
                .get("document/codebase_files/file")
                .await
                .unwrap();
            pool.writer().put("collection/codebase_chunks/properties", &json!({
                "schema":{"level":"strict","rule":{"type":"object","required":["fault_marker"]}}
            })).await.unwrap();
            let error = prepared(revision(pool.writer(), "file").await.unwrap(), "rejected")
                .store(&pool)
                .await
                .unwrap_err();
            assert!(error.to_string().contains("codebase_chunks"));
            assert_eq!(
                pool.reader()
                    .get("document/codebase_files/file")
                    .await
                    .unwrap(),
                before
            );
            assert_eq!(
                pool.reader()
                    .get("document/codebase_chunks/chunk")
                    .await
                    .unwrap()["text"],
                "original"
            );
        })
        .await;
    }
}
