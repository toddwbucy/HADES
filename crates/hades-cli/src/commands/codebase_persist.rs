//! Atomic persistence of already-prepared file replacements.
use hades_core::db::{ArangoClient, ArangoError, ArangoPool, collections::CODEBASE, transaction};
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
}

impl Replacement {
    pub async fn store(mut self, pool: &ArangoPool) -> Result<(), ArangoError> {
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
                    "@imports":CODEBASE.imports_edges,"files_name":CODEBASE.files,"key":self.key})).await?;
            }
            for collection in [CODEBASE.chunks, CODEBASE.embeddings] {
                query(&client, "FOR d IN @@collection FILTER d.file_key == @key REMOVE d IN @@collection",
                    json!({"@collection":collection,"key":self.key})).await?;
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
            self.file["_key"] = json!(self.key);
            let mode = if self.merge_file { "update" } else { "replace" };
            client.post(&format!("document/{}?overwriteMode={mode}", CODEBASE.files), &self.file).await?;
            Ok(())
        }).await
    }
}

async fn query(client: &ArangoClient, aql: &str, binds: Value) -> Result<(), ArangoError> {
    let result = client
        .post(
            "cursor",
            &json!({"query":aql,"bindVars":binds,"batchSize":1,
        "ttl":30,"options":{"maxRuntime":30,"failOnWarning":true}}),
        )
        .await?;
    if result["hasMore"] == true {
        return Err(ArangoError::Request(
            "unexpected persistence cursor continuation".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hades_core::test_support::{Fixtures, with_temp_db};

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
        }
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
