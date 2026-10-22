//! Bounded record I/O shared by managed document and code publication.
use crate::db::{ArangoClient, ArangoError, index::require_owner_index, query};
use serde_json::{Value, json};

fn invalid(message: &str) -> ArangoError {
    ArangoError::Request(format!("profile-bound storage: {message}"))
}

pub(super) async fn document(
    client: &ArangoClient,
    collection: &str,
    key: &str,
    bytes: usize,
) -> Result<Option<Value>, ArangoError> {
    match client
        .clone()
        .with_response_limit(bytes)?
        .get(&format!("document/{collection}/{key}"))
        .await
    {
        Ok(row) => Ok(Some(row)),
        Err(ArangoError::Api {
            status: 404,
            error_num: 1202,
            ..
        }) => Ok(None),
        Err(error) => Err(error),
    }
}

pub(super) fn revision(row: &Value) -> Result<&str, ArangoError> {
    row["_rev"]
        .as_str()
        .filter(|value| !value.is_empty() && value.len() <= 128)
        .ok_or_else(|| invalid("missing or invalid record revision"))
}

pub(super) fn acknowledged(
    ack: &Value,
    row: &Value,
    collection: &str,
) -> Result<String, ArangoError> {
    let key = row["_key"]
        .as_str()
        .ok_or_else(|| invalid("missing prepared key"))?;
    if ack["error"] == true || ack["_key"] != key || ack["_id"] != format!("{collection}/{key}") {
        return Err(invalid(
            "record insertion was not acknowledged for the requested collection/key",
        ));
    }
    Ok(revision(ack)?.into())
}

pub(super) fn allowed_fields(row: &Value, fields: &[&str]) -> Result<(), ArangoError> {
    if !row
        .as_object()
        .is_some_and(|row| row.keys().all(|key| fields.contains(&key.as_str())))
    {
        return Err(invalid("replacement would discard unknown fields"));
    }
    Ok(())
}

pub(super) async fn insert_rows(
    client: &ArangoClient,
    collection: &str,
    rows: &[Value],
) -> Result<(), ArangoError> {
    for batch in rows.chunks(32) {
        let response = client
            .clone()
            .with_response_limit(64 * 1024)?
            .post(&format!("document/{collection}"), &json!(batch))
            .await?;
        let results = response
            .as_array()
            .filter(|rows| rows.len() == batch.len())
            .ok_or_else(|| invalid("incomplete record insertion acknowledgment"))?;
        for (ack, row) in results.iter().zip(batch) {
            acknowledged(ack, row, collection)?;
        }
    }
    Ok(())
}

pub(super) async fn aql(
    client: &ArangoClient,
    statement: &str,
    vars: Value,
    batch: usize,
    bytes: usize,
) -> Result<Value, ArangoError> {
    client.clone().with_response_limit(bytes)?.post("cursor", &json!({
        "query":statement,"bindVars":vars,"batchSize":batch,"ttl":5,"memoryLimit":32*1024*1024,
        "options":{"maxRuntime":5,"failOnWarning":true}
    })).await
}

/// Resolve the actual index name again in the publication transaction. Forced hints
/// refuse missing/replaced indexes instead of silently falling back to a full scan.
pub(super) struct OwnedCollection {
    collection: String,
    field: &'static str,
    index: String,
}

impl OwnedCollection {
    pub(super) async fn require(
        client: &ArangoClient,
        collection: &str,
        field: &'static str,
    ) -> Result<Self, ArangoError> {
        if !["parent_key", "file_key"].contains(&field) {
            return Err(invalid("unknown owner field"));
        }
        Ok(Self {
            collection: collection.into(),
            field,
            index: require_owner_index(client, collection, field).await?,
        })
    }

    pub(super) async fn children(
        &self,
        client: &ArangoClient,
        key: &str,
        max_rows: usize,
        max_bytes: usize,
    ) -> Result<Vec<Value>, ArangoError> {
        let response = aql(client, &format!(
            "FOR r IN @@collection OPTIONS {{indexHint:@index, forceIndexHint:true}} FILTER r.{} == @owner LIMIT @limit RETURN r", self.field),
            json!({"@collection":self.collection,"index":self.index,"owner":key,"limit":max_rows+1}),
            max_rows+2, max_bytes).await?;
        let rows = query::completed_rows(&response)?;
        if rows.len() > max_rows || serde_json::to_vec(rows)?.len() > max_bytes {
            return Err(invalid("stored children exceed row/byte limits"));
        }
        Ok(rows.clone())
    }

    pub(super) async fn remove(
        &self,
        client: &ArangoClient,
        key: &str,
        bytes: usize,
    ) -> Result<(), ArangoError> {
        let response = aql(client, &format!(
            "FOR r IN @@collection OPTIONS {{indexHint:@index, forceIndexHint:true}} FILTER r.{} == @owner REMOVE r IN @@collection", self.field),
            json!({"@collection":self.collection,"index":self.index,"owner":key}), 1, bytes).await?;
        query::completed_mutation(&response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acknowledgments_require_the_exact_record_and_a_bounded_revision() {
        let row = json!({"_key":"one"});
        let valid = json!({"_key":"one","_id":"chunks/one","_rev":"revision"});
        assert!(acknowledged(&valid, &row, "chunks").is_ok());
        assert!(acknowledged(&valid, &row, "embeddings").is_err());
        for revision in [json!(null), json!(""), json!("x".repeat(129))] {
            let mut invalid = valid.clone();
            invalid["_rev"] = revision;
            assert!(acknowledged(&invalid, &row, "chunks").is_err());
        }
    }
}
