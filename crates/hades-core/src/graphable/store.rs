//! Append-only import with one atomic publication point and exact retry checks.
use super::{Plan, Result};
use crate::db::{ArangoClient, ArangoError, ArangoPool, crud, transaction};
use crate::dispatch as handlers;
use serde_json::{Value, json};

/// Import a validated plan into an explicitly selected database. Graph creation
/// is opt-in. Empty schema can survive a failed attempt, but content is atomic.
/// There is deliberately no force/overwrite option.
pub async fn import(pool: &ArangoPool, plan: &Plan, create_graph: bool) -> Result<Value> {
    import_inner(
        pool,
        plan,
        create_graph,
        #[cfg(test)]
        None,
    )
    .await
}

/// Exercise the normal importer, pausing after its first acknowledged node batch.
/// This checkpoint is invocation-local and absent from production builds.
#[cfg(test)]
pub(super) async fn import_until_node_write(
    pool: &ArangoPool,
    plan: &Plan,
    written: tokio::sync::oneshot::Sender<ArangoClient>,
) -> Result<Value> {
    import_inner(pool, plan, false, Some(written)).await
}

/// Prepare collections and retain one transaction owner for the complete import.
async fn import_inner(
    pool: &ArangoPool,
    plan: &Plan,
    create_graph: bool,
    #[cfg(test)] written: Option<tokio::sync::oneshot::Sender<ArangoClient>>,
) -> Result<Value> {
    if create_graph {
        let graphs = handlers::db_graph_list(pool).await?;
        if !graphs["graphs"]
            .as_array()
            .is_some_and(|gs| gs.iter().any(|g| g["name"] == plan.graph))
        {
            handlers::kg_create(pool, &plan.graph).await?;
        }
    }
    let (nodes, edges) = handlers::kg_layout(pool, &plan.graph).await?;
    let imports = format!("{}_imports", plan.graph);
    match pool
        .writer()
        .get(&format!("collection/{imports}/properties"))
        .await
    {
        Ok(v) => check_collection(&v)?,
        Err(e) if e.is_not_found() => {
            match crud::create_collection(pool, &imports, Some(2)).await {
                Ok(_)
                | Err(ArangoError::Api {
                    error_num: 1207, ..
                }) => {}
                Err(e) => return Err(e.into()),
            }
            check_collection(
                &pool
                    .writer()
                    .get(&format!("collection/{imports}/properties"))
                    .await?,
            )?;
        }
        Err(e) => return Err(e.into()),
    }
    let plan = plan.clone();
    let database = pool.database().to_owned();
    transaction::run(
        pool,
        vec![nodes.clone(), edges.clone(), imports.clone()],
        move |client| {
            write_plan(
                client,
                plan,
                database,
                nodes,
                edges,
                imports,
                #[cfg(test)]
                written,
            )
        },
    )
    .await
    .map_err(Into::into)
}

async fn write_plan(
    client: ArangoClient,
    plan: Plan,
    database: String,
    nodes: String,
    edges: String,
    imports: String,
    #[cfg(test)] mut written: Option<tokio::sync::oneshot::Sender<ArangoClient>>,
) -> std::result::Result<Value, ArangoError> {
    // Guard legacy/unmanaged rows too: an absent manifest does not grant
    // permission to import a conflicting source revision into a used graph.
    for collection in [&nodes, &edges] {
        let response = client.post("cursor", &json!({"query":
                "FOR n IN @@rows FILTER n.doc_id == @doc AND n.source_version == @version AND (n.source_sha256 != @hash OR n.mapping_version != @mapping) LIMIT 1 RETURN n._id",
                "bindVars":{"@rows":collection,"doc":plan.doc_id,"version":plan.version,
                    "hash":plan.source_sha256,"mapping":super::MAPPING_VERSION},
                "batchSize":2,"ttl":5,"options":{"maxRuntime":5}})).await?;
        if !crate::db::query::completed_rows(&response)?.is_empty() {
            return Err(conflict(
                "existing source version has a different hash or mapping; retain it and review the source version or use a separate graph",
            ));
        }
    }
    let mut inserted_nodes = 0;
    let mut inserted_edges = 0;
    // Compare every row even on a repeat. A receipt alone cannot prove the
    // graph is intact after an administrator edits or removes a row.
    let missing_manifest = missing(&client, &imports, &plan.manifest).await?;
    for (collection, rows, count) in [
        (&nodes, &plan.nodes, &mut inserted_nodes),
        (&edges, &plan.edges, &mut inserted_edges),
    ] {
        let mut pending = Vec::new();
        for batch in rows.chunks(500) {
            // One result per key, including null for absent rows. The lookup
            // uses the same transaction and cannot publish or overwrite data.
            let keys: Vec<_> = batch.iter().map(|row| &row["_key"]).collect();
            let response = client
                .post(
                    "cursor",
                    &json!({
                        "query": "FOR key IN @keys RETURN DOCUMENT(@collection, key)",
                        "bindVars": {"keys": keys, "collection": collection},
                        "batchSize": 501, "ttl": 5, "options": {"maxRuntime": 5}
                    }),
                )
                .await?;
            let existing = crate::db::query::completed_rows(&response)?;
            if existing.len() != batch.len() {
                return Err(ArangoError::Request("invalid batch lookup response".into()));
            }
            for (row, existing) in batch.iter().zip(existing) {
                if existing.is_null() {
                    pending.push(row.clone());
                } else {
                    compare_existing(collection, row, existing.clone())?;
                }
            }
        }
        *count = pending.len();
        for batch in pending.chunks(100) {
            let response = client
                .post(&format!("document/{collection}"), &json!(batch))
                .await?;
            let outcomes = response
                .as_array()
                .filter(|a| a.len() == batch.len())
                .ok_or_else(|| ArangoError::Request("invalid batch insert response".into()))?;
            if outcomes
                .iter()
                .any(|v| v["error"] == true || v["_id"].as_str().is_none())
            {
                return Err(ArangoError::Request(format!(
                    "import batch failed: {response}"
                )));
            }
            #[cfg(test)]
            if collection == &nodes
                && let Some(written) = written.take()
            {
                // Keep the transaction before commit until cancellation drops
                // this future. No sleep, global hook, or unrelated transaction scan.
                let _ = written.send(client.clone());
                std::future::pending::<()>().await;
            }
        }
    }
    if missing_manifest {
        client
            .post(&format!("document/{imports}"), &plan.manifest)
            .await?;
    }
    let mut result = plan.summary();
    result["database"] = json!(database);
    result["mapped"] = json!(true);
    result["inserted_nodes"] = json!(inserted_nodes);
    result["inserted_edges"] = json!(inserted_edges);
    result["unchanged"] = json!(!missing_manifest && inserted_nodes == 0 && inserted_edges == 0);
    result["source_archive_id"] = json!(format!(
        "{imports}/{}",
        plan.manifest["_key"].as_str().unwrap()
    ));
    Ok(result)
}

fn check_collection(value: &Value) -> std::result::Result<(), ArangoError> {
    if value["type"] != 2 {
        return Err(conflict(
            "import manifest collection must be a document collection",
        ));
    }
    Ok(())
}
fn conflict(message: impl Into<String>) -> ArangoError {
    ArangoError::Api {
        status: 409,
        error_num: 1200,
        message: message.into(),
        body: Value::Null,
    }
}
async fn missing(
    client: &ArangoClient,
    collection: &str,
    expected: &Value,
) -> std::result::Result<bool, ArangoError> {
    let path = format!(
        "document/{collection}/{}",
        expected["_key"].as_str().unwrap()
    );
    match client.get(&path).await {
        Ok(existing) => {
            compare_existing(collection, expected, existing)?;
            Ok(false)
        }
        Err(e) if e.is_not_found() => Ok(true),
        Err(e) => Err(e),
    }
}

/// Normalize exact integral floats for comparison after database JSON readback.
/// Integer values never pass through f64, which would lose precision above 2^53.
fn normalize_integral_numbers(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(normalize_integral_numbers),
        Value::Object(fields) => fields.values_mut().for_each(normalize_integral_numbers),
        Value::Number(number) if number.is_f64() => {
            let float = number.as_f64().unwrap();
            if !float.is_finite() || float.fract() != 0.0 {
                return;
            }
            // The upper bounds are exclusive: MAX rounds up when cast to f64.
            if float >= 0.0 && float < u64::MAX as f64 {
                let integer = float as u64;
                if integer as f64 == float {
                    *number = integer.into();
                }
            } else if float < 0.0 && float >= i64::MIN as f64 {
                let integer = float as i64;
                if integer as f64 == float {
                    *number = integer.into();
                }
            }
        }
        _ => {}
    }
}

/// Compare payloads allowing exact integral numeric readback normalization.
/// Source bytes, hashes, and mapping versions remain unchanged and compared.
/// Any authored edit is a conflict, never permission to replace the stored row.
fn compare_existing(
    collection: &str,
    expected: &Value,
    mut existing: Value,
) -> std::result::Result<(), ArangoError> {
    let object = existing
        .as_object_mut()
        .ok_or_else(|| ArangoError::Request("invalid document lookup response".into()))?;
    for field in ["_id", "_rev"] {
        object.remove(field);
    }
    normalize_integral_numbers(&mut existing);
    let mut normalized_expected = expected.clone();
    normalize_integral_numbers(&mut normalized_expected);
    if existing != normalized_expected {
        return Err(conflict(format!(
            "payload conflict at document/{collection}/{}; existing content preserved",
            expected["_key"].as_str().unwrap()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_comparison_is_exact_at_integer_boundaries() {
        for (float, integer) in [
            (json!(1.0), json!(1)),
            (json!(-1.0), json!(-1)),
            (json!(-0.0), json!(0)),
            (json!(i64::MIN as f64), json!(i64::MIN)),
            (
                json!(9223372036854775808.0_f64),
                json!(9223372036854775808_u64),
            ),
            (
                json!(18446744073709549568.0_f64),
                json!(18446744073709549568_u64),
            ),
        ] {
            let expected = json!({"_key":"sample", "metadata":{"nested":[float]}});
            let existing = json!({"_key":"sample", "_id":"nodes/sample", "_rev":"1", "metadata":{"nested":[integer]}});
            compare_existing("nodes", &expected, existing).unwrap();
        }
        for (left, right) in [
            (json!(1.5), json!(1)),
            (json!(1.5), json!(1.6)),
            (json!("1.0"), json!(1)),
            (json!(9007199254740993_u64), json!(9007199254740992.0_f64)),
            (json!(i64::MAX), json!(i64::MAX as f64)),
            (json!(u64::MAX), json!(u64::MAX as f64)),
            (json!(-9223372036854775807_i64), json!(i64::MIN as f64)),
            (json!(-18446744073709551616.0_f64), json!(i64::MIN)),
        ] {
            let expected = json!({"_key":"sample", "value":left});
            let existing = json!({"_key":"sample", "value":right});
            assert!(
                compare_existing("nodes", &expected, existing)
                    .unwrap_err()
                    .to_string()
                    .contains("payload conflict")
            );
        }
    }
}
