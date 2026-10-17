//! Append-only import with one atomic publication point and exact retry checks.
use super::{Plan, Result};
use crate::db::{ArangoClient, ArangoError, ArangoPool, crud, transaction};
use crate::dispatch as handlers;
use serde_json::{Value, json};

/// Import a validated plan into an explicitly selected database. Graph creation
/// is opt-in. Empty schema can survive a failed attempt, but content is atomic.
/// There is deliberately no force/overwrite option.
pub async fn import(pool: &ArangoPool, plan: &Plan, create_graph: bool) -> Result<Value> {
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
        move |client| write_plan(client, plan, database, nodes, edges, imports),
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
        for row in rows {
            if missing(&client, collection, row).await? {
                pending.push(row.clone());
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
        Ok(mut existing) => {
            for field in ["_id", "_rev"] {
                existing.as_object_mut().unwrap().remove(field);
            }
            if &existing != expected {
                return Err(conflict(format!(
                    "payload conflict at {path}; existing content preserved"
                )));
            }
            Ok(false)
        }
        Err(e) if e.is_not_found() => Ok(true),
        Err(e) => Err(e),
    }
}
