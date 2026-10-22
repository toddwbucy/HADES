//! Every graph write runs under the profile guard and one complete collection lock set.
use crate::{
    code::AnalysisTier,
    db::{ArangoClient, query},
};

use super::*;

const FILE_FIELDS: &[&str] = &[
    "_key",
    "_id",
    "_rev",
    "format",
    "kind",
    "namespace",
    "path",
    "full_text",
    "content_hash",
    "embedding_binding",
    "language",
    "analysis_tier",
    "analyzer",
    "fallback_reason",
    "metrics",
    "symbol_hash",
    "symbol_count",
    "chunk_count",
    "passage_plan",
    "record_manifest",
];

async fn file(client: &ArangoClient, key: &str) -> Result<Option<Value>, ArangoError> {
    match client
        .clone()
        .with_response_limit(MAX_PREPARED_BYTES)?
        .get(&format!("document/{}/{key}", CODEBASE.files))
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

fn revision(row: &Value) -> Result<&str, ArangoError> {
    row["_rev"]
        .as_str()
        .filter(|r| !r.is_empty())
        .ok_or_else(|| invalid("stored file has no revision"))
}

/// An old load is compatible only after deriving its exact saved binding from the current profile.
fn validate_file(
    row: &Value,
    key: &str,
    snapshot: &ProfileSnapshot,
    incoming_tier: AnalysisTier,
) -> Result<(), ArangoError> {
    let object = row
        .as_object()
        .ok_or_else(|| invalid("invalid stored code file"))?;
    ensure(
        object
            .keys()
            .all(|field| FILE_FIELDS.contains(&field.as_str())),
        "stored code file has unknown fields; replacement would discard them",
    )?;
    ensure(
        row["format"] == FORMAT && row["kind"] == "file" && row["_key"] == key,
        "stored file is not a verified code record; legacy adoption is unavailable",
    )?;
    let namespace = row["namespace"]
        .as_str()
        .ok_or_else(|| invalid("invalid stored namespace"))?;
    let path = row["path"]
        .as_str()
        .ok_or_else(|| invalid("invalid stored path"))?;
    ensure(
        keys::scoped_file_key(namespace, path) == key,
        "stored file identity differs",
    )?;
    let source = row["full_text"]
        .as_str()
        .ok_or_else(|| invalid("stored file has no source"))?;
    ensure(
        row["content_hash"] == compute_content_hash(source),
        "stored source digest differs",
    )?;
    let binding: BoundEmbeddingRequest = serde_json::from_value(row["embedding_binding"].clone())?;
    let expected = snapshot
        .profile()
        .bind(
            snapshot.manifest(),
            WorkloadKind::Code,
            EmbeddingRole::Passage,
            &binding.expected,
        )
        .map_err(|error| invalid(&format!("stored code binding differs: {error}")))?;
    ensure(
        binding == expected,
        "stored code binding differs from the active profile",
    )?;
    let plan: PassagePlan = serde_json::from_value(row["passage_plan"].clone())?;
    let workload = snapshot
        .profile()
        .workloads
        .code
        .as_ref()
        .ok_or_else(|| invalid("code is disabled_by_profile"))?;
    plan.validate(
        source,
        &binding,
        &workload.chunking,
        snapshot.profile().capabilities.max_input_tokens,
    )
    .map_err(|error| invalid(&format!("stored passage plan is invalid: {error}")))?;
    ensure(
        row["chunk_count"].as_u64() == Some(plan.spans.len() as u64),
        "stored passage count differs",
    )?;
    let tier = row["analysis_tier"]
        .as_str()
        .and_then(AnalysisTier::parse)
        .ok_or_else(|| invalid("stored file has no valid analysis tier"))?;
    ensure(
        incoming_tier >= tier,
        "replacement would downgrade code analysis",
    )?;
    let manifest: BTreeMap<String, BTreeMap<String, String>> =
        serde_json::from_value(row["record_manifest"].clone())?;
    let names: Vec<_> = CODEBASE
        .all_collections()
        .iter()
        .filter(|(name, _)| *name != CODEBASE.files)
        .map(|(name, _)| (*name).to_owned())
        .collect();
    ensure(
        manifest.len() == names.len() && names.iter().all(|name| manifest.contains_key(name)),
        "stored graph inventory has a different collection set",
    )?;
    ensure(
        manifest[CODEBASE.chunks].len() == plan.spans.len()
            && manifest[CODEBASE.embeddings].len() == plan.spans.len()
            && manifest[CODEBASE.symbols].len() <= MAX_SYMBOLS
            && row["symbol_count"].as_u64() == Some(manifest[CODEBASE.symbols].len() as u64)
            && manifest[CODEBASE.defines_edges].len() == manifest[CODEBASE.symbols].len()
            && [
                CODEBASE.calls_edges,
                CODEBASE.imports_edges,
                CODEBASE.implements_edges,
            ]
            .iter()
            .map(|name| manifest[*name].len())
            .sum::<usize>()
                <= MAX_RELATIONSHIPS,
        "stored graph inventory exceeds limits or differs from file counts",
    )?;
    Ok(())
}

pub(super) async fn admit(
    client: &ArangoClient,
    key: &str,
    replace: bool,
    snapshot: &ProfileSnapshot,
    tier: AnalysisTier,
    symbols: &[Value],
) -> Result<Option<String>, ArangoError> {
    let Some(row) = file(client, key).await? else {
        return Ok(None);
    };
    ensure(
        replace,
        "source already exists; code replacement must be explicit",
    )?;
    validate_file(&row, key, snapshot, tier)?;
    preserve_symbols(&row, symbols)?;
    Ok(Some(revision(&row)?.into()))
}

fn preserve_symbols(file: &Value, incoming: &[Value]) -> Result<(), ArangoError> {
    // Keep endpoints alive even when an authored relationship belongs to another collection.
    // The file inventory is validated before either call to this helper.
    let previous = file["record_manifest"][CODEBASE.symbols]
        .as_object()
        .ok_or_else(|| invalid("invalid stored symbol inventory"))?;
    ensure(
        previous
            .keys()
            .all(|key| incoming.iter().any(|row| row["_key"] == *key)),
        "replacement would retire or move symbol endpoints; explicit graph reconciliation is required",
    )
}

async fn aql(
    client: &ArangoClient,
    statement: &str,
    vars: Value,
    batch: usize,
) -> Result<Value, ArangoError> {
    client.clone().with_response_limit(MAX_PREPARED_BYTES)?.post("cursor", &json!({
        "query":statement,"bindVars":vars,"batchSize":batch,"ttl":5,"memoryLimit":32*1024*1024,
        "options":{"maxRuntime":5,"failOnWarning":true}
    })).await
}

async fn children(client: &ArangoClient, key: &str) -> Result<Rows, ArangoError> {
    let mut rows = Rows::new();
    let mut bytes = 0;
    for (collection, _) in CODEBASE.all_collections() {
        if collection == CODEBASE.files {
            continue;
        }
        let response = aql(
            client,
            "FOR r IN @@collection FILTER r.file_key == @file LIMIT @limit RETURN r",
            json!({"@collection":collection,"file":key,"limit":MAX_ROWS+1}),
            MAX_ROWS + 2,
        )
        .await?;
        let found = query::completed_rows(&response)?;
        ensure(
            found.len() <= MAX_ROWS,
            "stored code graph exceeds row limits",
        )?;
        bytes += serde_json::to_vec(found)?.len();
        ensure(
            bytes <= MAX_PREPARED_BYTES,
            "stored code graph exceeds 16 MiB",
        )?;
        rows.insert(collection.into(), found.clone());
    }
    Ok(rows)
}

fn acknowledged(ack: &Value, row: &Value, collection: &str) -> Result<String, ArangoError> {
    let key = row["_key"]
        .as_str()
        .ok_or_else(|| invalid("missing prepared key"))?;
    ensure(
        ack["error"] != true && ack["_key"] == key && ack["_id"] == format!("{collection}/{key}"),
        "code write was not acknowledged for the requested record",
    )?;
    Ok(revision(ack)?.into())
}

pub(super) async fn write(
    client: ArangoClient,
    prepared: PreparedCodeFile,
) -> Result<String, ArangoError> {
    let key = prepared.file_key();
    let existing = file(&client, key).await?;
    ensure(
        existing.as_ref().map(revision).transpose()? == prepared.previous_revision.as_deref(),
        "source changed after preparation; prepare it again",
    )?;
    let old = children(&client, key).await?;
    if let Some(row) = &existing {
        let incoming_tier = prepared.file["analysis_tier"]
            .as_str()
            .and_then(AnalysisTier::parse)
            .expect("constructed analysis tier");
        validate_file(row, key, &prepared.snapshot, incoming_tier)?;
        ensure(
            json!(graph::manifest(&old)?) == row["record_manifest"],
            "stored graph inventory differs; refusing to overwrite incomplete, foreign or edited records",
        )?;
        preserve_symbols(row, &prepared.rows[CODEBASE.symbols])?;
    } else {
        ensure(
            old.values().all(Vec::is_empty),
            "new code file has orphan or foreign graph records",
        )?;
    }
    for collection in old.keys() {
        let response = aql(
            &client,
            "FOR r IN @@collection FILTER r.file_key == @file REMOVE r IN @@collection",
            json!({"@collection":collection,"file":key}),
            1,
        )
        .await?;
        query::completed_mutation(&response)?;
    }
    let mode = if existing.is_some() {
        "?overwriteMode=replace"
    } else {
        ""
    };
    let ack = client
        .clone()
        .with_response_limit(64 * 1024)?
        .post(
            &format!("document/{}{mode}", CODEBASE.files),
            &prepared.file,
        )
        .await?;
    let revision = acknowledged(&ack, &prepared.file, CODEBASE.files)?;
    // Endpoint collections precede edge collections. Only validated old children were removed;
    // an unrelated key collision fails instead of enabling overwrite on child records.
    for (collection, _) in CODEBASE.all_collections() {
        if collection == CODEBASE.files {
            continue;
        }
        for batch in prepared.rows[collection].chunks(32) {
            let response = client
                .clone()
                .with_response_limit(64 * 1024)?
                .post(&format!("document/{collection}"), &json!(batch))
                .await?;
            let results = response
                .as_array()
                .filter(|rows| rows.len() == batch.len())
                .ok_or_else(|| invalid("incomplete code insertion acknowledgment"))?;
            for (ack, row) in results.iter().zip(batch) {
                acknowledged(ack, row, collection)?;
            }
        }
    }
    Ok(revision)
}
