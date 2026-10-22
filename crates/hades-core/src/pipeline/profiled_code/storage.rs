//! Every graph write runs under the profile guard and one complete collection lock set.
use crate::pipeline::profiled_storage::{
    self as records, OwnedCollection, acknowledged, allowed_fields, insert_rows, revision,
};
use crate::{code::AnalysisTier, db::ArangoClient};

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
    records::document(client, CODEBASE.files, key, MAX_PREPARED_BYTES).await
}

/// An old load is compatible only after deriving its exact saved binding from the current profile.
fn validate_file(
    row: &Value,
    key: &str,
    snapshot: &ProfileSnapshot,
    incoming_tier: AnalysisTier,
) -> Result<(), ArangoError> {
    allowed_fields(row, FILE_FIELDS)?;
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

async fn children(
    client: &ArangoClient,
    key: &str,
) -> Result<(Rows, Vec<OwnedCollection>), ArangoError> {
    let mut rows = Rows::new();
    let mut owners = Vec::new();
    let mut bytes = 0;
    for (collection, _) in CODEBASE.all_collections() {
        if collection == CODEBASE.files {
            continue;
        }
        let owner = OwnedCollection::require(client, collection, "file_key").await?;
        let found = owner
            .children(client, key, MAX_ROWS, MAX_PREPARED_BYTES)
            .await?;
        bytes += serde_json::to_vec(&found)?.len();
        ensure(
            bytes <= MAX_PREPARED_BYTES,
            "stored code graph exceeds 16 MiB",
        )?;
        rows.insert(collection.into(), found);
        owners.push(owner);
    }
    Ok((rows, owners))
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
    let (old, owners) = children(&client, key).await?;
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
    for owner in owners {
        owner.remove(&client, key, MAX_PREPARED_BYTES).await?;
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
        insert_rows(&client, collection, &prepared.rows[collection]).await?;
    }
    Ok(revision)
}
