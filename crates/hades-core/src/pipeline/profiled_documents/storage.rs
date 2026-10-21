//! Bounded storage operations. Every mutation and search uses the profile guard's
//! transaction client; external embedding never runs while these locks are held.
use super::*;
use crate::db::{ArangoClient, query};
use crate::retrieval::TopK;

const DOCUMENT_FIELDS: &[&str] = &[
    "_key",
    "_id",
    "_rev",
    "format",
    "source_id",
    "source_sha256",
    "full_text",
    "chunk_count",
    "embedding_binding",
    "passage_plan",
];
const CHUNK_FIELDS: &[&str] = &[
    "_key",
    "_id",
    "_rev",
    "format",
    "parent_key",
    "chunk_key",
    "chunk_index",
    "source_sha256",
    "embedding_binding",
    "text",
    "span",
];
const VECTOR_FIELDS: &[&str] = &[
    "_key",
    "_id",
    "_rev",
    "format",
    "parent_key",
    "chunk_key",
    "chunk_index",
    "source_sha256",
    "embedding_binding",
    "model",
    "dimension",
    "embedding",
];

/// Check an existing source before spending inference, without adopting legacy data.
pub(super) async fn previous_revision(
    client: &ArangoClient,
    key: &str,
    replace: bool,
    expected: &BoundEmbeddingRequest,
    snapshot: &ProfileSnapshot,
) -> Result<Option<String>, ArangoError> {
    let existing = document(client, key).await?;
    if let Some(row) = existing {
        ensure(
            replace,
            "source already exists; replacement must be explicit",
        )?;
        validate_document(&row, key, expected, snapshot)?;
        return Ok(Some(revision(&row)?.into()));
    }
    Ok(None)
}

/// A missing document is distinct from a missing collection or database.
async fn document(client: &ArangoClient, key: &str) -> Result<Option<Value>, ArangoError> {
    match client
        .clone()
        .with_response_limit(MAX_PREPARED_BYTES)?
        .get(&format!("document/documents/{key}"))
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

/// Reject content changes after preparation, and remove only a complete managed source.
pub(super) async fn write(
    client: ArangoClient,
    prepared: PreparedDocument,
) -> Result<(), ArangoError> {
    let key = prepared.document_key();
    let existing = document(&client, key).await?;
    let current_revision = existing.as_ref().map(revision).transpose()?;
    ensure(
        current_revision == prepared.previous_revision.as_deref(),
        "source changed after preparation; prepare it again",
    )?;
    if let Some(row) = &existing {
        validate_document(row, key, &prepared.binding, &prepared.snapshot)?;
    }
    // These bounded reads also detect stray/foreign children of a supposedly new source.
    let chunks = children(&client, "chunks", key).await?;
    let embeddings = children(&client, "embeddings", key).await?;
    if let Some(row) = &existing {
        let count = row["chunk_count"].as_u64().unwrap() as usize;
        ensure(
            chunks.len() == count && embeddings.len() == count,
            "existing source has incomplete or foreign children",
        )?;
        let normalization = prepared
            .snapshot
            .profile()
            .workloads
            .documents
            .as_ref()
            .expect("admitted documents")
            .space
            .representation
            .normalize;
        for embedding in &embeddings {
            allowed_fields(embedding, VECTOR_FIELDS)?;
            let chunk = chunks
                .iter()
                .find(|c| c["_key"] == embedding["chunk_key"])
                .ok_or_else(|| invalid("existing vector has no matching chunk"))?;
            allowed_fields(chunk, CHUNK_FIELDS)?;
            validate_triplet(
                embedding,
                chunk,
                row,
                &prepared.binding,
                normalization,
                &prepared.snapshot,
            )?;
            let index = embedding["chunk_index"].as_u64().unwrap() as usize;
            let span = &row["passage_plan"]["spans"][index];
            ensure(
                chunk["span"] == *span,
                "existing chunk span disagrees with source plan",
            )?;
            let start = span["start_byte"].as_u64().unwrap() as usize;
            let end = span["end_byte"].as_u64().unwrap() as usize;
            ensure(
                chunk["text"].as_str() == Some(&row["full_text"].as_str().unwrap()[start..end]),
                "existing chunk text disagrees with source plan",
            )?;
        }
    } else {
        ensure(
            chunks.is_empty() && embeddings.is_empty(),
            "new source has orphan children",
        )?;
    }
    for collection in ["chunks", "embeddings"] {
        let response = aql(
            &client,
            "FOR r IN @@collection FILTER r.parent_key == @parent REMOVE r IN @@collection",
            json!({"@collection":collection,"parent":key}),
            1,
            MAX_PREPARED_BYTES,
        )
        .await?;
        query::completed_mutation(&response)?;
    }
    // All prior children were validated above. Any collision with another source
    // remains an error; overwrite is enabled only for this metadata revision.
    let path = if existing.is_some() {
        "document/documents?overwriteMode=replace"
    } else {
        "document/documents"
    };
    let response = client
        .clone()
        .with_response_limit(64 * 1024)?
        .post(path, &prepared.document)
        .await?;
    acknowledged(&response, &prepared.document)?;
    for (collection, rows) in [
        ("chunks", prepared.chunks),
        ("embeddings", prepared.embeddings),
    ] {
        for batch in rows.chunks(32) {
            let response = client
                .clone()
                .with_response_limit(64 * 1024)?
                .post(&format!("document/{collection}"), &json!(batch))
                .await?;
            let results = response
                .as_array()
                .filter(|r| r.len() == batch.len())
                .ok_or_else(|| invalid("incomplete document insertion acknowledgment"))?;
            for (ack, row) in results.iter().zip(batch) {
                acknowledged(ack, row)?;
            }
        }
    }
    Ok(())
}

/// Bound children before deletion; a source cannot own more than the admission limit.
async fn children(
    client: &ArangoClient,
    collection: &str,
    key: &str,
) -> Result<Vec<Value>, ArangoError> {
    let response = aql(
        client,
        "FOR r IN @@collection FILTER r.parent_key == @parent LIMIT @limit RETURN r",
        json!({"@collection":collection,"parent":key,"limit":MAX_PASSAGES + 1}),
        MAX_PASSAGES + 2,
        MAX_PREPARED_BYTES,
    )
    .await?;
    let rows = query::completed_rows(&response)?;
    ensure(rows.len() <= MAX_PASSAGES, "source has too many children")?;
    Ok(rows.clone())
}

/// Enforce a complete scan with bounded pages and O(K) retained search results.
pub(super) async fn search(
    client: ArangoClient,
    expected: BoundEmbeddingRequest,
    vector: Vec<f32>,
    normalization: Normalization,
    top_k: usize,
    max_rows: u64,
    snapshot: ProfileSnapshot,
) -> Result<(Vec<Value>, u64), ArangoError> {
    let mut top = TopK::new(vector, expected.model.clone(), "parent_key", top_k)?;
    let mut after = String::new();
    let mut scanned = 0;
    loop {
        // Do not filter on provenance: an incompatible row must refuse the whole
        // result, rather than silently dropping unknown/legacy vectors.
        let response = aql(
            &client,
            r#"
            FOR e IN embeddings FILTER e._key > @after SORT e._key LIMIT 8
                LET c = DOCUMENT("chunks", e.chunk_key)
                LET d = DOCUMENT("documents", e.parent_key)
                RETURN {
                    vector: KEEP(e, "_key", "format", "parent_key", "chunk_key",
                        "chunk_index", "source_sha256", "embedding_binding",
                        "model", "dimension", "embedding"),
                    chunk: KEEP(c, "_key", "format", "parent_key", "chunk_key",
                        "chunk_index", "source_sha256", "embedding_binding", "span"),
                    document: KEEP(d, "_key", "format", "source_sha256",
                        "embedding_binding", "chunk_count", "passage_plan")
                }
        "#,
            json!({"after":after}),
            9,
            4 * 1024 * 1024,
        )
        .await?;
        let rows = query::completed_rows(&response)?;
        if rows.is_empty() {
            break;
        }
        for row in rows {
            ensure(
                scanned < max_rows,
                "search row budget exceeded; no partial ranking returned",
            )?;
            validate_triplet(
                &row["vector"],
                &row["chunk"],
                &row["document"],
                &expected,
                normalization,
                &snapshot,
            )?;
            let key = row["vector"]["_key"].as_str().unwrap();
            ensure(
                key > after.as_str(),
                "search page keys are not strictly increasing",
            )?;
            after = key.into();
            top.insert(row["vector"].clone())?;
            scanned += 1;
        }
    }
    // Group only hit indices: fetch/hash each parent once, holding at most one
    // full source at a time. A cache of K two-MiB sources would exceed admission.
    // Fill the original slots to preserve exact ranking across parent groups.
    let mut items = top.into_items();
    let mut parents = std::collections::BTreeMap::<String, Vec<usize>>::new();
    for (rank, item) in items.iter().enumerate() {
        parents
            .entry(item["parent_key"].as_str().unwrap().into())
            .or_default()
            .push(rank);
    }
    let mut bytes = 2 + items.len().saturating_sub(1); // JSON brackets and commas
    for (key, ranks) in parents {
        let document = document(&client, &key)
            .await?
            .ok_or_else(|| invalid("missing ranked source detail"))?;
        validate_document(&document, &key, &expected, &snapshot)?;
        for rank in ranks {
            let item = &mut items[rank];
            let chunk = client
                .clone()
                .with_response_limit(MAX_PREPARED_BYTES)?
                .get(&format!(
                    "document/chunks/{}",
                    item["chunk_key"].as_str().unwrap()
                ))
                .await?;
            let span: TokenPassage = serde_json::from_value(chunk["span"].clone())
                .map_err(|_| invalid("invalid ranked source span"))?;
            let text = document["full_text"]
                .as_str()
                .unwrap()
                .get(span.start_byte..span.end_byte)
                .ok_or_else(|| invalid("ranked source span is outside its UTF-8 text"))?;
            ensure(
                chunk["_key"] == item["chunk_key"]
                    && chunk["parent_key"] == item["parent_key"]
                    && chunk["text"].as_str() == Some(text),
                "ranked passage disagrees with source text",
            )?;
            item["text"] = json!(text);
            item["source_id"] = document["source_id"].clone();
            item["chunk_index"] = chunk["chunk_index"].clone();
            item["total_chunks"] = document["chunk_count"].clone();
            item["span"] = chunk["span"].clone();
            bytes += serde_json::to_vec(item)?.len();
            ensure(
                bytes <= 1024 * 1024,
                "search result byte budget exceeded; request fewer results",
            )?;
        }
    }
    Ok((items, scanned))
}

/// Compare persisted identity before touching vector values. Serving load IDs
/// describe observations, not embedding spaces; a verified identical reload is compatible.
fn binding(row: &Value, expected: &BoundEmbeddingRequest) -> Result<(), ArangoError> {
    ensure(row["format"] == FORMAT, "legacy or unknown document format")?;
    let stored: BoundEmbeddingRequest = serde_json::from_value(row["embedding_binding"].clone())
        .map_err(|_| invalid("invalid stored embedding binding"))?;
    stored
        .expected
        .validate()
        .map_err(|_| invalid("invalid stored provider receipt"))?;
    let mut compatible = expected.clone();
    compatible.expected.serving_generation = stored.expected.serving_generation.clone();
    ensure(
        stored == compatible,
        "stored vector belongs to an incompatible embedding binding",
    )
}

/// Validate all three provenance links, then vector representation, before ranking/deletion.
fn validate_triplet(
    vector: &Value,
    chunk: &Value,
    document: &Value,
    expected: &BoundEmbeddingRequest,
    normalization: Normalization,
    snapshot: &ProfileSnapshot,
) -> Result<(), ArangoError> {
    binding(vector, expected)?;
    let span: TokenPassage = serde_json::from_value(chunk["span"].clone())
        .map_err(|_| invalid("invalid stored token passage"))?;
    let policy = &snapshot
        .profile()
        .workloads
        .documents
        .as_ref()
        .expect("admitted documents")
        .chunking;
    let ChunkPolicy::Conventional {
        passage_tokens,
        overlap_tokens,
    } = policy
    else {
        return Err(invalid("stored passage requires conventional policy"));
    };
    ensure(
        span.start_byte < span.end_byte
            && span.end_byte <= MAX_TEXT_BYTES
            && span.body_tokens <= *passage_tokens
            && span.overlap_tokens <= *overlap_tokens
            && span.input_tokens > 0
            && span.input_tokens <= snapshot.profile().capabilities.max_input_tokens,
        "stored passage violates token policy",
    )?;
    ensure(
        chunk["format"] == FORMAT
            && document["format"] == FORMAT
            && chunk["embedding_binding"] == vector["embedding_binding"]
            && document["embedding_binding"] == vector["embedding_binding"],
        "stored source, chunk, and vector bindings disagree",
    )?;
    let count = document["chunk_count"]
        .as_u64()
        .filter(|n| (1..=MAX_PASSAGES as u64).contains(n))
        .ok_or_else(|| invalid("invalid source chunk count"))?;
    let index = vector["chunk_index"]
        .as_u64()
        .filter(|i| *i < count)
        .ok_or_else(|| invalid("invalid chunk ordinal"))?;
    let plan: PassagePlan = serde_json::from_value(document["passage_plan"].clone())
        .map_err(|_| invalid("invalid stored token passage plan"))?;
    let stored: BoundEmbeddingRequest = serde_json::from_value(vector["embedding_binding"].clone())
        .map_err(|_| invalid("invalid stored embedding binding"))?;
    plan.validate_metadata(
        document["source_sha256"].as_str().unwrap_or(""),
        &stored,
        policy,
        snapshot.profile().capabilities.max_input_tokens,
    )
    .map_err(|_| invalid("stored passage plan changes identity or token policy"))?;
    ensure(
        plan.spans.len() as u64 == count
            && plan.spans.get(index as usize) == Some(&span)
            && span.input_tokens <= plan.max_input_tokens
            && (index != 0 || (span.start_byte == 0 && span.overlap_tokens == 0)),
        "stored chunk disagrees with source passage plan",
    )?;
    let parent = document["_key"]
        .as_str()
        .filter(|key| valid_key(key, "bd"))
        .ok_or_else(|| invalid("invalid source key"))?;
    let key = digest_key("bc", &json!([expected.generation, parent, index]));
    let digest = vector["source_sha256"]
        .as_str()
        .ok_or_else(|| invalid("missing source digest"))?;
    ensure(
        valid_digest(digest)
            && chunk["source_sha256"] == digest
            && document["source_sha256"] == digest
            && vector["_key"] == key
            && vector["chunk_key"] == key
            && chunk["_key"] == key
            && chunk["chunk_key"] == key
            && vector["parent_key"] == parent
            && chunk["parent_key"] == parent
            && chunk["chunk_index"] == index,
        "stored source, chunk, and vector references disagree",
    )?;
    ensure(
        vector["model"] == expected.model && vector["dimension"] == expected.dimension,
        "stored model or vector width disagrees with profile",
    )?;
    let values: Vec<f32> = serde_json::from_value(vector["embedding"].clone())
        .map_err(|_| invalid("invalid stored vector values"))?;
    validate_vector(&values, expected.dimension, normalization)
}

/// Refuse replacing unknown fields or damaged source plans; no implicit repair/adoption.
fn validate_document(
    row: &Value,
    key: &str,
    expected: &BoundEmbeddingRequest,
    snapshot: &ProfileSnapshot,
) -> Result<(), ArangoError> {
    allowed_fields(row, DOCUMENT_FIELDS)?;
    binding(row, expected)?;
    let source_id = row["source_id"]
        .as_str()
        .ok_or_else(|| invalid("missing source identity"))?;
    let text = row["full_text"]
        .as_str()
        .ok_or_else(|| invalid("missing source text"))?;
    ensure(
        row["_key"] == key
            && digest_key("bd", &json!([source_id])) == key
            && row["source_sha256"] == hash(text.as_bytes()),
        "stored source identity is invalid",
    )?;
    let plan: PassagePlan = serde_json::from_value(row["passage_plan"].clone())
        .map_err(|_| invalid("invalid stored token passage plan"))?;
    let stored: BoundEmbeddingRequest = serde_json::from_value(row["embedding_binding"].clone())
        .map_err(|_| invalid("invalid stored embedding binding"))?;
    let profile = snapshot.profile();
    plan.validate(
        text,
        &stored,
        &profile
            .workloads
            .documents
            .as_ref()
            .expect("admitted documents")
            .chunking,
        profile.capabilities.max_input_tokens,
    )
    .map_err(|_| invalid("stored passage plan violates source identity or token policy"))?;
    ensure(
        row["chunk_count"].as_u64() == Some(plan.spans.len() as u64),
        "stored source chunk count disagrees with plan",
    )?;
    validate_input(&DocumentInput {
        source_id: source_id.into(),
        text: text.into(),
    })
    .map_err(|_| invalid("invalid stored extraction metadata"))?;
    revision(row)?;
    Ok(())
}

/// Protect additional authored metadata from destructive whole-record replacement.
fn allowed_fields(row: &Value, fields: &[&str]) -> Result<(), ArangoError> {
    ensure(
        row.as_object()
            .is_some_and(|r| r.keys().all(|k| fields.contains(&k.as_str()))),
        "replacement would discard unknown fields",
    )
}

/// Require the canonical lowercase SHA-256 spelling used by source and key digests.
fn valid_digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Distinguish generated source/chunk identities from arbitrary database keys.
fn valid_key(s: &str, prefix: &str) -> bool {
    s.strip_prefix(prefix)
        .and_then(|s| s.strip_prefix('_'))
        .is_some_and(valid_digest)
}

/// Require a bounded Arango revision before pinning or acknowledging a document.
fn revision(row: &Value) -> Result<&str, ArangoError> {
    row["_rev"]
        .as_str()
        .filter(|s| !s.is_empty() && s.len() <= 128)
        .ok_or_else(|| invalid("missing document revision"))
}

/// HTTP 2xx alone does not acknowledge every document in a bulk write.
fn acknowledged(ack: &Value, row: &Value) -> Result<(), ArangoError> {
    ensure(
        ack["error"].as_bool() != Some(true) && ack["_key"] == row["_key"],
        "document insertion was not acknowledged",
    )?;
    revision(ack)?;
    Ok(())
}

/// Every query is deliberately completed in one bounded response inside the transaction.
async fn aql(
    client: &ArangoClient,
    query: &str,
    bind: Value,
    batch: usize,
    bytes: usize,
) -> Result<Value, ArangoError> {
    client
        .clone()
        .with_response_limit(bytes)?
        .post(
            "cursor",
            &json!({
                "query":query, "bindVars":bind, "batchSize":batch, "ttl":5,
                "memoryLimit":32 * 1024 * 1024,
                "options":{"maxRuntime":5,"failOnWarning":true}
            }),
        )
        .await
}
