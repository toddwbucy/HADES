//! Preserve conventional records while requiring complete contextual provenance for late records.
use super::*;
use crate::persephone::{bound_embedding::LateTokenRange, late_passage_plan::LatePassagePlan};
use serde::Deserialize;

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub(super) enum DocumentPlan {
    Conventional(PassagePlan),
    Late(LatePassagePlan),
}

pub(super) fn policy(snapshot: &ProfileSnapshot) -> &ChunkPolicy {
    &snapshot
        .profile()
        .workloads
        .documents
        .as_ref()
        .expect("admitted documents")
        .chunking
}

pub(super) fn format(policy: &ChunkPolicy) -> &'static str {
    match policy {
        ChunkPolicy::Conventional { .. } => FORMAT,
        ChunkPolicy::Late { .. } => "hades-bound-document-v3",
    }
}

impl DocumentPlan {
    pub fn spans(&self) -> &[TokenPassage] {
        match self {
            Self::Conventional(plan) => &plan.spans,
            Self::Late(plan) => &plan.passages,
        }
    }

    pub fn ceiling(&self) -> u32 {
        match self {
            Self::Conventional(plan) => plan.max_input_tokens,
            Self::Late(plan) => plan.max_input_tokens,
        }
    }

    /// All persisted context metadata must agree before any associated vector is scored.
    pub fn stored(
        row: &Value,
        binding: &BoundEmbeddingRequest,
        snapshot: &ProfileSnapshot,
    ) -> Result<Self, ArangoError> {
        let plan: Self = serde_json::from_value(row["passage_plan"].clone())
            .map_err(|_| invalid("invalid stored passage plan"))?;
        let digest = row["source_sha256"].as_str().unwrap_or("");
        let ceiling = snapshot.profile().capabilities.max_input_tokens;
        let policy = policy(snapshot);
        ensure(
            row["format"] == format(policy),
            "stored document format disagrees with chunk policy",
        )?;
        match &plan {
            Self::Conventional(plan) => {
                plan.validate_metadata(digest, binding, policy, ceiling)
                    .map_err(|_| invalid("stored passage plan changes identity or token policy"))?;
                ensure(
                    row.get("late_token_ranges").is_none(),
                    "conventional source has contextual provenance",
                )?;
            }
            Self::Late(plan) => {
                plan.validate_metadata(digest, binding, policy, ceiling)
                    .map_err(|_| {
                        invalid("stored contextual plan changes identity or token policy")
                    })?;
                let ranges: Vec<LateTokenRange> =
                    serde_json::from_value(row["late_token_ranges"].clone())
                        .map_err(|_| invalid("missing or malformed contextual token ranges"))?;
                ensure(
                    ranges.len() == plan.passages.len(),
                    "incomplete contextual token ranges",
                )?;
                for (index, range) in ranges.iter().enumerate() {
                    let window = &plan.windows[plan.assignments[index]];
                    ensure(
                        range.token_start < range.token_end
                            && range.token_end <= window.input_tokens,
                        "contextual token range exceeds its encoding window",
                    )?;
                }
                // Check every span even if it belongs to an unranked chunk.
                coverage(
                    &plan.passages,
                    plan.passage_tokens,
                    plan.overlap_tokens,
                    plan.max_input_tokens,
                )?;
                coverage(
                    &plan.windows,
                    plan.max_input_tokens,
                    (plan.max_input_tokens - 1).min(
                        plan.minimum_window_overlap_tokens
                            .saturating_add(plan.passage_tokens),
                    ),
                    plan.max_input_tokens,
                )?;
                ensure(
                    plan.windows.last().unwrap().end_byte == plan.passages.last().unwrap().end_byte,
                    "context windows and passages cover different source extents",
                )?;
            }
        }
        ensure(
            row["chunk_count"].as_u64() == Some(plan.spans().len() as u64),
            "stored source chunk count disagrees with plan",
        )?;
        Ok(plan)
    }

    pub fn validate(
        &self,
        text: &str,
        binding: &BoundEmbeddingRequest,
        snapshot: &ProfileSnapshot,
    ) -> Result<(), ArangoError> {
        let policy = policy(snapshot);
        let ceiling = snapshot.profile().capabilities.max_input_tokens;
        match self {
            Self::Conventional(plan) => plan.validate(text, binding, policy, ceiling),
            Self::Late(plan) => plan.validate(text, binding, policy, ceiling),
        }
        .map_err(|_| invalid("stored passage plan violates source identity or token policy"))
    }

    pub fn check_context(
        &self,
        document: &Value,
        chunk: &Value,
        vector: &Value,
        index: usize,
    ) -> Result<(), ArangoError> {
        match self {
            Self::Conventional(_) => ensure(
                chunk.get("late_token_range").is_none() && vector.get("late_token_range").is_none(),
                "conventional rows have contextual provenance",
            ),
            Self::Late(_) => ensure(
                chunk["late_token_range"] == document["late_token_ranges"][index]
                    && vector["late_token_range"] == document["late_token_ranges"][index],
                "stored contextual source, chunk and vector token ranges disagree",
            ),
        }
    }

    pub fn context(&self, document: &Value, index: usize) -> Option<Value> {
        match self {
            Self::Conventional(_) => None,
            Self::Late(plan) => Some(json!({"window_index":plan.assignments[index],
                "window":plan.windows[plan.assignments[index]], "token_range":document["late_token_ranges"][index]})),
        }
    }
}

/// Check bounded forward coverage before scanning vectors without fetching source text.
fn coverage(
    spans: &[TokenPassage],
    body: u32,
    overlap: u32,
    ceiling: u32,
) -> Result<(), ArangoError> {
    let mut previous: Option<&TokenPassage> = None;
    let mut total = 0;
    for span in spans {
        ensure(
            span.start_byte < span.end_byte
                && span.end_byte <= MAX_TEXT_BYTES
                && span.body_tokens <= body
                && span.input_tokens > 0
                && span.input_tokens <= ceiling
                && span.overlap_tokens <= overlap,
            "stored contextual span violates token policy",
        )?;
        ensure(
            match previous {
                None => span.start_byte == 0 && span.overlap_tokens == 0,
                Some(last) => {
                    span.start_byte > last.start_byte
                        && span.start_byte <= last.end_byte
                        && span.end_byte > last.end_byte
                        && (span.start_byte != last.end_byte || span.overlap_tokens == 0)
                        && (overlap != 0 || span.start_byte == last.end_byte)
                }
            },
            "stored contextual spans have gaps or do not progress",
        )?;
        total += span.end_byte - span.start_byte;
        ensure(
            total <= 4 * 1024 * 1024,
            "stored contextual spans exceed byte budget",
        )?;
        previous = Some(span);
    }
    Ok(())
}
