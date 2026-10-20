//! Receipt-bound conventional plans. Counts come from the trusted serving tokenizer;
//! Rust independently checks policy, identity, bounds, and exact source coverage.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::embedding::EmbeddingError;
use crate::embedding_profile::{BoundEmbeddingRequest, ChunkPolicy, ProviderReceipt, require};

pub(crate) const MAX_TEXT_BYTES: usize = 2 * 1024 * 1024;
pub(crate) const MAX_PASSAGES: usize = 128;
const MAX_PASSAGE_BYTES: usize = 4 * 1024 * 1024;

/// Original source bytes, exclusive end; no normalized or token-index coordinates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenPassage {
    pub start_byte: usize,
    pub end_byte: usize,
    pub body_tokens: u32,
    pub input_tokens: u32,
    pub overlap_tokens: u32,
}

/// Complete policy and observed load, retained with source metadata for audit/replacement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PassagePlan {
    pub format: String,
    pub source_sha256: String,
    pub model: String,
    pub task: String,
    pub receipt: ProviderReceipt,
    pub passage_tokens: u32,
    pub overlap_tokens: u32,
    pub max_input_tokens: u32,
    pub spans: Vec<TokenPassage>,
}

/// Bound source size before a request or any substring allocation.
pub(crate) fn validate_source(text: &str) -> Result<(), EmbeddingError> {
    require(
        !text.trim().is_empty() && text.len() <= MAX_TEXT_BYTES,
        "passage planning requires nonblank source text of at most 2 MiB",
    )?;
    Ok(())
}

impl PassagePlan {
    /// Check persisted/wire identity and policy without loading the full source during search.
    pub(crate) fn validate_metadata(
        &self,
        source_sha256: &str,
        binding: &BoundEmbeddingRequest,
        policy: &ChunkPolicy,
        profile_ceiling: u32,
    ) -> Result<(), EmbeddingError> {
        let ChunkPolicy::Conventional {
            passage_tokens,
            overlap_tokens,
        } = policy
        else {
            return Err(EmbeddingError::InvalidResponse(
                "conventional passage plan required".into(),
            ));
        };
        require(
            self.format == "hades-token-passages-v1"
                && self.model == binding.model
                && self.task == binding.task
                && self.source_sha256 == source_sha256,
            "passage plan source/model/task identity differs",
        )?;
        self.receipt.check_matches(&binding.expected)?;
        require(
            self.passage_tokens == *passage_tokens
                && self.overlap_tokens == *overlap_tokens
                && *overlap_tokens < *passage_tokens
                && *passage_tokens <= profile_ceiling
                && self.max_input_tokens > 0
                && self.max_input_tokens <= profile_ceiling,
            "passage plan changes the admitted token policy",
        )?;
        require(
            !self.spans.is_empty() && self.spans.len() <= MAX_PASSAGES,
            "passage plan requires 1..=128 spans",
        )?;
        Ok(())
    }

    /// Refuse malformed counts, gaps, or unsafe UTF-8 spans in the complete source plan.
    /// Token counts are provider observations; this is not independent tokenizer attestation.
    pub(crate) fn validate(
        &self,
        text: &str,
        binding: &BoundEmbeddingRequest,
        policy: &ChunkPolicy,
        profile_ceiling: u32,
    ) -> Result<(), EmbeddingError> {
        validate_source(text)?;
        let digest = Sha256::digest(text.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        self.validate_metadata(&digest, binding, policy, profile_ceiling)?;
        let ChunkPolicy::Conventional {
            passage_tokens,
            overlap_tokens,
        } = policy
        else {
            unreachable!("metadata validation requires conventional policy")
        };
        let mut previous: Option<&TokenPassage> = None;
        let mut total_bytes = 0;
        for span in &self.spans {
            require(
                span.start_byte < span.end_byte
                    && span.end_byte <= text.len()
                    && text.is_char_boundary(span.start_byte)
                    && text.is_char_boundary(span.end_byte),
                "passage plan contains invalid UTF-8 byte offsets",
            )?;
            require(
                match previous {
                    None => span.start_byte == 0 && span.overlap_tokens == 0,
                    Some(last) => {
                        span.start_byte > last.start_byte
                            && span.start_byte <= last.end_byte
                            && span.end_byte > last.end_byte
                            && (span.start_byte != last.end_byte || span.overlap_tokens == 0)
                            && (*overlap_tokens != 0 || span.start_byte == last.end_byte)
                    }
                },
                "passage plan must cover source bytes in order with forward progress",
            )?;
            require(
                span.body_tokens <= *passage_tokens
                    && span.input_tokens > 0
                    && span.input_tokens <= self.max_input_tokens
                    && span.overlap_tokens <= *overlap_tokens,
                "passage plan exceeds token policy",
            )?;
            require(
                !text[span.start_byte..span.end_byte].trim().is_empty(),
                "passage plan contains a blank passage",
            )?;
            total_bytes += span.end_byte - span.start_byte;
            require(
                total_bytes <= MAX_PASSAGE_BYTES,
                "overlapping passages exceed 4 MiB",
            )?;
            previous = Some(span);
        }
        require(
            previous.is_some_and(|last| last.end_byte == text.len()),
            "passage plan omits the end of the source",
        )?;
        Ok(())
    }
}
