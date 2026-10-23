//! Independently checked source coverage and policy for verified contextual passages.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    embedding::EmbeddingError,
    passage_plan::{MAX_PASSAGES, PassagePlan, TokenPassage, validate_source},
};
use crate::embedding_profile::{BoundEmbeddingRequest, ChunkPolicy, ProviderReceipt, require};

/// Original byte spans; assignment indices select complete, containing encoding windows.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatePassagePlan {
    pub format: String,
    pub source_sha256: String,
    pub model: String,
    pub task: String,
    pub receipt: ProviderReceipt,
    pub passage_tokens: u32,
    pub overlap_tokens: u32,
    pub context_cap_tokens: u32,
    pub minimum_window_overlap_tokens: u32,
    pub max_input_tokens: u32,
    pub passages: Vec<TokenPassage>,
    pub windows: Vec<TokenPassage>,
    pub assignments: Vec<usize>,
}

impl LatePassagePlan {
    /// Validate persisted metadata before loading source bytes or scoring vectors.
    pub(crate) fn validate_metadata(
        &self,
        source_sha256: &str,
        binding: &BoundEmbeddingRequest,
        policy: &ChunkPolicy,
        profile_ceiling: u32,
    ) -> Result<(), EmbeddingError> {
        let ChunkPolicy::Late {
            context_cap_tokens,
            minimum_window_overlap_tokens,
            passages: Some(passages),
        } = policy
        else {
            return Err(EmbeddingError::InvalidResponse(
                "late documents require an explicit passage policy".into(),
            ));
        };
        require(
            self.format == "hades-late-passages-v1"
                && self.source_sha256 == source_sha256
                && self.model == binding.model
                && self.task == binding.task,
            "late plan source/model/task identity differs",
        )?;
        self.receipt.check_matches(&binding.expected)?;
        require(
            self.passage_tokens == passages.passage_tokens
                && self.overlap_tokens == passages.overlap_tokens
                && self.context_cap_tokens == *context_cap_tokens
                && self.minimum_window_overlap_tokens == *minimum_window_overlap_tokens
                && self.max_input_tokens > self.minimum_window_overlap_tokens
                && self.max_input_tokens <= self.context_cap_tokens
                && self.context_cap_tokens <= profile_ceiling,
            "late plan changes the admitted token policy",
        )?;
        require(
            !self.passages.is_empty()
                && self.passages.len() <= MAX_PASSAGES
                && !self.windows.is_empty()
                && self.windows.len() <= MAX_PASSAGES
                && self.assignments.len() == self.passages.len(),
            "late plan requires bounded complete passage/window assignments",
        )?;
        for (passage, &index) in self.passages.iter().zip(&self.assignments) {
            let window = self.windows.get(index).ok_or_else(|| {
                EmbeddingError::InvalidResponse("unknown late window assignment".into())
            })?;
            require(
                window.start_byte <= passage.start_byte && passage.end_byte <= window.end_byte,
                "encoding window does not contain its assigned passage",
            )?;
        }
        require(
            self.windows
                .iter()
                .skip(1)
                .all(|w| w.overlap_tokens >= self.minimum_window_overlap_tokens),
            "late windows violate the minimum token overlap",
        )?;
        Ok(())
    }

    /// Check exact UTF-8 coverage, counted budgets and overlap before sending any inference.
    pub(crate) fn validate(
        &self,
        text: &str,
        binding: &BoundEmbeddingRequest,
        policy: &ChunkPolicy,
        profile_ceiling: u32,
    ) -> Result<(), EmbeddingError> {
        validate_source(text)?;
        let digest = crate::embedding_profile::hex(&Sha256::digest(text.as_bytes()));
        self.validate_metadata(&digest, binding, policy, profile_ceiling)?;
        let upper_overlap = (self.max_input_tokens - 1).min(
            self.minimum_window_overlap_tokens
                .saturating_add(self.passage_tokens),
        );
        for (spans, tokens, overlap) in [
            (&self.passages, self.passage_tokens, self.overlap_tokens),
            (&self.windows, self.max_input_tokens, upper_overlap),
        ] {
            PassagePlan {
                format: "hades-token-passages-v1".into(),
                source_sha256: self.source_sha256.clone(),
                model: self.model.clone(),
                task: self.task.clone(),
                receipt: self.receipt.clone(),
                passage_tokens: tokens,
                overlap_tokens: overlap,
                max_input_tokens: self.max_input_tokens,
                spans: spans.clone(),
            }
            .validate(
                text,
                binding,
                &ChunkPolicy::Conventional {
                    passage_tokens: tokens,
                    overlap_tokens: overlap,
                },
                profile_ceiling,
            )?;
        }
        Ok(())
    }
}
