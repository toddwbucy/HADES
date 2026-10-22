//! Receipt-bound contextual planning and embedding; never fall back to ordinary passages.
use super::*;
use crate::persephone::late_passage_plan::LatePassagePlan;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Observed tokenizer coordinates within the selected prompted encoding window.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LateTokenRange {
    pub token_start: u32,
    pub token_end: u32,
}

/// Complete vectors and token ranges in original passage order.
pub struct LateDocumentEmbeddings {
    pub embeddings: Vec<Vec<f32>>,
    pub token_ranges: Vec<LateTokenRange>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WindowResponse {
    format: String,
    model: String,
    source_sha256: String,
    receipt: ProviderReceipt,
    input_tokens: u32,
    data: Vec<WindowRow>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WindowRow {
    index: usize,
    embedding: Vec<f32>,
    char_start: usize,
    char_end: usize,
    token_start: u32,
    token_end: u32,
}

impl BoundEmbeddingClient {
    /// Obtain counted source passages, context windows and assignments from the pinned tokenizer.
    pub async fn plan_late_passages(
        &self,
        text: &str,
    ) -> Result<BoundEmbeddingResult<LatePassagePlan>, EmbeddingError> {
        validate_source(text)?;
        require(
            self.role == EmbeddingRole::Passage,
            "only passage roles can plan late documents",
        )?;
        let ChunkPolicy::Late {
            context_cap_tokens,
            minimum_window_overlap_tokens,
            passages: Some(passages),
        } = self.chunking
        else {
            return Err(EmbeddingError::InvalidResponse(
                "late documents require an explicit passage policy".into(),
            ));
        };
        let output: LatePassagePlan = self.inner.request_typed(Method::POST, "/late-passages", Some(&json!({
            "model":self.binding.model, "task":self.binding.task, "input":text,
            "passage_tokens":passages.passage_tokens,"overlap_tokens":passages.overlap_tokens,
            "context_cap_tokens":context_cap_tokens,"minimum_window_overlap_tokens":minimum_window_overlap_tokens,
            "max_input_tokens":self.profile_max_input_tokens,
        }))).await?;
        output.validate(
            text,
            &self.binding,
            &self.chunking,
            self.profile_max_input_tokens,
        )?;
        self.check_plan_ceiling(output.max_input_tokens)?;
        Ok(BoundEmbeddingResult {
            output,
            binding: self.binding.clone(),
        })
    }

    /// Validate the entire plan, encode each used window once, and return only complete results.
    pub async fn embed_planned_late(
        &self,
        text: &str,
        plan: &LatePassagePlan,
    ) -> Result<BoundEmbeddingResult<LateDocumentEmbeddings>, EmbeddingError> {
        require(
            self.role == EmbeddingRole::Passage,
            "only passage roles can embed late documents",
        )?;
        plan.validate(
            text,
            &self.binding,
            &self.chunking,
            self.profile_max_input_tokens,
        )?;
        self.check_plan_ceiling(plan.max_input_tokens)?;
        let mut results = vec![None; plan.passages.len()];
        for (window_index, window) in plan.windows.iter().enumerate() {
            let assigned: Vec<_> = plan
                .assignments
                .iter()
                .enumerate()
                .filter_map(|(index, &selected)| (selected == window_index).then_some(index))
                .collect();
            if assigned.is_empty() {
                continue;
            }
            let input = &text[window.start_byte..window.end_byte];
            let boundaries: Vec<_> = assigned
                .iter()
                .map(|&index| {
                    let passage = &plan.passages[index];
                    (
                        text[window.start_byte..passage.start_byte].chars().count(),
                        text[window.start_byte..passage.end_byte].chars().count(),
                    )
                })
                .collect();
            let output: WindowResponse = self.inner.request_typed(Method::POST, "/late-embeddings", Some(&json!({
                "model":self.binding.model,"task":self.binding.task,"input":input,"boundaries":boundaries,
            }))).await?;
            output.receipt.check_matches(&self.binding.expected)?;
            require(
                output.format == "hades-late-embedding-v1"
                    && output.model == self.binding.model
                    && output.source_sha256
                        == crate::embedding_profile::hex(&Sha256::digest(input.as_bytes()))
                    && output.input_tokens == window.input_tokens
                    && output.data.len() == assigned.len(),
                "late embedding response changes its model, source, token count or passage count",
            )?;
            for (index, row) in output.data.into_iter().enumerate() {
                require(
                    row.index == index
                        && (row.char_start, row.char_end) == boundaries[index]
                        && row.token_start < row.token_end
                        && row.token_end <= output.input_tokens,
                    "late embedding response changes its character or token boundaries",
                )?;
                require(
                    row.embedding.len() == self.binding.dimension as usize
                        && row.embedding.iter().all(|v| v.is_finite()),
                    "late embedding has invalid dimension or nonfinite values",
                )?;
                let norm: f64 = row
                    .embedding
                    .iter()
                    .map(|&v| f64::from(v).powi(2))
                    .sum::<f64>()
                    .sqrt();
                require(
                    norm.is_finite()
                        && norm > 0.0
                        && (self.normalization != Normalization::L2 || (norm - 1.0).abs() <= 0.001),
                    "late embedding violates normalization",
                )?;
                results[assigned[index]] = Some((
                    row.embedding,
                    LateTokenRange {
                        token_start: row.token_start,
                        token_end: row.token_end,
                    },
                ));
            }
        }
        let complete: Vec<_> = results
            .into_iter()
            .map(|row| {
                row.ok_or_else(|| {
                    EmbeddingError::InvalidResponse("late embeddings are incomplete".into())
                })
            })
            .collect::<Result<_, _>>()?;
        let (embeddings, token_ranges) = complete.into_iter().unzip();
        Ok(BoundEmbeddingResult {
            output: LateDocumentEmbeddings {
                embeddings,
                token_ranges,
            },
            binding: self.binding.clone(),
        })
    }
}
