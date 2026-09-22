//! Backend-sized pre-chunking shared by document and code ingestion.
use super::TextChunk;
use crate::persephone::embedding::EmbeddingClient;
use serde::{Deserialize, Serialize};

/// The measured floor used by code ingestion, not a tokenizer guarantee.
pub const CHARS_PER_TOKEN_FLOOR: usize = 2;

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct ChunkingPolicy {
    pub prechunk: PrechunkConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct PrechunkConfig {
    pub overlap_tokens: u32,
    pub safety_margin_tokens: u32,
    pub max_window_tokens: Option<u32>,
}

impl Default for PrechunkConfig {
    fn default() -> Self {
        Self {
            overlap_tokens: 1000,
            safety_margin_tokens: 256,
            max_window_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrechunkWindow {
    pub tokens: u32,
    pub chars: usize,
    pub overlap_chars: usize,
}

impl PrechunkConfig {
    pub fn resolve(&self, ceiling: u32) -> anyhow::Result<PrechunkWindow> {
        if let Some(max) = self.max_window_tokens {
            anyhow::ensure!(
                max <= ceiling,
                "chunking.prechunk.max_window_tokens {max} exceeds backend max_seq_length {ceiling}"
            );
        }
        let safe = ceiling
            .checked_sub(self.safety_margin_tokens)
            .filter(|n| *n > 0)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "prechunk safety margin {} must be below backend ceiling {ceiling}",
                    self.safety_margin_tokens
                )
            })?;
        let tokens = self.max_window_tokens.unwrap_or(safe).min(safe);
        anyhow::ensure!(
            self.overlap_tokens < tokens,
            "prechunk overlap {} must be below effective window {tokens}",
            self.overlap_tokens
        );
        Ok(PrechunkWindow {
            tokens,
            chars: tokens as usize * CHARS_PER_TOKEN_FLOOR,
            overlap_chars: self.overlap_tokens as usize * CHARS_PER_TOKEN_FLOOR,
        })
    }

    pub async fn from_backend(&self, client: &EmbeddingClient) -> anyhow::Result<PrechunkWindow> {
        let ceiling =
            client.info().await?.max_seq_length.ok_or_else(|| {
                anyhow::anyhow!("backend must report max_seq_length for prechunking")
            })?;
        self.resolve(ceiling)
    }
}

/// Split oversized chunks; preserve byte offsets, parent indices and global order.
/// Fitting chunks keep their text and offsets unchanged.
pub fn prechunk(
    chunks: Vec<TextChunk>,
    window: usize,
    overlap: usize,
) -> anyhow::Result<(Vec<TextChunk>, Vec<usize>)> {
    anyhow::ensure!(
        window > 0 && overlap < window,
        "prechunk overlap must be below a positive window"
    );
    let mut out = Vec::new();
    let mut parents = Vec::new();
    for chunk in chunks {
        let mut offsets: Vec<usize> = chunk.text.char_indices().map(|(i, _)| i).collect();
        offsets.push(chunk.text.len());
        let length = offsets.len() - 1;
        let mut start: usize = 0;
        loop {
            let end = start.saturating_add(window).min(length);
            parents.push(chunk.chunk_index);
            out.push(TextChunk {
                text: chunk.text[offsets[start]..offsets[end]].to_owned(),
                start_char: chunk.start_char + offsets[start],
                end_char: if start == 0 && end == length {
                    chunk.end_char
                } else {
                    chunk.start_char + offsets[end]
                },
                chunk_index: out.len(),
                total_chunks: 0,
            });
            if end == length {
                break;
            }
            start = end - overlap;
        }
    }
    let total = out.len();
    for chunk in &mut out {
        chunk.total_chunks = total;
    }
    Ok((out, parents))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn chunk(text: String, index: usize) -> TextChunk {
        TextChunk {
            end_char: text.len(),
            text,
            start_char: 0,
            chunk_index: index,
            total_chunks: 1,
        }
    }
    #[test]
    fn policy_yaml_and_invalid_budgets() {
        let config: crate::config::HadesConfig = serde_yaml::from_str(
            "chunking:\n  prechunk:\n    overlap_tokens: 500\n    safety_margin_tokens: 256\n    max_window_tokens: 5946\n").unwrap();
        let window = config.chunking.prechunk.resolve(11892).unwrap();
        assert_eq!(
            (window.tokens, window.chars, window.overlap_chars),
            (5946, 11892, 1000)
        );
        assert!(PrechunkConfig::default().resolve(256).is_err());
        assert!(
            PrechunkConfig {
                overlap_tokens: 11636,
                ..Default::default()
            }
            .resolve(11892)
            .is_err()
        );
        assert!(PrechunkConfig::default().resolve(0).is_err());
    }
    #[test]
    fn document_windows_overlap_and_retain_parent_order() {
        let w = PrechunkConfig::default().resolve(11892).unwrap();
        assert_eq!(w.tokens, 11636);
        let (out, parents) =
            prechunk(vec![chunk("x".repeat(30000), 7)], w.chars, w.overlap_chars).unwrap();
        assert_eq!(parents, [7, 7]);
        assert_eq!((out[0].text.len(), out[1].text.len()), (23272, 8728));
        assert_eq!(out[0].end_char - out[1].start_char, 2000);
        assert_eq!((out[0].chunk_index, out[1].chunk_index), (0, 1));
        assert!(out.iter().all(|c| c.total_chunks == 2));
    }
    #[test]
    fn oversized_override_fails_with_both_numbers() {
        let c = PrechunkConfig {
            max_window_tokens: Some(12000),
            ..Default::default()
        };
        let error = c.resolve(11892).unwrap_err().to_string();
        assert!(error.contains("12000") && error.contains("11892"));
    }
    #[test]
    fn fitting_code_chunks_preserve_boundaries_and_order() {
        let chunks = vec![chunk("fn main() {}".into(), 0), chunk("// next".into(), 1)];
        let mut expected = chunks.clone();
        for c in &mut expected {
            c.total_chunks = 2;
        }
        let (actual, parents) = prechunk(chunks, 100, 20).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(parents, [0, 1]);
    }
    #[test]
    fn unicode_windows_use_characters_with_byte_offsets() {
        let (out, _) = prechunk(vec![chunk("é".repeat(9), 0)], 5, 2).unwrap();
        assert_eq!(
            out.iter()
                .map(|c| c.text.chars().count())
                .collect::<Vec<_>>(),
            [5, 5, 3]
        );
        assert_eq!((out[1].start_char, out[1].end_char), (6, 16));
    }
}
