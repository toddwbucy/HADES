//! Backend-sized pre-chunking shared by document and code ingestion.
use super::TextChunk;
use crate::persephone::embedding::{EmbedResult, EmbeddingClient, EmbeddingError};
use serde::{Deserialize, Serialize};

/// The measured floor used by code ingestion, not a tokenizer guarantee.
pub const BYTES_PER_TOKEN_FLOOR: usize = 2;

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
    pub bytes: usize,
    pub overlap_bytes: usize,
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
            bytes: tokens as usize * BYTES_PER_TOKEN_FLOOR,
            overlap_bytes: self.overlap_tokens as usize * BYTES_PER_TOKEN_FLOOR,
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
        let length = chunk.text.len();
        let mut start: usize = 0;
        loop {
            let mut end = start.saturating_add(window).min(length);
            while !chunk.text.is_char_boundary(end) {
                end -= 1;
            }
            anyhow::ensure!(
                end > start || length == 0,
                "prechunk window cannot hold one UTF-8 character"
            );
            parents.push(chunk.chunk_index);
            out.push(TextChunk {
                text: chunk.text[start..end].to_owned(),
                start_char: chunk.start_char + start,
                end_char: if start == 0 && end == length {
                    chunk.end_char
                } else {
                    chunk.start_char + end
                },
                chunk_index: out.len(),
                total_chunks: 0,
            });
            if end == length {
                break;
            }
            let mut next = end.saturating_sub(overlap).max(start + 1);
            while !chunk.text.is_char_boundary(next) {
                next += 1;
            }
            start = next;
        }
    }
    let total = out.len();
    for chunk in &mut out {
        chunk.total_chunks = total;
    }
    Ok((out, parents))
}

/// A packed request keeps its chunks and original parents through adaptive splits.
struct Window {
    text: String,
    chunks: Vec<TextChunk>,
    parents: Vec<usize>,
}

fn split_input(
    source: &str,
    mut chunks: Vec<TextChunk>,
    budget: usize,
    overlap: usize,
    late: bool,
) -> anyhow::Result<(Vec<TextChunk>, Vec<usize>)> {
    if late {
        for c in &mut chunks {
            let text = source
                .get(c.start_char..c.end_char)
                .ok_or_else(|| anyhow::anyhow!("invalid source offsets in embedding window"))?;
            if text.len() > budget {
                c.text = text.to_owned();
            }
        }
    }
    prechunk(chunks, budget, overlap)
}

fn pack(
    source: &str,
    chunks: Vec<TextChunk>,
    parents: Vec<usize>,
    budget: usize,
    late: bool,
) -> anyhow::Result<Vec<Window>> {
    let mut windows: Vec<Window> = Vec::new();
    for (chunk, parent) in chunks.into_iter().zip(parents) {
        let append = late
            && windows
                .last()
                .is_some_and(|w| chunk.end_char.saturating_sub(w.chunks[0].start_char) <= budget);
        if append {
            let w = windows.last_mut().unwrap();
            w.text = source
                .get(w.chunks[0].start_char..chunk.end_char)
                .ok_or_else(|| anyhow::anyhow!("invalid source offsets in embedding window"))?
                .to_owned();
            w.chunks.push(chunk);
            w.parents.push(parent);
        } else {
            windows.push(Window {
                text: if late {
                    source
                        .get(chunk.start_char..chunk.end_char)
                        .ok_or_else(|| {
                            anyhow::anyhow!("invalid source offsets in embedding window")
                        })?
                        .to_owned()
                } else {
                    chunk.text.clone()
                },
                chunks: vec![chunk],
                parents: vec![parent],
            });
        }
    }
    Ok(windows)
}

async fn request(
    client: &EmbeddingClient,
    windows: &[Window],
    task: &str,
    batch: Option<u32>,
    late: bool,
) -> Result<EmbedResult, EmbeddingError> {
    let texts: Vec<_> = windows.iter().map(|w| w.text.clone()).collect();
    if !late {
        return client.embed(&texts, task, batch).await;
    }
    let bounds: Vec<Vec<_>> = windows
        .iter()
        .map(|w| {
            let offsets: Vec<_> = w.text.char_indices().map(|(i, _)| i).collect();
            let count = |position: usize| {
                offsets.partition_point(|i| *i < position - w.chunks[0].start_char)
            };
            w.chunks
                .iter()
                .map(|c| (count(c.start_char), count(c.end_char)))
                .collect()
        })
        .collect();
    let r = client.embed_late_chunked(&texts, task, &bounds).await?;
    Ok(EmbedResult {
        embeddings: r
            .per_input
            .into_iter()
            .flatten()
            .map(|v| v.embedding)
            .collect(),
        model: r.model,
        dimension: r.dimension,
        duration_ms: 0,
    })
}

/// Only the documented size error with a reported token count permits re-splitting.
fn reported_tokens(error: &EmbeddingError) -> Option<usize> {
    let EmbeddingError::Http {
        status: 400,
        message,
    } = error
    else {
        return None;
    };
    let body: serde_json::Value = serde_json::from_str(message).ok()?;
    let detail = if body["error"]["code"] == "PE_INPUT_TOO_LARGE" {
        body["error"]["message"].as_str()?.strip_prefix("Input ")?
    } else {
        body["detail"]
            .as_str()?
            .strip_prefix("PE_INPUT_TOO_LARGE: input ")?
    };
    let (_, detail) = detail
        .split_once(" has ")
        .or_else(|| detail.split_once(" is "))?;
    detail
        .split_once(" tokens")?
        .0
        .parse()
        .ok()
        .filter(|n| *n > 0)
}

/// Shared preparation and the sole size-recovery path. Failed attempts never write.
/// The first byte/token estimate is refined only for the input the backend refused.
#[allow(clippy::too_many_arguments)]
pub async fn embed_chunks(
    client: &EmbeddingClient,
    source: &str,
    chunks: Vec<TextChunk>,
    policy: PrechunkWindow,
    task: &str,
    batch: Option<u32>,
    late: bool,
) -> anyhow::Result<(Vec<TextChunk>, Vec<usize>, EmbedResult)> {
    let (chunks, parents) = split_input(source, chunks, policy.bytes, policy.overlap_bytes, late)?;
    let windows = pack(source, chunks, parents, policy.bytes, late)?;
    let initial = request(client, &windows, task, batch, late).await;
    let mut out = Vec::new();
    let mut parents = Vec::new();
    let mut result = EmbedResult {
        embeddings: Vec::new(),
        model: String::new(),
        dimension: 0,
        duration_ms: 0,
    };
    let mut first_error = None;
    let mut queue: std::collections::VecDeque<_> = windows.into_iter().collect();
    match initial {
        Ok(r) => {
            result = r;
            for w in queue.drain(..) {
                out.extend(w.chunks);
                parents.extend(w.parents);
            }
        }
        Err(e) => {
            if !late && reported_tokens(&e).is_none() {
                return Err(e.into());
            }
            tracing::warn!(error=%e, "embedding batch failed; retrying individual windows");
        }
    }
    while let Some(w) = queue.pop_front() {
        match request(client, std::slice::from_ref(&w), task, batch, late).await {
            Ok(r) => {
                if !result.model.is_empty()
                    && (result.model != r.model || result.dimension != r.dimension)
                {
                    first_error.get_or_insert_with(|| {
                        anyhow::anyhow!(
                            "model identity or dimension changed across recovered windows"
                        )
                    });
                    continue;
                }
                result.model = r.model;
                result.dimension = r.dimension;
                result.embeddings.extend(r.embeddings);
                out.extend(w.chunks);
                parents.extend(w.parents);
            }
            Err(e) => {
                if let Some(tokens) = reported_tokens(&e) {
                    let budget = ((w.text.len() as u128 * policy.tokens as u128 / tokens as u128)
                        as usize)
                        .min(w.text.len().saturating_sub(1));
                    let overlap = ((w.text.len() as u128
                        * (policy.overlap_bytes / BYTES_PER_TOKEN_FLOOR) as u128
                        / tokens as u128) as usize)
                        .min(budget.saturating_sub(1));
                    let split = (|| {
                        let mut chunks = w.chunks.clone();
                        for (i, c) in chunks.iter_mut().enumerate() {
                            c.chunk_index = i;
                        }
                        let (chunks, mapping) = split_input(source, chunks, budget, overlap, late)?;
                        let parents = mapping.into_iter().map(|i| w.parents[i]).collect();
                        let next = pack(source, chunks, parents, budget, late)?;
                        anyhow::ensure!(
                            next.iter().all(|n| n.text.len() < w.text.len()),
                            "size recovery reached its UTF-8 floor"
                        );
                        Ok::<_, anyhow::Error>(next)
                    })();
                    if let Ok(next) = split {
                        for item in next.into_iter().rev() {
                            queue.push_front(item);
                        }
                        continue;
                    }
                }
                first_error.get_or_insert_with(|| e.into());
            }
        }
    }
    if let Some(e) = first_error {
        return Err(e);
    }
    anyhow::ensure!(
        out.len() == result.embeddings.len(),
        "embedding count mismatch after recovery"
    );
    let total = out.len();
    for (i, c) in out.iter_mut().enumerate() {
        c.chunk_index = i;
        c.total_chunks = total;
    }
    Ok((out, parents, result))
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
    fn packer_uses_the_same_byte_budget_and_source_boundaries() {
        for source in ["é".repeat(9), "a\r\nb\r\nc\r\n".into()] {
            let mut input = chunk(source.replace("\r\n", "\n"), 42);
            input.end_char = source.len();
            let (chunks, parents) = split_input(&source, vec![input], 5, 2, true).unwrap();
            let windows = pack(&source, chunks, parents, 5, true).unwrap();
            assert!(windows.iter().all(|w| w.text.len() <= 5));
            for w in windows {
                assert!(w.parents.iter().all(|p| *p == 42));
                for c in w.chunks {
                    assert_eq!(source.get(c.start_char..c.end_char), Some(c.text.as_str()));
                }
            }
        }
        let mut bad = chunk("x".into(), 3);
        bad.start_char = 2;
        bad.end_char = 4;
        assert!(split_input("a—b", vec![bad], 5, 2, true).is_err());
    }

    #[test]
    fn size_recovery_requires_the_reported_count_and_error_contract() {
        for message in [
            "{}",
            r#"{"detail":"PE_INPUT_TOO_LARGE: input 0 is null tokens"}"#,
            r#"{"detail":"PE_INPUT_TOO_LARGE: input 0 is 0 tokens"}"#,
            r#"{"detail":"PE_INFERENCE_FAILED: input 0 is 20 tokens"}"#,
        ] {
            assert_eq!(
                reported_tokens(&EmbeddingError::Http {
                    status: 400,
                    message: message.into()
                }),
                None
            );
        }
        let message = r#"{"detail":"PE_INPUT_TOO_LARGE: input 0 is 20 tokens"}"#;
        assert_eq!(
            reported_tokens(&EmbeddingError::Http {
                status: 400,
                message: message.into()
            }),
            Some(20)
        );
        assert_eq!(
            reported_tokens(&EmbeddingError::Http {
                status: 500,
                message: message.into()
            }),
            None
        );
    }

    #[test]
    fn refusal_sentence_forms_and_saturation_are_compatible() {
        for (body, expected) in [
            (
                serde_json::json!({"detail":"PE_INPUT_TOO_LARGE: input 0 is 13173 tokens, which exceeds this profile's 11892-token ceiling."}),
                Some(13173),
            ),
            (
                serde_json::json!({"error":{"code":"PE_INPUT_TOO_LARGE","message":"Input 0 has 13173 tokens; ceiling is 11892"}}),
                Some(13173),
            ),
            (
                serde_json::json!({"detail":"PE_INPUT_TOO_LARGE: input filled the 11892-token window and was truncated at character 40 of 50"}),
                None,
            ),
            (
                serde_json::json!({"error":{"code":"PE_INPUT_TOO_LARGE","message":"Input exceeds the model context ceiling"}}),
                None,
            ),
            (
                serde_json::json!({"error":{"code":"PE_INVALID_INPUT","message":"Input 0 has 13173 tokens; ceiling is 11892"}}),
                None,
            ),
            (
                serde_json::json!({"error":{"code":"PE_INPUT_TOO_LARGE","message":"Input 0 has 0 tokens; ceiling is 11892"}}),
                None,
            ),
        ] {
            assert_eq!(
                reported_tokens(&EmbeddingError::Http {
                    status: 400,
                    message: body.to_string()
                }),
                expected,
                "{body}"
            );
        }
    }

    #[test]
    fn policy_yaml_and_invalid_budgets() {
        let config: crate::config::HadesConfig = serde_yaml::from_str(
            "chunking:\n  prechunk:\n    overlap_tokens: 500\n    safety_margin_tokens: 256\n    max_window_tokens: 5946\n").unwrap();
        let window = config.chunking.prechunk.resolve(11892).unwrap();
        assert_eq!(
            (window.tokens, window.bytes, window.overlap_bytes),
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
            prechunk(vec![chunk("x".repeat(30000), 7)], w.bytes, w.overlap_bytes).unwrap();
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
    fn unicode_windows_use_the_packers_byte_budget() {
        let (out, _) = prechunk(vec![chunk("é".repeat(9), 0)], 5, 2).unwrap();
        assert_eq!(
            out.iter()
                .map(|c| c.text.chars().count())
                .collect::<Vec<_>>(),
            [2, 2, 2, 2, 2, 2, 2, 2]
        );
        assert_eq!((out[1].start_char, out[1].end_char), (2, 6));
    }
}
