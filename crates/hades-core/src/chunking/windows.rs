//! Full context windows with one selected context per stored passage.
//!
//! Budgets are UTF-8 bytes estimated from the backend token ceiling. They are
//! not tokenizer counts: only an explicit size refusal permits shrinking them.
use super::{TextChunk, prechunk::PrechunkWindow};
use crate::persephone::embedding::{EmbedResult, EmbeddingClient, EmbeddingError};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextWindow {
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowPlan {
    pub version: u32,
    pub backend_ceiling_tokens: u32,
    pub usable_window_tokens: u32,
    pub initial_window_bytes: usize,
    pub effective_window_bytes: usize,
    pub minimum_overlap_bytes: usize,
    pub size_refinements: usize,
    pub windows: Vec<ContextWindow>,
    /// Indexed by final file-order chunk_index, not by input or request order.
    pub chunk_windows: Vec<usize>,
}

impl WindowPlan {
    pub fn summary(&self) -> Value {
        let strides: Vec<_> = self
            .windows
            .windows(2)
            .map(|w| w[1].start_byte - w[0].start_byte)
            .collect();
        let overlaps: Vec<_> = self
            .windows
            .windows(2)
            .map(|w| w[0].end_byte.saturating_sub(w[1].start_byte))
            .collect();
        json!({"version":self.version,"budget_basis":"utf8_bytes_estimated_from_tokens",
            "backend_ceiling_tokens":self.backend_ceiling_tokens,
            "usable_window_tokens":self.usable_window_tokens,
            "initial_window_bytes":self.initial_window_bytes,
            "effective_window_bytes":self.effective_window_bytes,
            "minimum_overlap_bytes":self.minimum_overlap_bytes,
            "size_refinements":self.size_refinements,"window_count":self.windows.len(),
            "stride_bytes_min":strides.iter().min(),"stride_bytes_max":strides.iter().max(),
            "overlap_bytes_min":overlaps.iter().min(),"overlap_bytes_max":overlaps.iter().max()})
    }

    pub fn chunk_provenance(&self, index: usize, summary: &Value) -> Value {
        let selected = self.chunk_windows[index];
        json!({"plan":summary,"selected_window":selected,"window":self.windows[selected]})
    }
}

fn floor(text: &str, mut i: usize) -> usize {
    while !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}
fn ceil(text: &str, mut i: usize) -> usize {
    while !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Fewest evenly spread full windows for the budget and minimum overlap.
/// A Unicode window may be up to three bytes shorter to avoid cutting a scalar.
fn ranges(
    source: &str,
    start: usize,
    end: usize,
    budget: usize,
    overlap: usize,
) -> anyhow::Result<Vec<ContextWindow>> {
    anyhow::ensure!(
        budget > 0 && overlap < budget,
        "window overlap must be below a positive budget"
    );
    let text = source
        .get(start..end)
        .ok_or_else(|| anyhow::anyhow!("invalid source window offsets"))?;
    if text.len() <= budget {
        return Ok(vec![ContextWindow {
            start_byte: start,
            end_byte: end,
        }]);
    }
    let slack = text.chars().map(char::len_utf8).max().unwrap_or(1) - 1;
    let width = budget
        .checked_sub(slack)
        .filter(|w| *w > overlap)
        .ok_or_else(|| anyhow::anyhow!("window budget/overlap cannot cover UTF-8 boundaries"))?;
    let distance = text.len() - budget;
    let intervals = (text.len() - width).div_ceil(width - overlap);
    anyhow::ensure!(
        intervals < 100_000,
        "context plan exceeds 100000 windows; reduce overlap"
    );
    let mut windows: Vec<ContextWindow> = Vec::with_capacity(intervals + 1);
    for i in 0..=intervals {
        let relative = (distance as u128 * i as u128 / intervals as u128) as usize;
        let s = if i == intervals {
            ceil(text, relative)
        } else {
            floor(text, relative)
        };
        let e = floor(text, (s + budget).min(text.len()));
        let window = ContextWindow {
            start_byte: start + s,
            end_byte: start + e,
        };
        // Rounding can reduce the requested overlap, especially at the final
        // start. Insert full windows until the minimum is met; never drop text
        // or silently weaken the requested overlap to accommodate UTF-8.
        while let Some(previous) = windows.last()
            && previous.end_byte.saturating_sub(overlap) < window.start_byte
        {
            let bridge_start = floor(source, previous.end_byte.saturating_sub(overlap));
            let bridge_end = floor(source, bridge_start.saturating_add(budget).min(end));
            anyhow::ensure!(
                bridge_start > previous.start_byte && bridge_end > previous.end_byte,
                "window budget/overlap cannot bridge UTF-8 boundaries"
            );
            windows.push(ContextWindow {
                start_byte: bridge_start,
                end_byte: bridge_end,
            });
            anyhow::ensure!(
                windows.len() < 100_000,
                "context plan exceeds 100000 windows"
            );
        }
        if windows.last() != Some(&window) {
            windows.push(window);
        }
        anyhow::ensure!(
            windows.len() < 100_000,
            "context plan exceeds 100000 windows"
        );
    }
    anyhow::ensure!(
        windows.windows(2).all(|w| w[0].end_byte >= w[1].start_byte),
        "window plan has a source gap"
    );
    Ok(windows)
}

fn nearest(windows: &[ContextWindow], start: usize, end: usize) -> Option<usize> {
    let first = windows.partition_point(|w| w.end_byte < end);
    let last = windows.partition_point(|w| w.start_byte <= start);
    if first >= last {
        return None;
    }
    let center = start as u128 + end as u128;
    let candidates = &windows[first..last];
    let pivot =
        candidates.partition_point(|w| w.start_byte as u128 + (w.end_byte as u128) < center);
    [pivot.saturating_sub(1), pivot.min(candidates.len() - 1)]
        .into_iter()
        .min_by_key(|i| {
            (
                (candidates[*i].start_byte as u128 + candidates[*i].end_byte as u128)
                    .abs_diff(center),
                *i,
            )
        })
        .map(|i| first + i)
}

type Prepared = (Vec<TextChunk>, Vec<usize>, Vec<usize>);

/// Keep fitting passages intact. Split only a passage that no context contains,
/// into adjacent, non-overlapping fragments; preserve its original parent index.
fn assign(
    source: &str,
    chunks: &[TextChunk],
    windows: &[ContextWindow],
) -> anyhow::Result<Prepared> {
    let mut output = Vec::new();
    let mut parents = Vec::new();
    let mut selected = Vec::new();
    for chunk in chunks {
        anyhow::ensure!(
            chunk.start_char < chunk.end_char
                && source.get(chunk.start_char..chunk.end_char).is_some(),
            "invalid or empty source offsets in embedding chunk"
        );
        let mut pending = vec![(chunk.start_char, chunk.end_char)];
        while let Some((start, end)) = pending.pop() {
            if let Some(window) = nearest(windows, start, end) {
                let mut part = chunk.clone();
                part.start_char = start;
                part.end_char = end;
                if start != chunk.start_char || end != chunk.end_char {
                    part.text = source[start..end].to_owned();
                }
                part.chunk_index = output.len();
                output.push(part);
                parents.push(chunk.chunk_index);
                selected.push(window);
            } else {
                let middle = ceil(source, start + (end - start) / 2);
                anyhow::ensure!(
                    start < middle && middle < end,
                    "context window cannot hold one UTF-8 character"
                );
                pending.push((middle, end));
                pending.push((start, middle));
            }
        }
    }
    let total = output.len();
    for chunk in &mut output {
        chunk.total_chunks = total;
    }
    Ok((output, parents, selected))
}

pub(super) async fn embed(
    client: &EmbeddingClient,
    source: &str,
    input: Vec<TextChunk>,
    policy: PrechunkWindow,
    task: &str,
) -> anyhow::Result<(Vec<TextChunk>, Vec<usize>, EmbedResult, Option<WindowPlan>)> {
    anyhow::ensure!(!input.is_empty(), "late chunking requires passages");
    let start = input.iter().map(|c| c.start_char).min().unwrap();
    let end = input.iter().map(|c| c.end_char).max().unwrap();
    let mut budget = policy.bytes;
    let mut overlap = policy.overlap_bytes;
    for refinement in 0..32 {
        let windows = ranges(source, start, end, budget, overlap)?;
        let (chunks, parents, selected) = assign(source, &input, &windows)?;
        let mut embeddings: Vec<Option<Vec<f32>>> = vec![None; chunks.len()];
        let mut model: Option<String> = None;
        let mut dimension = 0;
        let mut duration = 0;
        let mut refused = None;
        let mut assigned = vec![Vec::new(); windows.len()];
        for (chunk, window) in selected.iter().enumerate() {
            assigned[*window].push(chunk);
        }
        for (window, indices) in windows.iter().zip(assigned) {
            if indices.is_empty() {
                continue;
            }
            let text = &source[window.start_byte..window.end_byte];
            let offsets: Vec<_> = text.char_indices().map(|(p, _)| p).collect();
            let position = |byte: usize| offsets.partition_point(|p| *p < byte - window.start_byte);
            let bounds: Vec<_> = indices
                .iter()
                .map(|c| {
                    (
                        position(chunks[*c].start_char),
                        position(chunks[*c].end_char),
                    )
                })
                .collect();
            let texts = [text.to_owned()];
            let boundaries = [bounds];
            let mut response = client.embed_late_chunked(&texts, task, &boundaries).await;
            // Keep one exact retry for a transport/temporary service failure.
            // Invalid vectors or identities never become a size-recovery signal.
            if matches!(
                &response,
                Err(EmbeddingError::Http {
                    status: 500..=599,
                    ..
                } | EmbeddingError::Connection(_)
                    | EmbeddingError::Timeout(_))
            ) {
                response = client.embed_late_chunked(&texts, task, &boundaries).await;
            }
            match response {
                Ok(result) => {
                    anyhow::ensure!(
                        model.as_ref().is_none_or(|m| m == &result.model)
                            && (dimension == 0 || dimension == result.dimension),
                        "model identity or dimension changed across context windows"
                    );
                    model = Some(result.model);
                    dimension = result.dimension;
                    duration += result.duration_ms;
                    for (index, vector) in indices
                        .into_iter()
                        .zip(result.per_input.into_iter().flatten())
                    {
                        embeddings[index] = Some(vector.embedding);
                    }
                }
                Err(error) => {
                    if let Some(tokens) = super::prechunk::reported_tokens(&error) {
                        let next = ((text.len() as u128 * policy.tokens as u128 / tokens as u128)
                            as usize)
                            .min(budget.saturating_sub(1))
                            .min(text.len().saturating_sub(1));
                        anyhow::ensure!(next > 0, "size recovery reached its UTF-8 floor");
                        refused = Some(next);
                        break;
                    }
                    return Err(error.into());
                }
            }
        }
        if let Some(next) = refused {
            overlap = ((overlap as u128 * next as u128 / budget as u128) as usize).min(next - 1);
            budget = next;
            continue; // discard every provisional vector and plan; nothing was written
        }
        let plan = WindowPlan {
            version: 1,
            backend_ceiling_tokens: policy.ceiling_tokens,
            usable_window_tokens: policy.tokens,
            initial_window_bytes: policy.bytes,
            effective_window_bytes: budget,
            minimum_overlap_bytes: overlap,
            size_refinements: refinement,
            windows,
            chunk_windows: selected,
        };
        let result = EmbedResult {
            embeddings: embeddings
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow::anyhow!("context window left a passage without a vector"))?,
            model: model.unwrap(),
            dimension,
            duration_ms: duration,
        };
        return Ok((chunks, parents, result, Some(plan)));
    }
    anyhow::bail!("context window size recovery exceeded 32 attempts")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn chunk(source: &str, start: usize, end: usize, index: usize) -> TextChunk {
        TextChunk {
            text: source[start..end].into(),
            start_char: start,
            end_char: end,
            chunk_index: index,
            total_chunks: 1,
        }
    }

    #[test]
    fn fitting_and_just_over_budget_have_no_short_tail() {
        let source = "x".repeat(101);
        assert_eq!(
            ranges(&source, 0, 99, 100, 20).unwrap(),
            vec![ContextWindow {
                start_byte: 0,
                end_byte: 99
            }]
        );
        assert_eq!(
            ranges(&source, 0, 100, 100, 20).unwrap(),
            vec![ContextWindow {
                start_byte: 0,
                end_byte: 100
            }]
        );
        assert_eq!(
            ranges(&source, 0, 101, 100, 20).unwrap(),
            vec![
                ContextWindow {
                    start_byte: 0,
                    end_byte: 100
                },
                ContextWindow {
                    start_byte: 1,
                    end_byte: 101
                }
            ]
        );
        let input = vec![chunk(&source, 0, 30, 7), chunk(&source, 40, 99, 42)];
        let (output, parents, selected) =
            assign(&source, &input, &ranges(&source, 0, 99, 100, 20).unwrap()).unwrap();
        assert_eq!(parents, [7, 42]);
        assert_eq!(selected, [0, 0]);
        for (a, b) in input.iter().zip(output) {
            assert_eq!(
                (a.start_char, a.end_char, &a.text),
                (b.start_char, b.end_char, &b.text)
            );
        }
    }

    #[test]
    fn multiple_ceilings_and_overlap_produce_distinct_even_plans() {
        // Abstract budget units; not a measurement of this text's tokenizer count.
        let source = "x".repeat(40000);
        for (budget, overlap, count) in [(32512, 1000, 2), (11644, 1000, 4), (11644, 6000, 7)] {
            let plan = ranges(&source, 0, source.len(), budget, overlap).unwrap();
            assert_eq!(plan.len(), count);
            assert!(plan.iter().all(|w| w.end_byte - w.start_byte == budget));
            assert_eq!(plan[0].start_byte, 0);
            assert_eq!(plan.last().unwrap().end_byte, source.len());
            let strides: Vec<_> = plan
                .windows(2)
                .map(|w| w[1].start_byte - w[0].start_byte)
                .collect();
            assert!(strides.iter().max().unwrap() - strides.iter().min().unwrap() <= 1);
            assert!(
                plan.windows(2)
                    .all(|w| w[0].end_byte - w[1].start_byte >= overlap)
            );
        }
    }

    #[test]
    fn nearest_containing_context_has_deterministic_ties() {
        let windows = vec![
            ContextWindow {
                start_byte: 0,
                end_byte: 100,
            },
            ContextWindow {
                start_byte: 20,
                end_byte: 120,
            },
        ];
        assert_eq!(nearest(&windows, 55, 65), Some(0)); // exactly between centers
        assert_eq!(nearest(&windows, 61, 69), Some(1));
        assert_eq!(nearest(&windows, 0, 5), Some(0));
        assert_eq!(nearest(&windows, 95, 115), Some(1));
        assert_eq!(nearest(&windows, 0, 120), None);
    }

    #[test]
    fn oversized_passages_split_without_duplicate_source_ranges() {
        for source in ["a".repeat(117), "é🦀\r\n".repeat(25)] {
            let windows = ranges(&source, 0, source.len(), 40, 10).unwrap();
            let input = vec![chunk(&source, 0, source.len(), 42)];
            let (chunks, parents, selected) = assign(&source, &input, &windows).unwrap();
            assert_eq!(
                chunks.iter().map(|c| c.text.as_str()).collect::<String>(),
                source
            );
            assert!(chunks.windows(2).all(|c| c[0].end_char == c[1].start_char));
            for (i, c) in chunks.iter().enumerate() {
                assert_eq!(c.chunk_index, i);
                assert_eq!(c.total_chunks, chunks.len());
                assert_eq!(parents[i], 42);
                assert_eq!(
                    Some(selected[i]),
                    nearest(&windows, c.start_char, c.end_char)
                );
                assert_eq!(source.get(c.start_char..c.end_char), Some(c.text.as_str()));
            }
        }
    }

    #[test]
    fn unicode_windows_never_drop_source_scalars() {
        for source in ["xé🦀\r\n".repeat(20), "🦀".repeat(20), "x".repeat(80)] {
            for budget in 8usize..40 {
                for overlap in 0..budget.saturating_sub(7) {
                    let windows = ranges(&source, 0, source.len(), budget, overlap).unwrap();
                    assert_eq!(windows[0].start_byte, 0);
                    assert_eq!(windows.last().unwrap().end_byte, source.len());
                    assert!(
                        windows
                            .windows(2)
                            .all(|w| w[0].end_byte >= w[1].start_byte + overlap)
                    );
                    for window in windows {
                        let text = &source[window.start_byte..window.end_byte];
                        assert!(
                            !text.is_empty() && text.len() <= budget && text.len() + 3 >= budget
                        );
                    }
                }
            }
        }
        assert!(ranges("xé🦀\r\nxé🦀\r\n", 0, 18, 11, 6).is_err());
        assert!(ranges("🦀", 0, 4, 2, 0).is_err());
        assert!(ranges("hello", 0, 5, 2, 2).is_err());
        assert!(ranges("é", 1, 2, 5, 0).is_err());
    }
}
