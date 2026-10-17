//! Bounded Markdown syntax reader. Extraction stays in the Python service.

use serde::Serialize;
use serde_json::json;

use super::{Error, Result};

pub const MAX_SOURCE: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, Serialize)]
pub struct Block {
    pub index: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub text: String,
    pub heading_level: Option<u8>,
    pub quoted: bool,
    pub literal: bool,
}

pub struct Source {
    pub text: String,
    pub blocks: Vec<Block>,
    pub format: String,
}

impl Source {
    pub fn location(&self, start: usize, end: usize) -> serde_json::Value {
        json!({"block_start": self.blocks[start].index, "block_end": self.blocks[end].index,
            "start_byte": self.blocks[start].start_byte, "end_byte": self.blocks[end].end_byte,
            "coordinate_system": "zero-based Markdown lines and UTF-8 bytes"})
    }
}

pub fn decode(bytes: &[u8], format: &str) -> Result<Source> {
    if bytes.len() > MAX_SOURCE {
        return Err(Error::Invalid("source exceeds 8 MiB import limit".into()));
    }
    match format {
        "md" | "markdown" => markdown(bytes),
        _ => Err(Error::Invalid("expected graphable .md or .markdown; use the Python extraction workflow for other formats".into())),
    }
}

fn markdown(bytes: &[u8]) -> Result<Source> {
    let text = String::from_utf8(bytes.to_vec()).map_err(|e| Error::Invalid(e.to_string()))?;
    let mut blocks = Vec::new();
    let mut start = 0;
    let mut fence: Option<(char, usize)> = None;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        if index >= 100_000 {
            return Err(Error::Invalid("source exceeds 100,000 lines".into()));
        }
        let raw = line.trim_end_matches(['\r', '\n']);
        let trimmed = raw.trim_start();
        let quoted = trimmed.starts_with('>');
        let body = if quoted {
            trimmed.strip_prefix('>').unwrap().trim_start()
        } else {
            raw
        };
        let marker = body.chars().next().filter(|c| matches!(c, '`' | '~'));
        let run = marker.map_or(0, |c| body.chars().take_while(|x| *x == c).count());
        let literal =
            fence.is_some() || run >= 3 || raw.starts_with("    ") || raw.starts_with('\t');
        if let Some((c, n)) = fence {
            if marker == Some(c) && run >= n && body[run..].trim().is_empty() {
                fence = None;
            }
        } else if run >= 3 {
            fence = Some((marker.unwrap(), run));
        }
        let hashes = body.chars().take_while(|c| *c == '#').count();
        let heading_level = (!literal
            && !quoted
            && (1..=6).contains(&hashes)
            && body.as_bytes().get(hashes) == Some(&b' '))
        .then_some(hashes as u8);
        blocks.push(Block {
            index,
            start_byte: start,
            end_byte: start + raw.len(),
            text: body.to_owned(),
            heading_level,
            quoted,
            literal,
        });
        start += line.len();
    }
    Ok(Source {
        text,
        blocks,
        format: "markdown".into(),
    })
}
