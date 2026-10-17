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

/// Count leading Markdown indentation, with tabs advancing to four-column stops.
/// Return its byte length and display width without changing source coordinates.
fn indentation(text: &str, start_column: usize) -> (usize, usize) {
    let mut column = start_column;
    let mut bytes = 0;
    for byte in text.bytes() {
        match byte {
            b' ' => column += 1,
            b'\t' => column += 4 - column % 4,
            _ => break,
        }
        bytes += 1;
    }
    (bytes, column - start_column)
}

fn markdown(bytes: &[u8]) -> Result<Source> {
    let text = String::from_utf8(bytes.to_vec()).map_err(|e| Error::Invalid(e.to_string()))?;
    let mut blocks = Vec::new();
    let mut start = 0;
    // The final field records whether the fence belongs to a blockquote.
    let mut fence: Option<(char, usize, bool)> = None;
    for (index, line) in text.split_inclusive('\n').enumerate() {
        if index >= 100_000 {
            return Err(Error::Invalid("source exceeds 100,000 lines".into()));
        }
        let raw = line.trim_end_matches(['\r', '\n']);
        let (prefix_bytes, prefix_columns) = indentation(raw, 0);
        // An outer fence contains literal quote markers. They cannot close
        // that fence or start a different Markdown container inside it.
        let quoted = !fence.is_some_and(|(_, _, quoted)| !quoted)
            && prefix_columns < 4
            && raw[prefix_bytes..].starts_with('>');
        if fence.is_some_and(|(_, _, quoted)| quoted) && !quoted {
            // A quoted fence ends with its containing blockquote, even when
            // its closing delimiter was omitted.
            fence = None;
        }
        let (body, syntax, indent) = if quoted {
            let content = &raw[prefix_bytes + 1..];
            let (bytes, columns) = indentation(content, prefix_columns + 1);
            // One optional whitespace column belongs to the quote marker.
            // A tab can span that column and leave indentation in the body.
            (
                content.strip_prefix(' ').unwrap_or(content),
                &content[bytes..],
                columns.saturating_sub(1),
            )
        } else {
            (raw, &raw[prefix_bytes..], prefix_columns)
        };
        // Four indentation columns indicate code, including mixed spaces/tabs.
        let fence_body = if indent < 4 { syntax } else { "" };
        let marker = fence_body.chars().next().filter(|c| matches!(c, '`' | '~'));
        let run = marker.map_or(0, |c| fence_body.chars().take_while(|x| *x == c).count());
        let opens_fence = run >= 3 && (marker != Some('`') || !fence_body[run..].contains('`'));
        let literal = fence.is_some() || opens_fence || indent >= 4;
        if let Some((c, n, _)) = fence {
            if marker == Some(c) && run >= n && fence_body[run..].trim().is_empty() {
                fence = None;
            }
        } else if opens_fence {
            fence = Some((marker.unwrap(), run, quoted));
        }
        // ATX headings permit the same zero-to-three-space indentation.
        let hashes = fence_body.chars().take_while(|c| *c == '#').count();
        let heading_level = (!literal
            && !quoted
            && (1..=6).contains(&hashes)
            && matches!(fence_body.as_bytes().get(hashes), Some(b' ' | b'\t')))
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
