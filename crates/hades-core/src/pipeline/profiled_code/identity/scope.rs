//! Recover lexical ownership from verified primitive spans, without ordinal guesses.
use super::*;

pub(super) fn scopes(
    symbols: &[Candidate<'_>],
    language: crate::code::Language,
) -> Result<Vec<Scope>, ArangoError> {
    // Hash each container once so long USRs are not copied into every descendant.
    let fingerprints: Vec<[u8; 32]> = symbols
        .iter()
        .map(|s| {
            let encoded = serde_json::to_vec(&(&s.qualified_name, s.kind, s.usr, s.definition))?;
            Ok(Sha256::digest(encoded).into())
        })
        .collect::<Result<_, ArangoError>>()?;
    Ok(symbols
        .iter()
        .enumerate()
        .map(|(i, symbol)| {
            let mut containers = Vec::new();
            for (j, parent) in symbols.iter().enumerate() {
                if i == j || !matches!(parent.universal_kind, "module" | "type" | "callable") {
                    continue;
                }
                if parent.start <= symbol.start && parent.end >= symbol.end {
                    // A declaration sharing either boundary line might be a
                    // sibling before the opening or after the closing brace.
                    // Python's indentation-based AST instead ends a container
                    // at its last child's body, without a closing delimiter.
                    if parent.start == symbol.start
                        || (parent.end == symbol.end && language != crate::code::Language::Python)
                    {
                        return None;
                    }
                    containers.push(j);
                } else if parent.start <= symbol.end
                    && parent.end >= symbol.start
                    && !(symbol.start <= parent.start && symbol.end >= parent.end)
                {
                    return None;
                }
            }
            containers.sort_by_key(|&j| (symbols[j].start, std::cmp::Reverse(symbols[j].end)));
            for pair in containers.windows(2) {
                let (outer, inner) = (&symbols[pair[0]], &symbols[pair[1]]);
                if outer.end < inner.end || (outer.start == inner.start && outer.end == inner.end) {
                    return None;
                }
            }
            let mut hash = Sha256::new();
            hash.update(b"hades:symbol-scope:v1\0");
            for j in containers {
                hash.update(fingerprints[j]);
            }
            Some(hash.finalize().into())
        })
        .collect())
}
