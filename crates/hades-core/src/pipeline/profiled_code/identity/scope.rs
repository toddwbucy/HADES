//! Recover lexical ownership from verified primitive spans, without ordinal guesses.
use super::*;

pub(super) fn scopes(
    symbols: &[Candidate<'_>],
    language: crate::code::Language,
) -> Result<Vec<Scope>, ArangoError> {
    // A container's optional compiler metadata must not change its children's
    // scope during a tier upgrade. Its own match still verifies USR and role.
    let fingerprints: Vec<[u8; 32]> = symbols
        .iter()
        .map(|s| {
            let encoded =
                serde_json::to_vec(&(&s.qualified_name, s.kind, s.metadata.implementation))?;
            Ok(Sha256::digest(encoded).into())
        })
        .collect::<Result<_, ArangoError>>()?;
    let chains: Vec<_> = (0..symbols.len())
        .map(|i| containers(i, symbols, language))
        .collect();
    let raw: Vec<Scope> = chains
        .iter()
        .map(|chain| {
            chain.as_ref().map(|chain| {
                let mut hash = Sha256::new();
                hash.update(b"hades:symbol-scope:v1\0");
                for &j in chain {
                    hash.update(fingerprints[j]);
                }
                hash.finalize().into()
            })
        })
        .collect();
    let mut counts = BTreeMap::<_, usize>::new();
    for (i, symbol) in symbols.iter().enumerate() {
        if is_container(symbol) {
            *counts.entry((raw[i], fingerprints[i])).or_default() += 1;
        }
    }
    Ok(chains
        .iter()
        .enumerate()
        .map(|(i, chain)| {
            // Dropping compiler discriminators from scope is safe only when each
            // owner is unique in its lexical chain. Otherwise two compiler-distinct
            // parents could silently exchange a weakly identified child.
            chain
                .as_ref()
                .filter(|chain| {
                    chain.iter().all(|&j| {
                        raw[j].is_some() && counts.get(&(raw[j], fingerprints[j])) == Some(&1)
                    })
                })
                .and(raw[i])
        })
        .collect())
}

fn is_container(symbol: &Candidate<'_>) -> bool {
    matches!(symbol.universal_kind, "module" | "type" | "callable")
}

fn containers(
    i: usize,
    symbols: &[Candidate<'_>],
    language: crate::code::Language,
) -> Option<Vec<usize>> {
    let symbol = &symbols[i];
    let mut containers = Vec::new();
    for (j, parent) in symbols.iter().enumerate() {
        if i == j || !is_container(parent) {
            continue;
        }
        if parent.start <= symbol.start && parent.end >= symbol.end {
            // A brace boundary can share a line with a sibling declaration.
            // Python's indentation-based AST instead ends a container at its
            // last child's body, without a closing delimiter.
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
    Some(containers)
}
