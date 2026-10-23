//! Retain existing endpoints; source locations are attributes, not replacement identities.
mod scope;

use std::collections::{BTreeMap, HashSet};

use serde::Deserialize;

use super::*;

#[derive(Deserialize)]
struct StoredSymbol {
    #[serde(rename = "_key")]
    key: String,
    qualified_name: String,
    lang_kind: String,
    kind: String,
    start_line: usize,
    end_line: usize,
}

struct Candidate<'a> {
    qualified_name: String,
    kind: &'a str,
    universal_kind: &'a str,
    usr: Option<&'a str>,
    definition: Option<bool>,
    start: usize,
    end: usize,
    index: usize,
}

type Scope = Option<[u8; 32]>;
type MatchKey<'a> = (&'a str, &'a str, Option<&'a str>, Option<bool>, Scope);

impl Candidate<'_> {
    fn key(&self, scope: Scope) -> MatchKey<'_> {
        (
            &self.qualified_name,
            self.kind,
            self.usr,
            self.definition,
            // Compiler identity already includes semantic ownership; an
            // out-of-line definition need not have the same lexical container.
            if self.usr.is_some() { None } else { scope },
        )
    }
}

/// Retain verified identities, never pair repeated definitions by body or order.
pub(super) fn reconcile(
    input: &CodeInput,
    previous_source: &str,
    previous_rows: &[Value],
    provisional: &[Option<String>],
) -> Result<Vec<Option<String>>, ArangoError> {
    let previous: Vec<StoredSymbol> = previous_rows
        .iter()
        .map(|row| StoredSymbol::deserialize(row).map_err(ArangoError::from))
        .collect::<Result<_, _>>()?;
    let old: Vec<_> = previous
        .iter()
        .zip(previous_rows)
        .enumerate()
        .map(|(index, (symbol, row))| {
            Ok(Candidate {
                qualified_name: symbol.qualified_name.clone(),
                kind: &symbol.lang_kind,
                universal_kind: &symbol.kind,
                usr: compiler_identity(&row["metadata"])?,
                definition: definition_role(&row["metadata"])?,
                start: symbol.start_line,
                end: symbol.end_line,
                index,
            })
        })
        .collect::<Result<_, ArangoError>>()?;
    let new: Vec<_> = input
        .analysis
        .symbols
        .iter()
        .enumerate()
        .filter(|(_, symbol)| symbol.kind.is_primitive())
        .map(|(index, symbol)| {
            Ok(Candidate {
                qualified_name: symbol.qualified_name(),
                kind: symbol.kind.lang_kind(),
                universal_kind: symbol.kind.universal_kind().unwrap_or_default(),
                usr: compiler_identity(&symbol.metadata)?,
                definition: definition_role(&symbol.metadata)?,
                start: symbol.start_line,
                end: symbol.end_line,
                index,
            })
        })
        .collect::<Result<_, ArangoError>>()?;
    let old_scopes = scope::scopes(&old, input.analysis.language)?;
    let new_scopes = scope::scopes(&new, input.analysis.language)?;
    let before = groups(&old, &old_scopes);
    let after = groups(&new, &new_scopes);
    let mut result = vec![None; provisional.len()];
    let mut retain = |old_index: usize, new_index: usize| {
        let slot = &mut result[new[new_index].index];
        ensure(slot.is_none(), "symbol identity matched more than once")?;
        *slot = Some(format!("{}/{}", CODEBASE.symbols, previous[old_index].key));
        Ok::<_, ArangoError>(())
    };
    for (identity, old_indices) in &before {
        let Some(new_indices) = after.get(identity) else {
            // A tier upgrade may add a USR only when the name/kind/scope is
            // unique on BOTH sides, including all compiler-identified overloads.
            if old_indices.len() == 1 {
                let i = old_indices[0];
                if let Some(j) = upgrade(i, &old, &new, &old_scopes, &new_scopes) {
                    retain(i, j)?;
                    continue;
                }
            }
            return Err(invalid(
                "replacement would retire or rename symbol endpoints; explicit graph reconciliation is required",
            ));
        };
        let i = old_indices[0];
        let j = new_indices[0];
        let uncertain_scope =
            old[i].usr.is_none() && (old_scopes[i].is_none() || new_scopes[j].is_none());
        if old_indices.len() == 1 && new_indices.len() == 1 && !uncertain_scope {
            retain(i, j)?;
            continue;
        }
        // Line-only spans cannot disambiguate repeated or overlapping owners.
        // Exact positions are evidence only when the entire source is unchanged.
        if previous_source != input.text || old_indices.len() != new_indices.len() {
            return Err(ambiguous());
        }
        let locations: BTreeMap<_, _> = new_indices
            .iter()
            .map(|&j| ((new[j].start, new[j].end), j))
            .collect();
        if locations.len() != new_indices.len() {
            return Err(ambiguous());
        }
        for &i in old_indices {
            let &j = locations
                .get(&(old[i].start, old[i].end))
                .ok_or_else(ambiguous)?;
            retain(i, j)?;
        }
    }
    let mut seen: HashSet<_> = result.iter().flatten().cloned().collect();
    let mut reserved = seen.clone();
    reserved.extend(provisional.iter().flatten().cloned());
    for (i, symbol) in new.iter().enumerate() {
        if result[symbol.index].is_some() {
            continue;
        }
        let candidate = provisional[symbol.index]
            .as_ref()
            .ok_or_else(|| invalid("missing provisional symbol endpoint"))?;
        let id = if seen.insert(candidate.clone()) {
            candidate.clone()
        } else {
            let salt = serde_json::to_string(&symbol.key(new_scopes[i]))?;
            (0..=MAX_SYMBOLS)
                .map(|attempt| {
                    format!(
                        "{}/{}",
                        CODEBASE.symbols,
                        keys::disambiguated_symbol_key(candidate, &salt, attempt)
                    )
                })
                .find(|id| reserved.insert(id.clone()))
                .ok_or_else(|| invalid("cannot allocate an unoccupied symbol endpoint"))?
        };
        seen.insert(id.clone());
        result[symbol.index] = Some(id);
    }
    Ok(result)
}

fn groups<'a>(
    symbols: &'a [Candidate<'a>],
    scopes: &[Scope],
) -> BTreeMap<MatchKey<'a>, Vec<usize>> {
    let mut groups = BTreeMap::<_, Vec<usize>>::new();
    for (index, symbol) in symbols.iter().enumerate() {
        groups
            .entry(symbol.key(scopes[index]))
            .or_default()
            .push(index);
    }
    groups
}

fn upgrade(
    i: usize,
    old: &[Candidate<'_>],
    new: &[Candidate<'_>],
    old_scopes: &[Scope],
    new_scopes: &[Scope],
) -> Option<usize> {
    let symbol = &old[i];
    if symbol.usr.is_some() || old_scopes[i].is_none() {
        return None;
    }
    let matches = |other: &Candidate<'_>, scope: Scope| {
        symbol.qualified_name == other.qualified_name
            && symbol.kind == other.kind
            && old_scopes[i] == scope
    };
    if old
        .iter()
        .zip(old_scopes)
        .filter(|(s, scope)| matches(s, **scope))
        .count()
        != 1
    {
        return None;
    }
    let mut candidates = new
        .iter()
        .enumerate()
        .filter(|(j, s)| matches(s, new_scopes[*j]));
    let (j, next) = candidates.next()?;
    (candidates.next().is_none()
        && next.usr.is_some()
        && (symbol.definition.is_none() || symbol.definition == next.definition))
        .then_some(j)
}

fn ambiguous() -> ArangoError {
    invalid("ambiguous symbol identity; explicit graph reconciliation is required")
}

/// Keep opaque USRs intact, bounded by the same budget as the entire analysis.
pub(super) fn compiler_identity(metadata: &Value) -> Result<Option<&str>, ArangoError> {
    let Some(value) = metadata.get("usr").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    value
        .as_str()
        .filter(|id| {
            !id.trim().is_empty()
                && id.len() <= MAX_ANALYSIS_BYTES
                && !id.chars().any(char::is_control)
        })
        .map(Some)
        .ok_or_else(|| {
            invalid("compiler symbol identity must contain 1..=4194304 printable UTF-8 bytes")
        })
}

pub(super) fn definition_role(metadata: &Value) -> Result<Option<bool>, ArangoError> {
    match metadata
        .get("is_definition")
        .filter(|value| !value.is_null())
    {
        None => Ok(None),
        Some(value) => value
            .as_bool()
            .map(Some)
            .ok_or_else(|| invalid("compiler definition role must be a boolean")),
    }
}
