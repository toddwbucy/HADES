//! Retain existing endpoints; source locations are attributes, not replacement identities.
use std::collections::{BTreeMap, HashSet};

use serde::Deserialize;

use super::*;

#[derive(Deserialize)]
struct StoredSymbol {
    #[serde(rename = "_key")]
    key: String,
    qualified_name: String,
    lang_kind: String,
    start_line: usize,
    end_line: usize,
}

/// Retain unique qualified-name/kind/compiler-identity matches. Repeated names
/// without distinct compiler identities must not be paired by body text or order.
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
    let mut before = BTreeMap::<_, Vec<usize>>::new();
    for (index, (symbol, row)) in previous.iter().zip(previous_rows).enumerate() {
        before
            .entry((
                symbol.qualified_name.clone(),
                symbol.lang_kind.clone(),
                compiler_identity(&row["metadata"])?,
            ))
            .or_default()
            .push(index);
    }
    let mut after = BTreeMap::<_, Vec<usize>>::new();
    for (index, symbol) in input.analysis.symbols.iter().enumerate() {
        if symbol.kind.is_primitive() {
            after
                .entry((
                    symbol.qualified_name(),
                    symbol.kind.lang_kind().to_owned(),
                    compiler_identity(&symbol.metadata)?,
                ))
                .or_default()
                .push(index);
        }
    }
    let mut result = provisional.to_vec();
    for (identity, old) in before {
        let new = after.get(&identity).ok_or_else(|| {
            invalid("replacement would retire or rename symbol endpoints; explicit graph reconciliation is required")
        })?;
        let mut retain = |old: usize, new: usize| {
            result[new] = Some(format!("{}/{}", CODEBASE.symbols, previous[old].key));
        };
        if old.len() == 1 && new.len() == 1 {
            retain(old[0], new[0]);
            continue;
        }
        // Even distinct source text cannot establish which repeated definition
        // moved: two owners might instead have swapped their method bodies.
        // Unchanged source can retain exact positions without such a guess.
        ensure(
            previous_source == input.text && old.len() == new.len(),
            "ambiguous symbol identity; explicit graph reconciliation is required",
        )?;
        let locations: BTreeMap<_, _> = new
            .iter()
            .map(|&index| {
                let symbol = &input.analysis.symbols[index];
                ((symbol.start_line, symbol.end_line), index)
            })
            .collect();
        for index in old {
            let symbol = &previous[index];
            let &new = locations
                .get(&(symbol.start_line, symbol.end_line))
                .ok_or_else(ambiguous)?;
            retain(index, new);
        }
    }
    // A new declaration can occupy a retained endpoint's historical line.
    // Refuse the collision rather than assigning two definitions the same ID.
    let mut seen = HashSet::new();
    ensure(
        result.iter().flatten().all(|id| seen.insert(id)),
        "retained symbol endpoint collides with a new symbol; explicit graph reconciliation is required",
    )?;
    Ok(result)
}

fn ambiguous() -> ArangoError {
    invalid("ambiguous symbol identity; explicit graph reconciliation is required")
}

/// The compiler-backed analyzer already records USRs in symbol metadata. Treat
/// them as opaque identities, never infer one from a signature or source text.
pub(super) fn compiler_identity(metadata: &Value) -> Result<Option<&str>, ArangoError> {
    let Some(value) = metadata.get("usr").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    value
        .as_str()
        .filter(|id| !id.trim().is_empty() && id.len() <= 4096 && !id.chars().any(char::is_control))
        .map(Some)
        .ok_or_else(|| {
            invalid("compiler symbol identity must contain 1..=4096 printable UTF-8 bytes")
        })
}
