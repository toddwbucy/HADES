//! File-local analyzer output has a closed endpoint vocabulary, never arbitrary graph IDs.
use std::collections::{BTreeMap, HashSet};

use super::*;

#[derive(Debug, Clone, Copy)]
pub enum LocalEndpoint {
    File,
    /// Index into CodeInput.analysis.symbols; must name a stored graph primitive.
    Symbol(usize),
}

#[derive(Debug, Clone, Copy)]
pub enum RelationshipKind {
    Calls,
    Imports,
    Implements,
}

impl RelationshipKind {
    fn collection(self) -> &'static str {
        match self {
            Self::Calls => CODEBASE.calls_edges,
            Self::Imports => CODEBASE.imports_edges,
            Self::Implements => CODEBASE.implements_edges,
        }
    }
}

pub struct LocalRelationship {
    pub kind: RelationshipKind,
    pub from: LocalEndpoint,
    pub to: LocalEndpoint,
}

pub(super) struct ResolvedGraph {
    pub(super) file_key: String,
    symbols: Vec<Option<String>>,
    relationships: Vec<ResolvedRelationship>,
}

impl ResolvedGraph {
    pub(super) fn retain_symbols(
        &mut self,
        input: &CodeInput,
        previous_source: &str,
        previous: &[Value],
    ) -> Result<(), ArangoError> {
        let symbols = identity::reconcile(input, previous_source, previous, &self.symbols)?;
        let remap: BTreeMap<_, _> = self
            .symbols
            .iter()
            .zip(&symbols)
            .filter_map(|(old, new)| old.as_ref().zip(new.as_ref()))
            .collect();
        // Resolve each original endpoint once: swaps and chains must not follow
        // an already rewritten ID into a different symbol.
        for edge in &mut self.relationships {
            if let Some(id) = remap.get(&edge.from) {
                edge.from = (*id).clone();
            }
            if let Some(id) = remap.get(&edge.to) {
                edge.to = (*id).clone();
            }
        }
        self.symbols = symbols;
        Ok(())
    }
}

struct ResolvedRelationship {
    collection: &'static str,
    from: String,
    to: String,
}

pub(super) fn validate_input(
    input: &CodeInput,
    ontology: &ontology::Ontology,
) -> Result<ResolvedGraph, PipelineError> {
    validate_source(&input.text)?;
    for value in [&input.namespace, &input.path] {
        ensure(
            !value.trim().is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control),
            "code namespace/path must contain 1..=4096 printable UTF-8 bytes",
        )?;
    }
    let analysis = &input.analysis;
    ensure(
        analysis.symbols.len() <= MAX_SYMBOLS && input.relationships.len() <= MAX_RELATIONSHIPS,
        "code graph exceeds symbol/relationship limits",
    )?;
    ensure(
        !analysis.analyzer.trim().is_empty() && analysis.analyzer.len() <= 256,
        "code analysis requires a bounded analyzer name",
    )?;
    let lines = input.text.lines().count();
    ensure(
        analysis.metrics.total_lines == lines,
        "analysis line count differs from source",
    )?;
    for symbol in &analysis.symbols {
        ensure(
            !symbol.name.trim().is_empty()
                && symbol.name.len() <= 4096
                && symbol.qualified_name().len() <= 4096
                && symbol.start_line > 0
                && symbol.start_line <= symbol.end_line
                && symbol.end_line <= lines,
            "code symbol has an invalid name or source span",
        )?;
        if symbol.kind.is_primitive() {
            identity::compiler_identity(&symbol.metadata)?;
            identity::definition_role(&symbol.metadata)?;
        }
    }
    // Catch duplicate keys and invalid endpoints before planning or inference.
    let file_key = keys::scoped_file_key(&input.namespace, &input.path);
    let symbols = symbol_ids(input, &file_key)?;
    if symbols.iter().any(Option::is_some) {
        ontology.check(CODEBASE.defines_edges, CODEBASE.files, CODEBASE.symbols)?;
    }
    let mut edges = HashSet::new();
    let mut relationships = Vec::new();
    for relationship in &input.relationships {
        let from = endpoint(relationship.from, &file_key, &symbols)?;
        let to = endpoint(relationship.to, &file_key, &symbols)?;
        ontology.check(
            relationship.kind.collection(),
            from.split_once('/').unwrap().0,
            to.split_once('/').unwrap().0,
        )?;
        ensure(
            edges.insert((relationship.kind.collection(), from.clone(), to.clone())),
            "duplicate code relationship",
        )?;
        relationships.push(ResolvedRelationship {
            collection: relationship.kind.collection(),
            from,
            to,
        });
    }
    ensure(
        serde_json::to_vec(&input.analysis)
            .map_err(ArangoError::from)?
            .len()
            <= MAX_ANALYSIS_BYTES,
        "analysis metadata exceeds 4 MiB",
    )?;
    Ok(ResolvedGraph {
        file_key,
        symbols,
        relationships,
    })
}

fn symbol_ids(input: &CodeInput, key: &str) -> Result<Vec<Option<String>>, ArangoError> {
    let mut seen = HashSet::new();
    input
        .analysis
        .symbols
        .iter()
        .map(|symbol| {
            if !symbol.kind.is_primitive() {
                return Ok(None);
            }
            let key = keys::symbol_key(key, &symbol.qualified_name(), symbol.start_line);
            ensure(seen.insert(key.clone()), "duplicate code symbol identity")?;
            Ok(Some(format!("{}/{key}", CODEBASE.symbols)))
        })
        .collect()
}

fn endpoint(
    endpoint: LocalEndpoint,
    key: &str,
    symbols: &[Option<String>],
) -> Result<String, ArangoError> {
    match endpoint {
        LocalEndpoint::File => Ok(format!("{}/{key}", CODEBASE.files)),
        LocalEndpoint::Symbol(index) => symbols
            .get(index)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or_else(|| {
                invalid("relationship endpoint does not name a file-local graph primitive")
            }),
    }
}

pub(super) fn records(
    input: &CodeInput,
    resolved: &ResolvedGraph,
    binding: &BoundEmbeddingRequest,
) -> (Value, Rows) {
    let key = &resolved.file_key;
    let analysis = &input.analysis;
    let hash = compute_content_hash(&input.text);
    let symbols = &resolved.symbols;
    let mut rows: Rows = CODEBASE
        .all_collections()
        .iter()
        .filter(|(name, _)| *name != CODEBASE.files)
        .map(|(name, _)| ((*name).to_string(), Vec::new()))
        .collect();
    let common = json!({"format":FORMAT,"file_key":key,"content_hash":hash,"embedding_binding":binding,
        "analysis_tier":analysis.analysis_tier,"analyzer":analysis.analyzer});
    let edge = |from: &str, to: &str, collection: &str| {
        let mut row = common.clone();
        row["_key"] = json!(format!(
            "brel_{}",
            digest(&json!([key, collection, from, to]))
        ));
        row["_from"] = json!(from);
        row["_to"] = json!(to);
        row
    };
    for (symbol, id) in analysis.symbols.iter().zip(symbols) {
        let Some(id) = id else {
            continue;
        };
        let mut row = common.clone();
        row["_key"] = json!(id.split_once('/').unwrap().1);
        row["name"] = json!(symbol.name);
        row["qualified_name"] = json!(symbol.qualified_name());
        row["kind"] = json!(symbol.kind.universal_kind());
        row["lang_kind"] = json!(symbol.kind.lang_kind());
        row["start_line"] = json!(symbol.start_line);
        row["end_line"] = json!(symbol.end_line);
        row["metadata"] = symbol.metadata.clone();
        rows.get_mut(CODEBASE.symbols).unwrap().push(row);
        rows.get_mut(CODEBASE.defines_edges).unwrap().push(edge(
            &format!("{}/{key}", CODEBASE.files),
            id,
            CODEBASE.defines_edges,
        ));
    }
    for relationship in &resolved.relationships {
        let collection = relationship.collection;
        rows.get_mut(collection).unwrap().push(edge(
            &relationship.from,
            &relationship.to,
            collection,
        ));
    }
    let file = json!({"_key":key,"format":FORMAT,"kind":"file","namespace":input.namespace,"path":input.path,
        "full_text":input.text,"content_hash":hash,"embedding_binding":binding,"language":analysis.language.name(),
        "analysis_tier":analysis.analysis_tier,"analyzer":analysis.analyzer,"fallback_reason":analysis.fallback_reason,
        "metrics":analysis.metrics,"symbol_hash":analysis.symbol_hash,"symbol_count":rows[CODEBASE.symbols].len()});
    (file, rows)
}

/// Exclude only Arango's own envelope, so additions to an owned record cannot be silently discarded.
pub(super) fn manifest(
    rows: &Rows,
) -> Result<BTreeMap<String, BTreeMap<String, String>>, ArangoError> {
    rows.iter()
        .map(|(collection, records)| Ok((collection.clone(), record_manifest(records)?)))
        .collect()
}

pub(super) fn record_manifest(records: &[Value]) -> Result<BTreeMap<String, String>, ArangoError> {
    let mut hashes = BTreeMap::new();
    for row in records {
        let mut row = row.clone();
        let object = row
            .as_object_mut()
            .ok_or_else(|| invalid("invalid code record"))?;
        object.remove("_id");
        object.remove("_rev");
        let key = object
            .get("_key")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("missing code record key"))?
            .to_owned();
        normalize_numbers(&mut row);
        ensure(
            hashes.insert(key, digest(&row)).is_none(),
            "duplicate code record key",
        )?;
    }
    Ok(hashes)
}

/// ArangoDB may return an integral JSON float as an integer. Preserve numerical
/// identity without converting large integers to lossy f64 values. This is a
/// storage inventory rule, not a change to portable model/space canonicalization.
fn normalize_numbers(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(normalize_numbers),
        Value::Object(fields) => fields.values_mut().for_each(normalize_numbers),
        Value::Number(number) if number.is_f64() => {
            let value = number.as_f64().unwrap();
            if value.fract() == 0.0 {
                if value >= i64::MIN as f64 && value < i64::MAX as f64 {
                    *number = (value as i64).into();
                } else if value >= 0.0 && value < u64::MAX as f64 {
                    *number = (value as u64).into();
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_accepts_storage_number_spelling_but_preserves_exact_values() {
        let hash = |mut row: Value| {
            normalize_numbers(&mut row);
            digest(&row)
        };
        assert_eq!(
            hash(json!({"a":[1.0, -0.0, -2.0]})),
            hash(json!({"a":[1, 0, -2]}))
        );
        assert_ne!(hash(json!([1.25])), hash(json!([1.25001])));
        assert_ne!(
            hash(json!(9007199254740993_u64)),
            hash(json!(9007199254740992.0_f64))
        );
        assert_ne!(hash(json!(u64::MAX)), hash(json!(u64::MAX as f64)));
        assert_eq!(hash(json!(i64::MIN)), hash(json!(i64::MIN as f64)));
    }
}
