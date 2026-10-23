//! The database owns endpoint permissions; there is no compiled-in ontology fallback.
use std::collections::BTreeSet;

use super::*;
use crate::{
    db::{ArangoClient, query},
    graph::runtime_schema::RuntimeEdgeDef,
    pipeline::profiled_storage::aql,
};

#[derive(Clone, PartialEq, Eq)]
struct Endpoints {
    from: BTreeSet<String>,
    to: BTreeSet<String>,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct Ontology(BTreeMap<String, Endpoints>);

impl Ontology {
    /// Only four direct edge definitions are relevant to this file-local writer.
    /// Bound the inventory and reject missing/ambiguous definitions before inference.
    pub(super) async fn load(client: &ArangoClient) -> Result<Self, ArangoError> {
        let names = CODEBASE.edge_collections();
        let response = aql(client,
            "FOR d IN hades_schema FILTER d.schema_type == 'edge_definition' AND d.name IN @names LIMIT 5 RETURN d",
            json!({"names":names}), 6, 64 * 1024).await?;
        let rows = query::completed_rows(&response)?;
        ensure(
            rows.len() == names.len(),
            "code ontology requires exactly one saved definition for each edge collection",
        )?;
        let mut definitions = BTreeMap::new();
        for row in rows {
            let definition: RuntimeEdgeDef = serde_json::from_value(row.clone())?;
            ensure(
                names.contains(&definition.name.as_str()) && definition.source_field.is_none(),
                "code ontology requires direct edge definitions",
            )?;
            let endpoints = Endpoints {
                from: definition.from_collections.into_iter().collect(),
                to: definition.to_collections.into_iter().collect(),
            };
            ensure(
                !endpoints.from.is_empty() && !endpoints.to.is_empty(),
                "code ontology has empty endpoint types",
            )?;
            ensure(
                definitions.insert(definition.name, endpoints).is_none(),
                "ambiguous code edge definition",
            )?;
        }
        Ok(Self(definitions))
    }

    pub(super) fn check(&self, collection: &str, from: &str, to: &str) -> Result<(), ArangoError> {
        let endpoints = self
            .0
            .get(collection)
            .ok_or_else(|| invalid("missing code edge definition"))?;
        ensure(
            endpoints.from.contains(from) && endpoints.to.contains(to),
            "relationship endpoints violate the saved code ontology",
        )
    }
}
