//! Resolve selected graph objects into immutable, database-scoped chat evidence.
use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub db: String,
    pub graph: String,
    pub kind: String,
    pub id: String,
    pub excerpt: String,
}

pub(super) async fn snapshot(
    pool: &ArangoPool,
    binding: &Value,
    selections: &[Selection],
    references: Vec<Value>,
) -> Result<Value, HandlerError> {
    if selections.len() > 6 {
        return Err(invalid("Select at most six workspace objects"));
    }
    let mut objects = Vec::new();
    for selection in selections {
        if selection.db != pool.database() {
            return Err(invalid("Workspace reference belongs to another database"));
        }
        let (collection, key) = selection
            .id
            .split_once('/')
            .ok_or_else(|| invalid("Invalid object ID"))?;
        if collection.is_empty()
            || key.is_empty()
            || key.contains('/')
            || selection.graph.is_empty()
            || selection.id.len() > 512
            || selection.graph.len() > 256
            || selection.excerpt.len() > 4800
        {
            return Err(invalid("Invalid or oversized workspace reference"));
        }
        let graph = pool
            .reader()
            .get(&format!(
                "gharial/{}",
                urlencoding::encode(&selection.graph)
            ))
            .await
            .map_err(db_error)?;
        let graph = &graph["graph"];
        let edges = graph["edgeDefinitions"]
            .as_array()
            .ok_or_else(|| invalid("Graph definition unavailable"))?;
        let member = match selection.kind.as_str() {
            "edge" => edges.iter().any(|e| e["collection"] == collection),
            "node" => {
                graph["orphanCollections"]
                    .as_array()
                    .is_some_and(|a| a.iter().any(|v| v == collection))
                    || edges.iter().any(|e| {
                        ["from", "to"].iter().any(|side| {
                            e[side]
                                .as_array()
                                .is_some_and(|a| a.iter().any(|v| v == collection))
                        })
                    })
            }
            _ => false,
        };
        if !member {
            return Err(invalid("Selected object does not belong to this graph"));
        }
        let row = pool
            .reader()
            .get(&format!(
                "document/{}/{}",
                urlencoding::encode(collection),
                urlencoding::encode(key)
            ))
            .await
            .map_err(db_error)?;
        // Accept only an exact excerpt of a current textual field. Full row
        // identity/digest lets subsequent retrieval distinguish newer content.
        if !selection.excerpt.is_empty()
            && !["content", "body", "definition", "text", "statement"]
                .iter()
                .any(|field| {
                    row[field]
                        .as_str()
                        .is_some_and(|text| text.contains(&selection.excerpt))
                })
        {
            return Err(invalid(
                "Selected excerpt changed or is unavailable; refresh the object",
            ));
        }
        objects.push(json!({"selection":selection,"revision":row["_rev"],"document_hash":record_hash(&row),
            "title":row.get("title").or_else(||row.get("label")),"source_version":row["source_version"],
            "source_location":row["source_location"],"assertion_origin":row["assertion_origin"],
            "from":row["_from"],"to":row["_to"],"relation":row.get("relation").or_else(||row.get("type"))}));
    }
    let context = json!({"format":"hades-workspace-context/v1","database":pool.database(),
        "origin_reading":binding["session"],"origin_source_hash":binding["source_hash"],
        "objects":objects,"conversation_references":references});
    if serde_json::to_vec(&context).unwrap().len() > 12000 {
        return Err(invalid(
            "Selected context exceeds 12,000 bytes; select fewer references",
        ));
    }
    Ok(context)
}
