//! Native Rust handlers for `hades db` read commands.
//!
//! Each function builds a [`DaemonCommand`], calls [`dispatch`], and
//! prints the result to stdout using the standard HADES JSON envelope.

use anyhow::{Context, Result};
use serde_json::json;

use hades_core::config::HadesConfig;
use hades_core::db::ArangoPool;
use hades_core::dispatch::{self, DaemonCommand};

use super::output::{self, OutputFormat};

/// `hades db get <collection> <key> [--format F]`
pub async fn run_get(
    config: &HadesConfig,
    collection: &str,
    key: &str,
    format: &str,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbGet(dispatch::DbGetParams {
            collection: collection.to_string(),
            key: key.to_string(),
        }),
    )
    .await?;
    output::print_output("db.get", data, &fmt);
    Ok(())
}

/// `hades db count <collection>`
pub async fn run_count(config: &HadesConfig, collection: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbCount(dispatch::DbCountParams {
            collection: collection.to_string(),
        }),
    )
    .await?;
    output::print_output("db.count", data, &OutputFormat::Json);
    Ok(())
}

/// `hades db collections [--format F]`
pub async fn run_collections(config: &HadesConfig, format: &str) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(&pool, config, DaemonCommand::DbCollections {}).await?;
    output::print_output("db.collections", data, &fmt);
    Ok(())
}

/// `hades db check <document_id>`
pub async fn run_check(config: &HadesConfig, document_id: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbCheck(dispatch::DbCheckParams {
            document_id: document_id.to_string(),
        }),
    )
    .await?;
    output::print_output("db.check", data, &OutputFormat::Json);
    Ok(())
}

/// `hades db recent [--limit N] [--format F]`
pub async fn run_recent(config: &HadesConfig, limit: u32, format: &str) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbRecent(dispatch::DbRecentParams { limit: Some(limit) }),
    )
    .await?;
    output::print_output("db.recent", data, &fmt);
    Ok(())
}

/// `hades db list [--collection PROFILE] [--limit N] [--paper ID] [--format F]`
pub async fn run_list(
    config: &HadesConfig,
    collection: Option<&str>,
    limit: u32,
    paper: Option<&str>,
    format: &str,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbList(dispatch::DbListParams {
            collection: collection.map(String::from),
            limit: Some(limit),
            paper: paper.map(String::from),
            // The CLI has always printed whole documents and its output is not
            // charged against a context window, so it keeps them.
            fields: None,
        }),
    )
    .await?;
    output::print_output("db.list", data, &fmt);
    Ok(())
}

/// `hades db aql <AQL> [--bind JSON] [--limit N] [--format F]`
pub async fn run_aql(
    config: &HadesConfig,
    aql: &str,
    bind: Option<&str>,
    limit: Option<u32>,
    format: &str,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let bind_value = match bind {
        Some(s) => {
            let v: serde_json::Value = serde_json::from_str(s).context("invalid --bind JSON")?;
            if !v.is_object() {
                anyhow::bail!(
                    "invalid --bind: must be a JSON object, got {}",
                    match &v {
                        serde_json::Value::Array(_) => "array",
                        serde_json::Value::String(_) => "string",
                        serde_json::Value::Number(_) => "number",
                        serde_json::Value::Bool(_) => "bool",
                        serde_json::Value::Null => "null",
                        _ => "non-object",
                    }
                );
            }
            Some(v)
        }
        None => None,
    };
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbAql(dispatch::DbAqlParams {
            aql: aql.to_string(),
            bind: bind_value,
            limit,
        }),
    )
    .await?;
    output::print_output("db.aql", data, &fmt);
    Ok(())
}

/// `hades db health [--verbose]`
pub async fn run_health(config: &HadesConfig, verbose: bool) -> Result<()> {
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(
        &pool,
        config,
        DaemonCommand::DbHealth(dispatch::DbHealthParams { verbose }),
    )
    .await?;
    output::print_output("db.health", data, &OutputFormat::Json);
    Ok(())
}

/// `hades db stats [--format F]`
pub async fn run_stats(config: &HadesConfig, format: &str) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let data = dispatch::dispatch(&pool, config, DaemonCommand::DbStats {}).await?;
    output::print_output("db.stats", data, &fmt);
    Ok(())
}

/// `hades db export <collection> [--limit N] [--format F] [--output PATH]`
///
/// Streams documents as JSONL to stdout (or to a file via --output).
/// Uses cursor pagination to avoid buffering the entire collection in memory.
/// This is CLI-only — not routed through the daemon dispatch.
pub async fn run_export(
    config: &HadesConfig,
    collection: &str,
    output: Option<&std::path::Path>,
    format: &str,
    limit: Option<u32>,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    if !matches!(fmt, OutputFormat::Jsonl) {
        anyhow::bail!(
            "export only supports jsonl format (streaming output) — \
             use --format jsonl or omit --format"
        );
    }
    use std::io::Write;

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    // Open writer BEFORE creating the cursor — if file creation fails,
    // no server-side cursor needs cleanup.
    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(
            std::fs::File::create(path)
                .with_context(|| format!("failed to create {}", path.display()))?,
        ),
        None => Box::new(std::io::stdout().lock()),
    };

    let aql = if let Some(lim) = limit {
        format!("FOR d IN @@col LIMIT {lim} RETURN d")
    } else {
        "FOR d IN @@col RETURN d".to_string()
    };

    let batch_size: u32 = 1000;
    let body = json!({
        "query": aql,
        "batchSize": batch_size,
        "bindVars": { "@col": collection },
    });

    // Cursor creation — route through writer when reader != writer,
    // because cursor state endpoints require the read-write socket.
    let initial_client = if pool.is_shared() {
        pool.reader()
    } else {
        pool.writer()
    };
    let resp = initial_client
        .post("cursor", &body)
        .await
        .with_context(|| format!("export cursor failed for '{collection}'"))?;

    let mut cursor_ids = Vec::new();
    // Capture safe IDs before validating pages so errors still clean up cursors.
    let stream_result: Result<u64> = async {
        let mut total: u64 = 0;
        let mut page = resp;
        loop {
            let (rows, more) = export_page(&page, &mut cursor_ids)?;
            for doc in rows {
                writeln!(writer, "{}", serde_json::to_string(doc)?)?;
                total += 1;
            }
            if !more {
                break;
            }
            let path = format!("cursor/{}", cursor_ids[0]);
            page = pool
                .writer()
                .post(&path, &json!({}))
                .await
                .context("export cursor pagination failed")?;
        }
        writer.flush().context("failed to flush export output")?;
        Ok(total)
    }
    .await;

    // A changed but safe ID is retained too; never interpolate unsafe IDs.
    for id in cursor_ids {
        let _ = pool.writer().delete(&format!("cursor/{id}")).await;
    }

    let total = stream_result?;
    eprintln!("exported {total} documents from '{collection}'");
    Ok(())
}

/// Validate a complete page before emitting it; earlier pages may already have
/// reached the output stream when a continuation fails.
fn export_page<'a>(
    page: &'a serde_json::Value,
    cursor_ids: &mut Vec<String>,
) -> Result<(&'a [serde_json::Value], bool)> {
    if let Some(value) = page.get("id") {
        let id = value
            .as_str()
            .filter(|id| {
                !id.is_empty()
                    && id.len() <= 128
                    && id
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
            })
            .context("invalid export cursor ID")?;
        if cursor_ids.first().is_some_and(|known| known != id) {
            cursor_ids.push(id.to_string());
            anyhow::bail!("export cursor ID changed during pagination");
        }
        if cursor_ids.is_empty() {
            cursor_ids.push(id.to_string());
        }
    }
    let more = page
        .get("hasMore")
        .and_then(serde_json::Value::as_bool)
        .context("export cursor response missing boolean hasMore")?;
    let rows = page
        .get("result")
        .and_then(serde_json::Value::as_array)
        .context("export cursor response missing result array")?;
    if !rows.iter().all(serde_json::Value::is_object) {
        anyhow::bail!("export cursor result contains a non-document value");
    }
    if more && cursor_ids.is_empty() {
        anyhow::bail!("hasMore=true but no cursor ID in export response");
    }
    Ok((rows, more))
}

/// `hades db index-status [--collection C] [--format F]`
///
/// CLI-only — shows vector index info for embedding collections.
pub async fn run_index_status(
    config: &HadesConfig,
    collection: Option<&str>,
    format: &str,
) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    use hades_core::db::collections::CollectionProfile;
    use hades_core::db::index;

    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;

    let collections: Vec<&str> = if let Some(c) = collection {
        vec![c]
    } else {
        CollectionProfile::all()
            .iter()
            .map(|(_, p)| p.embeddings)
            .collect()
    };

    let mut results = Vec::new();
    for col in &collections {
        let indexes = match index::list_indexes(&pool, col).await {
            Ok(idx) => idx,
            Err(e) if e.is_not_found() => Vec::new(),
            Err(e) => {
                anyhow::bail!("failed to list indexes for '{col}': {e}");
            }
        };
        let vector_idx = indexes.iter().find(|i| i.index_type == "vector");
        results.push(json!({
            "collection": col,
            "has_vector_index": vector_idx.is_some(),
            "vector_index": vector_idx.map(|i| json!({
                "id": i.id,
                "fields": i.fields,
                "params": i.params,
            })),
            "total_indexes": indexes.len(),
        }));
    }

    output::print_output("db.index-status", json!({ "indexes": results }), &fmt);
    Ok(())
}

/// `hades db databases [--format F]`
///
/// CLI-only — lists accessible databases.
pub async fn run_databases(config: &HadesConfig, format: &str) -> Result<()> {
    let fmt = OutputFormat::parse(format)?;
    let pool = ArangoPool::from_config(config).context("failed to connect to ArangoDB")?;
    let resp = pool
        .reader()
        .get("database/user")
        .await
        .context("failed to list databases")?;

    let databases: Vec<String> = serde_json::from_value(
        resp.get("result")
            .cloned()
            .context("database-list response missing result")?,
    )
    .context("database-list result must be an array of names")?;

    output::print_output(
        "db.databases",
        json!({
            "databases": databases,
            "count": databases.len(),
        }),
        &fmt,
    );
    Ok(())
}
