//! Thin administration/recovery adapter for guided reading.
use super::output::{self, OutputFormat};
use anyhow::Result;
use clap::Subcommand;
use hades_core::{
    HadesConfig,
    db::ArangoPool,
    reading::{self, ReadRequest},
};
use std::path::PathBuf;
#[derive(Debug, Subcommand)]
pub enum ReadingCmd {
    /// List sessions, inspect a session, or export its append-only record.
    Read {
        #[arg(long)]
        session: Option<String>,
        #[arg(long, default_value = "state")]
        view: String,
        #[arg(long)]
        source_hash: Option<String>,
    },
    /// Query the running extractor for document formats and reading limits.
    Capabilities,
    /// Retry conversion of an archived source without uploading or replacing its bytes.
    Extract { source_hash: String },
    /// Apply an exact saved JSON mutation (including request_id and expected_revision).
    /// This is an administrator recovery operation, not an agent acceptance path.
    Apply { file: PathBuf },
}
pub async fn run(config: &HadesConfig, command: ReadingCmd, format: &str) -> Result<()> {
    let pool = ArangoPool::from_config(config)?;
    let data = match command {
        ReadingCmd::Read {
            session,
            view,
            source_hash,
        } => {
            reading::read(
                &pool,
                ReadRequest {
                    session,
                    view,
                    source_hash,
                },
            )
            .await?
        }
        ReadingCmd::Capabilities => reading::capabilities(config).await?,
        ReadingCmd::Extract { source_hash } => reading::extract(&pool, config, source_hash).await?,
        ReadingCmd::Apply { file } => {
            let bytes = std::fs::read(file)?;
            if bytes.len() > 2 * 1024 * 1024 {
                anyhow::bail!("mutation file exceeds 2 MiB")
            };
            reading::mutate(&pool, serde_json::from_slice(&bytes)?, false).await?
        }
    };
    output::print_output("reading", data, &OutputFormat::parse(format)?);
    Ok(())
}
