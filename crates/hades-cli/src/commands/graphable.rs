//! Argument/output adapter. Mapping and persistence belong to hades-core.
use super::output::{self, OutputFormat};
use anyhow::Result;
use clap::Subcommand;
use hades_core::{
    HadesConfig,
    db::ArangoPool,
    graphable::{self, Plan},
};
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
pub enum GraphableCmd {
    /// Map declared structure. Does not interpret claims or request embeddings.
    Import {
        file: PathBuf,
        #[arg(long)]
        graph: String,
        /// Parse and report without connecting to the database.
        #[arg(long)]
        dry_run: bool,
        /// Create the authored graph if absent. Existing layouts must match.
        #[arg(long)]
        create_graph: bool,
        /// Preserve the original file behind a reviewed Markdown extraction.
        #[arg(long, requires = "extraction_note")]
        source_archive: Option<PathBuf>,
        /// Describe the extraction and review that links Markdown to the original.
        #[arg(long, requires = "source_archive")]
        extraction_note: Option<String>,
    },
}
pub async fn run(config: &HadesConfig, command: GraphableCmd, format: &str) -> Result<()> {
    let GraphableCmd::Import {
        file,
        graph,
        dry_run,
        create_graph,
        source_archive,
        extraction_note,
    } = command;
    let mut plan = Plan::from_path(&file, &graph)?;
    if let Some(archive) = source_archive {
        plan = plan.with_original_source(&archive, extraction_note.as_deref().unwrap_or(""))?;
    }
    let result = if dry_run {
        let mut summary = plan.summary();
        summary["mapped"] = serde_json::json!(false);
        summary["dry_run"] = serde_json::json!(true);
        summary
    } else {
        let pool = ArangoPool::from_config(config)?;
        graphable::import(&pool, &plan, create_graph).await?
    };
    output::print_output("graphable.import", result, &OutputFormat::parse(format)?);
    Ok(())
}
