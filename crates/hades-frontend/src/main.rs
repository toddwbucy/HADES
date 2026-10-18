//! `hades-viewer` — a local, WebGL graph-visualization frontend for HADES graphs.
//!
//! Graph reads consume the CLI JSON contract. Guided reading uses the shared
//! core contracts over the daemon Unix socket; the daemon owns all mutations.
//!
//! Two modes:
//!   * `hades-viewer dump  --db <db> --graph <g>`  -> print the graph-JSON to stdout
//!     (proves the coupling contract with no frontend involved).
//!   * `hades-viewer serve --db <db> [--bind ...]` -> serve the viewer + graph API.

mod assemble;
mod backend_process;
mod contract;
mod payload;
mod reading;
mod server;
mod transport_limits;

use anyhow::Result;
use assemble::Backend;
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "hades-viewer",
    version,
    about = "WebGL graph viewer for HADES graphs"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Assemble a graph snapshot and print it as JSON (no server, no frontend).
    Dump(DumpArgs),
    /// Serve the interactive viewer over loopback.
    Serve(ServeArgs),
}

/// Shared backend-invocation options.
#[derive(Args, Clone)]
struct BackendArgs {
    /// Source database (passed to `hades --db`).
    #[arg(long)]
    db: String,
    /// Path to the `hades` binary (defaults to `hades` on PATH).
    #[arg(long, default_value = "hades")]
    hades_bin: String,
    /// Cap documents per collection; marks the snapshot partial when hit.
    #[arg(long)]
    limit: Option<u32>,
}

impl BackendArgs {
    fn backend(&self) -> Backend {
        Backend {
            bin: self.hades_bin.clone(),
            database: self.db.clone(),
            limit: self.limit,
        }
    }
}

#[derive(Args)]
struct DumpArgs {
    #[command(flatten)]
    backend: BackendArgs,
    /// Named graph to assemble.
    #[arg(long)]
    graph: String,
    /// Pretty-print the JSON.
    #[arg(long)]
    pretty: bool,
}

#[derive(Args)]
struct ServeArgs {
    /// Databases the viewer may serve — repeat `--db` to restrict to a specific
    /// set (first is the default). Omit entirely to serve EVERY database the
    /// hades user can access (ArangoDB ACLs are the gate).
    #[arg(long = "db")]
    databases: Vec<String>,
    /// Path to the `hades` binary (defaults to `hades` on PATH).
    #[arg(long, default_value = "hades")]
    hades_bin: String,
    /// Cap documents per collection; marks the snapshot partial when hit.
    #[arg(long)]
    limit: Option<u32>,
    /// Require HTTP Basic auth with this password (any username). Mandatory when
    /// binding to a non-loopback address, since the viewer reads whole databases.
    #[arg(long)]
    password: Option<String>,
    /// Read authentication password from a local file. Required for LAN reading writes.
    #[arg(long, conflicts_with = "password")]
    password_file: Option<std::path::PathBuf>,
    /// Unix daemon serving the guided-reading database.
    #[arg(long, requires = "reading_db")]
    reading_socket: Option<std::path::PathBuf>,
    /// Explicit database served by the reading daemon.
    #[arg(long, requires = "reading_socket")]
    reading_db: Option<String>,
    /// Address to bind (loopback or a private LAN IP only).
    #[arg(long, default_value = "127.0.0.1:8088")]
    bind: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hades_viewer=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    backend_process::install_shutdown_signals()?;
    let result = async {
        match cli.command {
            Command::Dump(args) => {
                let snapshot = args.backend.backend().assemble(&args.graph).await?;
                let json = if args.pretty {
                    serde_json::to_string_pretty(&snapshot)?
                } else {
                    serde_json::to_string(&snapshot)?
                };
                println!("{json}");
                Ok(())
            }
            Command::Serve(args) => {
                if args.reading_socket.is_some() && args.password.is_some() {
                    anyhow::bail!("guided reading requires --password-file rather than a command-line password");
                }
                let password = load_password(args.password_file, args.password)?;
                server::serve(
                    args.hades_bin,
                    args.limit,
                    args.databases,
                    password,
                    args.reading_socket.zip(args.reading_db),
                    &args.bind,
                )
                .await
            }
        }
    }
    .await;
    backend_process::shutdown_and_wait().await;
    result
}

/// Read a file-backed credential without treating an empty file as authentication.
fn load_password(
    file: Option<std::path::PathBuf>,
    inline: Option<String>,
) -> anyhow::Result<Option<String>> {
    let password = match file {
        Some(path) => Some(
            std::fs::read_to_string(path)?
                .trim_end_matches(['\r', '\n'])
                .to_owned(),
        ),
        None => inline,
    };
    anyhow::ensure!(
        password.as_deref() != Some(""),
        "password must not be empty"
    );
    Ok(password)
}

#[cfg(test)]
mod password_tests {
    use super::load_password;

    #[test]
    fn empty_credentials_fail_without_requiring_reading_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("password");
        for value in ["", "\n", "\r\n"] {
            std::fs::write(&path, value).unwrap();
            assert!(load_password(Some(path.clone()), None).is_err());
        }
        assert!(load_password(None, Some(String::new())).is_err());
        assert_eq!(load_password(None, None).unwrap(), None);
        std::fs::write(&path, " fixture password \r\n").unwrap();
        assert_eq!(
            load_password(Some(path), None).unwrap().as_deref(),
            Some(" fixture password ")
        );
        assert_eq!(
            load_password(None, Some("nonempty".into()))
                .unwrap()
                .as_deref(),
            Some("nonempty")
        );
    }
}
