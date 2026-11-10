//! `hades db setup`: create a managed database through the recoverable setup ledger (#225).
//!
//! CLI-only and in-process, like `db create-database`: there is no daemon or MCP
//! variant. Every policy value is explicit; no default policy is silently chosen,
//! because the request digest must be reproducible for an identical retry.
use anyhow::{Result, bail};
use clap::Subcommand;
use hades_core::config::HadesConfig;
use hades_core::embedding_profile::{ChunkPolicy, PassagePolicy};

#[derive(Debug, Subcommand)]
pub enum DbSetupCmd {
    /// Create a managed database through the setup ledger (loads the model).
    Run(SetupRunArgs),
    /// Read the ledger receipt for an operation; no model load.
    Inspect {
        /// Operation ID given to `db setup run`.
        #[arg(long)]
        operation_id: String,
    },
    /// Advisory recovery report for an operation; no writes, no model load.
    Diagnose {
        /// Operation ID given to `db setup run`.
        #[arg(long)]
        operation_id: String,
    },
}

#[derive(Debug, clap::Args)]
pub struct SetupRunArgs {
    /// Database to create; it must not exist yet.
    #[arg(long)]
    pub database: String,
    /// Registered provider ID from `registered_providers`.
    #[arg(long)]
    pub provider: String,
    /// Operation ID; repeat the exact command with it to retry.
    #[arg(long)]
    pub operation_id: String,
    /// Profile generation label. Required: a default would change on every repeat
    /// and defeat the identical-retry rule.
    #[arg(long)]
    pub generation: String,
    /// Chunking space; must match the provider's served space.
    #[arg(long, value_enum)]
    pub chunking: Chunking,
    /// Late only: encoding window cap in tokens.
    #[arg(long, required_if_eq("chunking", "late"))]
    pub context_cap_tokens: Option<u32>,
    /// Late only: minimum overlap between encoding windows in tokens.
    #[arg(long, required_if_eq("chunking", "late"))]
    pub window_overlap_tokens: Option<u32>,
    /// Retrieval passage size in tokens.
    #[arg(long, required = true)]
    pub passage_tokens: Option<u32>,
    /// Overlap between retrieval passages in tokens.
    #[arg(long, required = true)]
    pub overlap_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Chunking {
    Late,
    Conventional,
}

/// Build the explicit chunk policy. clap already requires the flags for the chosen
/// space; this re-checks them for callers that build `SetupRunArgs` directly, and
/// refuses late-only flags with a conventional policy rather than ignoring them.
/// Numeric sanity is the coordinator's and provider's job.
pub fn policy(args: &SetupRunArgs) -> Result<ChunkPolicy> {
    match args.chunking {
        Chunking::Late => match (
            args.context_cap_tokens,
            args.window_overlap_tokens,
            args.passage_tokens,
            args.overlap_tokens,
        ) {
            (
                Some(context_cap_tokens),
                Some(window),
                Some(passage_tokens),
                Some(overlap_tokens),
            ) => Ok(ChunkPolicy::Late {
                context_cap_tokens,
                minimum_window_overlap_tokens: window,
                passages: Some(PassagePolicy {
                    passage_tokens,
                    overlap_tokens,
                }),
            }),
            _ => bail!(
                "--chunking late requires --context-cap-tokens, --window-overlap-tokens, \
                 --passage-tokens and --overlap-tokens"
            ),
        },
        Chunking::Conventional => match (
            args.context_cap_tokens,
            args.window_overlap_tokens,
            args.passage_tokens,
            args.overlap_tokens,
        ) {
            (None, None, Some(passage_tokens), Some(overlap_tokens)) => {
                Ok(ChunkPolicy::Conventional {
                    passage_tokens,
                    overlap_tokens,
                })
            }
            _ => bail!("--chunking conventional takes only --passage-tokens and --overlap-tokens"),
        },
    }
}

/// Create the managed database (Task 4).
pub async fn run_run(_config: &HadesConfig, args: SetupRunArgs) -> Result<()> {
    policy(&args)?;
    bail!("not implemented")
}

/// Print the ledger receipt for an operation (Task 4).
pub async fn run_inspect(_config: &HadesConfig, _operation_id: &str) -> Result<()> {
    bail!("not implemented")
}

/// Print the advisory recovery report for an operation (Task 4).
pub async fn run_diagnose(_config: &HadesConfig, _operation_id: &str) -> Result<()> {
    bail!("not implemented")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct T {
        #[command(subcommand)]
        cmd: DbSetupCmd,
    }

    fn args(
        chunking: Chunking,
        cap: Option<u32>,
        win: Option<u32>,
        pt: Option<u32>,
        ov: Option<u32>,
    ) -> SetupRunArgs {
        SetupRunArgs {
            database: "managed".into(),
            provider: "local-jina".into(),
            operation_id: "setup-one".into(),
            generation: "g1".into(),
            chunking,
            context_cap_tokens: cap,
            window_overlap_tokens: win,
            passage_tokens: pt,
            overlap_tokens: ov,
        }
    }

    #[test]
    fn late_policy_requires_all_four() {
        assert!(
            policy(&args(
                Chunking::Late,
                Some(16384),
                Some(1000),
                Some(500),
                None
            ))
            .unwrap_err()
            .to_string()
            .contains("--overlap-tokens")
        );
        assert_eq!(
            policy(&args(
                Chunking::Late,
                Some(16384),
                Some(1000),
                Some(500),
                Some(200)
            ))
            .unwrap(),
            ChunkPolicy::Late {
                context_cap_tokens: 16384,
                minimum_window_overlap_tokens: 1000,
                passages: Some(PassagePolicy {
                    passage_tokens: 500,
                    overlap_tokens: 200
                })
            }
        );
    }

    #[test]
    fn conventional_policy_refuses_late_flags() {
        assert!(
            policy(&args(
                Chunking::Conventional,
                Some(1),
                None,
                Some(500),
                Some(200)
            ))
            .is_err()
        );
        assert_eq!(
            policy(&args(
                Chunking::Conventional,
                None,
                None,
                Some(500),
                Some(200)
            ))
            .unwrap(),
            ChunkPolicy::Conventional {
                passage_tokens: 500,
                overlap_tokens: 200
            }
        );
    }

    #[test]
    fn clap_requires_generation_and_chunking() {
        let base = [
            "t",
            "run",
            "--database",
            "d",
            "--provider",
            "p",
            "--operation-id",
            "o",
        ];
        let policy = ["--passage-tokens", "500", "--overlap-tokens", "200"];
        let missing_generation = [&base[..], &["--chunking", "conventional"], &policy].concat();
        assert!(T::try_parse_from(missing_generation).is_err());
        let missing_chunking = [&base[..], &["--generation", "g"], &policy].concat();
        assert!(T::try_parse_from(missing_chunking).is_err());
        let complete = [
            &base[..],
            &["--generation", "g", "--chunking", "conventional"],
            &policy,
        ]
        .concat();
        assert!(T::try_parse_from(complete).is_ok());
    }

    /// Review focus 3: a missing late flag is a usage error from clap, naming the flag,
    /// with exit code 2, before any configuration, network or ledger access (#225).
    #[test]
    fn clap_names_a_missing_late_flag() {
        let error = T::try_parse_from([
            "t",
            "run",
            "--database",
            "d",
            "--provider",
            "p",
            "--operation-id",
            "o",
            "--generation",
            "g",
            "--chunking",
            "late",
            "--context-cap-tokens",
            "16384",
            "--passage-tokens",
            "500",
            "--overlap-tokens",
            "200",
        ])
        .err()
        .expect("missing --window-overlap-tokens must be refused");
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
        assert_eq!(error.exit_code(), 2);
        assert!(
            error.to_string().contains("--window-overlap-tokens"),
            "{error}"
        );
    }
}
