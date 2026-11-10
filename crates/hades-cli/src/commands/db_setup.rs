//! `hades db setup`: create a managed database through the recoverable setup ledger (#225).
//!
//! CLI-only and in-process, like `db create-database`: there is no daemon or MCP
//! variant. Every policy value is explicit; no default policy is silently chosen,
//! because the request digest must be reproducible for an identical retry.
use anyhow::{Context, Result, anyhow, bail};
use clap::Subcommand;
use hades_core::config::{HadesConfig, SetupConfig};
use hades_core::db::ArangoClient;
use hades_core::embedding_profile::{ChunkPolicy, PassagePolicy};
use hades_core::persephone::provider_registry::ProviderCatalog;
use hades_core::workspace_setup::{SetupRequest, SetupStatus, WorkspaceSetup};
use serde_json::json;

use super::output::{self, OutputFormat};

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

/// The installation sections setup needs, checked before any network or ledger access:
/// the `setup` section (validated) and the provider catalog from `registered_providers`.
fn sections(config: &HadesConfig) -> Result<(SetupConfig, ProviderCatalog)> {
    let setup = config.setup.clone().ok_or_else(|| {
        anyhow!(
            "db setup requires the setup section (ledger_root, installation_id) in \
             installation configuration"
        )
    })?;
    setup.validate().map_err(|e| anyhow!(e))?;
    let catalog = config
        .registered_providers
        .as_ref()
        .ok_or_else(|| {
            anyhow!("db setup requires registered_providers in installation configuration")
        })?
        .catalog()
        .context("invalid registered_providers configuration")?;
    Ok((setup, catalog))
}

/// Bind the coordinator to the configured ledger. The admin client targets `_system`
/// exactly as `db create-database` builds it, because the ledger scope hashes that
/// client's socket path and base URL. The owner is the ArangoDB user the CLI connects
/// as, so two users on one installation hold separate operations. Building the client
/// makes no request.
fn coordinator(
    config: &HadesConfig,
    setup: SetupConfig,
    catalog: ProviderCatalog,
) -> Result<WorkspaceSetup> {
    let mut system_config = config.clone();
    system_config.database.name = Some("_system".to_string());
    let admin = ArangoClient::from_config(&system_config, false)
        .context("failed to build the ArangoDB _system client for setup")?;
    WorkspaceSetup::new(
        setup.ledger_root,
        setup.installation_id,
        config.database.username.clone(),
        admin,
        catalog,
    )
    .context("failed to open the setup ledger")
}

/// Refuse a policy for the space the provider does not serve. The verified server serves
/// one space per process, so a late load cannot honour a conventional policy, and a
/// conventional load cannot plan late windows.
fn check_space(provider: &str, serves_late: bool, policy: &ChunkPolicy) -> Result<()> {
    match policy {
        ChunkPolicy::Conventional { .. } if serves_late => bail!(
            "provider {provider} serves the late space only; use --chunking late or a \
             conventional provider"
        ),
        ChunkPolicy::Late { .. } if !serves_late => bail!(
            "provider {provider} does not advertise late chunking; use --chunking conventional"
        ),
        _ => Ok(()),
    }
}

/// Map the liveness report's space to late or not, refusing a space this CLI does not know.
fn status_serves_late(provider: &str, space: &str) -> Result<bool> {
    match space {
        "late" => Ok(true),
        "conventional" => Ok(false),
        other => bail!("provider {provider} reports an unknown space {other:?}"),
    }
}

/// Create the managed database. A space mismatch is refused from `GET /v1/status` before
/// `verify`, so it never loads or observes the model; the verified observation is then
/// checked again, because liveness is not an observation.
pub async fn run_run(config: &HadesConfig, args: SetupRunArgs) -> Result<()> {
    let policy = policy(&args)?;
    let (setup_config, catalog) = sections(config)?;
    let provider = args.provider.as_str();
    let status = catalog
        .status(provider)
        .await
        .with_context(|| format!("provider {provider} status probe failed"))?;
    check_space(
        provider,
        status_serves_late(provider, &status.space)?,
        &policy,
    )?;
    let verified = catalog
        .verify(provider)
        .await
        .with_context(|| format!("provider {provider} verification failed"))?;
    check_space(provider, verified.serves_late_only(), &policy)?;
    let profile = verified
        .document_profile(args.generation.clone(), policy)
        .context("provider cannot serve the requested policy")?;
    let request = SetupRequest {
        operation_id: args.operation_id.clone(),
        database: args.database.clone(),
        profile,
        manifest: verified.manifest().clone(),
    };
    let setup = coordinator(config, setup_config, catalog)?;
    let inspect = format!(
        "inspect it with: hades db setup inspect --operation-id {}",
        args.operation_id
    );
    let receipt = match setup.run(request).await {
        Ok(receipt) => receipt,
        Err(error) => {
            // The coordinator reports every non-ready outcome as an error after saving
            // the receipt, so read it back: a failed command still shows where it
            // stopped. A ledger that cannot be read leaves only the error.
            if let Ok(Some(saved)) = setup.inspect(&args.operation_id).await {
                output::print_output_with_success(
                    "db.setup.run",
                    serde_json::to_value(&saved)?,
                    &OutputFormat::Json,
                    false,
                );
            }
            return Err(anyhow!(error).context(format!("setup did not complete; {inspect}")));
        }
    };
    let ready = receipt.status == SetupStatus::Ready;
    output::print_output_with_success(
        "db.setup.run",
        serde_json::to_value(&receipt)?,
        &OutputFormat::Json,
        ready,
    );
    if !ready {
        bail!(
            "setup is {:?} at {:?}; {inspect}",
            receipt.status,
            receipt.step
        );
    }
    Ok(())
}

/// Print the ledger receipt for an operation. No model or database request is made; an
/// operation this ledger does not hold is a failed envelope with a null receipt.
pub async fn run_inspect(config: &HadesConfig, operation_id: &str) -> Result<()> {
    let (setup_config, catalog) = sections(config)?;
    let receipt = coordinator(config, setup_config, catalog)?
        .inspect(operation_id)
        .await
        .context("failed to read the setup ledger")?;
    let found = receipt.is_some();
    output::print_output_with_success(
        "db.setup.inspect",
        json!({"operation_id": operation_id, "receipt": receipt}),
        &OutputFormat::Json,
        found,
    );
    if !found {
        bail!("no operation {operation_id} in this setup ledger");
    }
    Ok(())
}

/// Print the advisory recovery report. Writes nothing and loads no model; the report
/// cannot authorize or resume work by itself.
pub async fn run_diagnose(config: &HadesConfig, operation_id: &str) -> Result<()> {
    let (setup_config, catalog) = sections(config)?;
    let report = coordinator(config, setup_config, catalog)?
        .diagnose(operation_id)
        .await
        .context("failed to diagnose the setup operation")?;
    output::print_output_with_success(
        "db.setup.diagnose",
        serde_json::to_value(&report)?,
        &OutputFormat::Json,
        true,
    );
    Ok(())
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

    #[test]
    fn check_space_refuses_each_mismatch_by_name() {
        let late = policy(&args(
            Chunking::Late,
            Some(16384),
            Some(1000),
            Some(500),
            Some(200),
        ))
        .unwrap();
        let conventional = policy(&args(
            Chunking::Conventional,
            None,
            None,
            Some(500),
            Some(200),
        ))
        .unwrap();
        assert!(check_space("p", true, &late).is_ok());
        assert!(check_space("p", false, &conventional).is_ok());
        assert!(
            check_space("p", true, &conventional)
                .unwrap_err()
                .to_string()
                .contains("provider p serves the late space only")
        );
        assert!(
            check_space("p", false, &late)
                .unwrap_err()
                .to_string()
                .contains("provider p does not advertise late chunking")
        );
    }

    #[test]
    fn status_space_is_late_conventional_or_refused() {
        assert!(status_serves_late("p", "late").unwrap());
        assert!(!status_serves_late("p", "conventional").unwrap());
        assert!(status_serves_late("p", "Late").is_err());
    }
}
