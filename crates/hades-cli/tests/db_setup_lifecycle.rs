//! `hades db setup` end to end through the real binary against a disposable database
//! (#225). The provider is a synthetic registration plus the shared verified Peer; no
//! model is loaded. Run only through
//! `scripts/test_isolated_database.py --contract db_setup`, never the live server.
use hades_core::{
    config::{HadesConfig, SetupConfig},
    db::{ArangoClient, ArangoPool, crud},
    embedding_profile::{ProfileState, store::ProfileStore},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    sync::atomic::Ordering,
};
use tokio::process::Command;

#[allow(dead_code)]
#[path = "../../hades-core/tests/common/profiled_provider.rs"]
mod provider;
use provider::{Fixture, Peer};

#[allow(dead_code)]
#[path = "../../hades-core/tests/common/setup_proxy.rs"]
mod fault;
use fault::{Proxy, Rule};

/// The harness user; the CLI connects as it, so it is also the ledger owner.
fn user() -> String {
    std::env::var("HADES_TEST_USER").unwrap_or_else(|_| "root".into())
}

/// Let the coordinator create the exact database name the harness owns and will drop.
async fn remove_fixture_database(pool: &ArangoPool) {
    ArangoClient::with_socket(
        pool.writer().socket_path().unwrap().into(),
        "_system",
        &user(),
        &std::env::var("ARANGO_PASSWORD").unwrap(),
    )
    .delete(&format!("database/{}", pool.database()))
    .await
    .unwrap();
}

/// A private 0700 ledger, never the workstation's installation state.
fn ledger() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    directory
}

/// Every file in the ledger with its bytes, to prove a read-only verb wrote nothing.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    std::fs::read_dir(root)
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            let bytes = std::fs::read(&path).unwrap();
            (path, bytes)
        })
        .collect()
}

/// Installation configuration with both sections; `socket` is the database endpoint the
/// CLI uses, which the ledger scope freezes.
fn configuration(directory: &Path, socket: &Path, ledger: &Path, peer: &Peer) -> PathBuf {
    let mut config = HadesConfig::default();
    config.database.username = user();
    config.database.sockets.readonly = Some(socket.to_string_lossy().into());
    config.database.sockets.readwrite = Some(socket.to_string_lossy().into());
    config.registered_providers = Some(peer.configuration());
    config.setup = Some(SetupConfig {
        ledger_root: ledger.into(),
        installation_id: "test-installation".into(),
    });
    let path = directory.join("hades.json");
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    path
}

struct Cli {
    config: PathBuf,
    socket: PathBuf,
}

impl Cli {
    async fn run(&self, args: &[&str]) -> std::process::Output {
        self.raw(&[&["db", "setup"][..], args].concat()).await
    }

    async fn raw(&self, args: &[&str]) -> std::process::Output {
        Command::new(env!("CARGO_BIN_EXE_hades"))
            .args(args)
            .env("HADES_CONFIG", &self.config)
            .env("ARANGO_RO_SOCKET", &self.socket)
            .env("ARANGO_RW_SOCKET", &self.socket)
            .env("HADES_EMBEDDER_SOCKET", "/absent/legacy-provider.sock")
            .env("HADES_EXTRACTOR_SOCKET", "/absent/extractor.sock")
            .env_remove("HADES_DATABASE")
            .kill_on_drop(true)
            .output()
            .await
            .unwrap()
    }
}

fn envelope(output: &std::process::Output) -> Value {
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|e| panic!("no JSON envelope ({e}): {output:?}"))
}

fn stderr(output: &std::process::Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// The late policy the synthetic late fixture serves: 1000/1 windows, 500/0 passages.
fn run_args<'a>(database: &'a str, operation: &'a str, passage: &'a str) -> Vec<&'a str> {
    vec![
        "run",
        "--database",
        database,
        "--provider",
        "fixture-provider-2",
        "--operation-id",
        operation,
        "--generation",
        "g1",
        "--chunking",
        "late",
        "--context-cap-tokens",
        "1000",
        "--window-overlap-tokens",
        "1",
        "--passage-tokens",
        passage,
        "--overlap-tokens",
        "0",
    ]
}

#[tokio::test]
async fn run_retry_inspect_diagnose_and_refusals() {
    with_temp_db("cli_setup_lifecycle", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let peer = Peer::new(&Fixture::late());
        let root = ledger();
        let directory = tempfile::tempdir().unwrap();
        let socket = PathBuf::from(pool.writer().socket_path().unwrap());
        let cli = Cli {
            config: configuration(directory.path(), &socket, root.path(), &peer),
            socket,
        };
        let database = pool.database().to_string();

        // 1. Create: a Ready receipt in a successful envelope, exit 0.
        let first = cli.run(&run_args(&database, "setup-one", "500")).await;
        assert!(first.status.success(), "{}", stderr(&first));
        let created = envelope(&first);
        assert_eq!(created["success"], true);
        assert_eq!(created["command"], "db.setup.run");
        assert_eq!(created["data"]["status"], "ready");
        assert_eq!(created["data"]["step"], "complete");
        assert_eq!(created["data"]["database"], database.as_str());
        let receipt = created["data"].clone();

        // The database holds exactly the five setup collections and the profile rows.
        let mut names: Vec<_> = crud::list_collections(&pool, true)
            .await
            .unwrap()
            .into_iter()
            .map(|c| c.name)
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "chunks",
                "documents",
                "embeddings",
                "hades_setup",
                "hades_workspace"
            ]
        );
        assert_eq!(
            pool.writer()
                .get("collection/hades_workspace/count")
                .await
                .unwrap()["count"],
            5
        );
        let store = ProfileStore::new(pool.clone()).unwrap();
        let saved = store.load().await.unwrap();
        assert_eq!(saved.profile().state, ProfileState::Ready);
        assert_eq!(saved.profile().generation, "g1");
        assert_eq!(
            saved.profile().runtime_binding.provider_id,
            "fixture-provider-2"
        );

        // 2. The identical repeat returns the same receipt and leaves the profile alone.
        let repeat = cli.run(&run_args(&database, "setup-one", "500")).await;
        assert!(repeat.status.success(), "{}", stderr(&repeat));
        assert_eq!(envelope(&repeat)["data"], receipt);
        assert_eq!(store.load().await.unwrap(), saved);

        // 3. A changed request under the same operation id is refused, pointing at
        //    inspect, and the Ready receipt is not downgraded.
        let mut generation = run_args(&database, "setup-one", "500");
        generation[8] = "g2";
        let other = format!("{database}_other");
        // A refusal must not touch the ledger either: not the Ready record, and no
        // record for the refused request.
        let ledger_before_refusals = snapshot(root.path());
        for args in [
            run_args(&database, "setup-one", "400"),
            generation,
            run_args(&other, "setup-one", "500"),
        ] {
            let refused = cli.run(&args).await;
            assert!(!refused.status.success(), "{args:?}");
            // The stored receipt belongs to another request; printing it as this
            // run's result would report the wrong database as ready.
            assert!(refused.stdout.is_empty(), "{args:?}: {refused:?}");
            assert!(
                stderr(&refused).contains("different request"),
                "{args:?}: {}",
                stderr(&refused)
            );
            assert!(
                stderr(&refused).contains("hades db setup inspect --operation-id setup-one"),
                "{args:?}: {}",
                stderr(&refused)
            );
        }
        assert_eq!(store.load().await.unwrap(), saved);
        assert_eq!(snapshot(root.path()), ledger_before_refusals);

        // 4. inspect returns the original receipt; diagnose reports none/ready and
        //    writes nothing to the ledger.
        let inspected = cli.run(&["inspect", "--operation-id", "setup-one"]).await;
        assert!(inspected.status.success(), "{}", stderr(&inspected));
        let inspected = envelope(&inspected);
        assert_eq!(inspected["success"], true);
        assert_eq!(inspected["data"]["operation_id"], "setup-one");
        assert_eq!(inspected["data"]["receipt"], receipt);
        let before = snapshot(root.path());
        let diagnosed = cli.run(&["diagnose", "--operation-id", "setup-one"]).await;
        assert!(diagnosed.status.success(), "{}", stderr(&diagnosed));
        let report = envelope(&diagnosed);
        assert_eq!(report["success"], true);
        assert_eq!(report["data"]["action"], "none");
        assert_eq!(report["data"]["reason"], "ready");
        assert_eq!(report["data"]["receipt"], receipt);
        assert_eq!(snapshot(root.path()), before);
        // An unknown operation in an initialized ledger: as inspect, a failed
        // envelope and exit 1, still carrying the advisory report.
        let unknown = cli
            .run(&["diagnose", "--operation-id", "setup-unknown"])
            .await;
        assert_eq!(unknown.status.code(), Some(1), "{unknown:?}");
        let unknown = envelope(&unknown);
        assert_eq!(unknown["success"], false);
        assert_eq!(unknown["data"]["reason"], "operation_missing");
        assert!(unknown["data"]["receipt"].is_null());
        assert_eq!(snapshot(root.path()), before);

        // 5. A conventional policy against the late provider stops before verification
        //    and records no operation.
        let observations = peer.observations();
        let mismatch = cli
            .run(&[
                "run",
                "--database",
                &database,
                "--provider",
                "fixture-provider-2",
                "--operation-id",
                "setup-two",
                "--generation",
                "g1",
                "--chunking",
                "conventional",
                "--passage-tokens",
                "500",
                "--overlap-tokens",
                "0",
            ])
            .await;
        assert!(!mismatch.status.success());
        assert!(
            stderr(&mismatch).contains("serves the late space only"),
            "{}",
            stderr(&mismatch)
        );
        assert_eq!(peer.observations(), observations);
        assert_eq!(snapshot(root.path()), before);

        // 6. hades status reports the verified provider from GET /v1/status alone: no
        //    observation, no paths, and the legacy embedder block is still present.
        let status = cli.raw(&["--db", &database, "status"]).await;
        assert!(status.status.success(), "{}", stderr(&status));
        let status = envelope(&status);
        assert_eq!(
            status["data"]["verified_providers"],
            serde_json::json!([{"id":"fixture-provider-2","model":"fixture/model-2",
                "space":"late","loaded":true}])
        );
        assert_eq!(status["data"]["embedder"]["status"], "unavailable");
        assert_eq!(peer.observations(), observations);
    })
    .await;
}

/// A failed checkpoint is a failed envelope that still shows the receipt, and the
/// identical retry through the same endpoint resumes it to Ready.
#[tokio::test]
async fn failed_setup_prints_its_receipt_and_the_identical_retry_resumes() {
    with_temp_db("cli_setup_failure", Fixtures::Empty, |pool| async move {
        remove_fixture_database(&pool).await;
        let peer = Peer::new(&Fixture::late());
        let root = ledger();
        let directory = tempfile::tempdir().unwrap();
        // Lose the acknowledgement of the metadata collection's creation once; the
        // proxy forwards everything else to the disposable server.
        let proxy = Proxy::new(
            pool.writer(),
            Rule {
                method: "POST",
                path: "collection",
                name: Some("hades_workspace"),
                nth: 1,
                before: false,
                hold: false,
                fail: true,
                callback: None,
            },
        );
        let socket = PathBuf::from(proxy.admin.socket_path().unwrap());
        let cli = Cli {
            config: configuration(directory.path(), &socket, root.path(), &peer),
            socket,
        };
        let database = pool.database().to_string();

        let failed = cli.run(&run_args(&database, "setup-one", "500")).await;
        assert_eq!(failed.status.code(), Some(1), "{failed:?}");
        assert!(
            stderr(&failed).contains(
                "setup did not complete; inspect it with: hades db setup inspect \
                 --operation-id setup-one"
            ),
            "{}",
            stderr(&failed)
        );
        let shown = envelope(&failed);
        assert_eq!(shown["success"], false);
        assert_eq!(shown["command"], "db.setup.run");
        assert_eq!(shown["data"]["status"], "failed");
        assert_eq!(shown["data"]["step"], "profile");
        assert_eq!(proxy.hits.load(Ordering::SeqCst), 1);

        let inspected = cli.run(&["inspect", "--operation-id", "setup-one"]).await;
        assert!(inspected.status.success(), "{}", stderr(&inspected));
        assert_eq!(envelope(&inspected)["data"]["receipt"], shown["data"]);
        let diagnosed = cli.run(&["diagnose", "--operation-id", "setup-one"]).await;
        assert_eq!(
            envelope(&diagnosed)["data"]["action"],
            "retry_identical_request"
        );

        let resumed = cli.run(&run_args(&database, "setup-one", "500")).await;
        assert!(resumed.status.success(), "{}", stderr(&resumed));
        let resumed = envelope(&resumed);
        assert_eq!(resumed["data"]["status"], "ready");
        assert_eq!(
            resumed["data"]["request_sha256"],
            shown["data"]["request_sha256"]
        );
    })
    .await;
}
