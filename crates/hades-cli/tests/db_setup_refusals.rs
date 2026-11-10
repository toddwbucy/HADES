//! `hades db setup` refusals that need no database, no model and no registry (#225).
//! The provider is a private stub that answers only `GET /v1/status`, and the database
//! socket is a listener that must never be contacted. Each refusal is checked by its
//! message and exit code, and a space mismatch must stop before `POST /v1/verify`.
#![cfg(unix)]

use serde_json::{Value, json};
use std::{
    io::{BufRead, BufReader, Write},
    os::unix::net::UnixListener,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

/// A private Unix listener that records each request line and answers with `reply`.
struct Stub {
    path: PathBuf,
    contacts: Arc<Mutex<Vec<String>>>,
}

impl Stub {
    fn new(path: PathBuf, reply: Option<Value>) -> Self {
        let listener = UnixListener::bind(&path).unwrap();
        let contacts = Arc::new(Mutex::new(Vec::new()));
        let recorded = contacts.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                let _ = reader.read_line(&mut line);
                recorded.lock().unwrap().push(line.trim_end().to_string());
                loop {
                    let mut header = String::new();
                    if reader.read_line(&mut header).unwrap_or(0) == 0 || header == "\r\n" {
                        break;
                    }
                }
                if let Some(reply) = &reply {
                    let body = reply.to_string();
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
            }
        });
        Self { path, contacts }
    }

    fn contacts(&self) -> Vec<String> {
        self.contacts.lock().unwrap().clone()
    }
}

/// One isolated installation: private ledger, registry root, provider and database sockets.
struct Installation {
    root: tempfile::TempDir,
    provider: Stub,
    database: Stub,
}

impl Installation {
    fn new(space: &str) -> Self {
        let root = tempfile::tempdir().unwrap();
        for directory in ["ledger", "registry", "provider", "db"] {
            std::fs::create_dir(root.path().join(directory)).unwrap();
            set_mode(&root.path().join(directory), 0o700);
        }
        set_mode(root.path(), 0o700);
        let provider = Stub::new(
            root.path().join("provider/provider.sock"),
            Some(json!({"model":"fixture/model","space":space,"loaded":true})),
        );
        let database = Stub::new(root.path().join("db/arangodb.sock"), None);
        Self {
            root,
            provider,
            database,
        }
    }

    fn ledger(&self) -> PathBuf {
        self.root.path().join("ledger")
    }

    fn config(&self, setup: bool, providers: bool) -> PathBuf {
        let mut config = json!({"database":{"username":"fixture",
            "sockets":{"readonly":self.database.path,"readwrite":self.database.path}}});
        if setup {
            config["setup"] =
                json!({"ledger_root":self.ledger(),"installation_id":"fixture-installation"});
        }
        if providers {
            config["registered_providers"] = json!({"registry_root":self.root.path().join("registry"),
                "providers":[{"id":"local-jina","socket":self.provider.path}]});
        }
        let path = self.root.path().join("hades.json");
        std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
        path
    }

    async fn run(&self, config: &Path, args: &[&str]) -> std::process::Output {
        let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_hades"));
        command
            .env_clear()
            .env("HOME", self.root.path())
            .env("PATH", "/usr/bin:/bin")
            .env("HADES_CONFIG", config)
            .env("ARANGO_SOCKET", &self.database.path)
            .env("ARANGO_RO_SOCKET", &self.database.path)
            .env("ARANGO_RW_SOCKET", &self.database.path)
            .env("ARANGO_PASSWORD", "fixture-only")
            .env("TOKIO_WORKER_THREADS", "2")
            .current_dir(self.root.path())
            .args(["db", "setup"])
            .args(args)
            .kill_on_drop(true);
        tokio::time::timeout(Duration::from_secs(10), command.output())
            .await
            .expect("private CLI exceeded deadline")
            .unwrap()
    }
}

fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

const RUN: &[&str] = &[
    "run",
    "--database",
    "managed",
    "--operation-id",
    "setup-one",
    "--generation",
    "g1",
];
const LATE: &[&str] = &[
    "--chunking",
    "late",
    "--context-cap-tokens",
    "16384",
    "--window-overlap-tokens",
    "1000",
    "--passage-tokens",
    "500",
    "--overlap-tokens",
    "200",
];
const CONVENTIONAL: &[&str] = &[
    "--chunking",
    "conventional",
    "--passage-tokens",
    "500",
    "--overlap-tokens",
    "200",
];

/// Exit 1 with `needle` on stderr, and no ledger, provider or database contact.
fn refused(installation: &Installation, output: &std::process::Output, needle: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(stderr.contains(needle), "missing {needle:?} in {stderr}");
    assert!(installation.database.contacts().is_empty());
    assert_eq!(
        std::fs::read_dir(installation.ledger()).unwrap().count(),
        0,
        "a refusal must leave the ledger untouched"
    );
}

fn run_args<'a>(provider: &'a str, policy: &[&'a str]) -> Vec<&'a str> {
    [RUN, &["--provider", provider], policy].concat()
}

#[tokio::test]
async fn refuses_without_setup_section() {
    let installation = Installation::new("late");
    let config = installation.config(false, true);
    let output = installation
        .run(&config, &run_args("local-jina", LATE))
        .await;
    refused(&installation, &output, "requires the setup section");
    assert!(installation.provider.contacts().is_empty());
}

#[tokio::test]
async fn refuses_without_registered_providers() {
    let installation = Installation::new("late");
    let config = installation.config(true, false);
    let output = installation
        .run(&config, &run_args("local-jina", LATE))
        .await;
    refused(&installation, &output, "requires registered_providers");
    assert!(installation.provider.contacts().is_empty());
}

#[tokio::test]
async fn refuses_an_unknown_provider() {
    let installation = Installation::new("late");
    let config = installation.config(true, true);
    let output = installation.run(&config, &run_args("other", LATE)).await;
    refused(&installation, &output, "not in the installation allowlist");
    assert!(installation.provider.contacts().is_empty());
}

/// The space check reads only `GET /v1/status`; the model is never verified or loaded.
#[tokio::test]
async fn refuses_a_space_mismatch_before_verification() {
    for (space, policy, needle) in [
        (
            "late",
            CONVENTIONAL,
            "provider local-jina serves the late space only; use --chunking late or a conventional provider",
        ),
        (
            "conventional",
            LATE,
            "provider local-jina does not advertise late chunking; use --chunking conventional",
        ),
    ] {
        let installation = Installation::new(space);
        let config = installation.config(true, true);
        let output = installation
            .run(&config, &run_args("local-jina", policy))
            .await;
        refused(&installation, &output, needle);
        assert_eq!(
            installation.provider.contacts(),
            ["GET /v1/status HTTP/1.1"]
        );
    }
}

/// Inspect reads only the ledger: an unknown operation is a failed envelope with a null
/// receipt, and neither the provider nor the database is contacted.
#[tokio::test]
async fn inspect_unknown_operation_reports_null_receipt() {
    let installation = Installation::new("late");
    let config = installation.config(true, true);
    let output = installation
        .run(&config, &["inspect", "--operation-id", "setup-one"])
        .await;
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["success"], false);
    assert_eq!(envelope["command"], "db.setup.inspect");
    assert_eq!(envelope["data"]["operation_id"], "setup-one");
    assert!(envelope["data"]["receipt"].is_null());
    assert!(installation.provider.contacts().is_empty());
    assert!(installation.database.contacts().is_empty());
}
