// Provisioning writes land only in databases the endpoint could have created,
// and seeding never destroys a populated schema (#193).

async fn schema_rows(pool: &ArangoPool) -> Value {
    let result = hades_core::db::query::query(
        pool,
        "FOR d IN hades_schema SORT d._key RETURN d",
        None,
        None,
        false,
        hades_core::db::query::ExecutionTarget::Writer,
    )
    .await
    .unwrap();
    json!(result.results)
}

/// Drop a database this test created through MCP `create_database`, which
/// `with_temp_db` does not own. Uses the harness's own credentials.
async fn drop_created_database(name: &str) {
    let socket = std::env::var_os("ARANGO_SOCKET").unwrap();
    let password = std::env::var("ARANGO_PASSWORD").unwrap();
    let user = std::env::var("HADES_TEST_USER").unwrap_or_else(|_| "root".to_string());
    let sys = hades_core::db::ArangoClient::with_socket(socket.into(), "_system", &user, &password);
    if let Err(e) = sys.delete(&format!("database/{name}")).await {
        eprintln!("warning: failed to drop created database '{name}': {e}");
    }
}

fn refused_by_prefix(response: &Value, database: &str, prefix: &str) {
    assert_eq!(response["success"], false, "{response}");
    assert_eq!(response["error_code"], "ACCESS_DENIED", "{response}");
    let error = response["error"].as_str().unwrap();
    assert!(error.contains(&format!("'{database}'")), "{error}");
    assert!(error.contains("matches no provisioning prefix"), "{error}");
    assert!(error.contains(prefix), "{error}");
}

#[tokio::test]
async fn mcp_provisioning_writes_only_into_prefixed_databases() {
    // `served` stands in for production: the endpoint's default database, also
    // listed in --mcp-dbs for reading, and matching no provisioning prefix.
    with_temp_db("prov_served", Fixtures::Codebase, |served| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path().join("source");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("fixture.py"), "def target():\n    return 'quartz_prov'\n")
            .unwrap();

        // A live schema to protect, seeded the ordinary way, plus a row of its
        // own so a wipe followed by a reseed could not pass as unchanged.
        cli(&served, &embedder, &["db", "schema", "init", "--seed", "empty"], true).await;
        hades_core::db::crud::insert_document(
            &served,
            "hades_schema",
            &json!({"_key":"production_marker","kind":"edge_definition"}),
        )
        .await
        .unwrap();
        let before = schema_rows(&served).await;

        let prefix = format!("{}_new_", served.database());
        let fresh = format!("{prefix}graph");
        let token = tree.path().join("token");
        std::fs::write(&token, "private-fixture-token\n").unwrap();
        let socket = tree.path().join("daemon.sock");
        let log = tree.path().join("daemon.log");
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let child = cli_command(
            &served,
            &embedder,
            &[
                "daemon",
                "--socket",
                socket.to_str().unwrap(),
                "--mcp-bind",
                &address.to_string(),
                "--mcp-token-file",
                token.to_str().unwrap(),
                "--mcp-dbs",
                served.database(),
                "--mcp-db-prefix",
                &prefix,
                "--mcp-ingest-root",
                tree.path().to_str().unwrap(),
            ],
        )
        .env("HADES_USE_GPU", "false")
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("TOKIO_WORKER_THREADS", "2")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::fs::File::create(&log).unwrap())
        .spawn()
        .unwrap();
        let mut daemon = PrivateDaemon {
            child,
            ingests: Vec::new(),
        };
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while tokio::net::UnixStream::connect(&socket).await.is_err() {
                assert!(
                    daemon.child.try_wait().unwrap().is_none(),
                    "{}",
                    std::fs::read_to_string(&log).unwrap()
                );
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let mcp = PrivateMcp::connect(address).await;

        // The served database, named and by default, is refused for both
        // provisioning writes, and its schema is untouched.
        let path = root.to_str().unwrap();
        for db in [Some(served.database()), None] {
            let mut seed = json!({"seed":"empty"});
            let mut start = json!({"path":path});
            if let Some(db) = db {
                seed["db"] = json!(db);
                start["db"] = json!(db);
            }
            refused_by_prefix(&mcp.call_tool("db_schema_init", seed).await, served.database(), &prefix);
            refused_by_prefix(&mcp.call_tool("ingest_start", start).await, served.database(), &prefix);
        }
        assert_eq!(schema_rows(&served).await, before, "refused seeding changed the schema");
        assert_eq!(
            hades_core::db::crud::count_collection(&served, "codebase_files").await.unwrap(),
            0,
            "refused ingest wrote into the served database"
        );

        // The first-time path into a database this endpoint may create.
        let outcome = tokio::spawn({
            let fresh = fresh.clone();
            async move {
                let created = mcp.call_tool("create_database", json!({"name":fresh})).await;
                assert_eq!(created["success"], true, "{created}");
                let seeded = mcp
                    .call_tool("db_schema_init", json!({"db":fresh,"seed":"empty"}))
                    .await;
                assert_eq!(seeded["success"], true, "{seeded}");
                // Now populated, so a second seed is refused rather than a wipe.
                let again = mcp
                    .call_tool("db_schema_init", json!({"db":fresh,"seed":"empty"}))
                    .await;
                assert_eq!(again["success"], false, "{again}");
                assert_eq!(again["error_code"], "CONFLICT", "{again}");

                let started = mcp
                    .call_tool("ingest_start", json!({"db":fresh,"path":path_owned(&root)}))
                    .await;
                assert_eq!(started["success"], true, "{started}");
                let job = started["data"]["job_id"].as_str().unwrap().to_owned();
                let status = tokio::time::timeout(std::time::Duration::from_secs(60), async {
                    loop {
                        let status = mcp
                            .call_tool("ingest_status", json!({"db":fresh,"job_id":job}))
                            .await;
                        match status["data"]["status"].as_str() {
                            Some("completed" | "failed" | "recovery_required") => break status,
                            _ => tokio::time::sleep(std::time::Duration::from_millis(100)).await,
                        }
                    }
                })
                .await
                .expect("provisioned ingest did not finish");
                assert_eq!(status["data"]["status"], "completed", "{status}");

                let found = mcp
                    .call_tool(
                        "db_query",
                        json!({"db":fresh,"text":"quartz","collection":"codebase"}),
                    )
                    .await;
                assert_eq!(found["success"], true, "{found}");
                assert!(
                    found["data"]["results"][0]["text"]
                        .as_str()
                        .unwrap_or_default()
                        .contains("quartz_prov"),
                    "{found}"
                );
            }
        })
        .await;
        drop_created_database(&fresh).await;
        let _ = daemon.child.start_kill();
        if let Err(panic) = outcome {
            std::panic::resume_unwind(panic.into_panic());
        }
        assert_eq!(schema_rows(&served).await, before);
    })
    .await;
}

fn path_owned(path: &Path) -> String {
    path.to_str().unwrap().to_owned()
}

#[tokio::test]
async fn cli_schema_init_replaces_a_populated_schema_only_with_force() {
    with_temp_db("schema_force", Fixtures::Empty, |pool| async move {
        let embedder = Embedder::new().await;
        // Missing collection: seeded without --force.
        let first = cli(&pool, &embedder, &["db", "schema", "init", "--seed", "empty"], true).await;
        assert_eq!(first["documents_replaced"], 0, "{first}");
        hades_core::db::crud::insert_document(
            &pool,
            "hades_schema",
            &json!({"_key":"authored_marker","kind":"edge_definition"}),
        )
        .await
        .unwrap();
        let before = schema_rows(&pool).await;

        let output = cli_command(&pool, &embedder, &["db", "schema", "init", "--seed", "empty"])
            .output()
            .await
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "populated schema was reseeded: {stderr}");
        assert!(stderr.contains("already holds"), "{stderr}");
        assert!(stderr.contains("--force"), "{stderr}");
        assert_eq!(schema_rows(&pool).await, before, "refused seed changed the schema");

        let forced = cli(
            &pool,
            &embedder,
            &["db", "schema", "init", "--seed", "empty", "--force"],
            true,
        )
        .await;
        assert_eq!(forced["documents_replaced"], before.as_array().unwrap().len(), "{forced}");
        let after = schema_rows(&pool).await;
        assert!(
            !after.to_string().contains("authored_marker"),
            "--force kept the replaced schema: {after}"
        );
    })
    .await;

    // An existing but empty collection is seeded without --force too.
    with_temp_db("schema_empty", Fixtures::Empty, |pool| async move {
        let embedder = Embedder::new().await;
        hades_core::db::crud::create_collection(&pool, "hades_schema", None)
            .await
            .unwrap();
        let seeded = cli(&pool, &embedder, &["db", "schema", "init", "--seed", "empty"], true).await;
        assert!(seeded["documents_written"].as_u64().unwrap() > 0, "{seeded}");
    })
    .await;
}
