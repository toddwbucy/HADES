// Drift reads the raw-text extensions a root was ingested with from the graph,
// so it does not report those files stale when `--unparsed-ext` is omitted (#164).

#[tokio::test]
async fn drift_recognizes_raw_text_rows_without_the_ingest_flag() {
    with_temp_db("drift_unparsed", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::for_task("mixed").await;
        let tree = tempfile::tempdir().unwrap();
        // Keys are scoped by the canonical ingest root.
        let root = tree.path().canonicalize().unwrap().to_str().unwrap().to_owned();
        std::fs::write(tree.path().join("app.py"), "def target():\n    return 'quartz'\n").unwrap();
        std::fs::write(tree.path().join("Config.TOML"), "[fixture]\nname = \"quartz\"\n").unwrap();
        std::fs::write(tree.path().join("extra.toml"), "[extra]\nvalue = 1\n").unwrap();
        cli(&pool, &embedder, &["codebase", "ingest", &root, "--unparsed-ext", "toml"], true).await;
        let config_key = hades_core::db::keys::scoped_file_key(&root, "Config.TOML");

        // Without the flag: every raw-text row is found on disk.
        let clean = cli(&pool, &embedder, &["codebase", "drift", &root, "--full"], true).await;
        assert_eq!(clean["stale"]["count"], 0, "{clean}");
        assert_eq!(clean["uningested"]["count"], 0, "{clean}");
        assert_eq!(clean["changed"]["count"], 0, "{clean}");
        assert_eq!(clean["matched"], 3, "{clean}");
        assert_eq!(clean["clean"], true, "{clean}");

        // A genuinely changed raw-text file is still reported.
        std::fs::write(tree.path().join("Config.TOML"), "[fixture]\nname = \"sapphire\"\n").unwrap();
        let changed = cli(&pool, &embedder, &["codebase", "drift", &root, "--full"], true).await;
        assert_eq!(changed["changed"]["keys"], json!([config_key]), "{changed}");
        assert_eq!(changed["stale"]["count"], 0, "{changed}");

        // And a deleted one is stale, not silently dropped from the comparison.
        std::fs::remove_file(tree.path().join("extra.toml")).unwrap();
        let stale = cli(&pool, &embedder, &["codebase", "drift", &root, "--full"], true).await;
        assert_eq!(
            stale["stale"]["keys"],
            json!([hades_core::db::keys::scoped_file_key(&root, "extra.toml")]),
            "{stale}"
        );
    })
    .await;
}
