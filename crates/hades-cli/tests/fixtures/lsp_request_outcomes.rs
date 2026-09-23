#[tokio::test]
async fn semantic_request_failures_are_visible_and_preserve_stored_edges() {
    with_temp_db("semantic_requests", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        for go in [false, true] {
            let tree = tempfile::tempdir().unwrap();
            let root = tree.path();
            let source = root.join(if go { "main.go" } else { "lib.rs" });
            if go {
                std::fs::write(root.join("go.mod"), "module fixture\ngo 1.22\n").unwrap();
                std::fs::write(&source, "package fixture\nfunc caller() { target() }\nfunc target() {}\n").unwrap();
            } else {
                std::fs::write(root.join("Cargo.toml"), "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[lib]\npath=\"lib.rs\"\n").unwrap();
                std::fs::write(&source, "fn caller() { target(); }\nfn target() {}\n").unwrap();
            }
            let peer = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../hades-core/tests/fixtures/lsp_requests.py");
            let analyzer_env = if go { "HADES_GOPLS_PATH" } else { "HADES_RUST_ANALYZER_PATH" };
            let stats_key = if go { "gopls" } else { "rust_analyzer" };
            let mut stored = Value::Null;
            // Seed a real edge first, then exercise failure, acceptance, recovery,
            // and genuine no-calls through the same production ingest front door.
            for (mode, accept) in [("clean", false), ("empty-prepare", false), ("error", false), ("timeout", false), ("empty-prepare", true), ("recover", false), ("no-calls", false)] {
                if mode == "no-calls" {
                    std::fs::write(&source, if go { "package fixture\nfunc caller() {}\nfunc target() {}\n" } else { "fn caller() {}\nfn target() {}\n" }).unwrap();
                }
                std::fs::write(root.join(".lsp-mode"), mode).unwrap();
                let mut args = vec!["codebase", "ingest", root.to_str().unwrap(), "--force"];
                if accept { args.push("--allow-analysis-downgrade"); }
                let output = tokio::time::timeout(std::time::Duration::from_secs(90),
                    cli_command(&pool, &embedder, &args).env(analyzer_env, &peer).output()).await.expect("private analyzer exceeded its two request deadlines").unwrap();
                let report: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|_| panic!("{output:?}"));
                let failed = matches!(mode, "empty-prepare" | "error" | "timeout");
                assert_eq!(output.status.success(), !failed || accept, "go={go} mode={mode}: {report} {}", String::from_utf8_lossy(&output.stderr));
                assert_eq!(report["success"], !failed || accept, "{report}");
                let stats = &report["data"][stats_key];
                assert_eq!(stats["failed_request_count"], usize::from(failed), "{report}");
                if failed {
                    assert_eq!(stats["failed_requests"][0]["symbol"], "caller");
                    assert_eq!(stats["failed_requests"][0]["file"], source.file_name().unwrap().to_str().unwrap());
                    let diagnostic = String::from_utf8_lossy(&output.stderr);
                    assert!(diagnostic.contains("WARN") && diagnostic.contains("caller") && diagnostic.contains(source.file_name().unwrap().to_str().unwrap()), "{diagnostic}");
                    assert_eq!(snapshot_graph(&pool).await, stored, "failed requests must preserve all prior artifacts, even when accepted");
                } else {
                    stored = snapshot_graph(&pool).await;
                    let fkey = keys::scoped_file_key(root.to_str().unwrap(), source.file_name().unwrap().to_str().unwrap());
                    let symbols: Vec<_> = stored["codebase_symbols"].as_array().unwrap().iter().filter(|row| row["file_key"] == fkey).map(|row| row["_id"].as_str().unwrap()).collect();
                    let calls = stored["codebase_calls_edges"].as_array().unwrap().iter().filter(|row| symbols.contains(&row["_from"].as_str().unwrap())).count();
                    assert_eq!(calls, usize::from(mode != "no-calls"), "{stored}");
                }
            }
        }
    }).await;
}
