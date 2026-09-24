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
                // A failed request must not freeze otherwise valid new content.
                if matches!(mode, "empty-prepare" | "error" | "timeout") {
                    let text = std::fs::read_to_string(&source).unwrap();
                    std::fs::write(&source, format!("{text}// refreshed during {mode}\n")).unwrap();
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
                    let current = snapshot_graph(&pool).await;
                    assert_eq!(current["codebase_calls_edges"], stored["codebase_calls_edges"], "failed requests must preserve prior semantic edges, even when accepted");
                    assert!(current["codebase_chunks"].as_array().unwrap().iter().any(|c| c["text"].as_str().is_some_and(|text| text.contains(&format!("refreshed during {mode}")))), "content was withheld: {report} {current}");
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

#[tokio::test]
async fn incomplete_enrichment_refreshes_content_and_keeps_cross_file_targets() {
    with_temp_db("semantic_multi", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        for go in [false, true] {
            let tree = tempfile::tempdir().unwrap();
            let root = tree.path();
            std::fs::write(root.join(".lsp-multi"), "").unwrap();
            if go {
                std::fs::write(root.join("go.mod"), "module fixture\ngo 1.22\n").unwrap();
            } else {
                std::fs::write(root.join("Cargo.toml"), "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[lib]\npath=\"a.rs\"\n").unwrap();
            }
            for name in ["a", "b", "c"] {
                let text = if go { format!("package fixture\nfunc {name}() {{}}\n") } else { format!("fn {name}() {{}}\n{}", if name == "a" {"#[cfg(feature = \"off\")] mod b;\nmod c;\n"} else {""}) };
                std::fs::write(root.join(format!("{name}.{}", if go {"go"} else {"rs"})), text).unwrap();
            }
            let peer = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../hades-core/tests/fixtures/lsp_requests.py");
            let analyzer_env = if go { "HADES_GOPLS_PATH" } else { "HADES_RUST_ANALYZER_PATH" };
            for mode in if go { vec!["clean", "multi-error", "multi-document-error", "workspace-error", "bad-utf8"] } else { vec!["multi-cfg", "clean", "multi-error", "multi-document-error", "workspace-error", "bad-utf8"] } {
                std::fs::write(root.join(".lsp-mode"), mode).unwrap();
                if mode == "bad-utf8" { std::fs::write(root.join(if go {"bad.go"} else {"bad.rs"}), [0xff]).unwrap(); }
                let b = root.join(if go {"b.go"} else {"b.rs"});
                let text = std::fs::read_to_string(&b).unwrap();
                let text = if mode == "workspace-error" { text.replace(" b()", " renamed()") } else { text };
                std::fs::write(&b, format!("{}{text}// fresh {mode}\n", if mode == "multi-error" {"\n"} else {""})).unwrap();
                let output = cli_command(&pool, &embedder, &["codebase", "ingest", root.to_str().unwrap(), "--force"]).env(analyzer_env, &peer).output().await.unwrap();
                let report: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|_| panic!("{output:?}"));
                assert_eq!(report["success"], matches!(mode, "clean" | "multi-cfg"), "go={go} mode={mode}: {report} {}", String::from_utf8_lossy(&output.stderr));
                let graph = snapshot_graph(&pool).await;
                let mut ids = Vec::new();
                for name in ["a", "b", "c"] {
                    let key = keys::scoped_file_key(root.to_str().unwrap(), &format!("{name}.{}", if go {"go"} else {"rs"}));
                    assert!(graph["codebase_files"].as_array().unwrap().iter().any(|r| r["_key"] == key), "file missing: {graph}");
                    let symbol = graph["codebase_symbols"].as_array().unwrap().iter().find(|r| r["file_key"] == key && r["name"] == if name == "b" && matches!(mode,"workspace-error"|"bad-utf8") {"renamed"} else {name}).unwrap();
                    ids.push(symbol["_id"].clone());
                    assert!(symbol.get("enrichment_stale").is_none(), "stale endpoint must not be manufactured");
                    if name == "b" {
                        assert!(graph["codebase_chunks"].as_array().unwrap().iter().any(|r| r["file_key"] == key && r["text"].as_str().is_some_and(|s| s.contains(&format!("fresh {mode}")))), "B content withheld: {report} {graph}");
                    }
                }
                if mode == "multi-cfg" {
                    assert_eq!(report["data"]["rust_analyzer"]["no_calls"][0]["reason"], "cfg_inactive");
                    assert_eq!(report["data"]["rust_analyzer"]["no_calls"][0]["symbol"], "b");
                }
                assert_ownership_endpoints(&graph);
                if mode == "workspace-error" { continue; }
                if mode == "bad-utf8" {
                    assert!(report["data"]["results"].as_array().unwrap().iter().any(|r| r["path"] == (if go {"bad.go"} else {"bad.rs"}) && r["success"] == false), "{report}");
                    continue;
                }
                for pair in ids.windows(2).take(if mode == "multi-cfg" {1} else {2}) {
                    assert!(graph["codebase_calls_edges"].as_array().unwrap().iter().any(|e| e["_from"] == pair[0] && e["_to"] == pair[1]), "lost edge go={go} mode={mode}: {graph}");
                }
            }
        }
    }).await;
}
