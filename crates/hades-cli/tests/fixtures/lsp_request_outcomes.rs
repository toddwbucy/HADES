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
                assert_eq!(stats["failed_requests_truncated"], false, "{report}");
                if failed {
                    if !accept { assert!(String::from_utf8_lossy(&output.stderr).contains("--allow-analysis-downgrade"), "{output:?}"); }
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


// Exercise CLI, daemon and MCP with the same failed semantic request (#179).
#[tokio::test]
async fn unified_and_remote_degraded_enrichment_is_explicit() {
    with_temp_db("degraded_override", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path().join("source");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[lib]\npath=\"lib.rs\"\n").unwrap();
        std::fs::write(root.join("lib.rs"), "fn caller() { target(); }\nfn target() {}\n").unwrap();
        std::fs::write(root.join(".lsp-mode"), "error").unwrap();
        let peer = Path::new(env!("CARGO_MANIFEST_DIR")).join("../hades-core/tests/fixtures/lsp_requests.py");
        for accept in [false, true] {
            let mut args = vec!["ingest", root.to_str().unwrap(), "--force"];
            if accept { args.push("--allow-degraded-enrichment"); }
            let output = cli_command(&pool, &embedder, &args).env("HADES_RUST_ANALYZER_PATH", &peer).output().await.unwrap();
            let report: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|_| panic!("{output:?}"));
            assert_eq!(output.status.success(), accept, "{report}");
            assert_degraded_result(&report, accept);
            if !accept { assert!(String::from_utf8_lossy(&output.stderr).contains("--allow-degraded-enrichment"), "{output:?}"); }
        }
        let socket = tree.path().join("daemon.sock");
        let token = tree.path().join("token");
        std::fs::write(&token, "private-fixture-token").unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let child = cli_command(&pool, &embedder, &["daemon", "--socket", socket.to_str().unwrap(),
            "--mcp-bind", &address.to_string(), "--mcp-token-file", token.to_str().unwrap(),
            "--mcp-db-prefix", pool.database(), "--mcp-ingest-root", root.to_str().unwrap()])
            .env("HADES_RUST_ANALYZER_PATH", &peer).env("HADES_USE_GPU", "false")
            .env("CUDA_VISIBLE_DEVICES", "").env("TOKIO_WORKER_THREADS", "2")
            .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null())
            .stderr(std::fs::File::create(tree.path().join("daemon.log")).unwrap()).spawn().unwrap();
        let mut daemon = PrivateDaemon { child, ingests: Vec::new() };
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while tokio::net::UnixStream::connect(&socket).await.is_err() {
                assert!(daemon.child.try_wait().unwrap().is_none());
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }).await.unwrap();
        let mcp = PrivateMcp::connect(address).await;
        let listed = PrivateMcp::message(mcp.post(json!({"jsonrpc":"2.0","id":900,"method":"tools/list"})).await, 900).await;
        let tool = listed["result"]["tools"].as_array().unwrap().iter().find(|t| t["name"] == "ingest_start").unwrap();
        assert!(tool["inputSchema"]["properties"].get("allow_degraded_enrichment").is_some(), "{tool}");
        for remote in [false, true] {
            for accept in [false, true] {
                let mut params = json!({"path":root,"force":true});
                // Omission exercises the backwards-compatible default.
                if accept { params["allow_degraded_enrichment"] = json!(true); }
                let parsed: hades_core::dispatch::IngestStartParams = serde_json::from_value(params.clone()).unwrap();
                assert_eq!(hades_core::dispatch::DaemonCommand::IngestStart(parsed).access_tier(), hades_core::dispatch::AccessTier::Provisioning);
                let started = if remote {
                    params["db"] = json!(pool.database());
                    mcp.call_tool("ingest_start", params).await
                } else {
                    daemon_request(&socket, json!({"command":"ingest.start","params":params})).await
                };
                assert_eq!(started["success"], true, "{started}");
                let job = started["data"]["job_id"].as_str().unwrap();
                daemon.ingests.push((started["data"]["pid"].as_u64().unwrap() as u32, root.clone()));
                let row = tokio::time::timeout(std::time::Duration::from_secs(60), async {
                    loop {
                        let status = mcp.call_tool("ingest_status", json!({"db":pool.database(),"job_id":job})).await;
                        assert_eq!(status["success"], true, "{status}");
                        let row = status["data"].clone();
                        if matches!(row["status"].as_str(), Some("completed" | "failed")) { break row; }
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    }
                }).await.unwrap();
                assert_eq!(row["status"], if accept {"completed"} else {"failed"}, "{row}");
                assert_eq!(row["allow_degraded_enrichment"], accept);
                assert_degraded_result(&row["result"], accept);
                if !accept { assert!(row["detail"].as_str().unwrap().contains("allow_degraded_enrichment"), "{row}"); }
                let stored = hades_core::db::crud::get_document(&pool, "hades_ingest_jobs", job).await.unwrap();
                assert_eq!(stored["result"], row["result"]);
            }
        }
        unsafe { libc::kill(daemon.child.id().unwrap() as i32, libc::SIGTERM); }
        assert!(daemon.child.wait().await.unwrap().success());
    }).await;
}

fn assert_degraded_result(report: &Value, accept: bool) {
    assert_eq!(report["success"], accept, "{report}");
    let data = &report["data"];
    assert_eq!(data["enrichment_degraded"], true, "{report}");
    assert_eq!(data["failed_request_count"], 1);
    assert_eq!(data["failed_requests_truncated"], false);
    assert!(data["code"]["rust_analyzer"].get("failed_requests").is_none());
    assert_eq!(data["code"]["rust_analyzer"]["failed_request_count"], 1);
    let failures = data["failed_requests"].as_array().unwrap();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0]["file"], "lib.rs");
    assert_eq!(failures[0]["symbol"], "caller");
    assert_eq!(failures[0]["request"], "callHierarchy/outgoingCalls");
    assert!(!failures[0]["reason"].as_str().unwrap().is_empty());
}


// Diagnostics alone must not overflow an accepted MCP job (#185, #179).
#[tokio::test]
async fn mcp_accepts_ten_thousand_failures_with_bounded_result() {
    with_temp_db("bounded_degraded", Fixtures::Codebase, |pool| async move {
        let embedder = Embedder::new().await;
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path().join("source");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[lib]\npath=\"lib.rs\"\n").unwrap();
        // Small files stay below both graph and enrichment transaction budgets
        // while exercising 10,000 real request failures in one job (#185).
        for file in 0..20 {
            let name = if file == 0 { "lib.rs".to_owned() } else { format!("part{file}.rs") };
            std::fs::write(root.join(name), (0..500).map(|n| format!("fn caller{n}() {{ target(); }} // fixture padding\n")).collect::<String>()).unwrap();
        }
        std::fs::write(root.join(".lsp-mode"), "ten-thousand-error").unwrap();
        let peer = Path::new(env!("CARGO_MANIFEST_DIR")).join("../hades-core/tests/fixtures/lsp_requests.py");
        let socket = tree.path().join("daemon.sock");
        let token = tree.path().join("token");
        std::fs::write(&token, "private-fixture-token").unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let child = cli_command(&pool, &embedder, &["daemon", "--socket", socket.to_str().unwrap(),
            "--mcp-bind", &address.to_string(), "--mcp-token-file", token.to_str().unwrap(),
            "--mcp-db-prefix", pool.database(), "--mcp-ingest-root", root.to_str().unwrap()])
            // Embedding is optional for code. This regression needs no vectors.
            .env("HADES_EMBEDDER_SOCKET", tree.path().join("absent-embedder.sock"))
            .env("HADES_RUST_ANALYZER_PATH", &peer).env("HADES_USE_GPU", "false")
            .env("CUDA_VISIBLE_DEVICES", "").env("TOKIO_WORKER_THREADS", "2")
            .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null())
            .stderr(std::fs::File::create(tree.path().join("daemon.log")).unwrap()).spawn().unwrap();
        let mut daemon = PrivateDaemon { child, ingests: Vec::new() };
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while tokio::net::UnixStream::connect(&socket).await.is_err() {
                assert!(daemon.child.try_wait().unwrap().is_none());
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }).await.unwrap();
        let mcp = PrivateMcp::connect(address).await;
        let started = mcp.call_tool("ingest_start", json!({"db":pool.database(), "path":root, "force":true, "allow_degraded_enrichment":true})).await;
        assert_eq!(started["success"], true, "{started}");
        let job = started["data"]["job_id"].as_str().unwrap();
        daemon.ingests.push((started["data"]["pid"].as_u64().unwrap() as u32, root.clone()));
        let row = tokio::time::timeout(std::time::Duration::from_secs(240), async {
            loop {
                let status = mcp.call_tool("ingest_status", json!({"db":pool.database(), "job_id":job})).await;
                assert_eq!(status["success"], true, "{status}");
                let row = status["data"].clone();
                if matches!(row["status"].as_str(), Some("completed" | "failed")) { break row; }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }).await.unwrap();
        assert_eq!(row["status"], "completed", "detail={} code_results={}",
            row["detail"].as_str().unwrap_or("").chars().rev().take(2000).collect::<String>().chars().rev().collect::<String>(),
            row["result"]["data"]["code"]["results"]);
        assert_eq!(row["result"]["success"], true);
        let data = &row["result"]["data"];
        assert_eq!(data["enrichment_degraded"], true);
        assert_eq!(data["failed_request_count"], 10_000);
        assert_eq!(data["failed_requests_truncated"], true);
        assert!(!data["failed_requests"].as_array().unwrap().is_empty());
        assert!(data["failed_requests"].as_array().unwrap().len() <= 100);
        assert!(serde_json::to_vec(&data["failed_requests"]).unwrap().len() <= 64 * 1024);
        for analyzer in ["rust_analyzer", "gopls"] {
            assert!(data["code"][analyzer].get("failed_requests").is_none(), "duplicated diagnostics");
        }
        assert_eq!(data["code"]["rust_analyzer"]["failed_request_count"], 10_000);
        let stored = hades_core::db::crud::get_document(&pool, "hades_ingest_jobs", job).await.unwrap();
        assert_eq!(stored["result"], row["result"]);
        let bytes = serde_json::to_vec_pretty(&stored["result"]).unwrap().len() + 1;
        assert!(bytes < 8 * 1024 * 1024, "stored envelope is {bytes} bytes");
        assert!(bytes < 128 * 1024, "diagnostics should leave room for other fields");
        println!("Bounded MCP diagnostics: {}", json!({"status":row["status"],"success":row["result"]["success"],"failed_request_count":data["failed_request_count"],"retained":data["failed_requests"].as_array().unwrap().len(),"failed_requests_truncated":data["failed_requests_truncated"],"stored_result_bytes":bytes}));
        unsafe { libc::kill(daemon.child.id().unwrap() as i32, libc::SIGTERM); }
        assert!(daemon.child.wait().await.unwrap().success());
    }).await;
}
