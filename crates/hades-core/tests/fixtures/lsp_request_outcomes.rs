// Exercise real Rust and Go extractors against a controlled protocol peer.
use hades_core::code::lsp::{
    GoplsSession, RustAnalyzerSession, RustSymbolExtractor, go_symbols::GoSymbolExtractor,
};
use std::time::Duration;

#[tokio::test]
async fn rust_and_go_requests_distinguish_failure_recovery_and_no_calls() {
    for go in [false, true] {
        for mode in [
            "timeout",
            "empty-prepare",
            "cfg-inactive",
            "cfg-outside",
            "cfg-other-code",
            "error",
            "recover",
            "no-calls",
            "hover-error",
            "hover-recover",
            "impl-error",
            "impl-recover",
        ] {
            if !go && mode.starts_with("impl") {
                continue;
            }
            let root = tempfile::tempdir().unwrap();
            let path = root.path();
            std::fs::write(
                path.join("Cargo.toml"),
                "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\n",
            )
            .unwrap();
            if go {
                std::fs::write(path.join("go.mod"), "module fixture\n").unwrap();
            }
            std::fs::write(path.join(".lsp-mode"), mode).unwrap();
            let source = path.join(if go { "main.go" } else { "lib.rs" });
            std::fs::write(
                &source,
                if go {
                    "package fixture\nfunc caller() { target() }\nfunc target() {}\n"
                } else if mode.starts_with("hover") {
                    "pub fn caller() { target(); }\nfn target() {}\n"
                } else {
                    "fn caller() { target(); }\nfn target() {}\n"
                },
            )
            .unwrap();
            let peer = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/lsp_requests.py");
            let extraction = if go {
                let session = GoplsSession::start_with_options(path, peer.to_str(), 1)
                    .await
                    .unwrap()
                    .with_request_timeout(if mode == "timeout" { Duration::from_millis(100) } else { Duration::from_secs(30) });
                let result = GoSymbolExtractor::new(&session, true)
                    .extract_file(&source)
                    .await
                    .unwrap();
                session.shutdown().await.unwrap();
                result
            } else {
                let session = RustAnalyzerSession::start_with_options(path, peer.to_str(), 1)
                    .await
                    .unwrap()
                    .with_request_timeout(if mode == "timeout" { Duration::from_millis(100) } else { Duration::from_secs(30) });
                let result = RustSymbolExtractor::new(&session, true)
                    .extract_file(&source)
                    .await
                    .unwrap();
                session.shutdown().await.unwrap();
                result
            };
            let failed = matches!(
                mode,
                "timeout" | "empty-prepare" | "error" | "impl-error" | "cfg-outside" | "cfg-other-code"
            ) || (go && mode == "cfg-inactive");
            assert_eq!(extraction.no_call_count, usize::from(!go && mode == "cfg-inactive"));
            if !go && mode == "cfg-inactive" { assert_eq!(extraction.no_calls[0].reason, "cfg_inactive"); }
            assert_eq!(
                extraction.failed_request_count,
                usize::from(failed),
                "go={go} mode={mode}: {extraction:?}"
            );
            if failed {
                let failure = &extraction.failed_requests[0];
                assert_eq!(
                    failure.symbol,
                    if go && mode.starts_with("hover") {
                        "Caller"
                    } else {
                        "caller"
                    }
                );
                assert_eq!(failure.file, source.file_name().unwrap().to_str().unwrap());
                assert_eq!(
                    failure.request,
                    if mode.starts_with("hover") {
                        "textDocument/hover"
                    } else if mode.starts_with("impl") {
                        "textDocument/implementation"
                    } else if mode == "empty-prepare" || mode.starts_with("cfg-") {
                        "textDocument/prepareCallHierarchy"
                    } else {
                        "callHierarchy/outgoingCalls"
                    }
                );
                assert!(!failure.reason.is_empty());
            } else {
                assert_eq!(
                    extraction
                        .symbols
                        .iter()
                        .map(|symbol| symbol.calls.len())
                        .sum::<usize>(),
                    usize::from(mode != "no-calls" && mode != "cfg-inactive")
                );
            }
            let log: Vec<serde_json::Value> = std::fs::read_to_string(path.join(".lsp-requests"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let prepare: Vec<_> = log
                .iter()
                .filter(|row| row["method"] == "textDocument/prepareCallHierarchy")
                .map(|row| row["line"].as_u64().unwrap())
                .collect();
            if mode.starts_with("hover") || mode.starts_with("impl") {
                let method = if mode.starts_with("hover") {
                    "textDocument/hover"
                } else {
                    "textDocument/implementation"
                };
                let indices: Vec<_> = log
                    .iter()
                    .enumerate()
                    .filter(|(_, row)| row["method"] == method)
                    .map(|(index, _)| index)
                    .collect();
                assert_eq!(indices.len(), 2, "exactly one retry: {log:?}");
                assert_eq!(
                    indices[1],
                    log.iter()
                        .rposition(|row| row["method"] == "shutdown")
                        .unwrap()
                        - 1,
                    "retry after all file requests: {log:?}"
                );
                if mode == "impl-recover" {
                    assert_eq!(extraction.implementations.len(), 1);
                }
                if mode == "hover-recover" {
                    assert!(!extraction.symbols[0].signature.is_empty());
                }
            }
            let offset = u64::from(go);
            assert_eq!(
                prepare,
                if mode == "no-calls" || mode.starts_with("hover") || mode.starts_with("impl") {
                    vec![offset, offset + 1]
                } else {
                    vec![offset, offset + 1, offset]
                },
                "deferred retry: {log:?}"
            );
        }
    }
}

#[tokio::test]
async fn unlinked_file_needs_a_positive_inactive_parent_diagnostic() {
    for mode in ["parent-inactive", "parent-retry", "parent-empty-publish", "parent-unproven", "parent-outside"] {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("Cargo.toml"), "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\n").unwrap();
        std::fs::write(root.path().join("lib.rs"), "#[cfg(feature = \"off\")] mod gated;\n").unwrap();
        let source = root.path().join("gated.rs");
        std::fs::write(&source, "fn caller() { target(); }\nfn target() {}\n").unwrap();
        std::fs::write(root.path().join(".lsp-mode"), mode).unwrap();
        let peer = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lsp_requests.py");
        let session = RustAnalyzerSession::start_with_options(root.path(), peer.to_str(), 1).await.unwrap();
        let extraction = RustSymbolExtractor::new(&session, true).extract_file(&source).await.unwrap();
        assert_eq!(extraction.failed_request_count, usize::from(!matches!(mode,"parent-inactive"|"parent-retry"|"parent-empty-publish")), "{extraction:?}");
        assert_eq!(extraction.no_call_count, if mode == "parent-empty-publish" {2} else {usize::from(matches!(mode,"parent-inactive"|"parent-retry"))});
        session.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn document_symbols_reject_null_and_wrong_shape_but_accept_empty() {
    for go in [false, true] {
        for mode in ["document-null", "document-object", "document-empty"] {
            let root = tempfile::tempdir().unwrap();
            std::fs::write(root.path().join("Cargo.toml"), "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\n").unwrap();
            if go { std::fs::write(root.path().join("go.mod"), "module fixture\n").unwrap(); }
            let source = root.path().join(if go {"main.go"} else {"lib.rs"});
            std::fs::write(&source, if go {"package fixture\n"} else {""}).unwrap();
            std::fs::write(root.path().join(".lsp-mode"), mode).unwrap();
            let peer = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lsp_requests.py");
            let result = if go {
                let session = GoplsSession::start_with_options(root.path(), peer.to_str(), 1).await.unwrap();
                let result = session.document_symbols(&source).await;
                session.shutdown().await.unwrap();
                result
            } else {
                let session = RustAnalyzerSession::start_with_options(root.path(), peer.to_str(), 1).await.unwrap();
                let result = session.document_symbols(&source).await;
                session.shutdown().await.unwrap();
                result
            };
            if mode == "document-empty" { assert!(result.unwrap().is_empty()); }
            else { assert!(result.unwrap_err().to_string().contains("documentSymbol")); }
        }
    }
}

// The explicit degraded override must not lose failures after the old cap (#179).
#[tokio::test]
async fn all_failed_requests_survive_large_files() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("Cargo.toml"), "[package]\nname=\"fixture\"\nversion=\"0.1.0\"\n").unwrap();
    std::fs::write(root.path().join(".lsp-mode"), "many-error").unwrap();
    let source = root.path().join("lib.rs");
    std::fs::write(&source, (0..105).map(|n| format!("fn caller{n}() {{ target(); }}\n")).collect::<String>()).unwrap();
    let peer = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lsp_requests.py");
    let session = RustAnalyzerSession::start_with_options(root.path(), peer.to_str(), 1).await.unwrap();
    let extraction = RustSymbolExtractor::new(&session, true).extract_file(&source).await.unwrap();
    session.shutdown().await.unwrap();
    assert_eq!(extraction.failed_request_count, 105);
    assert_eq!(extraction.failed_requests.len(), 105);
    assert_eq!(extraction.failed_requests[104].symbol, "caller104");
}
