// Exercise the production store and transaction owner against a private HTTP DB.
#[tokio::test]
async fn enrichment_store_splits_over_cap_payload_without_partial_files() {
    use axum::{Json, Router, body::to_bytes, extract::Request, response::IntoResponse};
    use hades_core::code::lsp::symbols::{CallTarget, ExtractedSymbol};
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default, Debug)]
    struct FileState {
        revision: usize,
        symbols: usize,
        defines: usize,
        calls: usize,
        implements: usize,
        patched: bool,
        counted: bool,
    }
    #[derive(Default)]
    struct Mock {
        files: HashMap<String, FileState>,
        pending: Option<HashMap<String, FileState>>,
        begun: usize,
        commits: usize,
        aborts: usize,
        bytes: usize,
        committed_bytes: Vec<usize>,
        fail_second: bool,
    }
    let root = tempfile::tempdir().unwrap();
    let namespace = root.path().to_str().unwrap();
    let mut extractions = HashMap::new();
    let mut revisions = HashMap::new();
    let mut seed = HashMap::new();
    for i in 0..90 {
        let path = format!("file_{i:03}.rs");
        let source = "fn target() {}\ntrait Trait {}\n";
        fs::write(root.path().join(&path), source).unwrap();
        revisions.insert(
            path.clone(),
            EnrichmentInput {
                revision: "0".into(),
                content_hash: code::compute_content_hash(source),
                path: root.path().join(&path),
            },
        );
        seed.insert(
            keys::scoped_file_key(namespace, &path),
            FileState::default(),
        );
        let mut extraction = FileExtraction::empty();
        for (name, kind) in [("target", "function"), ("Trait", "interface")] {
            extraction.symbols.push(ExtractedSymbol {
                name: name.into(),
                qualified_name: format!("file_{i}::{name}"),
                kind: kind.into(),
                visibility: "public".into(),
                signature: "x".repeat(200 * 1024),
                start_line: if name == "target" { 0 } else { 1 },
                end_line: 2,
                parent_symbol: None,
                impl_trait: if name == "target" {
                    Some(format!("file_{i}::Trait"))
                } else {
                    None
                },
                is_pyo3: false,
                is_ffi: false,
                is_unsafe: false,
                derives: vec![],
                python_name: None,
                calls: if name == "target" {
                    vec![CallTarget {
                        qualified_name: format!("file_{i}::target"),
                        name: "target".into(),
                        file: path.clone(),
                        line: 0,
                    }]
                } else {
                    vec![]
                },
            });
        }
        extractions.insert(path, extraction);
    }
    let payload_bytes = serde_json::to_vec(
        &LspEdgeResolver::new_scoped(extractions.clone(), "fixture", namespace)
            .build_symbol_documents(),
    )
    .unwrap()
    .len();
    assert!(payload_bytes > 32 * 1024 * 1024);
    let mock = Arc::new(Mutex::new(Mock {
        files: seed.clone(),
        ..Default::default()
    }));
    let server_state = mock.clone();
    let socket = root.path().join("db.sock");
    let listener = tokio::net::UnixListener::bind(&socket).unwrap();
    let app = Router::new().fallback(move |request: Request| {
        let state = server_state.clone();
        async move {
            let (parts, body) = request.into_parts();
            let bytes = to_bytes(body, 80 * 1024 * 1024).await.unwrap();
            let body: Value = if bytes.is_empty() {Value::Null} else {serde_json::from_slice(&bytes).unwrap()};
            let path = parts.uri.path().split("/_api/").nth(1).unwrap();
            let method = parts.method.as_str();
            let mut state = state.lock().unwrap();
            let failure = || (axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error":true,"errorNum":32,"errorMessage":"controlled store failure"}))).into_response();
            let result = if path == "transaction/begin" {
                assert_eq!(body["maxTransactionSize"], 32 * 1024 * 1024);
                assert!(state.pending.is_none());
                state.begun += 1;
                state.bytes = 0;
                state.pending = Some(state.files.clone());
                json!({"result":{"id":state.begun.to_string()}})
            } else if path.starts_with("transaction/") {
                assert_eq!(path, format!("transaction/{}", state.begun));
                if method == "PUT" {
                    let pending = state.pending.take().unwrap();
                    for file in pending.values().filter(|f| f.patched) {
                        assert_eq!((file.symbols,file.defines,file.calls,file.implements), (2,2,1,1));
                        assert!(file.counted, "file's count update must commit with its documents");
                    }
                    state.files = pending;
                    state.commits += 1;
                    let bytes = state.bytes;
                    state.committed_bytes.push(bytes);
                    json!({"result":{"status":"committed"}})
                } else {
                    assert_eq!(method,"DELETE");
                    state.pending.take().unwrap();
                    state.aborts += 1;
                    json!({"result":{"status":"aborted"}})
                }
            } else {
                assert_eq!(parts.headers["x-arango-trx-id"], state.begun.to_string());
                state.bytes += bytes.len();
                if state.bytes > 32 * 1024 * 1024 { return failure(); }
                if path.starts_with("document/codebase_files/") {
                    let key = path.rsplit('/').next().unwrap();
                    if method == "PATCH" && state.fail_second && state.begun == 2 {return failure();}
                    let file = state.pending.as_mut().unwrap().get_mut(key).unwrap();
                    if method == "PATCH" {
                        assert_eq!(body["fixture_analyzed"],true);
                        assert_eq!(body["fixture_symbol_count"],2);
                        file.patched = true;
                        file.revision += 1;
                    } else { assert_eq!(method,"GET"); }
                    json!({"_rev":file.revision.to_string()})
                } else if path == "cursor" {
                    if body["query"].as_str().unwrap().contains("REMOVE e IN @@edges") {
                        assert!(body["bindVars"]["ids"].is_array());
                        return (axum::http::StatusCode::OK, Json(json!({"result":[],"hasMore":false}))).into_response();
                    }
                    assert!(body["query"].as_str().unwrap().contains("symbol_count: c"));
                    for key in body["bindVars"]["fkeys"].as_array().unwrap() {
                        let file = state.pending.as_mut().unwrap().get_mut(key.as_str().unwrap()).unwrap();
                        assert!(file.patched);
                        file.counted = true;
                        file.revision += 1;
                    }
                    json!({"result":[],"hasMore":false})
                } else {
                    assert_eq!(method,"POST");
                    let rows = body.as_array().unwrap();
                    for row in rows {
                        let owner = if path == "document/codebase_symbols" {row["file_key"].as_str().unwrap()}
                            else { let source = row["_from"].as_str().unwrap().split('/').nth(1).unwrap();
                                // Symbol keys start with the owning file key.
                                state.pending.as_ref().unwrap().keys().find(|key| source.starts_with(key.as_str())).unwrap() };
                        let owner = owner.to_owned();
                        let file = state.pending.as_mut().unwrap().get_mut(&owner).unwrap();
                        match path {
                            "document/codebase_symbols" => file.symbols += 1,
                            "document/codebase_defines_edges" => file.defines += 1,
                            "document/codebase_calls_edges" => file.calls += 1,
                            "document/codebase_implements_edges" => file.implements += 1,
                            _ => panic!("unexpected endpoint {path}"),
                        }
                    }
                    json!(rows.iter().map(|_|json!({"error":false})).collect::<Vec<_>>())
                }
            };
            Json(result).into_response()
        }
    });
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let config: HadesConfig = serde_json::from_value(json!({"database":{
        "name":"mock_enrichment", "username":"fixture", "sockets":{"readonly":socket,"readwrite":socket}
    }})).unwrap();
    let pool = ArangoPool::from_config(&config).unwrap();
    let result = store_lsp_extractions(
        &pool,
        extractions.clone(),
        revisions.clone(),
        1,
        1,
        "fixture",
        "fixture",
        namespace,
    )
    .await
    .unwrap();
    assert!(
        !result.store_failed,
        "oversized workspace must persist in whole-file groups"
    );
    assert_eq!((result.symbols, result.edges), (180, 360));
    {
        let state = mock.lock().unwrap();
        assert!(state.commits > 1);
        assert_eq!(state.begun, state.commits);
        assert_eq!(state.aborts, 0);
        assert!(
            state
                .files
                .values()
                .all(|file| file.patched && file.counted)
        );
        assert!(
            state
                .committed_bytes
                .iter()
                .all(|size| *size < 32 * 1024 * 1024)
        );
        println!(
            "ENRICHMENT_STORE_MOCK payload_bytes={payload_bytes} committed_transactions={} files=90",
            state.commits
        );
    }
    *mock.lock().unwrap() = Mock {
        files: seed,
        fail_second: true,
        ..Default::default()
    };
    let result = store_lsp_extractions(
        &pool,
        extractions,
        revisions,
        1,
        1,
        "fixture",
        "fixture",
        namespace,
    )
    .await
    .unwrap();
    assert!(result.store_failed);
    {
        let state = mock.lock().unwrap();
        assert_eq!((state.commits, state.aborts), (1, 1));
        assert!(state.pending.is_none());
        let complete = state.files.values().filter(|file| file.patched).count();
        assert!(complete > 0 && complete < 90);
        assert_eq!(result.symbols, complete * 2);
        for file in state.files.values() {
            assert_eq!(
                (
                    file.symbols,
                    file.defines,
                    file.calls,
                    file.implements,
                    file.counted
                ),
                if file.patched {
                    (2, 2, 1, 1, true)
                } else {
                    (0, 0, 0, 0, false)
                }
            );
        }
    }
    server.abort();
    let _ = server.await;
}
