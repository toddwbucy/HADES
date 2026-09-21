// Included in codebase_ingest::tests: real ingest_file, HTTP client and private DB.
#[tokio::test]
async fn recovered_embedding_windows_determine_file_outcome() {
    use axum::{Json, Router, response::IntoResponse, routing::post};
    use hades_core::persephone::embedding::{EmbeddingClientConfig, EmbeddingEndpoint};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    with_temp_db("retry_outcome", Fixtures::Codebase, |pool| async move {
        let tree = tempfile::tempdir().unwrap();
        let socket = tree.path().join("provider.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        // mode, request count. The initial multi-window client call fails on
        // its second request; calls 3 onward are its per-window retries.
        let state = Arc::new(Mutex::new((0usize, 0usize)));
        let provider_state = state.clone();
        let app = Router::new().route("/v1/embeddings", post(move |Json(body): Json<Value>| {
            let state = provider_state.clone();
            async move {
                let (mode, call) = {
                    let mut state = state.lock().unwrap();
                    state.1 += 1;
                    *state
                };
                if (mode > 0 && call == 2) || (mode == 2 && call == 4) {
                    return (axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        Json(json!({"error":{"message":"controlled window failure"}}))).into_response();
                }
                let late = body["late_chunk"]["boundaries"].as_array();
                let data: Vec<Value> = if let Some(bounds) = late {
                    bounds.iter().enumerate().map(|(i, b)| {
                        json!({"index":0,"chunk_index":i,"char_start":b[0],"char_end":b[1],
                            "embedding":if mode == 3 && call == 4 {vec![1.0]} else {vec![1.0, 0.0]}})
                    }).collect()
                } else {
                    vec![json!({"index":0,"embedding":[1.0,0.0]})]
                };
                let changed_identity = (mode == 4 && call == 4) || (mode == 5 && late.is_none());
                Json(json!({"model":if changed_identity {"/fixture/jinaai--jina-embeddings-v4"} else {"jinaai/jina-embeddings-v4"},"data":data})).into_response()
            }
        }));
        let peer = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let embedder = EmbeddingClient::connect(EmbeddingClientConfig {
            endpoint: EmbeddingEndpoint::Unix(socket), model:"jinaai/jina-embeddings-v4".into(),
            expected_dimension:2, timeout:Duration::from_secs(3), connect_timeout:Duration::from_secs(1),
        }).await.unwrap();
        let config = HadesConfig::default();
        let namespace = tree.path().to_str().unwrap();
        let mut violations = Vec::new();
        for (mode, case) in [(0,"initial_success"),(1,"complete_recovery"),(2,"terminal_retry_failure"),(3,"invalid_retry_vector"),(4,"changed_retry_model"),(1,"complete_recovery_with_oversized"),(5,"changed_oversized_model")] {
            let rel = format!("{case}.py");
            let path = tree.path().join(&rel);
            let mut body: String = (0..120).map(|i| format!(
                "# Documentation for generated function {i}: {}\ndef function_{i}():\n    return {i}\n\n",
                "private fixture padding ".repeat(7))).collect();
            if case.contains("oversized") {
                body.push_str(&format!("# {}\n", "oversized padding ".repeat(650)));
            }
            fs::write(&path, &body).unwrap();
            *state.lock().unwrap() = (0,0);
            let mut imports = ImportContext::default();
            let seed = ingest_file(&pool,Some(&embedder),&config,&path,&rel,None,&mut imports,None,
                true,false,false,9000,namespace).await.unwrap();
            assert!(seed.success, "seed must succeed: {:?}", seed.error);
            let windows = state.lock().unwrap().1;
            assert!(windows >= 2, "fixture must span multiple real late windows");
            let key = keys::scoped_file_key(namespace,&rel);
            let snapshot = |pool: ArangoPool, key: String| async move {
                let mut graph = serde_json::Map::new();
                for (col, _) in CODEBASE.all_collections() {
                    let rows = hades_core::db::query::query(&pool,
                        "FOR d IN @@col FILTER d.file_key == @key OR d._key == @key OR d._from == CONCAT('codebase_files/', @key) SORT d._key RETURN d",
                        Some(&json!({"@col":col,"key":key})),None,false,ExecutionTarget::Writer).await.unwrap();
                    graph.insert(col.to_string(),json!(rows.results));
                }
                Value::Object(graph)
            };
            let before = snapshot(pool.clone(),key.clone()).await;
            fs::write(&path, body.replace("return ","return 1000 + ")).unwrap();
            *state.lock().unwrap() = (mode,0);
            let mut imports = ImportContext::default();
            let result = ingest_file(&pool,Some(&embedder),&config,&path,&rel,None,&mut imports,None,
                true,false,false,9000,namespace).await.unwrap();
            let calls = state.lock().unwrap().1;
            let after = snapshot(pool.clone(),key.clone()).await;
            println!("EMBED_RETRY_OUTCOME {}",json!({"case":case,"success":result.success,
                "embedding_error":result.embedding_error,"windows":windows,"calls":calls,
                "graph_unchanged":before==after,"chunks":result.num_chunks,"embeddings":result.num_embeddings}));
            assert_eq!(calls, if mode == 0 { windows } else { windows + 2 });
            if mode < 2 {
                if !result.success { violations.push(format!("{case}: complete valid vectors rejected")); continue; }
                let file = &after[CODEBASE.files][0];
                assert_ne!(file["content_hash"],before[CODEBASE.files][0]["content_hash"]);
                assert!(result.num_embeddings.unwrap() > 1);
                assert_eq!(result.num_chunks,result.num_embeddings);
                assert_eq!(file["chunk_count"],file["embedding_count"]);
                assert_eq!(after[CODEBASE.embeddings].as_array().unwrap().len(),result.num_embeddings.unwrap());
                for row in after[CODEBASE.embeddings].as_array().unwrap() {
                    assert_eq!(row["embedding"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>(), vec![1.0, 0.0]);
                    assert_eq!(row["dimension"],2);
                    assert!(after[CODEBASE.chunks].as_array().unwrap().iter().any(|c| c["_key"]==row["chunk_key"]));
                }
                assert!(imports.committed_revisions.contains_key(&key));
            } else {
                assert!(!result.success && result.embedding_error.is_some());
                assert_eq!(before,after,"failed preparation must retain the committed graph");
                assert!(!imports.committed_revisions.contains_key(&key));
            }
        }
        peer.abort();
        let _ = peer.await;
        assert!(violations.is_empty(),"{}",violations.join("\n"));
    }).await;
}
