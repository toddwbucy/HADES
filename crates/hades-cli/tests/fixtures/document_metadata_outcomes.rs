mod document_metadata_outcomes {
    use super::*;
    use hades_proto::extraction::extraction_service_server::{ExtractionService, ExtractionServiceServer};
    use hades_proto::extraction::{CapabilitiesRequest,ExtractRequest,ExtractResponse,ExtractorInfo};
    struct Extractor;
    #[tonic::async_trait]
    impl ExtractionService for Extractor {
        async fn extract(&self, _:tonic::Request<ExtractRequest>) -> Result<tonic::Response<ExtractResponse>,tonic::Status> {
            Ok(tonic::Response::new(ExtractResponse {full_text:"Private document content for metadata outcome verification.".into(),..Default::default()}))
        }
        async fn capabilities(&self,_:tonic::Request<CapabilitiesRequest>) -> Result<tonic::Response<ExtractorInfo>,tonic::Status> {
            Ok(tonic::Response::new(ExtractorInfo::default()))
        }
    }
    struct Peer(JoinHandle<()>);
    impl Drop for Peer { fn drop(&mut self) { self.0.abort(); } }

    // A document-only path must not silently discard the semantic override (#185).
    #[tokio::test]
    async fn single_file_ingest_rejects_override_and_reports_clean_enrichment() {
        with_temp_db("file_override", Fixtures::Empty, |pool| async move {
            let tree = tempfile::tempdir().unwrap();
            let root = tree.path().join("source");
            std::fs::create_dir(&root).unwrap();
            let document = root.join("doc.md");
            std::fs::write(&document, "Private document fixture").unwrap();
            let extractor_socket = tree.path().join("extractor.sock");
            let listener = tokio::net::UnixListener::bind(&extractor_socket).unwrap();
            let incoming = futures::stream::unfold(listener, |listener| async {
                let next = listener.accept().await.map(|(stream,_)| stream); Some((next, listener))
            });
            let _peer = Peer(tokio::spawn(async move {
                tonic::transport::Server::builder().add_service(ExtractionServiceServer::new(Extractor))
                    .serve_with_incoming(incoming).await.unwrap();
            }));
            let embedder = Embedder::for_task("retrieval.passage").await;
            for paths in [vec![document.to_str().unwrap()], vec![document.to_str().unwrap(), document.to_str().unwrap()]] {
                let output = cli_command(&pool, &embedder, &["ingest"]).args(paths)
                    .arg("--allow-degraded-enrichment").env("HADES_EXTRACTOR_SOCKET", &extractor_socket).output().await.unwrap();
                assert!(!output.status.success(), "file override was silently accepted: {output:?}");
                let error = String::from_utf8_lossy(&output.stderr);
                assert!(error.contains("--allow-degraded-enrichment") && error.contains("directory"), "{error}");
            }
            let output = cli_command(&pool, &embedder, &["ingest", document.to_str().unwrap()])
                .env("HADES_EXTRACTOR_SOCKET", &extractor_socket).output().await.unwrap();
            assert!(output.status.success(), "{output:?}");
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_clean_enrichment(&result);

            let socket = tree.path().join("daemon.sock");
            let token = tree.path().join("token");
            std::fs::write(&token, "private-fixture-token").unwrap();
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap(); drop(listener);
            let child = cli_command(&pool, &embedder, &["daemon", "--socket", socket.to_str().unwrap(),
                "--mcp-bind", &address.to_string(), "--mcp-token-file", token.to_str().unwrap(),
                "--mcp-db-prefix", pool.database(), "--mcp-ingest-root", root.to_str().unwrap()])
                .env("HADES_EXTRACTOR_SOCKET", &extractor_socket).env("HADES_USE_GPU", "false")
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
            for remote in [false, true] {
                let mut params = json!({"path":document,"allow_degraded_enrichment":true});
                let rejected = if remote {
                    params["db"] = json!(pool.database());
                    mcp.call_tool("ingest_start", params).await
                } else { daemon_request(&socket, json!({"command":"ingest.start","params":params})).await };
                assert_eq!(rejected["success"], false, "{rejected}");
                assert!(rejected.to_string().contains("allow_degraded_enrichment"), "{rejected}");
                assert!(rejected.to_string().contains("directory"), "{rejected}");
            }
            // Rejection precedes record creation, not merely child execution.
            let collections = hades_core::db::crud::list_collections(&pool, false).await.unwrap();
            assert!(!collections.iter().any(|c| c.name == "hades_ingest_jobs"));
            for option in [None, Some(false)] {
                let mut params = json!({"db":pool.database(),"path":document,"force":true});
                if let Some(option) = option { params["allow_degraded_enrichment"] = json!(option); }
                let started = mcp.call_tool("ingest_start", params).await;
                assert_eq!(started["success"], true, "{started}");
                let job = started["data"]["job_id"].as_str().unwrap();
                daemon.ingests.push((started["data"]["pid"].as_u64().unwrap() as u32, document.clone()));
                let row = tokio::time::timeout(std::time::Duration::from_secs(30), async {
                    loop {
                        let status = mcp.call_tool("ingest_status", json!({"db":pool.database(),"job_id":job})).await;
                        assert_eq!(status["success"], true, "{status}");
                        let row = status["data"].clone();
                        if matches!(row["status"].as_str(), Some("completed" | "failed")) { break row; }
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    }
                }).await.unwrap();
                assert_eq!(row["status"], "completed", "{row}");
                assert_eq!(row["allow_degraded_enrichment"], false);
                assert_clean_enrichment(&row["result"]);
                let stored = hades_core::db::crud::get_document(&pool, "hades_ingest_jobs", job).await.unwrap();
                assert_eq!(stored["result"], row["result"]);
            }
            unsafe { libc::kill(daemon.child.id().unwrap() as i32, libc::SIGTERM); }
            assert!(daemon.child.wait().await.unwrap().success());
        }).await;
    }

    fn assert_clean_enrichment(result: &Value) {
        assert_eq!(result["success"], true, "{result}");
        assert_eq!(result["data"]["enrichment_degraded"], false);
        assert_eq!(result["data"]["failed_request_count"], 0);
        assert_eq!(result["data"]["failed_requests_truncated"], false);
        assert_eq!(result["data"]["failed_requests"], json!([]));
    }

    #[tokio::test]
    async fn final_document_metadata_failure_is_not_ingestion_success() {
        with_temp_db("document_metadata",Fixtures::Empty,|pool| async move {
            use hades_core::db::crud;
            for collection in ["documents","chunks","embeddings"] {
                crud::create_collection(&pool,collection,Some(2)).await.unwrap();
            }
            let root=tempfile::tempdir().unwrap();
            let socket=root.path().join("extract.sock");
            let listener=tokio::net::UnixListener::bind(&socket).unwrap();
            let incoming=futures::stream::unfold(listener,|listener|async {
                let next=listener.accept().await.map(|(stream,_)|stream);Some((next,listener))
            });
            let _peer=Peer(tokio::spawn(async move {
                tonic::transport::Server::builder().add_service(ExtractionServiceServer::new(Extractor))
                    .serve_with_incoming(incoming).await.unwrap();
            }));
            let embedder=Embedder::new().await;
            let control=root.path().join("control.md");
            std::fs::write(&control,"Control source content").unwrap();
            let output=cli_command(&pool,&embedder,&["ingest",control.to_str().unwrap(),"--task","code"])
                .env("HADES_EXTRACTOR_SOCKET",&socket).output().await.unwrap();
            assert!(output.status.success(),"{output:?}");
            let metadata=crud::get_document(&pool,"documents","control").await.unwrap();
            assert_eq!(metadata["source"],"local");
            assert!(metadata["content_hash"].is_string());
            // Optional source property: initial pipeline document lacks it and
            // passes; the post-commit PATCH sets source=local and is rejected.
            pool.writer().put("collection/documents/properties",&json!({"schema":{
                "level":"strict","rule":{"type":"object","properties":{"source":{"enum":["blocked"]}}},
                "message":"private source metadata rejection"}})).await.unwrap();
            let rejected=root.path().join("rejected.md");
            std::fs::write(&rejected,"Rejected source content").unwrap();
            let output=cli_command(&pool,&embedder,&["ingest",rejected.to_str().unwrap(),"--task","code"])
                .env("HADES_EXTRACTOR_SOCKET",&socket).output().await.unwrap();
            let stored=crud::get_document(&pool,"documents","rejected").await.unwrap();
            assert!(stored["full_text"].as_str().unwrap().contains("Private document content"));
            assert!(stored.get("source_path").is_none());
            assert!(stored.get("content_hash").is_none());
            assert_eq!(crud::count_collection(&pool,"chunks").await.unwrap(),2);
            assert_eq!(crud::count_collection(&pool,"embeddings").await.unwrap(),2);
            println!("Document metadata outcome: {}",json!({"exit":output.status.code(),
                "stdout":String::from_utf8_lossy(&output.stdout),"stderr":String::from_utf8_lossy(&output.stderr),
                "pipeline_document_persisted":true,"source_path_present":false,"content_hash_present":false,
                "chunks_including_control":2,"embeddings_including_control":2}));
            assert!(!output.status.success(),"a rejected final metadata write must fail the ingest item");
            let report:Value=serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(report["success"],false);
            assert_eq!(report["data"]["failed"],1);
            assert_eq!(report["data"]["results"][0]["success"],false);
            assert!(report["data"]["results"][0]["error"].as_str().unwrap().contains("already committed"));
            let args=["ingest",control.to_str().unwrap(),rejected.to_str().unwrap(),"--task","code","--concurrency","1"];
            let mixed=cli_command(&pool,&embedder,&args).env("HADES_EXTRACTOR_SOCKET",&socket)
                .current_dir(root.path()).output().await.unwrap();
            assert!(!mixed.status.success(),"{mixed:?}");
            let report:Value=serde_json::from_slice(&mixed.stdout).unwrap();
            assert_eq!(report["success"],false);
            assert_eq!(report["data"]["total"],2);
            assert_eq!(report["data"]["failed"],1);
            assert!(report["data"]["results"].as_array().unwrap().iter().any(|r|r["input"]==control.to_str().unwrap() && r["success"]==true));
            let checkpoint=root.path().join(".hades-batch-state.json");
            let state:Value=serde_json::from_slice(&std::fs::read(&checkpoint).unwrap()).unwrap();
            assert_eq!(state["completed"],json!([control.to_str().unwrap()]));
            assert!(state["failed"][rejected.to_str().unwrap()].is_string());
            // Resume must retry the failure even while the rejection remains.
            let again=cli_command(&pool,&embedder,&args).arg("--resume").env("HADES_EXTRACTOR_SOCKET",&socket)
                .current_dir(root.path()).output().await.unwrap();
            assert!(!again.status.success());
            let again:Value=serde_json::from_slice(&again.stdout).unwrap();
            assert_eq!(again["data"]["failed"],1);
            assert_eq!(again["data"]["skipped"],1);
            assert!(checkpoint.exists());
            pool.writer().put("collection/documents/properties",&json!({"schema":null})).await.unwrap();
            let retry=cli_command(&pool,&embedder,&args).arg("--resume").env("HADES_EXTRACTOR_SOCKET",&socket)
                .current_dir(root.path()).output().await.unwrap();
            assert!(retry.status.success(),"{retry:?}");
            let retry:Value=serde_json::from_slice(&retry.stdout).unwrap();
            assert_eq!(retry["success"],true);
            assert_eq!(retry["data"]["failed"],0);
            assert_eq!(retry["data"]["skipped"],1);
            assert!(!checkpoint.exists());
            let repaired=crud::get_document(&pool,"documents","rejected").await.unwrap();
            assert_eq!(repaired["source"],"local");
            assert_eq!(repaired["source_path"],rejected.to_str().unwrap());
            assert!(repaired["content_hash"].is_string());
            assert_eq!(crud::count_collection(&pool,"chunks").await.unwrap(),2);
            assert_eq!(crud::count_collection(&pool,"embeddings").await.unwrap(),2);
            println!("Metadata retry controls: mixed batch failure retained; failed checkpoint retried; repaired metadata and exact chunk/vector counts verified");

        }).await;
    }
    #[tokio::test]
    async fn source_git_reaches_all_ingest_writers_and_envelopes() {
        with_temp_db("source_git", Fixtures::Codebase, |pool| async move {
            use hades_core::db::{crud, query::{query, ExecutionTarget}};
            for collection in ["documents", "chunks", "embeddings"] {
                crud::create_collection(&pool, collection, Some(2)).await.unwrap();
            }
            let services = tempfile::tempdir().unwrap();
            let socket = services.path().join("extract.sock");
            let listener = tokio::net::UnixListener::bind(&socket).unwrap();
            let incoming = futures::stream::unfold(listener, |listener| async {
                Some((listener.accept().await.map(|(stream, _)| stream), listener))
            });
            let _peer = Peer(tokio::spawn(async move {
                tonic::transport::Server::builder().add_service(ExtractionServiceServer::new(Extractor))
                    .serve_with_incoming(incoming).await.unwrap();
            }));
            let embedder = Embedder::new().await;
            let tree = tempfile::tempdir().unwrap();
            std::fs::write(tree.path().join("source.py"), "def symbol():\n    return 1\n").unwrap();
            std::fs::write(tree.path().join("raw.toml"), "name = 'fixture'\n").unwrap();
            std::fs::write(tree.path().join("paper.md"), "# Private paper\nContent.\n").unwrap();
            let git = |args: &[&str]| {
                let out = std::process::Command::new("git").arg("-C").arg(tree.path()).args(args).output().unwrap();
                assert!(out.status.success(), "{out:?}"); out
            };
            let mut expected = Value::Null;
            for state in ["non_git", "clean", "dirty"] {
                if state == "clean" {
                    git(&["init", "-q"]); git(&["add", "."]);
                    git(&["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"]);
                    let commit = String::from_utf8(git(&["rev-parse", "HEAD"]).stdout).unwrap();
                    expected = json!({"commit":commit.trim(), "dirty":false});
                } else if state == "dirty" {
                    std::fs::write(tree.path().join("untracked"), "dirty").unwrap();
                    expected["dirty"] = json!(true);
                }
                let root = tree.path().to_str().unwrap();
                let paper = tree.path().join("paper.md");
                for args in [
                    vec!["ingest", root, "--unparsed-ext", "toml", "--task", "code", "--force"],
                    vec!["codebase", "ingest", root, "--unparsed-ext", "toml", "--force"],
                    vec!["ingest", paper.to_str().unwrap(), "--root", root, "--task", "code", "--force", "--metadata", "{\"source_git\":{\"commit\":\"spoof\"}}"],
                    vec!["ingest", paper.to_str().unwrap(), "--task", "code", "--force"],
                ] {
                    let output = cli_command(&pool, &embedder, &args)
                        .env("HADES_EXTRACTOR_SOCKET", &socket).current_dir(services.path()).output().await.unwrap();
                    assert!(output.status.success(), "{state}: {output:?}");
                    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
                    assert_eq!(envelope["data"].get("source_git"), Some(&expected), "{state}: {envelope}");
                    for collection in ["codebase_files", "documents"] {
                        let rows = query(&pool, "FOR row IN @@collection RETURN row", Some(&json!({"@collection":collection})), None, false, ExecutionTarget::Reader).await.unwrap().results;
                        assert_eq!(rows.len(), if collection == "documents" { 1 } else { 2 });
                        for row in rows {
                            assert_eq!(row.get("source_git"), Some(&expected), "{state}: {row}");
                        }
                    }
                }
            }
            // A later run that skips unchanged content must not relabel old rows.
            git(&["add", "."]);
            git(&["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "later"]);
            let output = cli_command(&pool, &embedder, &["ingest",tree.path().to_str().unwrap(),"--unparsed-ext","toml","--task","code"])
                .env("HADES_EXTRACTOR_SOCKET", &socket).current_dir(services.path()).output().await.unwrap();
            assert!(output.status.success(), "{output:?}");
            for collection in ["codebase_files", "documents"] {
                let rows = query(&pool, "FOR row IN @@collection RETURN row", Some(&json!({"@collection":collection})), None, false, ExecutionTarget::Reader).await.unwrap().results;
                for row in rows { assert_eq!(row.get("source_git"), Some(&expected)); }
            }
        }).await;
    }
}
