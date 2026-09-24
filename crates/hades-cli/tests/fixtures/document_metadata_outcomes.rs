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

    // N named files share one directory observation per batch (#171).
    #[tokio::test]
    async fn named_file_batch_observes_git_once_per_directory() {
        use std::os::unix::fs::PermissionsExt;
        with_temp_db("git_batch", Fixtures::Empty, |pool| async move {
            let services = tempfile::tempdir().unwrap();
            let tree = tempfile::tempdir().unwrap();
            for name in ["a.md", "b.md", "c.md"] { std::fs::write(tree.path().join(name), "Fixture document.").unwrap(); }
            let git = |args: &[&str]| {
                let out = std::process::Command::new("/usr/bin/git").arg("-C").arg(tree.path()).args(args).output().unwrap();
                assert!(out.status.success(), "{out:?}");
            };
            git(&["init", "-q"]); git(&["add", "."]);
            git(&["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"]);
            let bin = services.path().join("bin"); std::fs::create_dir(&bin).unwrap();
            let wrapper = bin.join("git");
            std::fs::write(&wrapper, "#!/bin/sh\nif [ \"$3\" = status ]; then printf '%s\\n' \"$2\" >> \"$HADES_GIT_STATUS_LOG\"; fi\nexec /usr/bin/git \"$@\"\n").unwrap();
            std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700)).unwrap();
            let count = services.path().join("status.log");
            let socket = services.path().join("extract.sock");
            let listener = tokio::net::UnixListener::bind(&socket).unwrap();
            let incoming = futures::stream::unfold(listener, |listener| async {
                let next = listener.accept().await.map(|(stream,_)|stream); Some((next,listener))
            });
            let _peer = Peer(tokio::spawn(async move {
                tonic::transport::Server::builder().add_service(ExtractionServiceServer::new(Extractor))
                    .serve_with_incoming(incoming).await.unwrap();
            }));
            let embedder = Embedder::new().await;
            for dirty in [false, true] {
                if dirty { std::fs::write(tree.path().join("untracked"), "new").unwrap(); }
                std::fs::write(&count, "").unwrap();
                let files: Vec<_> = ["a.md", "b.md", "c.md"].map(|name|tree.path().join(name)).into_iter().collect();
                let output = cli_command(&pool, &embedder, &["ingest", "--task", "code", "--force", "--concurrency", "3"])
                    .args(&files).env("HADES_EXTRACTOR_SOCKET", &socket)
                    .env("PATH", format!("{}:/usr/bin:/bin", bin.display())).env("HADES_GIT_STATUS_LOG", &count)
                    .current_dir(services.path()).output().await.unwrap();
                assert!(output.status.success(), "{output:?}");
                let report: Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(report["data"]["completed"], 3, "{report}");
                assert_eq!(report["data"]["source_git"]["dirty"], dirty, "batch must refresh observations");
                for row in report["data"]["results"].as_array().unwrap() {
                    assert_eq!(row["source_git"], report["data"]["source_git"]);
                }
                assert_eq!(std::fs::read_to_string(&count).unwrap().lines().count(), 1, "one git status per directory per batch");
            }
        }).await;
    }

    struct Peer(JoinHandle<()>);
    impl Drop for Peer { fn drop(&mut self) { self.0.abort(); } }

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
