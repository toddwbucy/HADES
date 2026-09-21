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
        }).await;
    }
}
