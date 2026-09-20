//! Audit reproduction with private extraction/embedding peers and disposable DB.
use hades_core::chunking::{ChunkingStrategy, TextChunk};
use hades_core::db::crud;
use hades_core::persephone::{embedding::EmbeddingClient, extraction::ExtractionClient};
use hades_core::pipeline::{Pipeline, PipelineConfig};
use hades_core::test_support::{Fixtures, with_temp_db};
use hades_proto::extraction::extraction_service_server::{
    ExtractionService, ExtractionServiceServer,
};
use hades_proto::extraction::{
    CapabilitiesRequest, ExtractRequest, ExtractResponse, ExtractorInfo,
};
use serde_json::json;
use std::time::Duration;
use tonic::{Request, Response, Status};
#[path = "common/embedding_mock.rs"]
mod embedding_mock;

struct Extractor;
#[tonic::async_trait]
impl ExtractionService for Extractor {
    async fn extract(
        &self,
        _: Request<ExtractRequest>,
    ) -> Result<Response<ExtractResponse>, Status> {
        Ok(Response::new(ExtractResponse {
            full_text: "replacement text".into(),
            ..Default::default()
        }))
    }
    async fn capabilities(
        &self,
        _: Request<CapabilitiesRequest>,
    ) -> Result<Response<ExtractorInfo>, Status> {
        Ok(Response::new(ExtractorInfo::default()))
    }
}
struct OneChunk;
impl ChunkingStrategy for OneChunk {
    fn chunk(&self, text: &str) -> Vec<TextChunk> {
        vec![TextChunk {
            text: text.into(),
            start_char: 0,
            end_char: text.len(),
            chunk_index: 0,
            total_chunks: 1,
        }]
    }
}
struct Peers(Vec<tokio::task::JoinHandle<()>>);
impl Drop for Peers {
    fn drop(&mut self) {
        for task in &self.0 {
            task.abort();
        }
    }
}

#[tokio::test]
async fn document_overwrite_failure_boundary() {
    with_temp_db("document_replace", Fixtures::Empty, |pool| async move {
        for collection in ["documents", "chunks", "embeddings"] {
            crud::create_collection(&pool, collection, Some(2)).await.unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let extract_path = dir.path().join("extract.sock");
        let embed_path = dir.path().join("embed.sock");
        let listener = tokio::net::UnixListener::bind(&extract_path).unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            let next = listener.accept().await.map(|(stream, _)| stream);
            Some((next, listener))
        });
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder().add_service(ExtractionServiceServer::new(Extractor))
                .serve_with_incoming(incoming).await.unwrap();
        });
        let embedder = embedding_mock::start_embedder(&embed_path, 3).await;
        let mut peers = Peers(vec![server, embedder]);
        let pipeline = Pipeline::new(
            ExtractionClient::connect_unix_at(&extract_path).await.unwrap(),
            EmbeddingClient::connect_unix_at(&embed_path).await.unwrap(),
            pool.clone(), PipelineConfig::default(),
        );
        for collection in ["documents", "chunks", "embeddings"] {
            crud::insert_documents(&pool, collection, &[json!({"_key":"unrelated","parent_key":"elsewhere","allow":true})], false).await.unwrap();
        }
        for rejected in [false, true] {
            let key = if rejected { "rejected" } else { "control" };
            crud::insert_documents(&pool, "documents", &[json!({"_key":key,"full_text":"old text"})], false).await.unwrap();
            crud::insert_documents(&pool, "chunks", &[json!({"_key":format!("{key}_chunk_0"),"doc_key":key,"parent_key":key,"text":"old text"})], false).await.unwrap();
            crud::insert_documents(&pool, "embeddings", &[json!({"_key":format!("{key}_chunk_0_emb"),"doc_key":key,"parent_key":key,"chunk_key":format!("{key}_chunk_0"),"embedding":[1,0],"allow":true})], false).await.unwrap();
            for collection in ["chunks", "embeddings"] {
                crud::insert_documents(&pool, collection, &[json!({"_key":format!("{key}_legacy"),"parent_key":key,"allow":true})], false).await.unwrap();
            }
            let old_metadata = crud::get_document(&pool, "documents", key).await.unwrap();
            let old_chunk = crud::get_document(&pool, "chunks", &format!("{key}_chunk_0")).await.unwrap();
            let old_embedding = crud::get_document(&pool, "embeddings", &format!("{key}_chunk_0_emb")).await.unwrap();
            if rejected {
                pool.writer().put("collection/embeddings/properties", &json!({"schema":{"level":"strict","rule":{"type":"object","required":["allow"]},"message":"audit rejection"}})).await.unwrap();
            }
            let result = tokio::time::timeout(Duration::from_secs(10), pipeline.process_document(&dir.path().join("fixture.txt"), key, &OneChunk)).await.unwrap();
            assert_eq!(result.success, !rejected, "{result:?}");
            let metadata = crud::get_document(&pool, "documents", key).await.unwrap();
            let chunk = crud::get_document(&pool, "chunks", &format!("{key}_chunk_0")).await.unwrap();
            let embedding = crud::get_document(&pool, "embeddings", &format!("{key}_chunk_0_emb")).await;
            assert_eq!(metadata["full_text"], if rejected { "old text" } else { "replacement text" });
            assert_eq!(chunk["text"], if rejected { "old text" } else { "replacement text" });
            let embedding = embedding.unwrap();
            if rejected {
                assert_eq!(metadata, old_metadata);
                assert_eq!(chunk, old_chunk);
                assert_eq!(embedding, old_embedding);
                assert_eq!(embedding["allow"], true);
                assert_eq!(embedding["embedding"], json!([1,0]));
            } else {
                assert_eq!(embedding["embedding"].as_array().unwrap().len(), 2048);
            }
        }
        let duplicate_pipeline = Pipeline::new(
            ExtractionClient::connect_unix_at(&extract_path).await.unwrap(),
            EmbeddingClient::connect_unix_at(&embed_path).await.unwrap(),
            pool.clone(), PipelineConfig { overwrite:false, ..PipelineConfig::default() },
        );
        let before = crud::get_document(&pool, "documents", "control").await.unwrap();
        let duplicate = tokio::time::timeout(Duration::from_secs(10), duplicate_pipeline.process_document(&dir.path().join("fixture.txt"), "control", &OneChunk)).await.unwrap();
        assert!(!duplicate.success);
        assert_eq!(crud::get_document(&pool, "documents", "control").await.unwrap(), before);
        for collection in ["chunks", "embeddings"] {
            assert!(crud::get_document(&pool, collection, "control_legacy").await.unwrap_err().is_not_found());
            assert!(crud::get_document(&pool, collection, "rejected_legacy").await.is_ok());
        }
        for collection in ["documents", "chunks", "embeddings"] {
            let unrelated = crud::get_document(&pool, collection, "unrelated").await.unwrap();
            assert_eq!(unrelated["parent_key"], "elsewhere");
        }
        for task in &peers.0 { task.abort(); }
        for task in peers.0.drain(..) { let _ = task.await; }
    }).await;
}

#[path = "common/document_gate.rs"]
mod document_gate;

#[tokio::test]
async fn cancelled_pipeline_rolls_back_acknowledged_writes() {
    with_temp_db("document_cancel", Fixtures::Empty, |pool| async move {
        for collection in ["documents", "chunks", "embeddings"] {
            crud::create_collection(&pool, collection, Some(2))
                .await
                .unwrap();
        }
        let before = [
            (
                "documents",
                "cancel",
                json!({"_key":"cancel","full_text":"old"}),
            ),
            (
                "chunks",
                "cancel_chunk_0",
                json!({"_key":"cancel_chunk_0","doc_key":"cancel","text":"old"}),
            ),
            (
                "embeddings",
                "cancel_chunk_0_emb",
                json!({"_key":"cancel_chunk_0_emb","doc_key":"cancel","embedding":[1,0]}),
            ),
        ];
        let mut saved = Vec::new();
        for (collection, key, doc) in before {
            crud::insert_documents(&pool, collection, &[doc], false)
                .await
                .unwrap();
            saved.push((
                collection,
                key,
                crud::get_document(&pool, collection, key).await.unwrap(),
            ));
        }
        let dir = tempfile::tempdir().unwrap();
        let extract_path = dir.path().join("extract.sock");
        let embed_path = dir.path().join("embed.sock");
        let listener = tokio::net::UnixListener::bind(&extract_path).unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            let next = listener.accept().await.map(|(stream, _)| stream);
            Some((next, listener))
        });
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(ExtractionServiceServer::new(Extractor))
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let embedder = embedding_mock::start_embedder(&embed_path, 1).await;
        let mut peers = Peers(vec![server, embedder]);
        let gate = document_gate::Gate::new(pool.writer().clone()).await;
        let pipeline = Pipeline::new(
            ExtractionClient::connect_unix_at(&extract_path)
                .await
                .unwrap(),
            EmbeddingClient::connect_unix_at(&embed_path).await.unwrap(),
            gate.pool.clone(),
            PipelineConfig::default(),
        );
        let path = dir.path().join("fixture.txt");
        let caller =
            tokio::spawn(
                async move { pipeline.process_document(&path, "cancel", &OneChunk).await },
            );
        tokio::time::timeout(Duration::from_secs(5), gate.entered.notified())
            .await
            .unwrap();
        // The private server has acknowledged metadata and chunk changes inside
        // the transaction, but the caller has not received the chunk response.
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(5), gate.aborted.notified())
            .await
            .unwrap();
        gate.release.notify_one();
        for (collection, key, doc) in saved {
            assert_eq!(
                crud::get_document(&pool, collection, key).await.unwrap(),
                doc
            );
        }
        // A following exclusive writer proves cleanup released the locks.
        hades_core::db::transaction::run(
            &pool,
            vec!["documents".into(), "chunks".into(), "embeddings".into()],
            |client| async move {
                client
                    .post("document/documents", &json!({"_key":"after_cancel"}))
                    .await?;
                Ok(())
            },
        )
        .await
        .unwrap();
        for task in &peers.0 {
            task.abort();
        }
        for task in peers.0.drain(..) {
            let _ = task.await;
        }
    })
    .await;
}
