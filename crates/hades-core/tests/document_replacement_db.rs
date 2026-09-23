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

struct Extractor(&'static str);
#[tonic::async_trait]
impl ExtractionService for Extractor {
    async fn extract(
        &self,
        _: Request<ExtractRequest>,
    ) -> Result<Response<ExtractResponse>, Status> {
        Ok(Response::new(ExtractResponse {
            full_text: self.0.into(),
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
            tonic::transport::Server::builder().add_service(ExtractionServiceServer::new(Extractor("replacement text")))
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
                .add_service(ExtractionServiceServer::new(Extractor("replacement text")))
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

#[path = "common/document_embedding.rs"]
mod document_embedding;

async fn writer(
    pool: hades_core::db::ArangoPool,
    text: &'static str,
    axis: usize,
    overwrite: bool,
) -> (Pipeline, Peers, tempfile::TempDir) {
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
            .add_service(ExtractionServiceServer::new(Extractor(text)))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });
    let embedder = document_embedding::start_embedder(&embed_path, 1, axis).await;
    let peers = Peers(vec![server, embedder]);
    let pipeline = Pipeline::new(
        ExtractionClient::connect_unix_at(&extract_path)
            .await
            .unwrap(),
        EmbeddingClient::connect_unix_at(&embed_path).await.unwrap(),
        pool,
        PipelineConfig {
            overwrite,
            ..PipelineConfig::default()
        },
    );
    (pipeline, peers, dir)
}

async fn assert_generation(pool: &hades_core::db::ArangoPool, text: &str, axis: usize) {
    assert_eq!(
        crud::get_document(pool, "documents", "shared")
            .await
            .unwrap()["full_text"],
        text
    );
    assert_eq!(
        crud::get_document(pool, "chunks", "shared_chunk_0")
            .await
            .unwrap()["text"],
        text
    );
    let embedding = crud::get_document(pool, "embeddings", "shared_chunk_0_emb")
        .await
        .unwrap();
    let values = embedding["embedding"].as_array().unwrap();
    assert_eq!(values.len(), 2048);
    for (index, value) in values.iter().enumerate() {
        assert_eq!(
            value.as_f64().unwrap(),
            if index == axis { 1.0 } else { 0.0 }
        );
    }
}

#[tokio::test]
async fn competing_pipelines_commit_complete_generations() {
    with_temp_db("document_writers", Fixtures::Empty, |pool| async move {
        for collection in ["documents", "chunks", "embeddings"] {
            crud::create_collection(&pool, collection, Some(2))
                .await
                .unwrap();
        }
        let first_gate = document_gate::Gate::new(pool.writer().clone()).await;
        let second_gate = document_gate::Gate::new(pool.writer().clone()).await;
        // The first writer also verifies overwrite=false can create an absent document.
        let (first, mut first_peers, first_dir) =
            writer(first_gate.pool.clone(), "first text", 0, false).await;
        let (second, mut second_peers, second_dir) =
            writer(second_gate.pool.clone(), "second text", 1, true).await;
        let path = first_dir.path().join("fixture.txt");
        let first =
            tokio::spawn(async move { first.process_document(&path, "shared", &OneChunk).await });
        tokio::time::timeout(Duration::from_secs(5), first_gate.entered.notified())
            .await
            .unwrap();
        let path = second_dir.path().join("fixture.txt");
        let second =
            tokio::spawn(async move { second.process_document(&path, "shared", &OneChunk).await });
        tokio::time::timeout(Duration::from_secs(5), second_gate.beginning.notified())
            .await
            .unwrap();
        assert!(!second.is_finished());
        first_gate.release.notify_one();
        assert!(
            tokio::time::timeout(Duration::from_secs(5), first)
                .await
                .unwrap()
                .unwrap()
                .success
        );
        tokio::time::timeout(Duration::from_secs(5), second_gate.entered.notified())
            .await
            .unwrap();
        // The second writer has changed metadata/chunks inside its transaction;
        // ordinary readers still observe the complete first committed generation.
        assert_generation(&pool, "first text", 0).await;
        second_gate.release.notify_one();
        assert!(
            tokio::time::timeout(Duration::from_secs(5), second)
                .await
                .unwrap()
                .unwrap()
                .success
        );
        assert_generation(&pool, "second text", 1).await;
        for peers in [&mut first_peers, &mut second_peers] {
            for task in &peers.0 {
                task.abort();
            }
            for task in peers.0.drain(..) {
                let _ = task.await;
            }
        }
    })
    .await;
}

use hades_core::db::{ArangoClient, ArangoPool};
#[allow(dead_code)] // Shared peer also supports cursor lifecycle tests.
#[path = "common/cursor_mock.rs"]
mod cursor_mock;

#[tokio::test]
async fn cleanup_acknowledgment_is_strict_before_document_writes() {
    use cursor_mock::{Mock, Reply};
    for response in [
        json!({}),
        json!({"result":[]}),
        json!({"result":[],"hasMore":null}),
        json!({"result":[],"hasMore":"false"}),
        json!({"result":[],"hasMore":0}),
        json!({"result":[],"hasMore":true}),
        json!({"result":[[]],"hasMore":false}),
        json!({"result":null,"hasMore":false}),
        json!({"result":[],"hasMore":false}),
    ] {
        let valid = response == json!({"result":[],"hasMore":false});
        let dir = tempfile::tempdir().unwrap();
        let extract_path = dir.path().join("extract.sock");
        let embed_path = dir.path().join("embed.sock");
        let listener = tokio::net::UnixListener::bind(&extract_path).unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(stream, _)| stream), listener))
        });
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(ExtractionServiceServer::new(Extractor("replacement text")))
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let embedder = embedding_mock::start_embedder(&embed_path, 1).await;
        let _peers = Peers(vec![server, embedder]);
        let mut replies = vec![
            Reply::page(json!({"result":{"id":"1"}})),
            Reply::page(response),
        ];
        if valid {
            replies.push(Reply::page(json!({"hasMore":false,"result":[]})));
            for key in ["doc", "doc_chunk_0", "doc_chunk_0_emb"] {
                replies.push(Reply::page(
                    json!([{"_key":key,"_rev":"fixture","error":false}]),
                ));
            }
            replies.push(Reply::page(json!({"result":{"status":"committed"}})));
        }
        let mut mock = Mock::new(replies).await;
        let pipeline = Pipeline::new(
            ExtractionClient::connect_unix_at(&extract_path)
                .await
                .unwrap(),
            EmbeddingClient::connect_unix_at(&embed_path).await.unwrap(),
            mock.pool.clone(),
            PipelineConfig::default(),
        );
        let result = pipeline
            .process_document(&dir.path().join("fixture.txt"), "doc", &OneChunk)
            .await;
        assert_eq!(result.success, valid, "{result:?}");
        mock.event("POST transaction/begin").await;
        mock.event("POST cursor").await;
        if valid {
            mock.event("POST cursor").await;
            for collection in ["documents", "chunks", "embeddings"] {
                mock.event(&format!("POST document/{collection}?overwriteMode=replace"))
                    .await;
            }
            mock.event("PUT transaction/1").await;
        } else {
            mock.event("DELETE transaction/1").await;
        }
        assert!(
            mock.events.try_recv().is_err(),
            "no writes may follow an invalid cleanup acknowledgment"
        );
    }
}
