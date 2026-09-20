//! Opt-in full-handler allocation measurement; disposable DB and fixed embedder.
use hades_core::{
    config::HadesConfig,
    db::crud,
    service::{ConnectionPolicy, handle_request},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::json;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
#[path = "common/embedding_mock.rs"]
mod embedding_mock;

fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")
                .map(|s| s.split_whitespace().next().unwrap().parse().unwrap())
        })
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "explicit isolated resource-limited full-handler benchmark only"]
async fn measure_full_search_handler() {
    assert_eq!(std::env::var("ARANGO_TESTS").as_deref(), Ok("1"));
    let rows: usize = std::env::var("HADES_BENCH_ROWS")
        .unwrap_or("1024".into())
        .parse()
        .unwrap();
    let trials: usize = std::env::var("HADES_BENCH_TRIALS")
        .unwrap_or("16".into())
        .parse()
        .unwrap();
    assert!((1000..=100_000).contains(&rows) && (4..=128).contains(&trials));
    with_temp_db("handler_bench", Fixtures::Empty, move |pool| async move {
        for collection in ["documents", "chunks", "embeddings"] {
            crud::create_collection(&pool, collection, None).await.unwrap();
        }
        let mut vector = vec![0.0; 2048];
        vector[0] = 1.0;
        for start in (0..rows).step_by(8) {
            let end = (start+8).min(rows);
            let embeddings: Vec<_> = (start..end).map(|i| json!({"_key":format!("e{i:06}"), "chunk_key":format!("c{i:06}"), "parent_key":format!("p{i:06}"), "model":"jinaai/jina-embeddings-v4", "dimension":2048, "embedding":vector})).collect();
            crud::insert_documents(&pool, "embeddings", &embeddings, false).await.unwrap();
            let documents: Vec<_> = (start..end).map(|i| json!({"_key":format!("p{i:06}"), "title":"fixture", "structural_embedding":vector})).collect();
            crud::insert_documents(&pool, "documents", &documents, false).await.unwrap();
            let chunks: Vec<_> = (start..end).map(|i| json!({"_key":format!("c{i:06}"), "text":"x".repeat(800), "chunk_index":0, "total_chunks":1})).collect();
            crud::insert_documents(&pool, "chunks", &chunks, false).await.unwrap();
        }
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("embedder.sock");
        let embedder = embedding_mock::start_embedder(&socket, trials*2).await;
        let mut config = HadesConfig::default();
        config.embedding.service.socket = socket.to_string_lossy().into_owned();
        let payload = Arc::new(serde_json::to_vec(&json!({"command":"db.query", "params":{
            "text":"fixture ".repeat(8192), "limit":1000, "hybrid":true, "structural":true
        }})).unwrap());
        for concurrency in [1, 4] {
            let baseline = rss_kib();
            let stop = Arc::new(AtomicBool::new(false));
            let flag = stop.clone();
            let sampler = std::thread::spawn(move || {
                let mut peak = rss_kib();
                while !flag.load(Ordering::Relaxed) {
                    peak = peak.max(rss_kib());
                    std::thread::sleep(Duration::from_millis(2));
                }
                peak
            });
            let mut times = Vec::new();
            let mut largest_response = 0;
            for start in (0..trials).step_by(concurrency) {
                let mut pending = tokio::task::JoinSet::new();
                for _ in start..(start+concurrency).min(trials) {
                    let pool = pool.clone(); let config = config.clone(); let payload = payload.clone();
                    pending.spawn(async move {
                        let start = Instant::now();
                        let response = handle_request(&pool, &config, ConnectionPolicy::agent_only(), &payload, Duration::from_secs(60)).await;
                        assert!(response.success, "{response:?}");
                        assert_eq!(response.data.as_ref().unwrap()["result_count"], 1000);
                        let wire = serde_json::to_vec(&response).unwrap();
                        (start.elapsed().as_secs_f64()*1000.0, response, wire)
                    });
                }
                let mut retained = Vec::new();
                while let Some(result) = pending.join_next().await {
                    let (ms, response, wire) = result.unwrap();
                    times.push(ms);
                    largest_response = largest_response.max(wire.len());
                    retained.push((response, wire));
                }
                // Include completed JSON values and their serialization buffers.
                tokio::time::sleep(Duration::from_millis(50)).await;
                drop(retained);
            }
            stop.store(true, Ordering::Relaxed);
            let peak = sampler.join().unwrap();
            times.sort_by(f64::total_cmp);
            let percentile = |fraction: f64| times[((times.len() as f64*fraction).ceil() as usize).saturating_sub(1)];
            println!("HANDLER_BENCH {}", json!({"rows":rows, "dimension":2048, "k":1000, "query_bytes":65536, "hybrid":true, "structural_dimension":2048,
                "concurrency":concurrency, "trials":trials, "baseline_rss_kib":baseline, "sampled_peak_rss_kib":peak, "largest_response_bytes":largest_response,
                "p50_ms":percentile(0.5), "p95_ms":percentile(0.95), "p99_ms":percentile(0.99),
                "scope":"full service handler and retained serialized results; fixed-vector embedder; excludes model inference and daemon/MCP socket queues"}));
        }
        embedder.await.unwrap();
    }).await;
}
