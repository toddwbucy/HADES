//! Opt-in, synthetic retrieval-engine comparison on a disposable database.
//! This measures vector ranking/index approximation, not language-model quality.
use hades_core::db::{ArangoPool, crud, index, query};
use hades_core::retrieval::TopK;
use hades_core::test_support::{Fixtures, with_temp_db};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

fn rss_kib_of(pid: &str) -> u64 {
    std::fs::read_to_string(format!("/proc/{pid}/status"))
        .unwrap()
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")
                .map(|n| n.split_whitespace().next().unwrap().parse().unwrap())
        })
        .unwrap()
}
fn vector(row: usize, dimension: usize) -> Vec<f32> {
    let mut state = row as u64 + 1;
    (0..dimension)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state % 10001) as f32 / 5000.0 - 1.0) * 0.1;
            noise + if i == row % 16 { 1.0 } else { 0.0 }
        })
        .collect()
}
fn limits(rows: u64) -> query::FoldLimits {
    query::FoldLimits {
        batch_size: 4,
        response_bytes: 256 * 1024,
        max_rows: rows,
        server_memory_bytes: 32 * 1024 * 1024,
    }
}
async fn search(pool: &ArangoPool, query: Vec<f32>, mode: &str, count: usize) -> Vec<Value> {
    if mode == "stream" {
        query::query_fold(
            pool,
            "FOR e IN vectors RETURN e",
            json!({}),
            limits(count as u64),
            TopK::new(query, "fixture-v1".into(), "parent_key", 10).unwrap(),
            |top, row| top.insert(row),
            (),
        )
        .await
        .unwrap()
        .into_items()
    } else {
        let probes = if mode == "index-4" { 4 } else { 16 };
        query::query_fold(pool,
            "FOR e IN vectors LET score = APPROX_NEAR_COSINE(e.embedding, @q, {nProbe: @probes}) SORT score DESC LIMIT 10 RETURN {chunk_key:e.chunk_key, score}",
            json!({"q":query,"probes":probes}), limits(10), Vec::new(),
            |rows, row| { rows.push(row); Ok(()) }, ()).await.unwrap()
    }
}
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    sorted[((sorted.len() as f64 * fraction).ceil() as usize).saturating_sub(1)]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "explicit isolated resource-limited benchmark only"]
async fn compare_streaming_and_indexed_retrieval() {
    assert_eq!(std::env::var("ARANGO_TESTS").as_deref(), Ok("1"));
    let dimension: usize = std::env::var("HADES_BENCH_DIMENSION")
        .unwrap_or("64".into())
        .parse()
        .unwrap();
    let count: usize = std::env::var("HADES_BENCH_ROWS")
        .unwrap_or("1024".into())
        .parse()
        .unwrap();
    let trials: usize = std::env::var("HADES_BENCH_TRIALS")
        .unwrap_or("16".into())
        .parse()
        .unwrap();
    assert!(
        (16..=2048).contains(&dimension)
            && (1024..=100_000).contains(&count)
            && (16..=512).contains(&trials)
    );
    with_temp_db("retrieval_bench", Fixtures::Empty, move |pool| async move {
        crud::create_collection(&pool, "vectors", None).await.unwrap();
        for start in (0..count).step_by(8) {
            let docs: Vec<_> = (start..(start+8).min(count)).map(|i| json!({
                "_key":format!("v{i}"), "chunk_key":format!("c{i}"), "parent_key":format!("p{i}"),
                "model":"fixture-v1", "dimension":dimension, "embedding":vector(i,dimension),
            })).collect();
            crud::insert_documents(&pool, "vectors", &docs, false).await.unwrap();
        }
        let queries: Vec<_> = (0..16).map(|i| vector(count+i,dimension)).collect();
        let mut truth = Vec::new();
        for q in &queries { truth.push(search(&pool,q.clone(),"stream",count).await); }
        let started = Instant::now();
        index::create_vector_index(&pool,"vectors","embedding",dimension as u32,Some(16),4,index::VectorMetric::Cosine).await.unwrap();
        let index_build_ms = started.elapsed().as_secs_f64()*1000.0;
        let plan = pool.writer().post("explain", &json!({"query":"FOR e IN vectors SORT APPROX_NEAR_COSINE(e.embedding, @q) DESC LIMIT 10 RETURN e._key", "bindVars":{"q":queries[0]}})).await.unwrap();
        assert!(plan["plan"]["rules"].as_array().unwrap().iter().any(|rule| rule == "use-vector-index"), "{plan}");
        for mode in ["stream", "index-4", "index-16"] {
            for concurrency in [1_usize,4,8] {
                let baseline_rss = rss_kib_of("self");
                let server_pid = std::env::var("HADES_BENCH_SERVER_PID").ok();
                let baseline_server_rss = server_pid.as_deref().map(rss_kib_of);
                let stop = Arc::new(AtomicBool::new(false));
                let flag = stop.clone();
                let sampler = std::thread::spawn(move || {
                    let mut peak = rss_kib_of("self");
                    let mut server_peak = baseline_server_rss;
                    while !flag.load(Ordering::Relaxed) {
                        peak = peak.max(rss_kib_of("self"));
                        if let Some(pid) = &server_pid {
                            server_peak = Some(server_peak.unwrap_or(0).max(rss_kib_of(pid)));
                        }
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    (peak, server_peak)
                });
                let mut elapsed = Vec::new();
                let mut recall = 0.0;
                let wall = Instant::now();
                for start in (0..trials).step_by(concurrency) {
                    let mut pending = tokio::task::JoinSet::new();
                    for trial in start..(start+concurrency).min(trials) {
                        let pool = pool.clone(); let q = queries[trial%16].clone();
                        pending.spawn(async move { let start=Instant::now(); let hits=search(&pool,q,mode,count).await; (trial,hits,start.elapsed().as_secs_f64()*1000.0) });
                    }
                    while let Some(result) = pending.join_next().await {
                        let (trial,hits,ms)=result.unwrap(); elapsed.push(ms);
                        recall += hits.iter().filter(|hit| truth[trial%16].iter().any(|expected| hit["chunk_key"]==expected["chunk_key"])).count() as f64/10.0;
                    }
                }
                stop.store(true,Ordering::Relaxed);
                let (peak, server_peak) = sampler.join().unwrap();
                elapsed.sort_by(f64::total_cmp);
                println!("BENCH {}",json!({"rows":count,"dimension":dimension,"mode":mode,"concurrency":concurrency,"trials":trials,
                    "p50_ms":percentile(&elapsed,0.5),"p95_ms":percentile(&elapsed,0.95),"p99_ms":percentile(&elapsed,0.99),
                    "baseline_rss_kib":baseline_rss,"sampled_peak_rss_kib":peak,
                    "baseline_server_rss_kib":baseline_server_rss,"sampled_peak_server_rss_kib":server_peak,"recall_at_10":recall/trials as f64,
                    "wall_ms":wall.elapsed().as_secs_f64()*1000.0,"index_build_ms":index_build_ms,
                    "scope":"retrieval engine; synthetic vectors; admission not exercised"}));
            }
        }
    }).await;
}
