//! Historical scheduling probe: copy to crates/hades-prefetch/tests/ to run.
//! No database, GPU, large graph or production endpoint is used.
use std::sync::Arc;
use std::time::{Duration, Instant};
use hades_core::graph::types::GraphData;
use hades_prefetch::prefetcher::{PrefetchConfig, Prefetcher};
use hades_prefetch::tensor::EdgeSplit;

#[test]
fn stopped_prefetch_retains_queued_sampling_graphs() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all().max_blocking_threads(1).build().unwrap();
    runtime.block_on(async {
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        // Occupy the single blocking worker. A finite timeout also releases it
        // if an assertion fails, so runtime destruction cannot hang forever.
        let gate = tokio::task::spawn_blocking(move || {
            entered_tx.send(()).unwrap();
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
        });
        entered_rx.await.unwrap();
        let mut data = GraphData::with_capacity(4, 2);
        data.add_edge(0, 1, 0);
        data.add_edge(1, 2, 0);
        let graph = Arc::new(data);
        let split = Arc::new(EdgeSplit {
            train_idx: vec![0], val_idx: vec![1], test_idx: vec![],
        });
        let prefetch = Prefetcher::start(graph.clone(), split,
            PrefetchConfig::default(), Some(1)).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Arc::strong_count(&graph) < 4 && Instant::now() < deadline {
            tokio::task::yield_now().await;
        }
        assert_eq!(Arc::strong_count(&graph), 4,
            "caller, producer and two queued samplers must retain the graph");
        prefetch.stop();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Arc::strong_count(&graph) > 3 && Instant::now() < deadline {
            tokio::task::yield_now().await;
        }
        assert_eq!(Arc::strong_count(&graph), 3,
            "stopping producer leaves both queued sampler captures alive");
        release_tx.send(()).unwrap();
        gate.await.unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Arc::strong_count(&graph) != 1 && Instant::now() < deadline {
            tokio::task::yield_now().await;
        }
        assert_eq!(Arc::strong_count(&graph), 1,
            "both real samplers must finish and release their captures");
        println!("before_stop=4 after_stop=3 after_sampler_completion=1");
    });
}
