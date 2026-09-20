//! Opt-in loopback transport measurement. Never connects to production.
use super::*;
use rmcp::model::{CallToolRequestParams, CallToolResult, ContentBlock};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone)]
struct FixtureServer {
    text_bytes: usize,
}
impl rmcp::ServerHandler for FixtureServer {
    async fn call_tool(
        &self,
        _request: CallToolRequestParams,
        _context: rmcp::service::RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        Ok(CallToolResult::success(vec![ContentBlock::text(
            "x".repeat(self.text_bytes),
        )]))
    }
}

fn rss_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")
                .map(|value| value.split_whitespace().next().unwrap().parse().unwrap())
        })
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "explicit resource-limited loopback MCP memory benchmark only"]
async fn measure_mcp_slow_readers() {
    assert_eq!(std::env::var("HADES_TRANSPORT_BENCH").as_deref(), Ok("1"));
    let text_bytes: usize = std::env::var("HADES_TRANSPORT_TEXT_BYTES")
        .unwrap_or_else(|_| (1024 * 1024).to_string())
        .parse()
        .unwrap();
    assert!((1024 * 1024..=MAX_MESSAGE_BYTES - 1024).contains(&text_bytes));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let manager = Arc::new(BoundedSessionManager::default());
    let service = rmcp::transport::StreamableHttpService::new(
        move || Ok(FixtureServer { text_bytes }),
        manager.clone(),
        Default::default(),
    );
    let app = axum::Router::new()
        .nest_service("/mcp", service)
        .layer(axum::middleware::from_fn(
            super::super::mcp_server::enforce_body_limit,
        ));
    let listener = super::super::transport_limits::AdmittedListener(listener);
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let url = format!("http://{address}/mcp");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let init = client.post(&url).header("accept", "application/json, text/event-stream").json(&serde_json::json!({
        "jsonrpc":"2.0", "id":0, "method":"initialize", "params":{
            "protocolVersion":"2025-03-26", "capabilities":{}, "clientInfo":{"name":"isolated-memory-fixture", "version":"1"}
        }
    })).send().await.unwrap().error_for_status().unwrap();
    let session = init
        .headers()
        .get("mcp-session-id")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    init.bytes().await.unwrap();
    client
        .post(&url)
        .header("mcp-session-id", &session)
        .header("accept", "application/json, text/event-stream")
        .json(&serde_json::json!({"jsonrpc":"2.0", "method":"notifications/initialized"}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .bytes()
        .await
        .unwrap();
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
    let mut calls = tokio::task::JoinSet::new();
    for id in 1..=64 {
        let client = client.clone();
        let url = url.clone();
        let session = session.clone();
        calls.spawn(async move {
            client.post(url).header("mcp-session-id", session).header("accept", "application/json, text/event-stream")
                .json(&serde_json::json!({"jsonrpc":"2.0", "id":id, "method":"tools/call", "params":{"name":"memory_fixture"}}))
                .send().await.unwrap().error_for_status().unwrap()
        });
    }
    let mut responses = Vec::new();
    while let Some(response) = calls.join_next().await {
        responses.push(response.unwrap());
    }
    // Keep all bodies unread after headers, exercising Hyper/SSE and socket queues.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let held_rss = rss_kib();
    let mut total_bytes = 0;
    for response in responses {
        let body = response.bytes().await.unwrap();
        assert!(
            body.len() > text_bytes,
            "fixture response must not be an error envelope"
        );
        total_bytes += body.len();
    }
    stop.store(true, Ordering::Relaxed);
    let peak = sampler.join().unwrap();
    manager.close_session(&session.into()).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !manager.closed.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    drop(client);
    server.abort();
    let _ = server.await;
    println!(
        "TRANSPORT_BENCH {}",
        serde_json::json!({"concurrent_unread_responses":64, "tool_text_bytes":text_bytes,
        "baseline_rss_kib":baseline, "held_rss_kib":held_rss, "sampled_peak_rss_kib":peak, "total_response_bytes":total_bytes,
        "scope":"loopback MCP HTTP/SSE with production session and connection admission; synthetic tool; client and server share process; excludes database/search handler and inference"})
    );
}
