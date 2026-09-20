//! Full service-envelope admission and timeout contracts; private sockets only.
use hades_core::config::HadesConfig;
use hades_core::db::{ArangoClient, ArangoPool};
use hades_core::service::{ConnectionPolicy, handle_request};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::Notify;
#[path = "common/cursor_mock.rs"]
mod cursor_mock;
use cursor_mock::{Mock, Reply};

const PAYLOAD: &[u8] = br#"{"command":"db.query","params":{"text":"fixture","limit":1}}"#;

async fn request(pool: &ArangoPool, config: &HadesConfig) -> hades_core::dispatch::DaemonResponse {
    handle_request(
        pool,
        config,
        ConnectionPolicy::agent_only(),
        PAYLOAD,
        Duration::from_secs(10),
    )
    .await
}

static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn start_embedder(socket: &std::path::Path, count: usize) -> tokio::task::JoinHandle<()> {
    let listener = tokio::net::UnixListener::bind(socket).unwrap();
    tokio::spawn(async move {
        for _ in 0..count {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(stream.read_u8().await.unwrap());
                assert!(headers.len() < 8192);
            }
            let len: usize = String::from_utf8(headers)
                .unwrap()
                .lines()
                .find_map(|line| {
                    line.to_lowercase()
                        .strip_prefix("content-length:")
                        .map(|n| n.trim().parse().unwrap())
                })
                .unwrap();
            let mut body = vec![0; len];
            stream.read_exact(&mut body).await.unwrap();
            let mut vector = vec![0.0; 2048];
            vector[0] = 1.0;
            let body = json!({"model":"jinaai/jina-embeddings-v4","data":[{"index":0,"embedding":vector}]}).to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        }
    })
}

#[tokio::test]
async fn overload_is_explicit_and_timeout_retains_budget_until_cursor_cleanup() {
    let _serial = SERIAL.lock().await;
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("embedder.sock");
    let embedder = start_embedder(&socket, 4).await;
    let gates: Vec<_> = (0..4).map(|_| Arc::new(Notify::new())).collect();
    let cleanup = Arc::new(Notify::new());
    let mut replies: Vec<_> = gates
        .iter()
        .map(|gate| Reply::blocked(json!({"id":"123","hasMore":true,"result":[]}), gate.clone()))
        .collect();
    replies.push(Reply::blocked(json!({"error":false}), cleanup.clone()));
    let mut mock = Mock::new(replies).await;
    let mut config = HadesConfig::default();
    config.embedding.service.socket = socket.to_string_lossy().into_owned();
    let mut callers = Vec::new();
    for i in 0..4 {
        let pool = mock.pool.clone();
        let config = config.clone();
        callers.push(tokio::spawn(async move {
            handle_request(
                &pool,
                &config,
                ConnectionPolicy::agent_only(),
                PAYLOAD,
                if i == 0 {
                    Duration::from_secs(1)
                } else {
                    Duration::from_secs(10)
                },
            )
            .await
        }));
        mock.event("POST cursor").await;
    }
    let response = request(&mock.pool, &config).await;
    assert_eq!(response.error_code.as_deref(), Some("SEARCH_OVERLOADED"));
    let timed_out = callers.remove(0).await.unwrap();
    assert_eq!(timed_out.error_code.as_deref(), Some("INTERNAL"));
    assert_eq!(
        request(&mock.pool, &config).await.error_code.as_deref(),
        Some("SEARCH_OVERLOADED")
    );
    assert_eq!(timed_out.error.as_deref(), Some("request timed out"));
    gates[0].notify_one();
    mock.event("DELETE cursor/123").await;
    assert_eq!(
        request(&mock.pool, &config).await.error_code.as_deref(),
        Some("SEARCH_OVERLOADED")
    );
    cleanup.notify_one();
    let mut absent = config.clone();
    absent.embedding.service.socket = directory
        .path()
        .join("absent.sock")
        .to_string_lossy()
        .into_owned();
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let response = request(&mock.pool, &absent).await;
            if response.error_code.as_deref() == Some("SERVICE_ERROR") {
                break;
            }
            assert_eq!(response.error_code.as_deref(), Some("SEARCH_OVERLOADED"));
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    for caller in callers {
        caller.abort();
        let _ = caller.await;
    }
    for gate in &gates[1..] {
        gate.notify_one();
    }
    for _ in 0..3 {
        mock.released().await;
    }
    embedder.await.unwrap();
}

#[tokio::test]
async fn oversized_query_text_is_rejected_before_service_io() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("absent.sock");
    let client = ArangoClient::with_socket(socket.clone(), "fixture", "fixture", "fixture");
    let pool = ArangoPool::new(client.clone(), client);
    let mut config = HadesConfig::default();
    config.embedding.service.socket = socket.to_string_lossy().into_owned();
    let payload =
        serde_json::to_vec(&json!({"command":"db.query", "params":{"text":"x".repeat(64*1024+1)}}))
            .unwrap();
    let response = handle_request(
        &pool,
        &config,
        ConnectionPolicy::agent_only(),
        &payload,
        Duration::from_secs(1),
    )
    .await;
    assert_eq!(response.error_code.as_deref(), Some("INVALID_PARAMS"));
    assert!(response.error.unwrap().contains("64 KiB"));
}

fn stored_vector(index: usize) -> serde_json::Value {
    let mut vector = vec![0.0; 2048];
    vector[0] = 1.0;
    json!({"chunk_key":format!("c{index}"),"parent_key":"p", "model":"jinaai/jina-embeddings-v4", "dimension":2048,"embedding":vector})
}

#[tokio::test]
async fn invalid_stored_vectors_fail_without_partial_results() {
    let _serial = SERIAL.lock().await;
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("embedder.sock");
    let embedder = start_embedder(&socket, 3).await;
    let mut config = HadesConfig::default();
    config.embedding.service.socket = socket.to_string_lossy().into_owned();
    for field in ["model", "dimension", "embedding"] {
        let mut invalid = stored_vector(1);
        invalid[field] = match field {
            "model" => json!("different-model"),
            "dimension" => json!(128),
            _ => json!([1, "bad"]),
        };
        let mut mock = Mock::new(vec![Reply::page(
            json!({"id":"123","hasMore":true,"result":[stored_vector(0),invalid]}),
        )])
        .await;
        let response = request(&mock.pool, &config).await;
        assert!(!response.success);
        assert_eq!(response.error_code.as_deref(), Some("QUERY_FAILED"));
        assert!(response.error.unwrap().contains("reingest the corpus"));
        assert!(response.data.is_none());
        mock.event("POST cursor").await;
        mock.released().await;
    }
    embedder.await.unwrap();
}

#[tokio::test]
async fn oversized_response_and_aggregate_details_fail_without_partial_results() {
    let _serial = SERIAL.lock().await;
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("embedder.sock");
    let embedder = start_embedder(&socket, 2).await;
    let mut config = HadesConfig::default();
    config.embedding.service.socket = socket.to_string_lossy().into_owned();
    let mut mock = Mock::new(vec![
        Reply::page(json!({"hasMore":false,"result":[stored_vector(0)]})),
        Reply::page(
            json!({"hasMore":false,"result":[{"parent_key":"p","text":"x".repeat(300_000)}]}),
        ),
    ])
    .await;
    let response = request(&mock.pool, &config).await;
    assert_eq!(response.error_code.as_deref(), Some("QUERY_FAILED"));
    assert!(response.error.unwrap().contains("length limit exceeded"));
    assert!(response.data.is_none());
    mock.event("POST cursor").await;
    mock.event("POST cursor").await;

    let mut replies = vec![Reply::page(
        json!({"hasMore":false,"result":(0..20).map(stored_vector).collect::<Vec<_>>()}),
    )];
    for page in 0..5 {
        replies.push(Reply::page(json!({"id":"123","hasMore":page<4,"result":(0..4).map(|_|json!({"parent_key":"p","text":"x".repeat(60_000)})).collect::<Vec<_>>()})));
    }
    let mut mock = Mock::new(replies).await;
    let payload =
        serde_json::to_vec(&json!({"command":"db.query","params":{"text":"fixture","limit":20}}))
            .unwrap();
    let response = handle_request(
        &mock.pool,
        &config,
        ConnectionPolicy::agent_only(),
        &payload,
        Duration::from_secs(10),
    )
    .await;
    assert_eq!(response.error_code.as_deref(), Some("QUERY_FAILED"));
    assert!(
        response
            .error
            .unwrap()
            .contains("result byte budget exceeded")
    );
    assert!(response.data.is_none());
    mock.event("POST cursor").await;
    mock.event("POST cursor").await;
    for _ in 0..4 {
        mock.event("POST cursor/123").await;
    }
    mock.released().await;
    embedder.await.unwrap();
}
