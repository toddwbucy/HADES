//! Private fixed-vector embedder for isolated contracts and memory measurements.
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
pub(super) async fn start_embedder(
    socket: &std::path::Path,
    count: usize,
) -> tokio::task::JoinHandle<()> {
    start_embedder_with_ceiling(socket, count, 11892).await
}
pub(super) async fn start_embedder_with_ceiling(
    socket: &std::path::Path,
    count: usize,
    ceiling: u32,
) -> tokio::task::JoinHandle<()> {
    let listener = tokio::net::UnixListener::bind(socket).unwrap();
    tokio::spawn(async move {
        let mut embedded = 0;
        while embedded < count {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut headers = Vec::new();
            while !headers.ends_with(b"\r\n\r\n") {
                headers.push(stream.read_u8().await.unwrap());
                assert!(headers.len() < 8192);
            }
            let headers = String::from_utf8(headers).unwrap();
            let models = headers.starts_with("GET /v1/models");
            let len: usize = headers
                .lines()
                .find_map(|line| {
                    line.to_lowercase()
                        .strip_prefix("content-length:")
                        .map(|n| n.trim().parse().unwrap())
                })
                .unwrap_or(0);
            let mut body = vec![0; len];
            stream.read_exact(&mut body).await.unwrap();
            let mut vector = vec![0.0; 2048];
            vector[0] = 1.0;
            let body = if models {
                json!({"data":[{"id":"jinaai/jina-embeddings-v4","dimension":2048,"max_seq_length":ceiling}]}).to_string()
            } else {
                embedded += 1;
                let request: serde_json::Value = serde_json::from_slice(&body).unwrap();
                let data = if let Some(bounds) = request["late_chunk"]["boundaries"].as_array() {
                    bounds
                        .iter()
                        .enumerate()
                        .map(|(i, b)| {
                            json!({"index":0,"chunk_index":i,
                        "char_start":b[0],"char_end":b[1],"embedding":vector})
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![json!({"index":0,"embedding":vector})]
                };
                json!({"model":"jinaai/jina-embeddings-v4","data":data}).to_string()
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        }
    })
}
