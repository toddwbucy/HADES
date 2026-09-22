//! Private fixed-vector embedder for isolated contracts and memory measurements.
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
pub(super) async fn start_embedder(
    socket: &std::path::Path,
    count: usize,
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
                json!({"data":[{"id":"jinaai/jina-embeddings-v4","dimension":2048,"max_seq_length":11892}]}).to_string()
            } else {
                embedded += 1;
                json!({"model":"jinaai/jina-embeddings-v4","data":[{"index":0,"embedding":vector}]})
                    .to_string()
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
