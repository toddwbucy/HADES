//! Private fixed-vector embedder for isolated contracts and memory measurements.
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
pub(super) async fn start_embedder(
    socket: &std::path::Path,
    count: usize,
    axis: usize,
) -> tokio::task::JoinHandle<()> {
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
            vector[axis] = 1.0;
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
