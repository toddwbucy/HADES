//! Shared private socket fixture; only test targets compile this module.
use super::{ArangoClient, ArangoPool};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::{Notify, mpsc};
use tokio::task::{JoinHandle, JoinSet};
type Event = (String, Value);
pub(super) struct Reply {
    pub(super) status: u16,
    pub(super) body: Value,
    pub(super) gate: Option<Arc<Notify>>,
}
impl Reply {
    pub(super) fn page(body: Value) -> Self {
        Self {
            status: 200,
            body,
            gate: None,
        }
    }
    pub(super) fn blocked(body: Value, gate: Arc<Notify>) -> Self {
        Self {
            status: 200,
            body,
            gate: Some(gate),
        }
    }
}
pub(super) struct Mock {
    pub(super) pool: ArangoPool,
    pub(super) events: mpsc::UnboundedReceiver<Event>,
    pub(super) live: Arc<AtomicBool>,
    pub(super) task: JoinHandle<()>,
    pub(super) _dir: tempfile::TempDir,
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Mock {
    pub(super) async fn new(replies: Vec<Reply>) -> Self {
        Self::with_backend(replies, None).await
    }
    pub(super) async fn with_backend(replies: Vec<Reply>, backend: Option<ArangoClient>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cursor.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let client = ArangoClient::with_socket(path, "fixture", "fixture", "fixture");
        let pool = ArangoPool::new(client.clone(), client);
        let replies = Arc::new(Mutex::new(VecDeque::from(replies)));
        let live = Arc::new(AtomicBool::new(false));
        let state = live.clone();
        let (events, receiver) = mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let mut requests = JoinSet::new();
            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let (mut stream, _) = accepted.unwrap();
                        let replies = replies.clone();
                        let backend = backend.clone();
                        let live = state.clone();
                        let events = events.clone();
                        requests.spawn(async move {
                            let mut bytes = Vec::new();
                            let (header_end, length) = loop {
                                let mut buf = [0; 4096];
                                let n = stream.read(&mut buf).await.unwrap();
                                if n == 0 { return; }
                                bytes.extend_from_slice(&buf[..n]);
                                if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                                    let headers = String::from_utf8_lossy(&bytes[..end]);
                                    let length = headers.lines().find_map(|line| {
                                        line.to_lowercase().strip_prefix("content-length:").map(|n| n.trim().parse::<usize>().unwrap())
                                    }).unwrap_or(0);
                                    if bytes.len() >= end + 4 + length { break (end, length); }
                                }
                            };
                            let line = String::from_utf8_lossy(&bytes[..header_end]).lines().next().unwrap().to_owned();
                            let mut parts = line.split_whitespace();
                            let method = parts.next().unwrap();
                            let path = parts.next().unwrap().strip_prefix("/_db/fixture/_api/").unwrap();
                            let mut body = if length == 0 { Value::Null } else { serde_json::from_slice(&bytes[header_end+4..header_end+4+length]).unwrap() };
                            let reply = if method == "DELETE" {
                                if let Some(backend) = &backend { backend.delete(path).await.unwrap(); }
                                let reply = replies.lock().unwrap().pop_front()
                                    .unwrap_or_else(|| Reply::page(json!({"error":false,"code":202})));
                                if reply.status < 300 { live.store(false, Ordering::SeqCst); }
                                reply
                            } else {
                                let mut reply = replies.lock().unwrap().pop_front().expect("unexpected request");
                                if path == "cursor" && let Some(backend) = &backend {
                                    reply.body = backend.post(path, &body).await.unwrap();
                                    body["_cursor_id"] = reply.body["id"].clone();
                                }
                                if path == "cursor" && reply.body.get("id").is_some() { live.store(true, Ordering::SeqCst); }
                                reply
                            };
                            events.send((format!("{method} {path}"), body)).unwrap();
                            if let Some(gate) = reply.gate { gate.notified().await; }
                            let body = reply.body.to_string();
                            let wire = format!("HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", reply.status, body.len(), body);
                            // Cancellation can close an in-flight continuation connection.
                            let _ = stream.write_all(wire.as_bytes()).await;
                        });
                    }
                    Some(result) = requests.join_next() => { result.unwrap(); }
                }
            }
        });
        Self {
            pool,
            events: receiver,
            live,
            task,
            _dir: dir,
        }
    }
    pub(super) async fn event(&mut self, expected: &str) -> Value {
        let (actual, body) = tokio::time::timeout(Duration::from_secs(2), self.events.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
        body
    }
    pub(super) async fn released(&mut self) {
        self.event("DELETE cursor/123").await;
        assert!(!self.live.load(Ordering::SeqCst));
    }
}
