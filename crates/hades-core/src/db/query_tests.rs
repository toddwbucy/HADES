//! Private mock-socket contracts; never connect to an installed database.
use super::*;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::{Notify, mpsc};
use tokio::task::{JoinHandle, JoinSet};

type Event = (String, Value);
struct Reply {
    status: u16,
    body: Value,
    gate: Option<Arc<Notify>>,
}
impl Reply {
    fn page(body: Value) -> Self {
        Self {
            status: 200,
            body,
            gate: None,
        }
    }
    fn blocked(body: Value, gate: Arc<Notify>) -> Self {
        Self {
            status: 200,
            body,
            gate: Some(gate),
        }
    }
}
struct Mock {
    pool: ArangoPool,
    events: mpsc::UnboundedReceiver<Event>,
    live: Arc<AtomicBool>,
    task: JoinHandle<()>,
    _dir: tempfile::TempDir,
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Mock {
    async fn new(replies: Vec<Reply>) -> Self {
        Self::with_backend(replies, None).await
    }
    async fn with_backend(replies: Vec<Reply>, backend: Option<ArangoClient>) -> Self {
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
    async fn event(&mut self, expected: &str) -> Value {
        let (actual, body) = tokio::time::timeout(Duration::from_secs(2), self.events.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
        body
    }
    async fn released(&mut self) {
        self.event("DELETE cursor/123").await;
        assert!(!self.live.load(Ordering::SeqCst));
    }
}
fn first() -> Value {
    json!({"id":"123", "hasMore":true, "result":[1],"extra":{"stats":{"fullCount":2}}})
}
fn last() -> Value {
    json!({"hasMore":false,"result":[2]})
}
fn run(pool: ArangoPool, limits: QueryLimits) -> JoinHandle<Result<QueryResult, ArangoError>> {
    tokio::spawn(async move {
        query_with_limits(
            &pool,
            "RETURN 1",
            None,
            Some(1),
            true,
            ExecutionTarget::Reader,
            limits,
        )
        .await
    })
}

#[tokio::test]
async fn completes_and_deletes_cursor_on_its_creating_endpoint() {
    let mut mock = Mock::new(vec![Reply::page(first()), Reply::page(last())]).await;
    // The reader must never be contacted when split sockets cannot handle cursor state.
    let absent = ArangoClient::with_socket(
        mock._dir.path().join("absent.sock"),
        "fixture",
        "fixture",
        "fixture",
    );
    mock.pool = ArangoPool::new(absent, mock.pool.writer().clone());
    let query = run(mock.pool.clone(), QueryLimits::default());
    let body = mock.event("POST cursor").await;
    assert_eq!(body["options"]["fullCount"], true);
    assert_eq!(body["options"]["maxRuntime"], 60.0);
    assert_eq!(body["ttl"], 60.0);
    mock.event("POST cursor/123").await;
    mock.released().await;
    let result = query.await.unwrap().unwrap();
    assert_eq!(result.results, vec![json!(1), json!(2)]);
    assert_eq!(result.full_count, Some(2));
}

#[tokio::test]
async fn malformed_first_page_still_releases_known_cursor() {
    for body in [
        json!({"id":"123","hasMore":true}),
        json!({"id":"123","hasMore":true,"result":{}}),
        json!({"id":"123","result":[]}),
    ] {
        let mut mock = Mock::new(vec![Reply::page(body)]).await;
        let query = run(mock.pool.clone(), QueryLimits::default());
        mock.event("POST cursor").await;
        mock.released().await;
        assert!(query.await.unwrap().is_err());
    }
}

#[tokio::test]
async fn continuation_errors_and_malformed_pages_release_cursor() {
    for reply in [
        Reply::page(json!({"id":"123","hasMore":false,"result":null})),
        Reply {
            status: 500,
            body: json!({"error":true,"errorNum":1500,"errorMessage":"fixture"}),
            gate: None,
        },
    ] {
        let mut mock = Mock::new(vec![Reply::page(first()), reply]).await;
        let query = run(mock.pool.clone(), QueryLimits::default());
        mock.event("POST cursor").await;
        mock.event("POST cursor/123").await;
        mock.released().await;
        assert!(query.await.unwrap().is_err());
    }
}

#[tokio::test]
async fn caller_cancellation_during_creation_keeps_response_ownership() {
    let gate = Arc::new(Notify::new());
    let mut mock = Mock::new(vec![Reply::blocked(first(), gate.clone())]).await;
    let query = run(mock.pool.clone(), QueryLimits::default());
    mock.event("POST cursor").await;
    query.abort();
    assert!(query.await.unwrap_err().is_cancelled());
    gate.notify_one();
    mock.released().await; // no continuation request after the caller disappears
}

#[tokio::test]
async fn caller_cancellation_during_pagination_deletes_without_waiting_for_page() {
    let mut mock = Mock::new(vec![
        Reply::page(first()),
        Reply::blocked(last(), Arc::new(Notify::new())),
    ])
    .await;
    let query = run(mock.pool.clone(), QueryLimits::default());
    mock.event("POST cursor").await;
    mock.event("POST cursor/123").await;
    query.abort();
    assert!(query.await.unwrap_err().is_cancelled());
    mock.released().await;
}

#[tokio::test]
async fn total_query_timeout_releases_cursor() {
    let mut mock = Mock::new(vec![
        Reply::page(first()),
        Reply::blocked(last(), Arc::new(Notify::new())),
    ])
    .await;
    let limits = QueryLimits {
        lifetime: Duration::from_millis(100),
        ..QueryLimits::default()
    };
    let query = run(mock.pool.clone(), limits);
    mock.event("POST cursor").await;
    mock.event("POST cursor/123").await;
    mock.released().await;
    assert!(
        query
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("lifetime exceeded")
    );
}

#[tokio::test]
async fn deletion_response_cannot_hold_caller_past_cleanup_budget() {
    let mut mock = Mock::new(vec![
        Reply::page(json!({"id":"123", "hasMore":false,"result":[1]})),
        Reply::blocked(json!({"error":false}), Arc::new(Notify::new())),
    ])
    .await;
    let limits = QueryLimits {
        cleanup: Duration::from_millis(50),
        ..QueryLimits::default()
    };
    let query = run(mock.pool.clone(), limits);
    mock.event("POST cursor").await;
    mock.released().await;
    let result = tokio::time::timeout(Duration::from_secs(1), query)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(result.results, vec![json!(1)]);
}

#[tokio::test]
async fn disposable_database_confirms_cursor_is_gone_after_cancellation() {
    use crate::test_support::{Fixtures, with_temp_db};
    with_temp_db("cursor_cancel", Fixtures::Empty, |database| async move {
        for during_creation in [true, false] {
            let gate = Arc::new(Notify::new());
            let replies = if during_creation {
                vec![Reply::blocked(first(), gate.clone())]
            } else {
                vec![Reply::page(first()), Reply::blocked(last(), gate.clone())]
            };
            let mut proxy = Mock::with_backend(replies, Some(database.writer().clone())).await;
            let pool = proxy.pool.clone();
            let caller = tokio::spawn(async move {
                query(
                    &pool,
                    "FOR n IN 1..5 RETURN n",
                    None,
                    Some(1),
                    false,
                    ExecutionTarget::Reader,
                )
                .await
            });
            let initial = proxy.event("POST cursor").await;
            let id = initial["_cursor_id"]
                .as_str()
                .expect("real database cursor ID");
            if !during_creation {
                proxy.event(&format!("POST cursor/{id}")).await;
            }
            caller.abort();
            assert!(caller.await.unwrap_err().is_cancelled());
            if during_creation {
                gate.notify_one();
            }
            proxy.event(&format!("DELETE cursor/{id}")).await;
            let error = database
                .writer()
                .post(&format!("cursor/{id}"), &json!({}))
                .await
                .unwrap_err();
            assert!(error.is_not_found(), "server cursor still exists: {error}");
        }
    })
    .await;
}
