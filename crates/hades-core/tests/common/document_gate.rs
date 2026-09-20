//! Forward real private database operations, holding one acknowledged chunk write.
use hades_core::db::{ArangoClient, ArangoPool};
use http_body_util::{BodyExt, Full};
use hyper::{
    Request, Response,
    body::{Bytes, Incoming},
    service::service_fn,
};
use hyper_util::rt::TokioIo;
use serde_json::Value;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::Notify;

pub struct Gate {
    pub pool: ArangoPool,
    pub entered: Arc<Notify>,
    pub release: Arc<Notify>,
    pub aborted: Arc<Notify>,
    task: tokio::task::JoinHandle<()>,
    _dir: tempfile::TempDir,
}
impl Drop for Gate {
    fn drop(&mut self) {
        self.release.notify_one();
        self.task.abort();
    }
}
impl Gate {
    pub async fn new(backend: ArangoClient) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("database.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let aborted = Arc::new(Notify::new());
        let state = (
            entered.clone(),
            release.clone(),
            aborted.clone(),
            Arc::new(AtomicBool::new(true)),
        );
        let client = reqwest::Client::builder()
            .unix_socket(backend.socket_path().unwrap())
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let base = format!("http://localhost/_db/{}/_api/", backend.database());
        let user = std::env::var("HADES_TEST_USER").unwrap_or_else(|_| "root".into());
        let password = std::env::var("ARANGO_PASSWORD").unwrap();
        let task = tokio::spawn(async move {
            let mut peers = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    incoming = listener.accept() => {
                        let (socket, _) = incoming.unwrap();
                        let client = client.clone(); let base = base.clone(); let user = user.clone(); let password = password.clone();
                        let state = state.clone();
                        peers.spawn(async move {
                            let service = service_fn(move |request: Request<Incoming>| {
                                let client = client.clone(); let base = base.clone(); let user = user.clone(); let password = password.clone(); let state = state.clone();
                                async move {
                                    let method = request.method().clone();
                                    let path = request.uri().path_and_query().unwrap().as_str().split_once("/_api/").unwrap().1.to_owned();
                                    let transaction = request.headers().get("x-arango-trx-id").cloned();
                                    let bytes = request.into_body().collect().await.unwrap().to_bytes();
                                    let mut forward = client.request(method.clone(), format!("{base}{path}"))
                                        .basic_auth(user, Some(password)).header("content-type","application/json").body(bytes);
                                    if let Some(id) = transaction { forward = forward.header("x-arango-trx-id",id); }
                                    let response = forward.send().await.unwrap();
                                    let status = response.status().as_u16();
                                    let response: Value = response.json().await.unwrap();
                                    if status < 300 && path.starts_with("document/chunks?") && state.3.swap(false, Ordering::SeqCst) {
                                        state.0.notify_one();
                                        tokio::time::timeout(std::time::Duration::from_secs(5), state.1.notified()).await.unwrap();
                                    }
                                    if method == "DELETE" && path.starts_with("transaction/") && response["result"]["status"] == "aborted" { state.2.notify_one(); }
                                    Ok::<_, std::convert::Infallible>(Response::builder().status(status).body(Full::new(Bytes::from(response.to_string()))).unwrap())
                                }
                            });
                            // A cancelled pipeline deliberately closes its response connection.
                            let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(socket), service).await;
                        });
                    }
                    Some(result) = peers.join_next() => { result.unwrap(); }
                }
            }
        });
        let client = ArangoClient::with_socket(path, "fixture", "fixture", "fixture");
        Self {
            pool: ArangoPool::new(client.clone(), client),
            entered,
            release,
            aborted,
            task,
            _dir: dir,
        }
    }
}
