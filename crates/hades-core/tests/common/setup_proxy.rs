//! Forward requests only to the current disposable database or its private admin endpoint.
//! Inject failures before/after actual side effects, including lost commit acknowledgements.
use hades_core::db::ArangoClient;
use http_body_util::{BodyExt, Full};
use hyper::{
    Request, Response,
    body::{Bytes, Incoming},
    service::service_fn,
};
use hyper_util::rt::TokioIo;
use serde_json::{Value, json};
use std::{
    os::unix::fs::PermissionsExt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::Notify;

pub struct Rule {
    pub method: &'static str,
    pub path: &'static str,
    pub name: Option<&'static str>,
    pub nth: usize,
    pub before: bool,
    pub hold: bool,
    pub fail: bool,
    pub callback: Option<Box<dyn FnOnce() + Send>>,
}

pub struct Proxy {
    pub admin: ArangoClient,
    pub entered: Arc<Notify>,
    pub release: Arc<Notify>,
    pub hits: Arc<AtomicUsize>,
    #[allow(dead_code)] // Used by request-count contracts; fault-only callers inspect hits.
    pub matched: Arc<AtomicUsize>,
    broken: Arc<AtomicBool>,
    task: tokio::task::JoinHandle<()>,
    _directory: tempfile::TempDir,
}
impl Proxy {
    /// Proxy one test's explicit server; use matching request occurrences to target real DDL/commits.
    pub fn new(backend: &ArangoClient, rule: Rule) -> Self {
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let socket = directory.path().join("proxy.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let client = reqwest::Client::builder()
            .unix_socket(backend.socket_path().unwrap())
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let database = backend.database().to_string();
        let user = std::env::var("HADES_TEST_USER").unwrap_or_else(|_| "root".into());
        let password = std::env::var("ARANGO_PASSWORD").unwrap();
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let hits = Arc::new(AtomicUsize::new(0));
        let matched = Arc::new(AtomicUsize::new(0));
        let broken = Arc::new(AtomicBool::new(false));
        let state = (
            Arc::new(Mutex::new((Some(rule), 0))),
            entered.clone(),
            release.clone(),
            hits.clone(),
            broken.clone(),
            matched.clone(),
        );
        let task = tokio::spawn(async move {
            let mut peers = tokio::task::JoinSet::new();
            loop {
                tokio::select! {
                    incoming = listener.accept() => {
                        let (stream, _) = match incoming {
                            Ok(peer) => peer,
                            Err(_) => { state.4.store(true, Ordering::SeqCst); break; }
                        };
                        let client=client.clone(); let database=database.clone(); let user=user.clone(); let password=password.clone(); let state=state.clone();
                        peers.spawn(async move {
                            let service = service_fn(move |request: Request<Incoming>| {
                                let client=client.clone(); let database=database.clone(); let user=user.clone(); let password=password.clone(); let state=state.clone();
                                async move {
                                    let method=request.method().clone();
                                    let Some(path)=request.uri().path_and_query().map(|p|p.as_str().to_owned()) else {
                                        return broken_response(&state.4,"missing request path");
                                    };
                                    if !(path.starts_with(&format!("/_db/{database}/_api/")) || path.starts_with("/_db/_system/_api/")) {
                                        return broken_response(&state.4,"request escaped disposable scope");
                                    }
                                    let api=path.split_once("/_api/").unwrap().1;
                                    let transaction=request.headers().get("x-arango-trx-id").cloned();
                                    let bytes=match request.into_body().collect().await {
                                        Ok(body)=>body.to_bytes(),
                                        Err(_)=>return broken_response(&state.4,"request body failed"),
                                    };
                                    let body: Value=if bytes.is_empty() { Value::Null } else {
                                        match serde_json::from_slice(&bytes) {
                                            Ok(value)=>value,
                                            Err(_)=>return broken_response(&state.4,"request JSON failed"),
                                        }
                                    };
                                    let fault = {
                                        let mut locked=match state.0.lock() {
                                            Ok(state)=>state,
                                            Err(_)=>return broken_response(&state.4,"proxy rule lock poisoned"),
                                        };
                                        if locked.0.as_ref().is_some_and(|r| r.method == method.as_str() && api.starts_with(r.path) && r.name.is_none_or(|n| body["name"] == n)) {
                                            state.5.fetch_add(1,Ordering::SeqCst);
                                            locked.1 += 1;
                                            if locked.1 == locked.0.as_ref().unwrap().nth { locked.0.take() } else { None }
                                        } else { None }
                                    };
                                    let mut status=500;
                                    let mut response=json!({"error":true,"errorNum":1,"code":500,"errorMessage":"injected setup fault"}).to_string();
                                    if !fault.as_ref().is_some_and(|r| r.before) {
                                        let mut forwarded=client.request(method, format!("http://localhost{path}")).basic_auth(user,Some(password)).header("content-type","application/json").body(bytes);
                                        if let Some(id)=transaction { forwarded=forwarded.header("x-arango-trx-id",id); }
                                        let actual=match forwarded.send().await {
                                            Ok(response)=>response,
                                            Err(_)=>return broken_response(&state.4,"database forwarding failed"),
                                        };
                                        status=actual.status().as_u16();
                                        response=match actual.text().await {
                                            Ok(body)=>body,
                                            Err(_)=>return broken_response(&state.4,"database response body failed"),
                                        };
                                    }
                                    if let Some(mut rule)=fault {
                                        state.3.fetch_add(1,Ordering::SeqCst);
                                        if let Some(callback)=rule.callback.take()
                                            && std::panic::catch_unwind(std::panic::AssertUnwindSafe(callback)).is_err() {
                                            return broken_response(&state.4,"fixture callback panicked");
                                        }
                                        state.1.notify_one();
                                        if rule.hold && tokio::time::timeout(Duration::from_secs(15),state.2.notified()).await.is_err() {
                                            return broken_response(&state.4,"fixture hold timed out");
                                        }
                                        if rule.fail { status=500; response=json!({"error":true,"errorNum":1,"code":500,"errorMessage":"injected setup fault"}).to_string(); }
                                    }
                                    Ok::<_,std::convert::Infallible>(Response::builder().status(status).body(Full::new(Bytes::from(response))).unwrap())
                                }
                            });
                            let _=hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(stream),service).await;
                        });
                    }
                    Some(result)=peers.join_next() => {
                        if result.is_err() { state.4.store(true,Ordering::SeqCst); }
                    }
                }
            }
        });
        Self {
            admin: ArangoClient::with_socket(socket, "_system", "fixture", "fixture"),
            entered,
            release,
            hits,
            matched,
            broken,
            task,
            _directory: directory,
        }
    }
}
impl Drop for Proxy {
    /// Reap only this test's proxy tasks; the shared harness owns database cleanup.
    fn drop(&mut self) {
        self.release.notify_one();
        self.task.abort();
        if !std::thread::panicking() {
            assert!(
                !self.broken.load(Ordering::SeqCst),
                "setup proxy failed internally"
            );
        }
    }
}

/// Separate unexpected infrastructure failures from deliberately injected setup refusals.
fn broken_response(
    broken: &AtomicBool,
    reason: &str,
) -> Result<Response<Full<Bytes>>, std::convert::Infallible> {
    broken.store(true, Ordering::SeqCst);
    let mut response = Response::new(Full::new(Bytes::from(
        json!({"error":true,"errorNum":1,
        "code":502,"errorMessage":format!("unexpected setup proxy failure: {reason}")})
        .to_string(),
    )));
    *response.status_mut() = http::StatusCode::BAD_GATEWAY;
    Ok(response)
}
