//! Local HTTP server: serves the viewer assets and a graph-data API.
//!
//! The viewer (a WebGL single-page app) fetches `/api/graphs` to populate a
//! graph picker, `/api/graph-data?graph=<name>&limit=<n>` to load a (optionally
//! capped) snapshot, and `/api/expand?graph=&node=` to grow a node's
//! neighborhood on demand. Every route delegates to the [`Backend`] assembler,
//! so the server is a thin shell over the same coupling layer the `dump`
//! subcommand uses.
//!
//! Binding is guarded to loopback or RFC1918/ULA private ranges only — the viewer
//! reads a database and has no auth, so it must never land on a public address.
//! The pinned database is fixed at startup and not client-selectable, containing
//! blast radius to the one graph store this server was pointed at.

use crate::assemble::{Backend, discover_databases};
use anyhow::{Context, Result, anyhow};
use axum::{
    Router,
    extract::{Query, Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
};
use base64::Engine;
use serde::Deserialize;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    bin: String,
    limit: Option<u32>,
    request_timeout: std::time::Duration,
    /// Databases the client may select. Exact-match validated per request, so the
    /// UI's database picker can never reach a database outside this set.
    databases: Arc<Vec<String>>,
    /// When set, every request must present HTTP Basic credentials whose password
    /// matches. Protects the viewer when it's exposed on a LAN address.
    password: Option<Arc<String>>,
    /// Host header values this server will answer to. A browser enforces
    /// same-origin by *name*, not address, so without this an attacker page can
    /// point a short-TTL hostname at 127.0.0.1 (DNS rebinding), reach an
    /// unauthenticated loopback bind as a same-origin request, and read every
    /// served database. Checking Host closes that: the rebound name never matches.
    allowed_hosts: Arc<Vec<String>>,
}

impl AppState {
    /// Resolve a request's target database to a [`Backend`], defaulting to the
    /// first allowlisted database and rejecting anything outside the allowlist.
    #[allow(clippy::result_large_err)] // Err is an axum Response by design (one call site each)
    fn backend_for(&self, db: Option<&str>) -> std::result::Result<Backend, Response> {
        let db = db.unwrap_or(&self.databases[0]);
        if !self.databases.iter().any(|d| d == db) {
            return Err((
                StatusCode::BAD_REQUEST,
                crate::payload::BoundedJson(serde_json::json!({ "error": format!("database `{db}` is not in the allowlist") })),
            )
                .into_response());
        }
        Ok(Backend {
            bin: self.bin.clone(),
            database: db.to_string(),
            limit: self.limit,
        })
    }
}

pub async fn serve(
    bin: String,
    limit: Option<u32>,
    databases: Vec<String>,
    password: Option<String>,
    bind: &str,
) -> Result<()> {
    // No explicit --db list => serve every database the hades user can access.
    let databases = if databases.is_empty() {
        let all = discover_databases(&bin)
            .await
            .context("failed to discover databases (pass --db to list them explicitly)")?;
        tracing::info!(count = all.len(), "serving all accessible databases");
        all
    } else {
        databases
    };
    if databases.is_empty() {
        return Err(anyhow!("no databases to serve"));
    }

    let addr = ensure_private(bind)?;
    // Loud refusal to expose an unauthenticated viewer beyond loopback: with no
    // password, anyone on the LAN could read every served database.
    if password.is_none() && !addr.ip().is_loopback() {
        return Err(anyhow!(
            "refusing to serve {} databases unauthenticated on {} — pass --password <PASS> \
             (browser prompts once), or bind loopback (127.0.0.1)",
            databases.len(),
            addr.ip()
        ));
    }

    let state = AppState {
        bin,
        limit,
        request_timeout: std::time::Duration::from_secs(60),
        databases: Arc::new(databases),
        password: password.map(Arc::new),
        allowed_hosts: Arc::new(allowed_hosts_for(&addr)),
    };
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    tracing::info!(%addr, "hades-viewer serving");
    axum::serve(crate::transport_limits::AdmittedListener(listener), app)
        .with_graceful_shutdown(crate::backend_process::shutdown_requested())
        .await
        .context("server error")?;
    Ok(())
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .route(
            "/",
            get(|| async { asset(INDEX_HTML, "text/html; charset=utf-8") }),
        )
        .route("/viewer.js", get(|| async { asset(VIEWER_JS, JS) }))
        .route(
            "/vendor/graphology.min.js",
            get(|| async { asset(GRAPHOLOGY_JS, JS) }),
        )
        .route(
            "/vendor/sigma.min.js",
            get(|| async { asset(SIGMA_JS, JS) }),
        )
        .route(
            "/vendor/fa2.bundle.min.js",
            get(|| async { asset(FA2_JS, JS) }),
        )
        .route("/api/databases", get(list_databases))
        .route("/api/graphs", get(list_graphs))
        .route("/api/graph-data", get(graph_data))
        .route("/api/expand", get(expand))
        .route("/api/node", get(node))
        .route("/api/content", get(content))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            request_deadline,
        ))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_auth))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_known_host,
        ))
        .with_state(state)
}

/// The Host header values this server answers to: the literal bind address, plus
/// the loopback spellings when bound to loopback.
fn allowed_hosts_for(addr: &SocketAddr) -> Vec<String> {
    let port = addr.port();
    let mut hosts = vec![addr.to_string(), format!("{}:{}", addr.ip(), port)];
    if addr.ip().is_loopback() {
        hosts.push(format!("localhost:{port}"));
        hosts.push(format!("127.0.0.1:{port}"));
        hosts.push(format!("[::1]:{port}"));
    }
    hosts.sort();
    hosts.dedup();
    hosts
}

/// Reject requests whose Host header isn't one this server was bound as, which
/// defeats DNS rebinding against the unauthenticated loopback bind.
async fn require_known_host(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default();
    if state.allowed_hosts.iter().any(|h| h == host) {
        return next.run(req).await;
    }
    tracing::warn!(%host, "rejected request with unrecognized Host header");
    (
        StatusCode::MISDIRECTED_REQUEST,
        crate::payload::BoundedJson(serde_json::json!({ "error": "unrecognized Host header" })),
    )
        .into_response()
}

/// HTTP Basic auth gate (no-op when no password is configured). The username is
/// ignored; only the password must match, compared in constant time.
async fn require_auth(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let Some(expected) = state.password.as_ref() else {
        return next.run(req).await; // auth disabled
    };
    let supplied = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Basic "))
        .and_then(|b64| base64::engine::general_purpose::STANDARD.decode(b64).ok())
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .and_then(|creds| creds.split_once(':').map(|(_, p)| p.to_string()));

    if supplied.map(|p| constant_time_eq(p.as_bytes(), expected.as_bytes())) == Some(true) {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Basic realm=\"hades-viewer\"")],
            "authentication required",
        )
            .into_response()
    }
}

/// Length-independent, content-constant-time byte comparison for the password.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Parse and gate the bind address: loopback or a private LAN range only. The
/// viewer is unauthenticated, so a public bind is refused outright rather than
/// silently exposing a database to the internet.
fn ensure_private(bind: &str) -> Result<SocketAddr> {
    let addr: SocketAddr = bind
        .parse()
        .with_context(|| format!("invalid bind address `{bind}` (want IP:PORT)"))?;
    let private = match addr.ip() {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
        // Stable-Rust equivalents of ULA (fc00::/7) and link-local (fe80::/10),
        // since Ipv6Addr::is_unique_local / is_unicast_link_local are unstable.
        IpAddr::V6(v6) => {
            let o = v6.octets();
            v6.is_loopback() || (o[0] & 0xfe) == 0xfc || (o[0] == 0xfe && (o[1] & 0xc0) == 0x80)
        }
    };
    if !private {
        return Err(anyhow!(
            "refusing to bind non-private address {} — use loopback or a private LAN IP \
             (10/8, 172.16/12, 192.168/16)",
            addr.ip()
        ));
    }
    Ok(addr)
}

// Frontend assets, embedded at compile time so the binary is self-contained
// (no CDN, no external files at runtime). Vendored libs are pinned; see
// web/vendor/PROVENANCE.md.
const INDEX_HTML: &str = include_str!("../web/index.html");
const VIEWER_JS: &str = include_str!("../web/viewer.js");
const GRAPHOLOGY_JS: &str = include_str!("../web/vendor/graphology.min.js");
const SIGMA_JS: &str = include_str!("../web/vendor/sigma.min.js");
const FA2_JS: &str = include_str!("../web/vendor/fa2.bundle.min.js");
const JS: &str = "text/javascript; charset=utf-8";

fn asset(body: &'static str, content_type: &'static str) -> Response {
    ([(header::CONTENT_TYPE, content_type)], body).into_response()
}

/// The databases this server may serve; the first is the client default.
async fn list_databases(State(state): State<AppState>) -> Response {
    crate::payload::BoundedJson(serde_json::json!({ "databases": &*state.databases }))
        .into_response()
}

#[derive(Deserialize)]
struct DbQuery {
    db: Option<String>,
}

async fn list_graphs(State(state): State<AppState>, Query(q): Query<DbQuery>) -> Response {
    let backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match backend.list_graphs().await {
        Ok(graphs) => {
            let names: Vec<&str> = graphs.iter().map(|g| g.name.as_str()).collect();
            crate::payload::BoundedJson(serde_json::json!({ "graphs": names })).into_response()
        }
        Err(e) => error_response(e),
    }
}

#[derive(Deserialize)]
struct GraphQuery {
    db: Option<String>,
    graph: String,
    /// Optional per-collection cap for server-side slicing of large graphs.
    limit: Option<u32>,
}

async fn graph_data(State(state): State<AppState>, Query(q): Query<GraphQuery>) -> Response {
    let mut backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    // Per-request limit may only narrow the operator's cap, never widen it.
    // An unclamped override would let any client erase `--limit` and force a
    // full in-memory export of every collection.
    backend.limit = match (state.limit, q.limit) {
        (Some(op), Some(client)) => Some(op.min(client)),
        (Some(op), None) => Some(op),
        (None, client) => client,
    };
    match backend.assemble(&q.graph).await {
        Ok(snapshot) => crate::payload::BoundedJson(snapshot).into_response(),
        Err(e) => error_response(e),
    }
}

#[derive(Deserialize)]
struct ExpandQuery {
    db: Option<String>,
    graph: String,
    node: String,
    #[serde(default = "default_direction")]
    direction: String,
    #[serde(default = "default_expand_limit")]
    limit: u32,
}

fn default_direction() -> String {
    "any".to_string()
}
fn default_expand_limit() -> u32 {
    50
}

/// Grow one node's immediate neighborhood — the interactive click-to-expand path
/// for graphs too large to load whole.
async fn expand(State(state): State<AppState>, Query(q): Query<ExpandQuery>) -> Response {
    let backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match backend
        .neighbors(&q.graph, &q.node, &q.direction, q.limit)
        .await
    {
        Ok(delta) => crate::payload::BoundedJson(delta).into_response(),
        Err(e) => error_response(e),
    }
}

#[derive(Deserialize)]
struct NodeQuery {
    db: Option<String>,
    /// Full ArangoDB `_id` (`collection/key`).
    id: String,
}

/// Fetch one node by `_id` — lets a deep link resolve a node that isn't part of
/// the currently loaded slice.
async fn node(State(state): State<AppState>, Query(q): Query<NodeQuery>) -> Response {
    let backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match backend.fetch_node(&q.id).await {
        Ok(Some(n)) => crate::payload::BoundedJson(n).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            crate::payload::BoundedJson(
                serde_json::json!({ "error": format!("node `{}` not found", q.id) }),
            ),
        )
            .into_response(),
        Err(e) => error_response(e),
    }
}

#[derive(Deserialize)]
struct ContentQuery {
    db: Option<String>,
    /// The `file_key` whose source chunks to fetch (codebase-schema graphs).
    file_key: String,
}

/// Fetch a node's associated source text (ordered chunks). Codebase-schema-aware;
/// returns an empty list for graphs without a `codebase_chunks` collection.
async fn content(State(state): State<AppState>, Query(q): Query<ContentQuery>) -> Response {
    let backend = match state.backend_for(q.db.as_deref()) {
        Ok(b) => b,
        Err(r) => return r,
    };
    match backend.file_content(&q.file_key).await {
        Ok(chunks) => {
            crate::payload::BoundedJson(serde_json::json!({ "chunks": chunks })).into_response()
        }
        Err(e) => error_response(e),
    }
}

fn error_response(e: anyhow::Error) -> Response {
    tracing::error!(error = %e, "request failed");
    let status = if e.is::<crate::backend_process::Overloaded>() {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::BAD_GATEWAY
    };
    (
        status,
        crate::payload::BoundedJson(serde_json::json!({ "error": e.to_string() })),
    )
        .into_response()
}

/// Bound the whole assembly, including sequences of individually bounded children.
async fn request_deadline(State(state): State<AppState>, req: Request, next: Next) -> Response {
    match tokio::time::timeout(state.request_timeout, next.run(req)).await {
        Ok(response) => response,
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            "viewer request deadline exceeded",
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use std::os::unix::fs::PermissionsExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn router_enforces_host_auth_database_and_bounded_backend_errors() {
        let _fixture = crate::backend_process::SCRIPT_FIXTURE.lock().await;
        let root = tempfile::tempdir().unwrap();
        let bin = root.path().join("backend");
        let marker = root.path().join("called");
        let script = format!(
            "#!/usr/bin/python3\nimport pathlib\npathlib.Path({}).touch()\nprint('{{\"data\":{{\"graphs\":[{{\"name\":\"fixture\",\"edge_definitions\":[]}}]}}}}')\n",
            serde_json::to_string(marker.to_str().unwrap()).unwrap()
        );
        std::fs::write(&bin, script).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o700)).unwrap();
        let router = build_router(AppState {
            bin: bin.to_str().unwrap().into(),
            limit: Some(1),
            request_timeout: std::time::Duration::from_secs(60),
            databases: Arc::new(vec!["fixture".into()]),
            password: Some(Arc::new("fixture-password".into())),
            allowed_hosts: Arc::new(vec!["localhost:12345".into()]),
        });
        let authorization = format!(
            "Basic {}",
            base64::engine::general_purpose::STANDARD.encode("u:fixture-password")
        );
        for (uri, host, auth, expected) in [
            (
                "/api/graphs",
                "rebound.invalid",
                authorization.as_str(),
                StatusCode::MISDIRECTED_REQUEST,
            ),
            ("/", "localhost:12345", "", StatusCode::UNAUTHORIZED),
            (
                "/api/graphs",
                "localhost:12345",
                "Basic bad",
                StatusCode::UNAUTHORIZED,
            ),
            (
                "/api/graphs?db=outside",
                "localhost:12345",
                authorization.as_str(),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let request = Request::builder()
                .uri(uri)
                .header(header::HOST, host)
                .header(header::AUTHORIZATION, auth)
                .body(Body::empty())
                .unwrap();
            assert_eq!(
                router.clone().oneshot(request).await.unwrap().status(),
                expected
            );
            assert!(!marker.exists(), "rejected route launched a backend");
        }
        for db in ["", "?db=fixture"] {
            let request = Request::builder()
                .uri(format!("/api/graphs{db}"))
                .header(header::HOST, "localhost:12345")
                .header(header::AUTHORIZATION, &authorization)
                .body(Body::empty())
                .unwrap();
            let response = router.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = to_bytes(response.into_body(), 1024).await.unwrap();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
                serde_json::json!({"graphs":["fixture"]})
            );
        }
        assert!(marker.exists());
        for script in [
            "#!/usr/bin/python3\nprint('not json')\n",
            "#!/usr/bin/python3\nimport sys\nsys.stderr.write('synthetic-private-diagnostic')\nsys.exit(1)\n",
        ] {
            std::fs::write(&bin, script).unwrap();
            let request = Request::builder()
                .uri("/api/graphs")
                .header(header::HOST, "localhost:12345")
                .header(header::AUTHORIZATION, &authorization)
                .body(Body::empty())
                .unwrap();
            let response = router.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
            let bytes = to_bytes(response.into_body(), 1024).await.unwrap();
            assert!(!String::from_utf8_lossy(&bytes).contains("synthetic-private-diagnostic"));
        }
    }
    #[tokio::test]
    async fn graph_route_rejects_cumulative_collection_document_and_byte_overflow() {
        let _fixture = crate::backend_process::SCRIPT_FIXTURE.lock().await;
        for (mode, expected) in [
            ("collections", "32-collection"),
            ("documents", "document budget"),
            ("bytes", "aggregate byte budget"),
        ] {
            let root = tempfile::tempdir().unwrap();
            let bin = root.path().join("backend");
            let script = format!(
                r#"#!/usr/bin/python3
import json, pathlib, sys
mode = {mode:?}
args = sys.argv[1:]
with pathlib.Path(__file__).with_name('calls').open('a') as f:
    f.write(json.dumps(args) + '\n')
if 'graph' in args:
    names = ['n' + str(i) for i in range(33)] if mode == 'collections' else ['left', 'right']
    print(json.dumps({{'data':{{'graphs':[{{'name':'fixture','edge_definitions':[{{'collection':'edges','from':names,'to':[]}}]}}]}}}}))
elif 'export' in args:
    col = args[args.index('export') + 1]
    if mode == 'documents':
        for i in range(30000): print(json.dumps({{'_id': col + '/' + str(i)}}))
    elif mode == 'bytes':
        print(json.dumps({{'_id': col + '/one', 'text':'X' * (4 * 1024 * 1024 + 1024)}}))
else:
    sys.exit(9)
"#
            );
            std::fs::write(&bin, script).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o700)).unwrap();
            let router = build_router(AppState {
                bin: bin.to_str().unwrap().into(),
                limit: None,
                request_timeout: std::time::Duration::from_secs(60),
                databases: Arc::new(vec!["fixture".into()]),
                password: None,
                allowed_hosts: Arc::new(vec!["localhost:12345".into()]),
            });
            let request = Request::builder()
                .uri("/api/graph-data?graph=fixture")
                .header(header::HOST, "localhost:12345")
                .body(Body::empty())
                .unwrap();
            let response =
                tokio::time::timeout(std::time::Duration::from_secs(10), router.oneshot(request))
                    .await
                    .unwrap()
                    .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{mode}");
            let bytes = to_bytes(response.into_body(), 1024).await.unwrap();
            assert!(
                String::from_utf8_lossy(&bytes).contains(expected),
                "{mode}: {bytes:?}"
            );
            let calls = std::fs::read_to_string(root.path().join("calls")).unwrap();
            let count = calls.lines().count();
            assert_eq!(
                count,
                if mode == "collections" { 1 } else { 3 },
                "{mode}: {calls}"
            );
            assert!(
                !calls.contains("collections"),
                "overflow must stop before count lookup"
            );
        }
    }
    #[tokio::test]
    async fn concurrent_routes_cap_actual_children_and_readmit_after_completion() {
        let _fixture = crate::backend_process::SCRIPT_FIXTURE.lock().await;
        let root = tempfile::tempdir().unwrap();
        let bin = root.path().join("backend");
        std::fs::write(
            &bin,
            r#"#!/usr/bin/python3
import os, pathlib, time
root = pathlib.Path(__file__).parent
(root / ('started-' + str(os.getpid()))).touch()
deadline = time.monotonic() + 10
while not (root / 'release').exists():
    if time.monotonic() > deadline: raise SystemExit(9)
    time.sleep(0.01)
print('{"data":{"graphs":[]}}')
"#,
        )
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o700)).unwrap();
        let router = build_router(AppState {
            bin: bin.to_str().unwrap().into(),
            limit: Some(1),
            request_timeout: std::time::Duration::from_secs(60),
            databases: Arc::new(vec!["fixture".into()]),
            password: None,
            allowed_hosts: Arc::new(vec!["localhost:12345".into()]),
        });
        let request = || {
            Request::builder()
                .uri("/api/graphs")
                .header(header::HOST, "localhost:12345")
                .body(Body::empty())
                .unwrap()
        };
        let mut calls = tokio::task::JoinSet::new();
        for _ in 0..8 {
            calls.spawn(router.clone().oneshot(request()));
        }
        // Four admitted children are held by the fixture gate. The remaining
        // requests must fail promptly without launching additional processes.
        for _ in 0..4 {
            let response =
                tokio::time::timeout(std::time::Duration::from_secs(3), calls.join_next())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap()
                    .unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        }
        let pids = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                let pids: Vec<u32> = std::fs::read_dir(root.path())
                    .unwrap()
                    .filter_map(|entry| {
                        entry
                            .unwrap()
                            .file_name()
                            .to_str()?
                            .strip_prefix("started-")?
                            .parse()
                            .ok()
                    })
                    .collect();
                if pids.len() == 4 {
                    break pids;
                }
                assert!(pids.len() < 4, "admission launched too many children");
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(
            pids.iter()
                .all(|pid| std::path::Path::new(&format!("/proc/{pid}")).exists())
        );
        std::fs::write(root.path().join("release"), "").unwrap();
        while let Some(result) = calls.join_next().await {
            assert_eq!(result.unwrap().unwrap().status(), StatusCode::OK);
        }
        assert!(
            pids.iter()
                .all(|pid| !std::path::Path::new(&format!("/proc/{pid}")).exists()),
            "responses returned before direct children were reaped"
        );
        assert_eq!(
            router.oneshot(request()).await.unwrap().status(),
            StatusCode::OK
        );
    }
    #[tokio::test]
    async fn whole_request_deadline_spans_multiple_backend_calls() {
        let _fixture = crate::backend_process::SCRIPT_FIXTURE.lock().await;
        let root = tempfile::tempdir().unwrap();
        let bin = root.path().join("backend");
        std::fs::write(&bin, r#"#!/usr/bin/python3
import json, os, pathlib, sys, time
root = pathlib.Path(__file__).parent
(root / ('pid-' + str(os.getpid()))).touch()
time.sleep(0.1)
if 'graph' in sys.argv:
    print(json.dumps({'data':{'graphs':[{'name':'fixture','edge_definitions':[{'collection':'edges','from':['left'],'to':['right']}]}]}}))
elif 'export' in sys.argv:
    col = sys.argv[sys.argv.index('export') + 1]
    print(json.dumps({'_id':col + '/one'}))
else:
    print(json.dumps({'data':{'collections':[]}}))
"#).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o700)).unwrap();
        let router = build_router(AppState {
            bin: bin.to_str().unwrap().into(),
            limit: None,
            request_timeout: std::time::Duration::from_millis(350),
            databases: Arc::new(vec!["fixture".into()]),
            password: None,
            allowed_hosts: Arc::new(vec!["localhost:12345".into()]),
        });
        let request = Request::builder()
            .uri("/api/graph-data?graph=fixture")
            .header(header::HOST, "localhost:12345")
            .body(Body::empty())
            .unwrap();
        let response = router.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let pids: Vec<u32> = std::fs::read_dir(root.path())
            .unwrap()
            .filter_map(|entry| {
                entry
                    .unwrap()
                    .file_name()
                    .to_str()?
                    .strip_prefix("pid-")?
                    .parse()
                    .ok()
            })
            .collect();
        assert!(
            pids.len() >= 2,
            "deadline did not exercise multiple children"
        );
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while pids
                .iter()
                .any(|pid| std::path::Path::new(&format!("/proc/{pid}")).exists())
            {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("request deadline left an unreaped direct child");
    }
}
