//! Guided-reading HTTP adapter. Domain operations stay in the daemon/core.
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Query, Request, State},
    http::{Method, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine;
use hades_core::{
    daemon_client::DaemonClient,
    dispatch::DaemonCommand,
    reading::{Mutation, OpenRequest, ReadRequest, SourceRequest},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::PathBuf, time::Duration};
#[derive(Clone)]
pub struct ReadingBackend {
    pub socket: PathBuf,
    pub database: String,
}
impl ReadingBackend {
    pub async fn verify(&self) -> Result<(), String> {
        let mut client = DaemonClient::connect(&self.socket)
            .await
            .map_err(|e| e.to_string())?;
        let reply = client
            .request_once(DaemonCommand::ReadingRead(ReadRequest {
                session: None,
                view: String::new(),
                source_hash: None,
            }))
            .await
            .map_err(|e| e.to_string())?;
        if !reply.success {
            return Err(reply
                .error
                .unwrap_or_else(|| "reading service unavailable".into()));
        }
        if reply.data.as_ref().and_then(|v| v["database"].as_str()) != Some(self.database.as_str())
        {
            return Err("reading daemon database does not match viewer configuration".into());
        }
        Ok(())
    }
    async fn call(&self, cmd: DaemonCommand) -> Result<Value, String> {
        // Verify on the same connection so a socket retarget/restart cannot change the database silently.
        let mut client = DaemonClient::connect_with_timeout(&self.socket, Duration::from_secs(620))
            .await
            .map_err(|e| e.to_string())?;
        let reply = client
            .request_once(DaemonCommand::ReadingRead(ReadRequest {
                session: None,
                view: String::new(),
                source_hash: None,
            }))
            .await
            .map_err(|e| e.to_string())?;
        if !reply.success
            || reply.data.as_ref().and_then(|v| v["database"].as_str())
                != Some(self.database.as_str())
        {
            return Err("reading database verification failed".into());
        }
        let reply = client.request_once(cmd).await.map_err(|e| e.to_string())?;
        if reply.success {
            Ok(reply.data.unwrap_or(Value::Null))
        } else {
            Err(reply
                .error
                .unwrap_or_else(|| "reading operation failed".into()))
        }
    }
}
#[derive(Clone)]
struct ReadingState(Option<ReadingBackend>);
pub fn router<S: Clone + Send + Sync + 'static>(backend: Option<ReadingBackend>) -> Router<S> {
    Router::new()
        .route(
            "/reading",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                    include_str!("../web/reading.html"),
                )
            }),
        )
        .route(
            "/reading.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    include_str!("../web/reading.js"),
                )
            }),
        )
        .route(
            "/reading.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                    include_str!("../web/reading.css"),
                )
            }),
        )
        .route(
            "/vendor/pdf_viewer.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css")],
                    include_str!("../web/vendor/pdf_viewer.css"),
                )
            }),
        )
        .route(
            "/vendor/pdf.mjs",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript")],
                    include_str!("../web/vendor/pdf.mjs"),
                )
            }),
        )
        .route(
            "/vendor/pdf.worker.mjs",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript")],
                    include_str!("../web/vendor/pdf.worker.mjs"),
                )
            }),
        )
        .route(
            "/reading-chat.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "application/javascript")],
                    include_str!("../web/reading-chat.js"),
                )
            }),
        )
        .route("/api/reading/chat", post(chat))
        .route("/api/reading/config", get(configuration))
        .route("/api/reading/capabilities", get(capabilities))
        .route("/api/reading/state", get(read))
        .route("/api/reading/original", get(original))
        .route("/api/reading/upload", post(upload))
        .route("/api/reading/extract", post(extract))
        .route("/api/reading/open", post(open))
        .route("/api/reading/open-or-resume", post(open_or_resume))
        .route("/api/reading/mutate", post(mutate))
        .layer(DefaultBodyLimit::max(12 * 1024 * 1024))
        .route_layer(middleware::from_fn_with_state(
            ReadingState(backend.clone()),
            browser_write_gate,
        ))
        .with_state(ReadingState(backend))
}
async fn browser_write_gate(
    State(state): State<ReadingState>,
    req: Request,
    next: Next,
) -> Response {
    if req.method() != Method::GET && req.method() != Method::HEAD {
        let headers = req.headers();
        let origin = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok());
        let host = headers.get(header::HOST).and_then(|v| v.to_str().ok());
        let same = origin
            .zip(host)
            .is_some_and(|(o, h)| o == format!("http://{h}"));
        let custom = headers.get("x-hades-workspace").is_some_and(|h| h == "1");
        let fetch = headers
            .get("sec-fetch-site")
            .is_none_or(|h| h == "same-origin");
        if !same || !custom || !fetch {
            return (
                StatusCode::FORBIDDEN,
                Json(json!({"error":"same-origin workspace write required"})),
            )
                .into_response();
        }
    }
    if req.method() != Method::GET
        && req.method() != Method::HEAD
        && let Some(backend) = &state.0
        && req
            .headers()
            .get("x-hades-database")
            .and_then(|v| v.to_str().ok())
            != Some(backend.database.as_str())
    {
        return (
            StatusCode::CONFLICT,
            Json(json!({"error":"reading database changed; reload before writing"})),
        )
            .into_response();
    }
    // No cross-origin allow headers, including OPTIONS/preflight.
    let mut response = next.run(req).await;
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
        .headers_mut()
        .insert(header::X_CONTENT_TYPE_OPTIONS, "nosniff".parse().unwrap());
    response
}
async fn call(state: ReadingState, cmd: DaemonCommand) -> Response {
    let Some(backend) = state.0 else {
        return (StatusCode::SERVICE_UNAVAILABLE,Json(json!({"error":"guided reading is not configured; start viewer with --reading-socket and --reading-db"}))).into_response();
    };
    match backend.call(cmd).await {
        Ok(v) => Json(v).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error":e}))).into_response(),
    }
}
async fn chat(
    State(s): State<ReadingState>,
    Json(p): Json<hades_core::reading::chat::Request>,
) -> Response {
    call(s, DaemonCommand::ReadingChat(p)).await
}
async fn configuration(State(s): State<ReadingState>) -> Json<Value> {
    Json(json!({"enabled":s.0.is_some(),"database":s.0.map(|b|b.database)}))
}
async fn capabilities(State(s): State<ReadingState>) -> Response {
    call(s, DaemonCommand::ReadingCapabilities {}).await
}
async fn read(State(s): State<ReadingState>, Query(p): Query<ReadRequest>) -> Response {
    call(s, DaemonCommand::ReadingRead(p)).await
}
async fn upload(State(s): State<ReadingState>, Json(p): Json<SourceRequest>) -> Response {
    call(s, DaemonCommand::ReadingUpload(p)).await
}
async fn open(State(s): State<ReadingState>, Json(p): Json<OpenRequest>) -> Response {
    call(s, DaemonCommand::ReadingOpen(p)).await
}
async fn open_or_resume(State(s): State<ReadingState>, Json(p): Json<OpenRequest>) -> Response {
    call(s, DaemonCommand::ReadingOpenOrResume(p)).await
}
async fn mutate(State(s): State<ReadingState>, Json(p): Json<Mutation>) -> Response {
    call(s, DaemonCommand::ReadingMutate(p)).await
}
#[derive(Deserialize)]
struct SourceId {
    source_hash: String,
}
async fn extract(State(s): State<ReadingState>, Json(p): Json<SourceId>) -> Response {
    call(
        s,
        DaemonCommand::ReadingExtract {
            source_hash: p.source_hash,
        },
    )
    .await
}
async fn original(State(s): State<ReadingState>, Query(p): Query<SourceId>) -> Response {
    let Some(b) = s.0 else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    match b
        .call(DaemonCommand::ReadingRead(ReadRequest {
            session: None,
            view: "original".into(),
            source_hash: Some(p.source_hash),
        }))
        .await
    {
        Ok(v) => {
            let Ok(bytes) = base64::engine::general_purpose::STANDARD
                .decode(v["content_base64"].as_str().unwrap_or(""))
            else {
                return StatusCode::BAD_GATEWAY.into_response();
            };
            let pdf = v["filename"]
                .as_str()
                .is_some_and(|n| n.to_ascii_lowercase().ends_with(".pdf"));
            (
                [
                    (
                        header::CONTENT_TYPE,
                        if pdf {
                            "application/pdf"
                        } else {
                            "application/octet-stream"
                        },
                    ),
                    (
                        header::CONTENT_DISPOSITION,
                        if pdf { "inline" } else { "attachment" },
                    ),
                ],
                bytes,
            )
                .into_response()
        }
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error":e}))).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use tower::ServiceExt;
    #[tokio::test]
    async fn stale_tabs_cannot_write_after_database_switch() {
        for (database, expected) in [
            (None, 409),
            (Some("old_database"), 409),
            (Some("current_database"), 204),
        ] {
            let backend = ReadingState(Some(ReadingBackend {
                socket: "/unused".into(),
                database: "current_database".into(),
            }));
            let app = Router::new()
                .route("/probe", post(|| async { StatusCode::NO_CONTENT }))
                .layer(middleware::from_fn_with_state(backend, browser_write_gate));
            let mut request = Request::builder()
                .method("POST")
                .uri("/probe")
                .header("host", "localhost:8088")
                .header("origin", "http://localhost:8088")
                .header("x-hades-workspace", "1");
            if let Some(database) = database {
                request = request.header("x-hades-database", database);
            }
            let response = app
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected);
        }
    }

    #[tokio::test]
    async fn browser_writes_require_origin_custom_header_and_same_origin_metadata_when_present() {
        for (origin, custom, site, expected) in [
            (None, false, None, 403),
            (Some("http://evil.test"), true, None, 403),
            (Some("http://localhost:8088"), false, None, 403),
            (Some("http://localhost:8088"), true, Some("cross-site"), 403),
            (Some("null"), true, None, 403),
            (Some("http://localhost:8088"), true, None, 503),
            (
                Some("http://localhost:8088"),
                true,
                Some("same-origin"),
                503,
            ),
        ] {
            let mut req = Request::builder()
                .method("POST")
                .uri("/api/reading/extract")
                .header("host", "localhost:8088")
                .header("content-type", "application/json");
            if let Some(v) = origin {
                req = req.header("origin", v)
            }
            if custom {
                req = req.header("x-hades-workspace", "1")
            }
            if let Some(v) = site {
                req = req.header("sec-fetch-site", v)
            }
            let response = router::<()>(None)
                .oneshot(
                    req.body(Body::from(r#"{"source_hash":"fixture"}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected);
            assert!(
                !response
                    .headers()
                    .contains_key("access-control-allow-origin")
            );
        }
    }
}
