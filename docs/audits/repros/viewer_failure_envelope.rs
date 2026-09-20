
#[cfg(test)]
mod audit_envelope_baseline {
    use super::*;
    use axum::body::{Body, to_bytes};
    use std::os::unix::fs::PermissionsExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn private_backend_failure_envelope_is_rendered_as_success() {
        let _fixture = crate::backend_process::SCRIPT_FIXTURE.lock().await;
        let root = tempfile::tempdir().unwrap();
        let bin = root.path().join("backend");
        for success in [true, false] {
            let envelope = serde_json::json!({
                "success": success,
                "data": {"graphs":[{"name":"synthetic-graph","edge_definitions":[]}]},
                "error": if success { None } else { Some("synthetic backend failure") }
            });
            let script = format!("#!/usr/bin/python3\nprint({:?})\n", envelope.to_string());
            std::fs::write(&bin, script).unwrap();
            std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o700)).unwrap();
            let router = build_router(AppState {
                bin: bin.to_str().unwrap().into(),
                limit: None,
                request_timeout: std::time::Duration::from_secs(5),
                databases: Arc::new(vec!["fixture".into()]),
                password: None,
                allowed_hosts: Arc::new(vec!["localhost:12345".into()]),
            });
            let response = router.oneshot(Request::builder()
                .uri("/api/graphs")
                .header(header::HOST, "localhost:12345")
                .body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = to_bytes(response.into_body(), 1024).await.unwrap();
            let data: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(data, serde_json::json!({"graphs":["synthetic-graph"]}));
        }
    }
}
