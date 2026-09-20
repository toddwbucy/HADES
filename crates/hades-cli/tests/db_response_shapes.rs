//! Actual CLI response validation against private Unix HTTP peers only.
use axum::{Json, Router, extract::Request, response::IntoResponse};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

async fn run(args: &[&str], pages: Vec<Value>) -> (std::process::Output, Vec<String>) {
    let root = tempfile::tempdir().unwrap();
    let socket = root.path().join("db.sock");
    let listener = tokio::net::UnixListener::bind(&socket).unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let recorded = calls.clone();
    let responses = Arc::new(Mutex::new(VecDeque::from(pages)));
    let app = Router::new().fallback(move |request: Request| {
        let calls = recorded.clone();
        let responses = responses.clone();
        async move {
            let method = request.method().as_str();
            calls
                .lock()
                .unwrap()
                .push(format!("{method} {}", request.uri().path()));
            if method == "DELETE" {
                Json(json!({"error":false})).into_response()
            } else {
                let response = responses
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("unexpected request");
                Json(response).into_response()
            }
        }
    });
    let peer = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let config = root.path().join("config.json");
    std::fs::write(
        &config,
        serde_json::to_vec(&json!({"database":{
            "name":"fixture", "username":"fixture", "sockets":{"readonly":socket,"readwrite":socket}
        }}))
        .unwrap(),
    )
    .unwrap();
    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_hades"));
    command
        .env_clear()
        .env("HOME", root.path())
        .env("PATH", "/usr/bin:/bin")
        .env("HADES_CONFIG", config)
        .env("ARANGO_SOCKET", &socket)
        .env("ARANGO_RO_SOCKET", &socket)
        .env("ARANGO_RW_SOCKET", &socket)
        .env("ARANGO_PASSWORD", "fixture-only")
        .env("TOKIO_WORKER_THREADS", "2")
        .current_dir(root.path())
        .args(["--db", "fixture", "db"])
        .args(args)
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(5), command.output()).await;
    peer.abort();
    let _ = peer.await;
    let output = output.expect("private CLI exceeded deadline").unwrap();
    let calls = calls.lock().unwrap().clone();
    (output, calls)
}
fn failed(output: &std::process::Output) {
    assert!(!output.status.success(), "unexpected success: {:?}", output);
    assert!(!String::from_utf8_lossy(&output.stderr).contains("exported "));
}

#[tokio::test]
async fn database_list_requires_an_array_of_names() {
    for bad in [
        json!({}),
        json!({"result":null}),
        json!({"result":{}}),
        json!({"result":[1]}),
        json!({"result":["valid",null]}),
    ] {
        let (output, calls) = run(&["databases"], vec![bad]).await;
        failed(&output);
        assert!(output.stdout.is_empty());
        assert_eq!(calls, ["GET /_db/fixture/_api/database/user"]);
    }
    for names in [json!([]), json!(["one", "two"])] {
        let (output, _) = run(&["databases"], vec![json!({"result":names})]).await;
        assert!(output.status.success(), "{:?}", output);
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["data"]["databases"], names);
        assert_eq!(value["data"]["count"], names.as_array().unwrap().len());
    }
}

#[tokio::test]
async fn malformed_first_export_pages_fail_and_clean_safe_ids() {
    for bad in [
        json!({}),
        json!({"result":[]}),
        json!({"result":[],"hasMore":"true"}),
        json!({"result":{},"hasMore":false}),
        json!({"result":[1],"hasMore":false}),
    ] {
        for with_id in [false, true] {
            let mut page = bad.clone();
            if with_id {
                page["id"] = json!("123");
            }
            let (output, calls) = run(&["export", "docs"], vec![page]).await;
            failed(&output);
            assert!(output.stdout.is_empty());
            assert_eq!(calls.len(), if with_id { 2 } else { 1 });
            if with_id {
                assert_eq!(calls[1], "DELETE /_db/fixture/_api/cursor/123");
            }
        }
    }
    for id in [json!(""), json!("../database"), json!(1), json!(null)] {
        let (output, calls) = run(
            &["export", "docs"],
            vec![json!({"id":id,"result":[],"hasMore":true})],
        )
        .await;
        failed(&output);
        assert_eq!(calls.len(), 1, "unsafe ID must not be used in a request");
    }
    let (output, calls) = run(
        &["export", "docs"],
        vec![json!({"result":[{}],"hasMore":true})],
    )
    .await;
    failed(&output);
    assert!(output.stdout.is_empty());
    assert_eq!(calls.len(), 1);
}

#[tokio::test]
async fn export_pagination_and_late_failure_preserve_stream_status_and_cleanup() {
    let first = json!({"id":"123","hasMore":true,"result":[{"_key":"first"}]});
    for last in [
        json!({}),
        json!({"result":null,"hasMore":false}),
        json!({"result":[],"hasMore":"false"}),
        json!({"id":"456","result":[],"hasMore":false}),
    ] {
        let changed = last.get("id").is_some();
        let (output, calls) = run(&["export", "docs"], vec![first.clone(), last]).await;
        failed(&output);
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "{\"_key\":\"first\"}\n"
        );
        assert_eq!(
            &calls[..3],
            [
                "POST /_db/fixture/_api/cursor",
                "POST /_db/fixture/_api/cursor/123",
                "DELETE /_db/fixture/_api/cursor/123"
            ]
        );
        if changed {
            assert_eq!(calls[3], "DELETE /_db/fixture/_api/cursor/456");
        }
    }
    let (output, calls) = run(
        &["export", "docs"],
        vec![first, json!({"result":[{"_key":"last"}],"hasMore":false})],
    )
    .await;
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "{\"_key\":\"first\"}\n{\"_key\":\"last\"}\n"
    );
    assert_eq!(calls.last().unwrap(), "DELETE /_db/fixture/_api/cursor/123");
    let (output, _) = run(
        &["export", "docs"],
        vec![json!({"result":[],"hasMore":false})],
    )
    .await;
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
}

#[tokio::test]
async fn export_output_failure_still_releases_cursor() {
    let (output, calls) = run(
        &["export", "docs", "--output", "/dev/full"],
        vec![json!({"id":"123","hasMore":false,"result":[{}]})],
    )
    .await;
    failed(&output);
    assert_eq!(calls.last().unwrap(), "DELETE /_db/fixture/_api/cursor/123");
}

#[tokio::test]
async fn materialize_scan_failure_is_not_success() {
    let collections = json!({"result":[{"name":"hades_schema","type":2},{"name":"docs","type":2}]});
    let schema = json!({"hasMore":false,"result":[
        {"schema_type":"schema_meta"},
        {"schema_type":"edge_definition","name":"edges","source_field":"related",
         "from_collections":["docs"],"to_collections":["docs"]}
    ]});
    let mut incorrect_successes = 0;
    for args in [
        vec!["graph", "materialize"],
        vec!["graph", "materialize", "--dry-run"],
    ] {
        let (control, _) = run(
            &args,
            vec![
                collections.clone(),
                schema.clone(),
                collections.clone(),
                json!({"hasMore":false,"result":[]}),
            ],
        )
        .await;
        assert!(control.status.success(), "{control:?}");
        let (output, calls) = run(
            &args,
            vec![
                collections.clone(),
                schema.clone(),
                collections.clone(),
                json!({"hasMore":false,"result":null}),
            ],
        )
        .await;
        println!(
            "materialize failure probe: {}",
            json!({"args":args,"exit":output.status.code(),
            "stdout":String::from_utf8_lossy(&output.stdout),"stderr":String::from_utf8_lossy(&output.stderr),"calls":calls})
        );
        assert_eq!(
            calls,
            [
                "GET /_db/fixture/_api/collection",
                "POST /_db/fixture/_api/cursor",
                "GET /_db/fixture/_api/collection",
                "POST /_db/fixture/_api/cursor"
            ]
        );
        if output.status.success() {
            incorrect_successes += 1;
        }
    }
    assert_eq!(
        incorrect_successes, 0,
        "failed source scans must not report successful materialization"
    );
}

#[tokio::test]
async fn graph_list_rejects_malformed_success_responses() {
    // Empty is valid, but absent or malformed graph metadata is not evidence
    // that the selected database has no graphs.
    for graphs in [
        json!([]),
        json!([{"_key":"fixture_graph", "edgeDefinitions":[]}]),
        json!([{"name":"legacy", "edgeDefinitions":[{"collection":"edges","from":["docs"],"to":["docs"]}]}]),
    ] {
        let (output, calls) = run(&["graph", "list"], vec![json!({"graphs":graphs})]).await;
        assert!(output.status.success(), "{output:?}");
        let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
        let expected: Vec<Value> = graphs
            .as_array()
            .unwrap()
            .iter()
            .map(|graph| {
                json!({"name":graph.get("_key").or_else(|| graph.get("name")).unwrap(),
                "edge_definitions":graph["edgeDefinitions"]})
            })
            .collect();
        assert_eq!(envelope["data"]["graphs"], json!(expected));
        assert_eq!(envelope["success"], true);
        assert_eq!(calls, ["GET /_db/fixture/_api/gharial"]);
    }
    for bad in [
        json!({}),
        json!({"graphs":null}),
        json!({"graphs":{}}),
        json!({"graphs":[1]}),
        json!({"graphs":[{}]}),
        json!({"graphs":[{"_key":"", "edgeDefinitions":[]}]}),
        json!({"graphs":[{"_key":"g"}]}),
        json!({"graphs":[{"_key":"g", "edgeDefinitions":{}}]}),
        json!({"graphs":[{"_key":"g", "edgeDefinitions":[{}]}]}),
        json!({"graphs":[{"_key":"g", "edgeDefinitions":[{"collection":"e","from":[1],"to":[]}]}]}),
    ] {
        let (output, calls) = run(&["graph", "list"], vec![bad.clone()]).await;
        println!(
            "graph response probe: {}",
            json!({"response":bad,
            "exit":output.status.code(), "stdout":String::from_utf8_lossy(&output.stdout),
            "stderr":String::from_utf8_lossy(&output.stderr), "calls":calls})
        );
        failed(&output);
        assert!(output.stdout.is_empty());
    }
}
