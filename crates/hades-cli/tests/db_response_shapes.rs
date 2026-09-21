//! Actual CLI response validation against private Unix HTTP peers only.
use axum::{Json, Router, extract::Request, response::IntoResponse};
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};

async fn run(args: &[&str], pages: Vec<Value>) -> (std::process::Output, Vec<String>) {
    let mut command = vec!["db"];
    command.extend_from_slice(args);
    run_root(&command, pages).await
}

async fn run_root(args: &[&str], pages: Vec<Value>) -> (std::process::Output, Vec<String>) {
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
        .args(["--db", "fixture"])
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

#[tokio::test]
async fn orientation_does_not_hide_metadata_read_failures() {
    let good = vec![
        json!({"count":2}),
        json!({"hasMore":false,"result":[{"_key":"one","title":"Fixture"}]}),
        json!({"hasMore":false,"result":[{"_key":"one","title":"Fixture"}]}),
        json!({"indexes":[]}),
    ];
    let (control, _) = run_root(&["orient", "--collection", "docs"], good.clone()).await;
    assert!(control.status.success(), "{control:?}");
    let data: Value = serde_json::from_slice(&control.stdout).unwrap();
    assert_eq!(data["data"]["count"], 2);
    assert_eq!(data["data"]["recent"].as_array().unwrap().len(), 1);
    let mut incorrect_successes = 0;
    for (stage, index, bad) in [
        ("sample", 1, json!({"hasMore":false,"result":null})),
        ("recent", 2, json!({"hasMore":false,"result":null})),
        ("indexes", 3, json!({})),
    ] {
        let mut replies = good.clone();
        replies[index] = bad;
        let (output, calls) = run_root(&["orient", "--collection", "docs"], replies).await;
        println!(
            "Orientation failure probe: {}",
            json!({"stage":stage,"exit":output.status.code(),
            "stdout":String::from_utf8_lossy(&output.stdout),"stderr":String::from_utf8_lossy(&output.stderr),"calls":calls})
        );
        if output.status.success() {
            incorrect_successes += 1;
        } else {
            assert!(output.stdout.is_empty());
            let diagnostic = String::from_utf8_lossy(&output.stderr);
            assert!(diagnostic.contains("orientation"), "{diagnostic}");
            assert!(diagnostic.contains(stage), "{diagnostic}");
            assert_eq!(calls.len(), index + 1, "stop after failed metadata stage");
        }
    }
    assert_eq!(
        incorrect_successes, 0,
        "metadata read failures must not become successful empty data"
    );
}

#[tokio::test]
async fn schema_apply_requires_confirmed_metadata_import() {
    let root = tempfile::tempdir().unwrap();
    let schema = root.path().join("schema.yaml");
    std::fs::write(&schema, "{}\n").unwrap();
    let args = ["schema", "apply", schema.to_str().unwrap()];
    let acknowledged =
        json!({"error":false,"created":1,"updated":0,"errors":0,"ignored":0,"empty":0});
    let (control, calls) = run_root(&args, vec![json!({}), acknowledged]).await;
    assert!(control.status.success(), "{control:?}");
    assert_eq!(
        calls,
        [
            "POST /_db/fixture/_api/collection",
            "POST /_db/fixture/_api/import"
        ]
    );
    let mut incorrect_successes = 0;
    for (case, reply) in [
        (
            "reported_errors",
            json!({"error":false,"created":0,"updated":0,"errors":1,"ignored":0,"empty":0}),
        ),
        ("missing_counts", json!({})),
        (
            "unaccounted_document",
            json!({"error":false,"created":0,"updated":0,"errors":0,"ignored":0,"empty":0}),
        ),
    ] {
        let (output, calls) = run_root(&args, vec![json!({}), reply]).await;
        println!(
            "schema_case={case} observed={}",
            json!({"exit_code":output.status.code(),
            "stdout":String::from_utf8_lossy(&output.stdout),"stderr":String::from_utf8_lossy(&output.stderr),"calls":calls})
        );
        assert_eq!(calls.len(), 2);
        if output.status.success() {
            incorrect_successes += 1;
        }
    }
    assert_eq!(
        incorrect_successes, 0,
        "unconfirmed schema metadata must not be reported applied"
    );
}

#[tokio::test]
async fn schema_apply_validates_every_import_stage_and_preserves_controls() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("schema.yaml");
    std::fs::write(
        &file,
        r#"
collections:
  - {name: docs, type: document}
  - {name: links, type: edge}
docs:
  - {_key: one}
edge_definitions:
  - {name: links, from_collections: [docs], to_collections: [docs]}
named_graphs:
  - {name: example, edges: [links]}
"#,
    )
    .unwrap();
    let stages = [
        "document seeds",
        "edge definition",
        "named graph metadata",
        "schema metadata",
    ];
    let created = json!({"error":false,"created":1,"updated":0,"errors":0,"ignored":0,"empty":0});
    let updated = json!({"error":false,"created":0,"updated":1,"errors":0,"ignored":0,"empty":0});
    for (stage_index, stage) in stages.iter().enumerate() {
        let mut pages = vec![json!({}); 3];
        pages.extend(vec![created.clone(); stage_index]);
        pages.push(json!({"error":false,"created":0,"updated":0,"errors":1,"ignored":0,"empty":0}));
        let (output, calls) = run_root(
            &["schema", "apply", file.to_str().unwrap(), "--force"],
            pages,
        )
        .await;
        assert!(!output.status.success(), "{stage}: {output:?}");
        assert!(output.stdout.is_empty());
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        assert!(diagnostic.contains(stage), "{diagnostic}");
        assert!(diagnostic.contains("earlier operations may have committed"));
        assert_eq!(calls.len(), 4 + stage_index, "stop after rejected import");
    }
    for acknowledgment in [created, updated] {
        let mut pages = vec![json!({}); 3];
        pages.extend(vec![acknowledgment; 4]);
        pages.push(json!({}));
        let (output, calls) = run_root(
            &["schema", "apply", file.to_str().unwrap(), "--force"],
            pages,
        )
        .await;
        assert!(output.status.success(), "{output:?}");
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["data"]["applied"], true);
        assert_eq!(value["data"]["result"]["documents_upserted"], 1);
        assert_eq!(value["data"]["result"]["edge_definitions_registered"], 1);
        assert_eq!(calls.len(), 8);
    }
    let (dry, calls) = run_root(
        &["schema", "apply", file.to_str().unwrap(), "--dry-run"],
        vec![],
    )
    .await;
    assert!(dry.status.success(), "{dry:?}");
    assert!(calls.is_empty());
    let (guard, calls) = run_root(
        &["schema", "apply", file.to_str().unwrap()],
        vec![json!({"result":[1],"hasMore":false})],
    )
    .await;
    assert!(!guard.status.success());
    assert!(String::from_utf8_lossy(&guard.stderr).contains("is in use"));
    assert_eq!(calls, ["POST /_db/fixture/_api/cursor"]);
}

#[tokio::test]
async fn schema_apply_rejects_malformed_or_inconsistent_import_counts() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("schema.yaml");
    std::fs::write(&file, "{}\n").unwrap();
    let good = json!({"error":false,"created":1,"updated":0,"errors":0,"ignored":0,"empty":0});
    let mut replies = Vec::new();
    for field in ["created", "updated", "errors", "ignored", "empty"] {
        let mut missing = good.clone();
        missing.as_object_mut().unwrap().remove(field);
        replies.push(missing);
        for bad in [json!(null), json!(-1), json!("0"), json!(0.5)] {
            let mut reply = good.clone();
            reply[field] = bad;
            replies.push(reply);
        }
    }
    for (field, value) in [
        ("ignored", json!(1)),
        ("empty", json!(1)),
        ("created", json!(2)),
        ("error", json!(true)),
        ("error", json!(null)),
    ] {
        let mut reply = good.clone();
        reply[field] = value;
        replies.push(reply);
    }
    let mut overflow = good;
    overflow["created"] = json!(u64::MAX);
    overflow["updated"] = json!(2);
    replies.push(overflow);
    for reply in replies {
        let (output, calls) = run_root(
            &["schema", "apply", file.to_str().unwrap()],
            vec![json!({}), reply.clone()],
        )
        .await;
        assert!(!output.status.success(), "{reply}: {output:?}");
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("schema metadata"));
        assert_eq!(calls.len(), 2);
    }
}
