use base64::Engine;
use hades_core::{
    db::ArangoPool,
    reading::{
        self,
        chat::{self, ChatConfig, Request},
        *,
    },
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};
use std::time::Duration;

async fn call(
    pool: &ArangoPool,
    cfg: &ChatConfig,
    value: Value,
) -> Result<Value, hades_core::dispatch::HandlerError> {
    chat::handle(pool, cfg, serde_json::from_value::<Request>(value).unwrap()).await
}
async fn wait(pool: &ArangoPool, cfg: &ChatConfig, predicate: impl Fn(&Value) -> bool) -> Value {
    for _ in 0..100 {
        let state = call(pool, cfg, json!({"action":"state","binding":"discussion"}))
            .await
            .unwrap();
        if predicate(&state) {
            return state;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("conversation state did not converge")
}
fn send(id: &str, revision: u64, message: &str) -> Value {
    json!({"action":"send","binding":"discussion","request_id":id,"expected_revision":revision,"message":message,"highlights":["h1"],"knowledge":[],"references":[]})
}

#[tokio::test]
async fn chat_journal_ownership_context_approval_and_disconnect_recovery() {
    with_temp_db("chat",Fixtures::Empty,|pool|async move{
        let directory=tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("hades-test-fixture"),"").unwrap();
        let cfg=ChatConfig{enabled:true,executable:std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/tests/fake_codex.py").canonicalize().unwrap().display().to_string(),home:directory.path().display().to_string(),cwd:directory.path().display().to_string(),mcp_url:"http://127.0.0.1:1/mcp".into(),mcp_token_env:"PATH".into()};
        let source=reading::upload(&pool,SourceRequest{filename:"chat.txt".into(),content_base64:base64::engine::general_purpose::STANDARD.encode(b"The selected passage.")}).await.unwrap();
        let hash=source["source_hash"].as_str().unwrap().to_owned();
        pool.writer().patch(&format!("document/hades_reading_sources/{hash}"),&json!({"status":"ready","structured":{"format":"docling-blocks-v1","warnings":[],"blocks":[{"id":"b1","kind":"text","text":"The selected passage."}]}})).await.unwrap();
        reading::open(&pool,OpenRequest{id:"fixture".into(),title:"Chat fixture".into(),source_hash:hash.clone(),actor:"human".into()}).await.unwrap();
        for (rev,action) in [Action::Bound{is:"A fixture".into(),is_not:"Research data".into()},Action::Advance,Action::Highlight{id:"h1".into(),label:"question".into(),anchor:Anchor{source_hash:hash,passage_revision:0,correction_session:None,figure_box:None,representation:"docling-blocks-v1".into(),page:None,block:Some("b1".into()),exact:"selected passage".into(),prefix:"The ".into(),suffix:".".into()}}].into_iter().enumerate(){
            reading::mutate(&pool,Mutation{session:"fixture".into(),request_id:format!("mutation_{rev}"),expected_revision:rev as u64,actor:"human".into(),action},false).await.unwrap();
        }
        let open=json!({"action":"open","id":"discussion","session":"fixture","thread_id":null});
        let b=call(&pool,&cfg,open.clone()).await.unwrap();
        assert_eq!(call(&pool,&cfg,open.clone()).await.unwrap()["thread_id"],b["thread_id"]);
        let competitor=json!({"action":"open","id":"other","session":"fixture","thread_id":b["thread_id"]});
        assert!(call(&pool,&cfg,competitor).await.unwrap_err().to_string().contains("active writer"));
        let (first,second)=tokio::join!(call(&pool,&cfg,send("message_one",3,"Discuss this")),call(&pool,&cfg,send("message_one",3,"Discuss this")));
        assert_eq!(first.as_ref().unwrap()["_key"],second.unwrap()["_key"]);
        reading::mutate(&pool,Mutation{session:"fixture".into(),request_id:"correction".into(),expected_revision:3,actor:"human".into(),action:Action::CorrectPassage{block:"b1".into(),passage_revision:0,text:"The corrected passage.".into(),restore_revision:None}},false).await.unwrap();
        let state=wait(&pool,&cfg,|v|v["exchanges"][0]["status"]=="completed").await;
        assert_eq!(state["exchanges"].as_array().unwrap().len(),1);
        assert_eq!(state["exchanges"][0]["context"]["highlights"][0]["anchor"]["exact"],"selected passage");
        let exchange=chat::exchange_context(&pool,first.unwrap()["_key"].as_str().unwrap()).await.unwrap();
        assert_eq!(exchange["context"],state["exchanges"][0]["context"]);
        assert_eq!(call(&pool,&cfg,send("message_one",3,"Different payload")).await.unwrap()["rejected"],true);
        assert_eq!(call(&pool,&cfg,send("stale",3,"Discuss stale")).await.unwrap()["rejected"],true);

        call(&pool,&cfg,send("approval",4,"/approval")).await.unwrap();
        let state=wait(&pool,&cfg,|v|v["live"]["approvals"].as_array().is_some_and(|a|!a.is_empty())).await;
        let result=json!({"answers":{"scope":{"answers":["Selected passage"]}}});
        assert!(call(&pool,&cfg,json!({"action":"approval","binding":"discussion","generation":"old","request_id":900,"result":result})).await.is_err());
        call(&pool,&cfg,json!({"action":"approval","binding":"discussion","generation":state["live"]["generation"],"request_id":900,"result":result})).await.unwrap();
        wait(&pool,&cfg,|v|v["exchanges"][1]["status"]=="completed").await;
        call(&pool,&cfg,send("interrupt",4,"/slow")).await.unwrap();
        wait(&pool,&cfg,|v|v["live"]["status"]=="inProgress").await;
        assert!(call(&pool,&cfg,json!({"action":"release","binding":"discussion"})).await.is_err());
        call(&pool,&cfg,json!({"action":"interrupt","binding":"discussion"})).await.unwrap();
        wait(&pool,&cfg,|v|v["exchanges"][2]["status"]=="interrupted").await;
        call(&pool,&cfg,json!({"action":"position","binding":"discussion","page":2,"block":"b1"})).await.unwrap();
        call(&pool,&cfg,json!({"action":"release","binding":"discussion"})).await.unwrap();
        let b=call(&pool,&cfg,open.clone()).await.unwrap();assert_eq!(b["position"]["page"],2);
        call(&pool,&cfg,send("disconnect",4,"/disconnect")).await.unwrap();
        wait(&pool,&cfg,|v|v["exchanges"][3]["status"]=="unknown").await;
        let recovered=call(&pool,&cfg,json!({"action":"reconcile","binding":"discussion"})).await.unwrap();assert_eq!(recovered["unresolved"],0);assert_eq!(recovered["resent"],false);
        let state=call(&pool,&cfg,json!({"action":"state","binding":"discussion"})).await.unwrap();assert_eq!(state["exchanges"][3]["status"],"completed");
        call(&pool,&cfg,open).await.unwrap();
        let retry=call(&pool,&cfg,send("disconnect",4,"/disconnect")).await.unwrap();assert_eq!(retry["status"],"completed");
        call(&pool,&cfg,json!({"action":"release","binding":"discussion"})).await.unwrap();
        let history:Value=serde_json::from_slice(&std::fs::read(directory.path().join(format!("{}.json",b["thread_id"].as_str().unwrap()))).unwrap()).unwrap();
        assert_eq!(history.as_array().unwrap().len(),4,"no duplicated provider turns");
        let reference = &history[0]["items"][1];
        call(&pool,&cfg,json!({"action":"open","id":"reference_binding","session":"fixture"})).await.unwrap();
        let mut request=send("referenced",4,"Compare the earlier answer");
        request["binding"]=json!("reference_binding");
        request["references"]=json!([{"thread_id":b["thread_id"],"turn_id":history[0]["id"],"item_id":reference["id"]}]);
        let reference_exchange=call(&pool,&cfg,request).await.unwrap();
        assert_eq!(reference_exchange["context"]["conversation_references"][0]["item"],*reference);
        for _ in 0..100 {
            let status=call(&pool,&cfg,json!({"action":"state","binding":"reference_binding"})).await.unwrap();
            if status["exchanges"][0]["status"]=="completed" {break;}
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        call(&pool,&cfg,json!({"action":"release","binding":"reference_binding"})).await.unwrap();
        // Stored projection pagination must not omit or repeat equal-date rows.
        for i in 0..23 {
            let mut row=reference_exchange.clone(); row["_key"]=json!(format!("page_{i:02}"));
            row["created_at"]=json!("2000-01-01T00:00:00Z"); row["status"]=json!("completed");
            pool.writer().post("document/hades_reading_exchanges",&row).await.unwrap();
        }
        let page=call(&pool,&cfg,json!({"action":"state","binding":"reference_binding"})).await.unwrap();
        assert_eq!(page["exchanges"].as_array().unwrap().len(),20);
        let older=call(&pool,&cfg,json!({"action":"state","binding":"reference_binding","before":page["next_before"]})).await.unwrap();
        assert_eq!(older["exchanges"].as_array().unwrap().len(),4); assert!(older["next_before"].is_null());
        let ids:std::collections::BTreeSet<_>=page["exchanges"].as_array().unwrap().iter().chain(older["exchanges"].as_array().unwrap()).map(|v|v["_key"].as_str().unwrap()).collect();
        assert_eq!(ids.len(),24);
        // Controls use the whole journal, including rows outside the latest page.
        for (status, unresolved, active) in [("acknowledged",true,true),("unknown",true,false),("completed",false,false)] {
            pool.writer().patch("document/hades_reading_exchanges/page_00",&json!({"status":status})).await.unwrap();
            let current=call(&pool,&cfg,json!({"action":"state","binding":"reference_binding"})).await.unwrap();
            assert!(!current["exchanges"].as_array().unwrap().iter().any(|e|e["_key"]=="page_00"));
            assert_eq!(current["pending"],json!({"unresolved":unresolved,"active":active}));
            let other=call(&pool,&cfg,json!({"action":"state","binding":"discussion"})).await.unwrap();
            assert_eq!(other["pending"],json!({"unresolved":false,"active":false}));
        }
        review_regressions(&pool, &cfg, json!({"thread_id":b["thread_id"],"turn_id":history[0]["id"],"item_id":reference["id"]})).await;
    }).await;
}

async fn fixture_barrier(cfg: &ChatConfig, method: &str, count: usize) {
    let path = std::path::Path::new(&cfg.home).join(format!("entered_{method}"));
    tokio::time::timeout(Duration::from_secs(5), async {
        while std::fs::read_to_string(&path)
            .unwrap_or_default()
            .lines()
            .count()
            < count
        {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("Codex did not reach the fixture barrier");
}

async fn responsive(pool: &ArangoPool, cfg: &ChatConfig, request: Value) -> Value {
    tokio::time::timeout(Duration::from_secs(2), call(pool, cfg, request))
        .await
        .expect("another conversation blocked this request")
        .unwrap()
}

async fn review_regressions(pool: &ArangoPool, cfg: &ChatConfig, reference: Value) {
    let root = std::path::Path::new(&cfg.home);
    let open = |id: &str| json!({"action":"open","id":id,"session":"fixture"});
    let release = |id: &str| json!({"action":"release","binding":id});
    let state = |id: &str| json!({"action":"state","binding":id});

    // Version/startup and profile failures must leave the same binding ID usable.
    let mut missing = cfg.clone();
    missing.executable = root.join("missing_codex").display().to_string();
    assert!(call(pool, &missing, open("retry_startup")).await.is_err());
    assert!(call(pool, cfg, state("retry_startup")).await.is_err());
    call(pool, cfg, open("retry_startup")).await.unwrap();
    call(pool, cfg, release("retry_startup")).await.unwrap();
    std::fs::write(root.join("fail_config_read"), "").unwrap();
    assert!(call(pool, cfg, open("retry_profile")).await.is_err());
    assert!(call(pool, cfg, state("retry_profile")).await.is_err());
    std::fs::remove_file(root.join("fail_config_read")).unwrap();
    let (a, b) = tokio::join!(
        call(pool, cfg, open("retry_profile")),
        call(pool, cfg, open("retry_profile"))
    );
    assert_eq!(
        a.unwrap()["thread_id"],
        b.unwrap()["thread_id"],
        "concurrent opens must share one worker"
    );
    call(pool, cfg, release("retry_profile")).await.unwrap();

    // A failed post-start audit must retain the known thread, without owning it.
    std::fs::write(root.join("fail_mcpServerStatus_list"), "").unwrap();
    assert!(call(pool, cfg, open("retry_audit")).await.is_err());
    let audited = call(pool, cfg, state("retry_audit")).await.unwrap();
    assert!(audited["binding"]["thread_id"].is_string());
    assert_eq!(audited["live"]["status"], "disconnected");
    std::fs::remove_file(root.join("fail_mcpServerStatus_list")).unwrap();
    assert_eq!(
        call(pool, cfg, open("retry_audit")).await.unwrap()["thread_id"],
        audited["binding"]["thread_id"]
    );
    call(pool, cfg, release("retry_audit")).await.unwrap();

    // Lost thread/start acknowledgement stays unknown until explicitly linked.
    std::fs::write(root.join("disconnect_after_thread_start"), "").unwrap();
    assert!(call(pool, cfg, open("unknown_open")).await.is_err());
    std::fs::remove_file(root.join("disconnect_after_thread_start")).unwrap();
    assert!(
        call(pool, cfg, state("unknown_open")).await.unwrap()["binding"]["thread_id"].is_null()
    );
    assert!(
        call(pool, cfg, open("unknown_open"))
            .await
            .unwrap_err()
            .to_string()
            .contains("outcome is unknown")
    );
    let thread = std::fs::read_to_string(root.join("started_thread")).unwrap();
    let mut link = open("unknown_open");
    link["thread_id"] = json!(thread);
    assert_eq!(call(pool, cfg, link).await.unwrap()["thread_id"], thread);
    let mut wrong = open("unknown_open");
    wrong["thread_id"] = json!("different_thread");
    assert!(call(pool, cfg, wrong).await.is_err());
    call(pool, cfg, release("unknown_open")).await.unwrap();

    // Three in-flight opens reserve the remaining slots, but never the registry.
    call(pool, cfg, open("discussion")).await.unwrap();
    call(pool, cfg, send("responsive_stop", 4, "/slow"))
        .await
        .unwrap();
    wait(pool, cfg, |s| s["live"]["status"] == "inProgress").await;
    std::fs::write(root.join("pause_config_read"), "").unwrap();
    let mut openings = Vec::new();
    for id in ["opening_a", "opening_b", "opening_cancelled"] {
        let pool = pool.clone();
        let cfg = cfg.clone();
        let request = open(id);
        openings.push(tokio::spawn(
            async move { call(&pool, &cfg, request).await },
        ));
    }
    fixture_barrier(cfg, "config_read", 3).await;
    assert_eq!(
        responsive(pool, cfg, state("discussion")).await["live"]["status"],
        "inProgress"
    );
    responsive(
        pool,
        cfg,
        json!({"action":"interrupt","binding":"discussion"}),
    )
    .await;
    assert!(
        call(pool, cfg, open("retry_capacity"))
            .await
            .unwrap_err()
            .to_string()
            .contains("four-worker limit")
    );
    assert!(call(pool, cfg, state("retry_capacity")).await.is_err());
    let cancelled = openings.pop().unwrap();
    cancelled.abort();
    assert!(cancelled.await.unwrap_err().is_cancelled());
    assert!(call(pool, cfg, state("opening_cancelled")).await.is_err());
    std::fs::remove_file(root.join("pause_config_read")).unwrap();
    for task in openings {
        task.await.unwrap().unwrap();
    }
    // The cancelled reservation is available even with three other workers alive.
    call(pool, cfg, open("retry_capacity")).await.unwrap();
    for id in ["opening_a", "opening_b", "retry_capacity"] {
        call(pool, cfg, release(id)).await.unwrap();
    }
    wait(pool, cfg, |s| {
        s["exchanges"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["request_id"] == "responsive_stop" && e["status"] == "interrupted")
    })
    .await;

    // Pause reference retrieval after the session snapshot, then mutate it.
    // State, approval, and release on another binding must remain responsive.
    call(pool, cfg, send("responsive_approval", 4, "/approval"))
        .await
        .unwrap();
    let pending = wait(pool, cfg, |s| {
        s["live"]["approvals"]
            .as_array()
            .is_some_and(|a| !a.is_empty())
    })
    .await;
    call(pool, cfg, open("revision_race")).await.unwrap();
    std::fs::write(root.join("pause_thread_turns_list"), "").unwrap();
    let mut request = send("revision_race", 4, "Compare this earlier answer");
    request["binding"] = json!("revision_race");
    request["references"] = json!([reference]);
    let pool_copy = pool.clone();
    let cfg_copy = cfg.clone();
    let sending = tokio::spawn(async move { call(&pool_copy, &cfg_copy, request).await });
    fixture_barrier(cfg, "thread_turns_list", 1).await;
    responsive(pool, cfg, state("discussion")).await;
    responsive(pool, cfg, json!({"action":"approval","binding":"discussion","generation":pending["live"]["generation"],"request_id":900,"result":{"answers":{"scope":{"answers":["Selected passage"]}}}})).await;
    wait(pool, cfg, |s| {
        s["exchanges"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["request_id"] == "responsive_approval" && e["status"] == "completed")
    })
    .await;
    responsive(pool, cfg, release("discussion")).await;
    reading::mutate(
        pool,
        Mutation {
            session: "fixture".into(),
            request_id: "revision_race".into(),
            expected_revision: 4,
            actor: "human".into(),
            action: Action::CorrectPassage {
                block: "b1".into(),
                passage_revision: 1,
                text: "The revised passage.".into(),
                restore_revision: None,
            },
        },
        false,
    )
    .await
    .unwrap();
    std::fs::remove_file(root.join("pause_thread_turns_list")).unwrap();
    let rejected = sending.await.unwrap().unwrap();
    assert_eq!(rejected["rejected"], true);
    assert!(
        rejected["error"]
            .as_str()
            .unwrap()
            .contains("Reading changed before dispatch")
    );
    let before = call(pool, cfg, state("revision_race")).await.unwrap();
    assert!(
        before["exchanges"].as_array().unwrap().is_empty(),
        "rejected request must not be journaled"
    );
    let history: Value = serde_json::from_slice(
        &std::fs::read(root.join(format!(
            "{}.json",
            before["binding"]["thread_id"].as_str().unwrap()
        )))
        .unwrap(),
    )
    .unwrap();
    assert!(
        history.as_array().unwrap().is_empty(),
        "rejected request must not reach Codex"
    );
    let mut corrected = send("corrected_revision", 5, "Discuss the updated passage");
    corrected["binding"] = json!("revision_race");
    assert_eq!(
        call(pool, cfg, corrected).await.unwrap()["status"],
        "dispatching"
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if call(pool, cfg, state("revision_race")).await.unwrap()["exchanges"][0]["status"]
                == "completed"
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    call(pool, cfg, release("revision_race")).await.unwrap();
}

/// Browser dependencies and optional real inference are explicit opt-ins.
#[tokio::test]
#[ignore = "use scripts/test_isolated_database.py --reading-chat-browser"]
async fn reading_chat_browser() {
    with_temp_db("chat_browser", Fixtures::Empty, |pool| async move {
        let source = reading::upload(&pool, SourceRequest {
            filename: "conversation-fixture.txt".into(),
            content_base64: base64::engine::general_purpose::STANDARD.encode(b"The reader checks a claim against the selected passage."),
        }).await.unwrap();
        let hash = source["source_hash"].as_str().unwrap();
        pool.writer().patch(&format!("document/hades_reading_sources/{hash}"), &json!({"status":"ready","structured":{"format":"docling-blocks-v1","warnings":[],"blocks":[{"id":"b1","kind":"text","text":"The reader checks a claim against the selected passage."}]}})).await.unwrap();
        reading::open(&pool, OpenRequest {id:"browser_fixture".into(), title:"Discussing a source passage".into(), source_hash:hash.into(), actor:"human".into()}).await.unwrap();
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/test_reading_chat.cjs");
        let result = tokio::process::Command::new("node").arg(script).arg(pool.database()).status().await.unwrap();
        assert!(result.success(), "browser conversation contract failed");
    }).await;
}
