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

    }).await;
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
