use base64::Engine;
use hades_core::{
    reading::{self, *},
    test_support::{Fixtures, with_temp_db},
};
use serde_json::json;
fn request(session: &str, revision: u64, action: Action) -> Mutation {
    Mutation {
        session: session.into(),
        request_id: format!("request_{revision}"),
        expected_revision: revision,
        actor: "human:fixture".into(),
        action,
    }
}
fn draft() -> Draft {
    Draft {
        title: "Scope matters".into(),
        note: "A note authored from the selected passage.".into(),
        is: Some("A source-specific observation".into()),
        is_not: None,
        standing: "source claim".into(),
        highlight: "h1".into(),
        basis: "This addresses the reading question.".into(),
        from: None,
        to: None,
        relation: None,
        inferred: false,
        rationale: None,
    }
}
#[tokio::test]
async fn acceptance_is_atomic_revisioned_retry_safe_and_retains_attribution() {
    with_temp_db("reading",Fixtures::Empty,|pool|async move{
    let uploaded=reading::upload(&pool,SourceRequest{filename:"fixture.txt".into(),content_base64:base64::engine::general_purpose::STANDARD.encode(b"The selected passage.")}).await.unwrap();
    let hash=uploaded["source_hash"].as_str().unwrap().to_owned();
    // Explicit isolated fixture replaces extraction, which has separate gRPC/browser tests.
    let path=format!("document/hades_reading_sources/{hash}");
    let mut row=pool.writer().get(&path).await.unwrap();row["status"]=json!("ready");row["structured"]=json!({"format":"docling-blocks-v1","blocks":[{"id":"b1","text":"The selected passage."}]});pool.writer().put(&path,&row).await.unwrap();
    let session=OpenRequest{id:"fixture".into(),title:"Fixture".into(),source_hash:hash.clone(),actor:"human:fixture".into()};
    // A pre-existing graph is available for reviewed cross-document connections.
    for (name,kind) in [("prior_nodes",2),("prior_edges",3)] {
        hades_core::db::crud::create_collection(&pool,name,Some(kind)).await.unwrap();
    }
    pool.writer().post("document/prior_nodes",&json!({"_key":"concept","label":"Prior source concept"})).await.unwrap();
    pool.writer().post("gharial",&json!({"name":"prior","edgeDefinitions":[{"collection":"prior_edges","from":["prior_nodes"],"to":["prior_nodes"]}]})).await.unwrap();
    reading::open(&pool,session.clone()).await.unwrap();reading::open(&pool,session).await.unwrap();
    let apply=|r,a|request("fixture",r,a);
    reading::mutate(&pool,apply(0,Action::Bound{is:"A reading fixture".into(),is_not:"A scientific claim".into()}),false).await.unwrap();
    reading::mutate(&pool,apply(1,Action::Advance),false).await.unwrap();
    let highlight=Action::Highlight{id:"h1".into(),label:"node".into(),anchor:Anchor{passage_revision:0,correction_session:None,source_hash:hash,figure_box:None,representation:"docling-blocks-v1".into(),page:None,block:Some("b1".into()),exact:"selected passage".into(),prefix:"The ".into(),suffix:".".into()}};
    assert!(reading::mutate(&pool,apply(2,highlight.clone()),true).await.is_err());
    reading::mutate(&pool,apply(2,highlight),false).await.unwrap();
    let mut propose=apply(3,Action::Propose{id:"p1".into(),draft:draft()});propose.actor="model:fixture".into();
    reading::mutate(&pool,propose,true).await.unwrap();
    let mut edited=draft();edited.note="The human changed the proposed wording.".into();
    reading::mutate(&pool,apply(4,Action::Edit{id:"p1".into(),proposal_revision:1,draft:edited}),false).await.unwrap();
    let stale=apply(5,Action::Accept{id:"p1".into(),proposal_revision:1});assert!(reading::mutate(&pool,stale,false).await.is_err());
    let accept=apply(5,Action::Accept{id:"p1".into(),proposal_revision:2});
    assert!(reading::mutate(&pool,accept.clone(),true).await.is_err());
    let (a,b)=tokio::join!(reading::mutate(&pool,accept.clone(),false),reading::mutate(&pool,accept.clone(),false));
    let a=a.unwrap();assert_eq!(a,b.unwrap());assert_eq!(a["created_ids"].as_array().unwrap().len(),2);
    let mut collision=accept;collision.actor="someone else".into();assert!(reading::mutate(&pool,collision,false).await.is_err());
    let stored=pool.writer().get("document/reading_fixture_nodes/note_p1").await.unwrap();
    assert_eq!(stored["attrs"]["model_origin"],"model:fixture");assert_eq!(stored["attrs"]["reviewer"],"human:fixture");assert_eq!(stored["text"],"The human changed the proposed wording.");assert_eq!(stored["attrs"]["boundaries_unresolved"],true);
    let count=pool.writer().get("collection/reading_fixture_nodes/count").await.unwrap();assert_eq!(count["count"],2);
    // An endpoint failure occurs inside the transaction and publishes no decision/event.
    let mut edge=draft();edge.from=Some("reading_fixture_nodes/note_p1".into());edge.to=Some("reading_fixture_nodes/missing".into());edge.relation=Some("supports".into());
    reading::mutate(&pool,apply(6,Action::Propose{id:"p2".into(),draft:edge}),true).await.unwrap();
    assert!(reading::mutate(&pool,apply(7,Action::Accept{id:"p2".into(),proposal_revision:1}),false).await.is_err());
    let data=reading::read(&pool,ReadRequest{session:Some("fixture".into()),view:"export".into(),source_hash:None}).await.unwrap();
    assert_eq!(data["session"]["revision"],7);assert_eq!(data["events"].as_array().unwrap().len(),7);assert_eq!(data["session"]["proposals"]["p2"]["status"],"pending");
    let mut prior="";for e in data["events"].as_array().unwrap(){assert_eq!(e["previous_hash"],prior);prior=e["event_hash"].as_str().unwrap();}assert_eq!(data["session"]["last_event_hash"],prior);assert_eq!(data["chain_valid"],true);
    let mut connection=draft();connection.from=Some("reading_fixture_nodes/note_p1".into());connection.to=Some("prior_nodes/concept".into());connection.relation=Some("related_under_source_scope".into());connection.inferred=true;connection.rationale=Some("The selected passage supports this proposed comparison; the human reviews it.".into());
    reading::mutate(&pool,apply(7,Action::Propose{id:"p3".into(),draft:connection}),true).await.unwrap();
    reading::mutate(&pool,apply(8,Action::Accept{id:"p3".into(),proposal_revision:1}),false).await.unwrap();
    let edge=pool.writer().get("document/reading_fixture_edges/note_p3").await.unwrap();assert_eq!(edge["_to"],"prior_nodes/concept");assert_eq!(edge["attrs"]["evidence"]["inferred"],true);
    // Page-edge figure coordinates must survive the database JSON number round trip.
    reading::mutate(&pool,apply(9,Action::Highlight{id:"whole_page".into(),label:"question".into(),anchor:Anchor{passage_revision:0,correction_session:None,source_hash:uploaded["source_hash"].as_str().unwrap().into(),figure_box:Some([0.0,0.0,1.0,1.0]),representation:"pdf-figure-v1".into(),page:Some(1),block:None,exact:String::new(),prefix:String::new(),suffix:String::new()}}),false).await.unwrap();
    let export=reading::read(&pool,ReadRequest{session:Some("fixture".into()),view:"export".into(),source_hash:None}).await.unwrap();assert_eq!(export["chain_valid"],true);
    // Administrators can tamper with storage, but an unchanged hash-chain head detects it.
    let path="document/hades_reading_events/fixture_request_0";
    let mut event=pool.writer().get(path).await.unwrap();event["actor"]=json!("tampered");pool.writer().put(path,&event).await.unwrap();
    let export=reading::read(&pool,ReadRequest{session:Some("fixture".into()),view:"export".into(),source_hash:None}).await.unwrap();assert_eq!(export["chain_valid"],false);
}).await;
}

#[test]
fn state_machine_does_not_invent_boundaries_or_allow_agent_acceptance() {
    let mut s = Session::new(&OpenRequest {
        id: "fixture".into(),
        title: "Fixture".into(),
        source_hash: "a".repeat(64),
        actor: "human".into(),
    })
    .unwrap();
    assert!(
        s.apply(&request("fixture", 0, Action::Advance), false)
            .is_err()
    );
    assert!(
        s.apply(
            &request(
                "fixture",
                0,
                Action::Bound {
                    is: "scope".into(),
                    is_not: "".into()
                }
            ),
            false
        )
        .is_err()
    );
    assert!(
        s.apply(
            &request(
                "fixture",
                0,
                Action::Bound {
                    is: "scope".into(),
                    is_not: "exclusion".into()
                }
            ),
            true
        )
        .is_err()
    );
    s.apply(
        &request(
            "fixture",
            0,
            Action::Bound {
                is: "scope".into(),
                is_not: "exclusion".into(),
            },
        ),
        false,
    )
    .unwrap();
    assert!(
        s.apply(&request("fixture", 0, Action::Advance), false)
            .is_err()
    );
}

#[tokio::test]
async fn corrections_preserve_source_and_references_and_restore_as_new_revisions() {
    with_temp_db("reading_corrections", Fixtures::Empty, |pool| async move {
        let uploaded = reading::upload(&pool, SourceRequest { filename:"corrections.txt".into(), content_base64:base64::engine::general_purpose::STANDARD.encode(b"A typ0.") }).await.unwrap();
        let hash=uploaded["source_hash"].as_str().unwrap().to_owned();
        let path=format!("document/hades_reading_sources/{hash}");
        let mut source=pool.writer().get(&path).await.unwrap();
        source["status"]=json!("ready");source["structured"]=json!({"format":"docling-blocks-v1","blocks":[{"id":"b1","kind":"text","text":"A typ0."},{"id":"t1","kind":"table","text":"cell"}]});
        pool.writer().put(&path,&source).await.unwrap();
        let original=pool.writer().get(&path).await.unwrap();
        reading::open(&pool,OpenRequest{id:"corrections".into(),title:"Fixture".into(),source_hash:hash.clone(),actor:"human:fixture".into()}).await.unwrap();
        let apply=|r,a|request("corrections",r,a);
        reading::mutate(&pool,apply(0,Action::Bound{is:"A test".into(),is_not:"Research".into()}),false).await.unwrap();
        reading::mutate(&pool,apply(1,Action::Advance),false).await.unwrap();
        let anchor=Anchor{source_hash:hash.clone(),passage_revision:0,correction_session:None,figure_box:None,representation:"docling-blocks-v1".into(),page:None,block:Some("b1".into()),exact:"A typ0.".into(),prefix:String::new(),suffix:String::new()};
        let legacy=serde_json::to_value(&anchor).unwrap();assert!(legacy.get("passage_revision").is_none());assert!(legacy.get("correction_session").is_none());
        reading::mutate(&pool,apply(2,Action::Highlight{id:"h1".into(),anchor:anchor.clone(),label:"node".into()}),false).await.unwrap();
        reading::mutate(&pool,apply(3,Action::Propose{id:"p1".into(),draft:draft()}),true).await.unwrap();
        reading::mutate(&pool,apply(4,Action::Accept{id:"p1".into(),proposal_revision:1}),false).await.unwrap();
        let note=pool.writer().get("document/reading_corrections_nodes/note_p1").await.unwrap();
        let correction=apply(5,Action::CorrectPassage{block:"b1".into(),passage_revision:0,text:"A typo.".into(),restore_revision:None});
        assert!(reading::mutate(&pool,correction.clone(),true).await.is_err());
        let receipt=reading::mutate(&pool,correction.clone(),false).await.unwrap();
        assert_eq!(reading::mutate(&pool,correction,false).await.unwrap(),receipt);
        assert!(reading::mutate(&pool,apply(6,Action::CorrectPassage{block:"b1".into(),passage_revision:0,text:"Stale edit".into(),restore_revision:None}),false).await.is_err());
        assert!(reading::mutate(&pool,apply(6,Action::Highlight{id:"stale".into(),anchor:anchor.clone(),label:"node".into()}),false).await.is_err());
        let corrected=Anchor{passage_revision:1,correction_session:Some("corrections".into()),exact:"A typo.".into(),..anchor.clone()};
        let wrong_session=Anchor{correction_session:Some("another".into()),..corrected.clone()};
        assert!(reading::mutate(&pool,apply(6,Action::Highlight{id:"bad".into(),anchor:wrong_session,label:"node".into()}),false).await.is_err());
        reading::mutate(&pool,apply(6,Action::Highlight{id:"h2".into(),anchor:corrected,label:"node".into()}),false).await.unwrap();
        assert!(reading::mutate(&pool,apply(7,Action::CorrectPassage{block:"b1".into(),passage_revision:1,text:"Invented restore".into(),restore_revision:Some(0)}),false).await.is_err());
        reading::mutate(&pool,apply(7,Action::CorrectPassage{block:"b1".into(),passage_revision:1,text:"A typ0.".into(),restore_revision:Some(0)}),false).await.unwrap();
        reading::mutate(&pool,apply(8,Action::Focus{references:vec!["h2".into()]}),false).await.unwrap();
        for block in ["missing","t1"] {assert!(reading::mutate(&pool,apply(9,Action::CorrectPassage{block:block.into(),passage_revision:0,text:"Changed".into(),restore_revision:None}),false).await.is_err());}
        let read=|view:&str|ReadRequest{session:Some("corrections".into()),view:view.into(),source_hash:None};
        let state=reading::read(&pool,read("state")).await.unwrap();
        assert_eq!(state["state"]["corrections"]["b1"].as_array().unwrap().len(),2);
        assert_eq!(state["state"]["corrections"]["b1"][1]["restored_from"],0);
        assert_eq!(state["reference_warnings"].as_array().unwrap().len(),2);
        let context=reading::read(&pool,read("context")).await.unwrap();
        assert_eq!(context["references"][0]["highlight"]["anchor"]["passage_revision"],1);
        assert_eq!(context["references"][0]["current_passage_revision"],2);
        assert_eq!(context["references"][0]["passage_changed"],true);
        let export=reading::read(&pool,read("export")).await.unwrap();assert_eq!(export["chain_valid"],true);
        assert_eq!(pool.writer().get(&path).await.unwrap(),original);
        assert_eq!(pool.writer().get("document/reading_corrections_nodes/note_p1").await.unwrap(),note);
        reading::open(&pool,OpenRequest{id:"other_reading".into(),title:"Other reading".into(),source_hash:hash,actor:"human:fixture".into()}).await.unwrap();
        let other=reading::read(&pool,ReadRequest{session:Some("other_reading".into()),view:"state".into(),source_hash:None}).await.unwrap();assert!(other["state"]["corrections"].as_object().unwrap().is_empty());
    }).await;
}

#[tokio::test]
async fn invalid_extractions_are_persisted_and_recoverable() {
    use hades_core::config::HadesConfig;
    use hades_proto::extraction::extraction_service_server::{
        ExtractionService, ExtractionServiceServer,
    };
    use hades_proto::extraction::{
        CapabilitiesRequest, ExtractRequest, ExtractResponse, ExtractorInfo,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use tonic::{Request, Response, Status};

    struct Extractor(Arc<AtomicBool>);
    #[tonic::async_trait]
    impl ExtractionService for Extractor {
        async fn extract(
            &self,
            request: Request<ExtractRequest>,
        ) -> Result<Response<ExtractResponse>, Status> {
            let content = request.into_inner().content;
            let document = if self.0.load(Ordering::SeqCst) {
                r#"{"format":"docling-blocks-v1","blocks":[{"id":"b1","text":"Recovered passage"}]}"#.into()
            } else {
                match content.as_slice() {
                    b"empty" => String::new(),
                    b"oversized" => " ".repeat(4 * 1024 * 1024 + 1) + "{}",
                    b"malformed" => "{".into(),
                    b"structure" => r#"{"format":"other","blocks":{}}"#.into(),
                    _ => return Err(Status::unavailable("fixture transport failure")),
                }
            };
            Ok(Response::new(ExtractResponse {
                reading_document_json: document,
                full_text: "Extracted text without a usable reading surface".into(),
                ..Default::default()
            }))
        }
        async fn capabilities(
            &self,
            _: Request<CapabilitiesRequest>,
        ) -> Result<Response<ExtractorInfo>, Status> {
            Ok(Response::new(ExtractorInfo::default()))
        }
    }
    with_temp_db("reading_failure", Fixtures::Empty, |pool| async move {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("extract.sock");
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(stream, _)| stream), listener))
        });
        let recovered = Arc::new(AtomicBool::new(false));
        let service = Extractor(recovered.clone());
        let server = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(
                    ExtractionServiceServer::new(service)
                        .max_encoding_message_size(16 * 1024 * 1024),
                )
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        struct Cleanup(tokio::task::JoinHandle<()>);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                self.0.abort();
            }
        }
        let _cleanup = Cleanup(server);
        let mut config = HadesConfig::default();
        config.extraction.service.socket = socket.to_string_lossy().into_owned();
        let mut hashes = vec![];
        for (input, expected) in [
            ("empty", "no structured reading surface"),
            ("oversized", "exceeds 4 MiB"),
            ("malformed", "invalid structured"),
            ("structure", "invalid structured"),
            ("transport", "fixture transport failure"),
        ] {
            let encoded = base64::engine::general_purpose::STANDARD.encode(input);
            let uploaded = reading::upload(
                &pool,
                SourceRequest {
                    filename: "fixture.txt".into(),
                    content_base64: encoded.clone(),
                },
            )
            .await
            .unwrap();
            let hash = uploaded["source_hash"].as_str().unwrap().to_owned();
            let result = reading::extract(&pool, &config, hash.clone())
                .await
                .unwrap();
            assert_eq!(result["status"], "failed");
            assert!(
                result["error"].as_str().unwrap().contains(expected),
                "{result}"
            );
            let stored = pool
                .writer()
                .get(&format!("document/hades_reading_sources/{hash}"))
                .await
                .unwrap();
            assert_eq!(stored["status"], "failed");
            assert_eq!(stored["content_base64"], encoded);
            let listed = reading::read(
                &pool,
                ReadRequest {
                    session: None,
                    source_hash: None,
                    view: "sources".into(),
                },
            )
            .await
            .unwrap();
            assert!(
                listed["sources"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|s| s["source_hash"] == hash && s["error"] == result["error"])
            );
            hashes.push(hash);
        }
        recovered.store(true, Ordering::SeqCst);
        let hash = hashes.remove(0);
        assert_eq!(
            reading::extract(&pool, &config, hash.clone())
                .await
                .unwrap()["status"],
            "ready"
        );
        recovered.store(false, Ordering::SeqCst);
        assert_eq!(
            reading::extract(&pool, &config, hash).await.unwrap()["unchanged"],
            true
        );
    })
    .await;
}
