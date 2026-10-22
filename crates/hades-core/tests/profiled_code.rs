//! Verified code preparation and file-local graph publication on disposable databases.
#![cfg(unix)]
use std::sync::Arc;

use hades_core::{
    code::{self, AnalysisTier},
    db::{ArangoPool, collections::CODEBASE, query},
    embedding_profile::{ChunkPolicy, store::COLLECTION},
    pipeline::profiled_code::{
        CodeInput, LocalEndpoint, LocalRelationship, ProfiledCode, RelationshipKind,
    },
    test_support::{Fixtures, with_temp_db},
};
use serde_json::{Value, json};

#[path = "common/profiled_provider.rs"]
#[allow(dead_code)]
mod provider;
use provider::{Fault, Fixture, Peer};

#[path = "profiled_code/admission.rs"]
mod admission;

fn input(path: &str, apple: &str) -> CodeInput {
    let text = format!("fn apple() {{ {apple} }}\nfn banana() {{ apple(); }}\n");
    let analysis = code::analyze(&text, "fixture.rs").unwrap();
    CodeInput {
        namespace: "fixture/project".into(),
        path: path.into(),
        text,
        analysis,
        relationships: vec![LocalRelationship {
            kind: RelationshipKind::Calls,
            from: LocalEndpoint::Symbol(1),
            to: LocalEndpoint::Symbol(0),
        }],
    }
}

async fn rows(pool: &ArangoPool) -> Value {
    let mut result = json!({});
    for (collection, _) in CODEBASE.all_collections() {
        result[collection] = json!(
            query::query(
                pool,
                "FOR r IN @@c SORT r._key RETURN r",
                Some(&json!({"@c":collection})),
                None,
                false,
                query::ExecutionTarget::Writer
            )
            .await
            .unwrap()
            .results
        );
    }
    result
}

async fn touch_profile(pool: &ArangoPool) {
    let path = format!("document/{COLLECTION}/embedding_profile");
    let before = pool.writer().get(&path).await.unwrap();
    pool.writer()
        .patch(&path, &json!({"payload":before["payload"]}))
        .await
        .unwrap();
}

async fn connect(pool: &ArangoPool, peer: &Peer) -> ProfiledCode {
    ProfiledCode::connect(pool.clone(), &peer.catalog())
        .await
        .unwrap()
}

#[tokio::test]
async fn complete_code_graph_uses_selected_task_and_original_source_spans() {
    with_temp_db("code_roundtrip", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(3);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool,&peer).await;
        let source = input("日本語.rs", "/* café */");
        let source_text = source.text.clone();
        let before = rows(&pool).await;
        let prepared = code.prepare(source,false).await.unwrap();
        assert_eq!(rows(&pool).await,before,"preparation must perform no writes");
        let receipt = code.commit(prepared).await.unwrap();
        assert_eq!((receipt.chunks,receipt.symbols,receipt.relationships),(2,2,3));
        assert_eq!(receipt.binding.expected,fixture.receipt);
        assert_eq!(receipt.binding.task,"fixture.code.passage");
        let stored = rows(&pool).await;
        let file = &stored[CODEBASE.files][0];
        assert_eq!(file["full_text"],source_text);
        assert_eq!(file["passage_plan"]["task"],"fixture.code.passage");
        assert_eq!(file["_rev"],receipt.revision);
        for (collection,_) in CODEBASE.all_collections() {
            for row in stored[collection].as_array().unwrap() {
                assert_eq!(row["format"],"hades-bound-code-v1");
                assert_eq!(row["embedding_binding"],json!(receipt.binding));
                if collection != CODEBASE.files { assert_eq!(row["file_key"],receipt.file_key); }
            }
        }
        for row in stored[CODEBASE.chunks].as_array().unwrap() {
            let span = &row["span"];
            let start = span["start_byte"].as_u64().unwrap() as usize;
            let end = span["end_byte"].as_u64().unwrap() as usize;
            assert_eq!(row["text"].as_str().unwrap(),&source_text[start..end]);
        }
        assert_eq!(peer.plans()[0]["task"],"fixture.code.passage");
        assert_eq!(peer.requests()[0]["expected_receipt"],json!(fixture.receipt));
        assert_eq!(peer.requests()[0]["task"],"fixture.code.passage");
        assert_eq!(stored[CODEBASE.embeddings][0]["embedding"].as_array().unwrap().len(),3);
        // A real external edge into a managed symbol must survive a compatible replacement.
        let symbol_id = stored[CODEBASE.symbols][0]["_id"].as_str().unwrap();
        let external = pool.writer().post(&format!("document/{}",CODEBASE.imports_edges),
            &json!({"_key":"authored_reference","_from":format!("{}/{}",CODEBASE.files,receipt.file_key),
                "_to":symbol_id,"note":"preserve this authored relationship"})).await.unwrap();
        let prepared = code.prepare(input("日本語.rs","let x = 1;"),true).await.unwrap();
        code.commit(prepared).await.unwrap();
        let retained = pool.writer().get(&format!("document/{}/authored_reference",CODEBASE.imports_edges)).await.unwrap();
        assert_eq!(retained["_rev"],external["_rev"]);
    }).await;
}

#[tokio::test]
async fn explicit_replacement_conflicts_and_late_write_failure_preserve_entire_graph() {
    with_temp_db("code_atomic", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2); fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool,&peer).await;
        code.commit(code.prepare(input("one.rs",""),false).await.unwrap()).await.unwrap();
        code.commit(code.prepare(input("other.rs",""),false).await.unwrap()).await.unwrap();
        let requests = peer.requests().len();
        assert!(code.prepare(input("one.rs",""),false).await.is_err());
        assert_eq!(peer.requests().len(),requests);
        let first = code.prepare(input("one.rs","let x = 1;"),true).await.unwrap();
        let stale = code.prepare(input("one.rs","let y = 2;"),true).await.unwrap();
        code.commit(first).await.unwrap();
        let before = rows(&pool).await;
        assert!(code.commit(stale).await.unwrap_err().to_string().contains("source changed"));
        assert_eq!(rows(&pool).await,before);
        let prepared = code.prepare(input("one.rs","let x = 3;"),true).await.unwrap();
        pool.writer().put(&format!("collection/{}/properties",CODEBASE.calls_edges), &json!({
            "schema":{"rule":{"type":"object","required":["injected_missing_field"]},"level":"strict","message":"fixture rejection"}
        })).await.unwrap();
        assert!(code.commit(prepared).await.is_err());
        assert_eq!(rows(&pool).await,before,"all earlier file/chunk/vector/symbol writes must roll back");
    }).await;
}

#[tokio::test]
async fn drift_during_planning_or_before_commit_never_publishes_graph() {
    with_temp_db("code_drift", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = Arc::new(connect(&pool, &peer).await);
        let before = rows(&pool).await;
        let (started, release) = peer.hold_next_plan();
        let task = {
            let code = code.clone();
            tokio::spawn(async move { code.prepare(input("one.rs", ""), false).await })
        };
        started.await.unwrap();
        touch_profile(&pool).await;
        release.send(()).unwrap();
        assert!(task.await.unwrap().is_err());
        assert!(peer.requests().is_empty());
        assert_eq!(rows(&pool).await, before);
        let code = connect(&pool, &peer).await;
        let prepared = code.prepare(input("one.rs", ""), false).await.unwrap();
        touch_profile(&pool).await;
        assert!(code.commit(prepared).await.is_err());
        assert_eq!(rows(&pool).await, before);
    })
    .await;
}

#[tokio::test]
async fn verification_profile_race_and_disabled_or_contextual_code_fail_before_inference() {
    with_temp_db("code_verify_race", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let (started, release) = peer.hold_next_verification();
        let catalog = peer.catalog();
        let connection_pool = pool.clone();
        let task =
            tokio::spawn(async move { ProfiledCode::connect(connection_pool, &catalog).await });
        started.await.unwrap();
        touch_profile(&pool).await;
        release.send(()).unwrap();
        assert!(task.await.unwrap().is_err());
        assert!(peer.plans().is_empty() && peer.requests().is_empty());
    })
    .await;
    for disabled in [true, false] {
        with_temp_db("code_policy", Fixtures::Empty, move |pool| async move {
            let mut fixture = if disabled {
                Fixture::new(2)
            } else {
                Fixture::code(2)
            };
            if !disabled {
                fixture.profile.capabilities.late_chunking = true;
                fixture.profile.capabilities.tokenizer_offsets = true;
                fixture.profile.workloads.code.as_mut().unwrap().chunking = ChunkPolicy::Late {
                    context_cap_tokens: 1000,
                    minimum_window_overlap_tokens: 1,
                    passages: None,
                };
            }
            fixture.install_code(&pool).await;
            let peer = Peer::with_workload_description(&fixture);
            assert!(ProfiledCode::connect(pool, &peer.catalog()).await.is_err());
            assert_eq!(peer.observations(), 0);
        })
        .await;
    }
}

#[tokio::test]
async fn incomplete_batches_bad_vectors_and_wrong_workload_receipts_never_write() {
    with_temp_db("code_provider_faults", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let before = rows(&pool).await;
        for fault in [
            Fault::MissingReceipt,
            Fault::WrongModel,
            Fault::Short,
            Fault::Zero,
            Fault::NotNormalized,
            Fault::Reject,
        ] {
            peer.fault_after(1, fault);
            assert!(code.prepare(input("one.rs", ""), false).await.is_err());
            assert_eq!(rows(&pool).await, before);
        }
        let mut many = input("many.rs", "");
        for index in 0..20 {
            many.text.push_str(&format!("fn extra_{index}() {{}}\n"));
        }
        many.analysis = code::analyze(&many.text, "fixture.rs").unwrap();
        let requests = peer.requests().len();
        peer.fault_after(2, Fault::MissingReceipt);
        assert!(code.prepare(many, false).await.is_err());
        assert_eq!(peer.requests().len(), requests + 2);
        assert_eq!(
            rows(&pool).await,
            before,
            "a later failed batch must discard earlier vectors"
        );
        peer.patch_next_plan(json!({"task":"retrieval.passage"}));
        let requests = peer.requests().len();
        assert!(code.prepare(input("one.rs", ""), false).await.is_err());
        assert_eq!(peer.requests().len(), requests);
        let mut wrong = peer.receipt();
        wrong.space_id = Fixture::new(2).receipt.space_id;
        peer.set_receipt(wrong);
        assert!(code.prepare(input("one.rs", ""), false).await.is_err());
        assert_eq!(rows(&pool).await, before);
    })
    .await;
}

#[tokio::test]
async fn ordinary_text_code_uses_its_declared_asymmetric_workload_without_specialization() {
    with_temp_db("code_ordinary", Fixtures::Empty, |pool| async move {
        let mut fixture = Fixture::new(2);
        fixture.profile.workloads.code = fixture.profile.workloads.documents.take();
        assert!(
            !fixture
                .profile
                .workloads
                .code
                .as_ref()
                .unwrap()
                .specialized_code
        );
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let receipt = code
            .commit(code.prepare(input("ordinary.rs", ""), false).await.unwrap())
            .await
            .unwrap();
        assert_eq!(receipt.binding.task, "retrieval.passage");
        assert_eq!(receipt.binding.expected.space_id, fixture.receipt.space_id);
        assert_eq!(peer.requests()[0]["task"], "retrieval.passage");
    })
    .await;
}

#[tokio::test]
async fn invalid_graph_endpoints_and_duplicate_symbols_fail_before_planning() {
    with_temp_db("code_invalid_graph", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let mut invalid = input("one.rs", "");
        invalid.relationships[0].to = LocalEndpoint::Symbol(500);
        assert!(code.prepare(invalid, false).await.is_err());
        let mut invalid = input("one.rs", "");
        invalid
            .analysis
            .symbols
            .push(invalid.analysis.symbols[0].clone());
        assert!(code.prepare(invalid, false).await.is_err());
        assert!(peer.plans().is_empty() && peer.requests().is_empty());
    })
    .await;
}

#[tokio::test]
async fn unknown_metadata_or_edited_children_are_never_discarded() {
    with_temp_db("code_edited", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        let receipt = code
            .commit(code.prepare(input("one.rs", ""), false).await.unwrap())
            .await
            .unwrap();
        let prepared = code
            .prepare(input("one.rs", "let x = 1;"), true)
            .await
            .unwrap();
        let initial = rows(&pool).await;
        let symbol = initial[CODEBASE.symbols][0]["_key"].as_str().unwrap();
        pool.writer()
            .patch(
                &format!("document/{}/{symbol}", CODEBASE.symbols),
                &json!({"authored_note":"keep"}),
            )
            .await
            .unwrap();
        let edited = rows(&pool).await;
        assert!(
            code.commit(prepared)
                .await
                .unwrap_err()
                .to_string()
                .contains("inventory differs")
        );
        assert_eq!(rows(&pool).await, edited);
        pool.writer()
            .patch(
                &format!("document/{}/{}", CODEBASE.files, receipt.file_key),
                &json!({"authored_note":"keep"}),
            )
            .await
            .unwrap();
        let requests = peer.requests().len();
        assert!(code.prepare(input("one.rs", ""), true).await.is_err());
        assert_eq!(peer.requests().len(), requests);
    })
    .await;
}

#[tokio::test]
async fn symbol_retirement_and_analysis_downgrades_preserve_existing_endpoints() {
    with_temp_db("code_retirement", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = connect(&pool, &peer).await;
        code.commit(code.prepare(input("one.rs", ""), false).await.unwrap())
            .await
            .unwrap();
        let before = rows(&pool).await;
        let mut shifted = input("one.rs", "");
        shifted
            .text
            .insert_str(0, "// inserted above every symbol\n");
        shifted.analysis = code::analyze(&shifted.text, "fixture.rs").unwrap();
        let plans = peer.plans().len();
        let requests = peer.requests().len();
        assert!(
            code.prepare(shifted, true)
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("symbol endpoints")
        );
        assert_eq!(
            peer.plans().len(),
            plans,
            "line shifts require reconciliation before planning"
        );
        assert_eq!(peer.requests().len(), requests);
        assert_eq!(rows(&pool).await, before);
        let mut next = input("one.rs", "");
        next.analysis.symbols.pop();
        next.relationships.clear();
        let requests = peer.requests().len();
        assert!(
            code.prepare(next, true)
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("symbol endpoints")
        );
        assert_eq!(peer.requests().len(), requests);
        assert_eq!(rows(&pool).await, before);
        let mut next = input("one.rs", "");
        next.analysis.analysis_tier = AnalysisTier::Text;
        assert!(code.prepare(next, true).await.is_err());
        assert_eq!(rows(&pool).await, before);
    })
    .await;
}

#[tokio::test]
async fn database_and_provider_admissions_cannot_exchange_prepared_graphs() {
    with_temp_db("code_scope_a", Fixtures::Empty, |a| async move {
        with_temp_db("code_scope_b", Fixtures::Empty, |b| async move {
            let fixture = Fixture::code(2);
            fixture.install_code(&a).await;
            fixture.install_code(&b).await;
            let peer = Peer::with_workload_description(&fixture);
            let first = connect(&a, &peer).await;
            let other = connect(&b, &peer).await;
            let prepared = first.prepare(input("one.rs", ""), false).await.unwrap();
            assert!(
                other
                    .commit(prepared)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("different database")
            );
            assert!(
                rows(&a).await[CODEBASE.files]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            assert!(
                rows(&b).await[CODEBASE.files]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
            let receipt = first
                .commit(first.prepare(input("one.rs", ""), false).await.unwrap())
                .await
                .unwrap();
            let mut reloaded = peer.receipt();
            reloaded.serving_generation = "fixture-reloaded".into();
            peer.set_receipt(reloaded);
            assert!(first.prepare(input("one.rs", ""), true).await.is_err());
            let fresh = connect(&a, &peer).await;
            let current = fresh
                .commit(
                    fresh
                        .prepare(input("one.rs", "let x = 1;"), true)
                        .await
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                receipt.binding.expected.space_id,
                current.binding.expected.space_id
            );
            assert_ne!(
                receipt.binding.expected.serving_generation,
                current.binding.expected.serving_generation
            );
        })
        .await;
    })
    .await;
}

#[tokio::test]
async fn cancellation_during_inference_and_legacy_or_orphan_conflicts_never_publish() {
    with_temp_db("code_cancel", Fixtures::Empty, |pool| async move {
        let fixture = Fixture::code(2);
        fixture.install_code(&pool).await;
        let peer = Peer::with_workload_description(&fixture);
        let code = Arc::new(connect(&pool, &peer).await);
        let before = rows(&pool).await;
        let (started, release) = peer.hold_next();
        let task = {
            let code = code.clone();
            tokio::spawn(async move { code.prepare(input("one.rs", ""), false).await })
        };
        started.await.unwrap();
        task.abort();
        assert!(task.await.is_err_and(|error| error.is_cancelled()));
        release.send(()).unwrap();
        assert_eq!(rows(&pool).await, before);
        let prepared = code.prepare(input("orphan.rs", ""), false).await.unwrap();
        pool.writer()
            .post(
                &format!("document/{}", CODEBASE.chunks),
                &json!({"_key":"orphan","file_key":prepared.file_key(),"authored":"keep"}),
            )
            .await
            .unwrap();
        let before = rows(&pool).await;
        assert!(
            code.commit(prepared)
                .await
                .unwrap_err()
                .to_string()
                .contains("orphan")
        );
        assert_eq!(rows(&pool).await, before);
        let key = hades_core::db::keys::scoped_file_key("fixture/project", "legacy.rs");
        pool.writer()
            .post(
                &format!("document/{}", CODEBASE.files),
                &json!({"_key":key,"legacy":true}),
            )
            .await
            .unwrap();
        assert!(code.prepare(input("legacy.rs", ""), true).await.is_err());
    })
    .await;
}
