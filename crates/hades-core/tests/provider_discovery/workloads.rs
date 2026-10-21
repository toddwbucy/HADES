//! Independent workload observations at the registered provider boundary.

use super::{Fixture, Peer, Reply};
use hades_core::embedding_profile::{ChunkPolicy, EmbeddingRole, EmbeddingSpace, WorkloadKind};
use serde_json::{Value, json};

/// Same artifact and dimension, deliberately different numerical tasks and spaces.
fn mixed() -> Fixture {
    let mut fixture = Fixture::new();
    let documents = json!({
        "capabilities":fixture.description["capabilities"],
        "space":fixture.description["space"], "receipt":fixture.description["receipt"],
    });
    let mut code = fixture.profile.workloads.documents.clone().unwrap();
    code.specialized_code = true;
    code.space.passage.task = "code".into();
    code.space.passage.prompt = String::new();
    code.space.passage.adapter = "fixture-code".into();
    code.space.query = code.space.passage.clone();
    let mut capabilities = fixture.profile.capabilities.clone();
    capabilities.tasks = vec!["code".into()];
    capabilities.specialized_code = true;
    let mut receipt = fixture.description["receipt"].clone();
    receipt["space_id"] = json!(code.space.id().unwrap());
    fixture.description = json!({
        "format":"hades-provider-description-v2", "model":fixture.profile.model,
        "workloads": {"documents":documents, "code":{
            "capabilities":capabilities, "space":code.space, "receipt":receipt,
        }},
    });
    fixture.profile.capabilities.tasks.push("code".into());
    fixture.profile.capabilities.specialized_code = true;
    fixture.profile.workloads.code = Some(code);
    fixture.profile.validate(&fixture.manifest).unwrap();
    fixture
}

/// Build independently returned embedding rows; the peer never echoes the request's identity.
fn embedding(fixture: &Fixture, workload: &str) -> Value {
    let observation = &fixture.description["workloads"][workload];
    let width = observation["space"]["representation"]["dimension"]
        .as_u64()
        .unwrap() as usize;
    json!({"model":fixture.profile.model.id,"receipt":observation["receipt"],
        "data":[{"index":0,"embedding":vec![0.5; width]}]})
}

#[tokio::test]
async fn mixed_workloads_bind_their_own_tasks_dimensions_and_receipts() {
    for width in [384, 768] {
        let mut fixture = mixed();
        let code = fixture.profile.workloads.code.as_mut().unwrap();
        code.space.representation.dimension = width;
        fixture.description["workloads"]["code"]["space"] = json!(code.space);
        fixture.description["workloads"]["code"]["receipt"]["space_id"] =
            json!(code.space.id().unwrap());
        let roles = [
            (
                WorkloadKind::Documents,
                "documents",
                EmbeddingRole::Passage,
                "retrieval.passage",
                384,
            ),
            (
                WorkloadKind::Code,
                "code",
                EmbeddingRole::Passage,
                "code",
                width,
            ),
            (
                WorkloadKind::Documents,
                "documents",
                EmbeddingRole::Query,
                "retrieval.query",
                384,
            ),
            (
                WorkloadKind::Code,
                "code",
                EmbeddingRole::Query,
                "code",
                width,
            ),
        ];
        let mut replies = vec![Reply::json(fixture.description.clone())];
        replies.extend(
            roles
                .iter()
                .map(|(_, name, ..)| Reply::json(embedding(&fixture, name))),
        );
        let peer = Peer::new(replies);
        let observed = fixture
            .catalog(&peer)
            .verify_profile(&fixture.profile, &fixture.manifest)
            .await
            .unwrap();
        assert_eq!(json!(observed.description()), fixture.description);
        let document_profile = observed
            .document_profile(
                "new-corpus".into(),
                ChunkPolicy::Conventional {
                    passage_tokens: 100,
                    overlap_tokens: 0,
                },
            )
            .unwrap();
        assert!(document_profile.workloads.code.is_none());
        assert!(!document_profile.capabilities.specialized_code);
        assert_eq!(
            document_profile.workloads.documents.unwrap().space,
            fixture.profile.workloads.documents.as_ref().unwrap().space
        );
        for (index, (kind, name, role, task, dimension)) in roles.into_iter().enumerate() {
            let client = observed
                .client(&fixture.profile, &fixture.manifest, kind, role)
                .await
                .unwrap();
            let result = client.embed(&["same input".into()], None).await.unwrap();
            assert_eq!(result.binding.dimension, dimension);
            assert_eq!(result.output.embeddings[0].len(), dimension as usize);
            assert_eq!(result.binding.task, task);
            assert_eq!(
                json!(result.binding.expected),
                fixture.description["workloads"][name]["receipt"]
            );
            let request: Value = serde_json::from_slice(&peer.requests()[index + 1].1).unwrap();
            assert_eq!(request["task"], task);
            assert_eq!(
                request["expected_receipt"],
                fixture.description["workloads"][name]["receipt"]
            );
        }
        assert_eq!(peer.requests().len(), 5);
        assert!(peer.requests()[0].1.is_empty());
    }
}

#[tokio::test]
async fn code_only_observation_does_not_require_or_create_document_intent() {
    let mut fixture = mixed();
    fixture.description["workloads"]["documents"] = Value::Null;
    fixture.profile.workloads.documents = None;
    let peer = Peer::new(vec![
        Reply::json(fixture.description.clone()),
        Reply::json(embedding(&fixture, "code")),
    ]);
    let observed = fixture
        .catalog(&peer)
        .verify_profile(&fixture.profile, &fixture.manifest)
        .await
        .unwrap();
    let error = observed
        .document_profile(
            "new".into(),
            ChunkPolicy::Conventional {
                passage_tokens: 20,
                overlap_tokens: 0,
            },
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("does not advertise the Documents workload")
    );
    assert!(
        observed
            .description()
            .workload(WorkloadKind::Documents)
            .is_none()
    );
    observed
        .client(
            &fixture.profile,
            &fixture.manifest,
            WorkloadKind::Code,
            EmbeddingRole::Query,
        )
        .await
        .unwrap()
        .embed(&["find a function".into()], None)
        .await
        .unwrap();
    assert_eq!(peer.requests().len(), 2);
}

#[tokio::test]
async fn legacy_observation_preserves_explicit_ordinary_text_code_with_the_same_space() {
    let mut fixture = Fixture::new();
    fixture.profile.workloads.code = fixture.profile.workloads.documents.clone();
    let response = json!({"model":fixture.profile.model.id,"receipt":fixture.description["receipt"],
        "data":[{"index":0,"embedding":vec![0.5;384]}]});
    let peer = Peer::new(vec![
        Reply::json(fixture.description.clone()),
        Reply::json(response),
    ]);
    let observed = fixture
        .catalog(&peer)
        .verify_profile(&fixture.profile, &fixture.manifest)
        .await
        .unwrap();
    let client = observed
        .client(
            &fixture.profile,
            &fixture.manifest,
            WorkloadKind::Code,
            EmbeddingRole::Query,
        )
        .await
        .unwrap();
    let result = client
        .embed(&["ordinary text code".into()], None)
        .await
        .unwrap();
    assert_eq!(result.binding.task, "retrieval.query");
    assert_eq!(
        json!(result.binding.expected),
        fixture.description["receipt"]
    );
    let request: Value = serde_json::from_slice(&peer.requests()[1].1).unwrap();
    assert_eq!(request["task"], "retrieval.query");
    assert_eq!(request["expected_receipt"], fixture.description["receipt"]);
    let mut code_only = fixture.profile.clone();
    code_only.workloads.documents = None;
    assert_eq!(
        observed
            .client(
                &code_only,
                &fixture.manifest,
                WorkloadKind::Code,
                EmbeddingRole::Query
            )
            .await
            .unwrap()
            .binding(),
        client.binding()
    );
    let mut disabled = fixture.profile.clone();
    disabled.workloads.code = None;
    let error = observed
        .client(
            &disabled,
            &fixture.manifest,
            WorkloadKind::Code,
            EmbeddingRole::Query,
        )
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("disabled_by_profile"));
    assert_eq!(peer.requests().len(), 2);
}

#[tokio::test]
async fn legacy_code_compatibility_cannot_change_space_or_enable_unsupported_capabilities() {
    for mode in ["space", "specialized", "late"] {
        let mut fixture = Fixture::new();
        fixture.profile.workloads.code = fixture.profile.workloads.documents.clone();
        let code = fixture.profile.workloads.code.as_mut().unwrap();
        let expected = match mode {
            "space" => {
                code.space.query.prompt = "different code prompt".into();
                "provider embedding behavior differs"
            }
            "specialized" => {
                fixture.profile.capabilities.specialized_code = true;
                fixture.description["capabilities"]["specialized_code"] = json!(true);
                code.specialized_code = true;
                "does not advertise the Code workload"
            }
            "late" => {
                fixture.profile.capabilities.late_chunking = true;
                fixture.profile.capabilities.tokenizer_offsets = true;
                code.chunking = ChunkPolicy::Late {
                    context_cap_tokens: 1000,
                    minimum_window_overlap_tokens: 100,
                    passages: None,
                };
                "provider no longer supports the saved late-chunk policy"
            }
            _ => unreachable!(),
        };
        fixture.profile.validate(&fixture.manifest).unwrap();
        let peer = Peer::new(vec![Reply::json(fixture.description.clone())]);
        let error = fixture
            .catalog(&peer)
            .verify_profile(&fixture.profile, &fixture.manifest)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains(expected), "{mode}: {error}");
        assert_eq!(peer.requests().len(), 1);
    }
}

#[tokio::test]
async fn ordinary_text_code_requires_explicit_observation_and_database_intent() {
    let mut fixture = mixed();
    fixture.profile.workloads.code = fixture.profile.workloads.documents.clone();
    fixture.profile.capabilities.specialized_code = false;
    fixture.description["workloads"]["code"] =
        fixture.description["workloads"]["documents"].clone();
    let peer = Peer::new(vec![
        Reply::json(fixture.description.clone()),
        Reply::json(embedding(&fixture, "code")),
    ]);
    let observed = fixture
        .catalog(&peer)
        .verify_profile(&fixture.profile, &fixture.manifest)
        .await
        .unwrap();
    observed
        .client(
            &fixture.profile,
            &fixture.manifest,
            WorkloadKind::Code,
            EmbeddingRole::Query,
        )
        .await
        .unwrap()
        .embed(&["ordinary text code".into()], None)
        .await
        .unwrap();
    let mut disabled = fixture.profile.clone();
    disabled.workloads.code = None;
    let error = observed
        .client(
            &disabled,
            &fixture.manifest,
            WorkloadKind::Code,
            EmbeddingRole::Query,
        )
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("disabled_by_profile"));
    assert_eq!(peer.requests().len(), 2);
    // V2 explicitly declares workloads: matching document space cannot replace an omitted code entry.
    fixture.description["workloads"]["code"] = Value::Null;
    let peer = Peer::new(vec![Reply::json(fixture.description.clone())]);
    let error = fixture
        .catalog(&peer)
        .verify_profile(&fixture.profile, &fixture.manifest)
        .await
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("does not advertise the Code workload")
    );
    assert_eq!(peer.requests().len(), 1);
}

#[tokio::test]
async fn capabilities_are_checked_per_workload_even_when_binding_the_other_one() {
    for mode in ["specialized", "late", "offsets", "space", "missing"] {
        let mut fixture = mixed();
        match mode {
            "specialized" => {
                fixture.description["workloads"]["code"]["capabilities"]["specialized_code"] =
                    json!(false)
            }
            "late" | "offsets" => {
                fixture.profile.capabilities.late_chunking = true;
                fixture.profile.capabilities.tokenizer_offsets = true;
                fixture.profile.workloads.code.as_mut().unwrap().chunking = ChunkPolicy::Late {
                    context_cap_tokens: 1000,
                    minimum_window_overlap_tokens: 100,
                    passages: None,
                };
                fixture.description["workloads"]["documents"]["capabilities"]["late_chunking"] =
                    json!(true);
                fixture.description["workloads"]["documents"]["capabilities"]["tokenizer_offsets"] =
                    json!(true);
                if mode == "offsets" {
                    fixture.description["workloads"]["code"]["capabilities"]["late_chunking"] =
                        json!(true);
                }
            }
            "space" => {
                fixture
                    .profile
                    .workloads
                    .code
                    .as_mut()
                    .unwrap()
                    .space
                    .query
                    .prompt = "different prompt".into();
            }
            "missing" => fixture.description["workloads"]["code"] = Value::Null,
            _ => unreachable!(),
        }
        fixture.profile.validate(&fixture.manifest).unwrap();
        let peer = Peer::new(vec![
            Reply::json(fixture.description.clone()),
            Reply::json(fixture.description.clone()),
        ]);
        let catalog = fixture.catalog(&peer);
        let observed = catalog.verify("fixture-local-provider").await.unwrap();
        let expected = match mode {
            "specialized" => "provider no longer supports an enabled workload",
            "late" | "offsets" => "provider no longer supports the saved late-chunk policy",
            "space" => "provider embedding behavior differs from the saved workload",
            _ => "does not advertise the Code workload",
        };
        let error = observed
            .client(
                &fixture.profile,
                &fixture.manifest,
                WorkloadKind::Documents,
                EmbeddingRole::Query,
            )
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains(expected), "{mode}: {error}");
        let error = catalog
            .verify_profile(&fixture.profile, &fixture.manifest)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains(expected), "{mode}: {error}");
        assert_eq!(peer.requests().len(), 2);
    }
}

#[tokio::test]
async fn mixed_receipts_cannot_cross_workloads_or_survive_a_reload() {
    for (kind, name, wrong_name) in [
        (WorkloadKind::Documents, "documents", "code"),
        (WorkloadKind::Code, "code", "documents"),
    ] {
        for reload in [false, true] {
            let fixture = mixed();
            let mut wrong = embedding(&fixture, name);
            if reload {
                wrong["receipt"]["serving_generation"] = json!("new-load");
            } else {
                wrong["receipt"] = fixture.description["workloads"][wrong_name]["receipt"].clone();
            }
            let peer = Peer::new(vec![
                Reply::json(fixture.description.clone()),
                Reply::json(wrong),
                Reply::json(embedding(&fixture, name)),
            ]);
            let observed = fixture
                .catalog(&peer)
                .verify_profile(&fixture.profile, &fixture.manifest)
                .await
                .unwrap();
            let client = observed
                .client(
                    &fixture.profile,
                    &fixture.manifest,
                    kind,
                    EmbeddingRole::Query,
                )
                .await
                .unwrap();
            let pin = client.binding().clone();
            let error = client
                .embed(&["same width, wrong identity".into()], None)
                .await
                .err()
                .unwrap();
            assert!(
                error.to_string().contains("provider receipt changed"),
                "{error}"
            );
            // A clone keeps its original pin; the failed response did not refresh it.
            client
                .clone()
                .embed(&["original receipt".into()], None)
                .await
                .unwrap();
            assert_eq!(client.binding(), &pin);
            assert_eq!(peer.requests().len(), 3);
        }
    }
}

#[tokio::test]
async fn malformed_or_incoherent_workload_observations_are_refused_in_full() {
    let fixture = mixed();
    let original = fixture.description.to_string();
    let duplicate_receipt = format!(
        "\"receipt\":{},\"receipt\":",
        fixture.description["workloads"]["code"]["receipt"]
    );
    let mut replies = vec![
        (
            original.replacen(
                "\"format\":\"hades-provider-description-v2\"",
                "\"format\":null",
                1,
            ),
            "invalid",
        ),
        (
            original.replacen("\"workloads\":", "\"workloads\":{},\"workloads\":", 1),
            "duplicate field",
        ),
        (
            original.replacen("\"code\":", "\"code\":null,\"code\":", 1),
            "duplicate field",
        ),
        (
            original.replacen("\"receipt\":", &duplicate_receipt, 1),
            "duplicate field",
        ),
        (
            original.replacen(
                "\"dimension\":384",
                "\"dimension\":384,\"dimension\":384",
                1,
            ),
            "duplicate field",
        ),
    ];
    for (pointer, value, expected) in [
        (
            "/format",
            json!("hades-provider-description-v99"),
            "unknown variant",
        ),
        ("/workloads", json!({}), "at least one workload"),
        (
            "/workloads",
            json!({"documents":null,"code":null}),
            "at least one workload",
        ),
        ("/model/id", json!("unregistered"), "model identity differs"),
        (
            "/model/tokenizer_files",
            json!(["missing.json"]),
            "tokenizer asset is absent",
        ),
        (
            "/workloads/code/receipt/artifact_md5",
            json!("0".repeat(32)),
            "identities disagree",
        ),
        (
            "/workloads/code/receipt/space_id",
            json!("0".repeat(64)),
            "identities disagree",
        ),
        (
            "/workloads/code/receipt/serving_generation",
            json!("another-load"),
            "different serving generations",
        ),
        (
            "/workloads/code/receipt/serving_generation",
            json!(""),
            "label",
        ),
        (
            "/workloads/code/capabilities/max_input_tokens",
            json!(0),
            "context token limit",
        ),
        (
            "/workloads/code/capabilities/text",
            json!(false),
            "require text capability",
        ),
        (
            "/workloads/code/capabilities/tasks",
            json!(["retrieval.query"]),
            "unsupported task",
        ),
        (
            "/workloads/code/capabilities/tasks",
            json!(["code", "code"]),
            "duplicate supported task",
        ),
    ] {
        let mut wrong = fixture.description.clone();
        *wrong.pointer_mut(pointer).unwrap() = value;
        replies.push((wrong.to_string(), expected));
    }
    for variant in [
        "mixed",
        "missing_version",
        "unknown",
        "unknown_workload",
        "unknown_capability",
        "bad_tokenizer",
        "bad_artifact",
    ] {
        let mut wrong = fixture.description.clone();
        let expected = match variant {
            "mixed" => {
                wrong["space"] = wrong["workloads"]["documents"]["space"].clone();
                "exactly one complete wire format"
            }
            "missing_version" => {
                wrong.as_object_mut().unwrap().remove("format");
                "exactly one complete wire format"
            }
            "unknown" => {
                wrong["unknown"] = json!(true);
                "unknown field"
            }
            "unknown_workload" => {
                wrong["workloads"]["images"] = json!({});
                "unknown field"
            }
            "unknown_capability" => {
                wrong["workloads"]["code"]["capabilities"]["unknown"] = json!(true);
                "unknown field"
            }
            "bad_tokenizer" | "bad_artifact" => {
                let field = if variant == "bad_tokenizer" {
                    "tokenizer_manifest_md5"
                } else {
                    "artifact_md5"
                };
                wrong["workloads"]["code"]["space"][field] = json!("0".repeat(32));
                let space: EmbeddingSpace =
                    serde_json::from_value(wrong["workloads"]["code"]["space"].clone()).unwrap();
                wrong["workloads"]["code"]["receipt"]["space_id"] = json!(space.id().unwrap());
                if variant == "bad_tokenizer" {
                    "tokenizer differs"
                } else {
                    "artifact differs"
                }
            }
            _ => unreachable!(),
        };
        replies.push((wrong.to_string(), expected));
    }
    for (body, expected) in replies {
        let peer = Peer::new(vec![Reply {
            body,
            status: 200,
            before: None,
            delay: std::time::Duration::ZERO,
        }]);
        let error = fixture
            .catalog(&peer)
            .verify("fixture-local-provider")
            .await
            .err()
            .expect("malformed observation must fail");
        assert!(
            error.to_string().contains(expected),
            "expected {expected}: {error}"
        );
        assert_eq!(peer.requests().len(), 1);
    }
}
