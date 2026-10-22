//! Synthetic verified HTTP peer. Receipts are provider-owned observations, never
//! copied from the client's expected identity. All sockets are private and temporary.
use std::{
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use hades_core::{
    db::{ArangoPool, crud},
    embedding_profile::{
        ArtifactManifest, EmbeddingProfile, ProfileState, ProviderReceipt, store::ProfileStore,
    },
    persephone::{
        bound_embedding::ProviderConnection, embedding::EmbeddingEndpoint,
        provider_registry::ProviderCatalog,
    },
    pipeline::profiled_documents::ProfiledDocuments,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::oneshot,
};

#[path = "registered_provider.rs"]
mod registration_fixture;

pub struct Fixture {
    pub profile: EmbeddingProfile,
    pub manifest: ArtifactManifest,
    pub receipt: ProviderReceipt,
}

impl Fixture {
    /// One model load serves independently bound document and code spaces.
    #[allow(dead_code)]
    pub fn mixed(document_dimension: u32, code_dimension: u32) -> Self {
        let mut fixture = Self::new(document_dimension);
        let code = Self::code(code_dimension);
        fixture.profile.capabilities.specialized_code = true;
        fixture
            .profile
            .capabilities
            .tasks
            .extend(code.profile.capabilities.tasks);
        fixture.profile.workloads.code = code.profile.workloads.code;
        fixture
    }

    /// A separate, asymmetric code space with no document workload.
    #[allow(dead_code)]
    pub fn code(dimension: u32) -> Self {
        let mut fixture = Self::new(dimension);
        let mut workload = fixture.profile.workloads.documents.take().unwrap();
        workload.specialized_code = true;
        workload.space.passage.task = "fixture.code.passage".into();
        workload.space.query.task = "fixture.code.query".into();
        fixture.profile.capabilities.specialized_code = true;
        fixture.profile.capabilities.tasks =
            vec!["fixture.code.passage".into(), "fixture.code.query".into()];
        fixture.receipt.space_id = workload.space.id().unwrap();
        fixture.profile.workloads.code = Some(workload);
        fixture
    }

    /// A distinct synthetic contextual space; no numerical model claim is made.
    pub fn late() -> Self {
        let mut fixture = Self::new(2);
        fixture.profile.capabilities.late_chunking = true;
        fixture.profile.capabilities.tokenizer_offsets = true;
        let workload = fixture.profile.workloads.documents.as_mut().unwrap();
        workload.chunking = hades_core::embedding_profile::ChunkPolicy::Late {
            context_cap_tokens: 1000,
            minimum_window_overlap_tokens: 1,
            passages: Some(hades_core::embedding_profile::PassagePolicy {
                passage_tokens: 500,
                overlap_tokens: 0,
            }),
        };
        workload.space.representation.pooling = "fixture-contextual-token-mean-v1".into();
        fixture.receipt.space_id = workload.space.id().unwrap();
        fixture
    }

    /// Derive a self-consistent model, provider, and space for a synthetic width.
    pub fn new(dimension: u32) -> Self {
        let value: Value =
            serde_json::from_str(include_str!("../fixtures/embedding_profile.json")).unwrap();
        let mut profile: EmbeddingProfile =
            serde_json::from_value(value["profile"].clone()).unwrap();
        profile.state = ProfileState::Initializing;
        profile.model.id = format!("fixture/model-{dimension}");
        profile.runtime_binding.provider_id = format!("fixture-provider-{dimension}");
        let space = &mut profile.workloads.documents.as_mut().unwrap().space;
        space.representation.dimension = dimension;
        let mut receipt: ProviderReceipt =
            serde_json::from_value(value["receipt"].clone()).unwrap();
        receipt.space_id = space.id().unwrap();
        Self {
            profile,
            receipt,
            manifest: serde_json::from_value(value["manifest"].clone()).unwrap(),
        }
    }

    /// Publish the fixture only inside the caller's freshly allocated test database.
    pub async fn install(&self, pool: &ArangoPool) {
        self.install_collections(pool, &[("documents", 2), ("chunks", 2), ("embeddings", 2)])
            .await;
    }

    /// Install code storage explicitly, preserving document admission fixtures.
    #[allow(dead_code)]
    pub async fn install_code(&self, pool: &ArangoPool) {
        self.install_collections(
            pool,
            &hades_core::db::collections::CODEBASE.all_collections(),
        )
        .await;
    }

    #[allow(dead_code)]
    pub async fn install_mixed(&self, pool: &ArangoPool) {
        let mut collections = vec![("documents", 2), ("chunks", 2), ("embeddings", 2)];
        collections.extend(hades_core::db::collections::CODEBASE.all_collections());
        self.install_collections(pool, &collections).await;
    }

    async fn install_collections(&self, pool: &ArangoPool, collections: &[(&str, u32)]) {
        let store = ProfileStore::new(pool.clone()).unwrap();
        let initial = store
            .initialize(self.profile.clone(), self.manifest.clone())
            .await
            .unwrap();
        for &(collection, kind) in collections {
            crud::create_collection(pool, collection, Some(kind))
                .await
                .unwrap();
            let owner = match collection {
                "chunks" | "embeddings" => Some("parent_key"),
                "codebase_files" | "documents" => None,
                _ => Some("file_key"),
            };
            if let Some(field) = owner {
                hades_core::db::index::ensure_lookup_index(pool, collection, field)
                    .await
                    .unwrap();
            }
        }
        store.mark_ready(&initial).await.unwrap();
    }

    /// Resolve the saved profile using the peer's independently held observation.
    pub async fn connect(&self, pool: &ArangoPool, peer: &Peer) -> ProfiledDocuments {
        ProfiledDocuments::connect(pool.clone(), &peer.catalog())
            .await
            .unwrap()
    }
}

#[derive(Clone, Copy)]
pub enum Fault {
    MissingReceipt,
    WrongModel,
    Short,
    Zero,
    NotNormalized,
    Reject,
}

struct Gate {
    started: oneshot::Sender<()>,
    release: oneshot::Receiver<()>,
}

struct State {
    receipt: ProviderReceipt,
    code_receipt: Option<ProviderReceipt>,
    requests: Vec<Value>,
    plans: Vec<Value>,
    plan_patch: Option<Value>,
    plan_gate: Option<Gate>,
    fault: Option<(usize, Fault)>,
    gate: Option<Gate>,
    observations: usize,
    verify_gate: Option<Gate>,
}

pub struct Peer {
    path: PathBuf,
    provider_id: String,
    registry: PathBuf,
    state: Arc<Mutex<State>>,
    server: tokio::task::JoinHandle<()>,
    _directory: tempfile::TempDir,
}

impl Peer {
    /// Serve deterministic normalized vectors with injectable failures over a private socket.
    pub fn new(fixture: &Fixture) -> Self {
        Self::start(fixture, false)
    }

    /// Advertise every enabled workload through the explicit workload wire format.
    pub fn with_workload_description(fixture: &Fixture) -> Self {
        Self::start(fixture, true)
    }

    fn start(fixture: &Fixture, workload_description: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let registry =
            registration_fixture::create(directory.path(), &fixture.profile, &fixture.manifest);
        let path = directory.path().join("provider.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let model = fixture.profile.model.id.clone();
        let workload_key = if fixture.profile.workloads.documents.is_some() {
            "documents"
        } else {
            "code"
        };
        let primary = fixture
            .profile
            .workloads
            .documents
            .as_ref()
            .or(fixture.profile.workloads.code.as_ref())
            .unwrap()
            .clone();
        let secondary = fixture
            .profile
            .workloads
            .documents
            .as_ref()
            .and(fixture.profile.workloads.code.as_ref())
            .cloned();
        let code_receipt = secondary.as_ref().map(|workload| {
            let mut receipt = fixture.receipt.clone();
            receipt.space_id = workload.space.id().unwrap();
            receipt
        });
        let description = json!({"model":fixture.profile.model,"capabilities":fixture.profile.capabilities,
            "space":primary.space,"receipt":fixture.receipt});
        let secondary_description = secondary.as_ref().map(|workload| {
            json!({
            "capabilities":fixture.profile.capabilities,"space":workload.space})
        });
        let state = Arc::new(Mutex::new(State {
            receipt: fixture.receipt.clone(),
            code_receipt,
            requests: Vec::new(),
            plans: Vec::new(),
            plan_patch: None,
            plan_gate: None,
            fault: None,
            gate: None,
            observations: 0,
            verify_gate: None,
        }));
        let observed = state.clone();
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                tokio::time::timeout(Duration::from_secs(15), async {
                    let mut headers = Vec::new();
                    while !headers.ends_with(b"\r\n\r\n") {
                        headers.push(stream.read_u8().await.unwrap());
                        assert!(headers.len() < 8192);
                    }
                    let headers = String::from_utf8(headers).unwrap();
                    let late_planning = headers.starts_with("POST /v1/late-passages HTTP/1.1\r\n");
                    let late_embedding = headers.starts_with("POST /v1/late-embeddings HTTP/1.1\r\n");
                    let planning = late_planning || headers.starts_with("POST /v1/passages HTTP/1.1\r\n");
                    let verifying = headers.starts_with("POST /v1/verify HTTP/1.1\r\n");
                    assert!(verifying || planning || late_embedding || headers.starts_with("POST /v1/embeddings HTTP/1.1\r\n"));
                    let length: usize = headers.lines().find_map(|line| {
                        line.to_lowercase().strip_prefix("content-length:")
                            .map(|n| n.trim().parse().unwrap())
                    }).unwrap_or(0);
                    assert!(length < 8 * 1024 * 1024);
                    let mut body = vec![0; length];
                    stream.read_exact(&mut body).await.unwrap();
                    let request: Value = if verifying { json!({}) } else { serde_json::from_slice(&body).unwrap() };
                    let is_code = secondary.as_ref().is_some_and(|workload|
                        request["task"] == workload.space.passage.task || request["task"] == workload.space.query.task);
                    let workload = if is_code { secondary.as_ref().unwrap() } else { &primary };
                    let dimension = workload.space.representation.dimension as usize;
                    let passage_task = &workload.space.passage.task;
                    let (receipt, code_receipt, fault, gate, patch) = {
                        let mut state = observed.lock().unwrap();
                        let receipt = if is_code { state.code_receipt.as_ref().unwrap() } else { &state.receipt }.clone();
                        let code_receipt = state.code_receipt.clone();
                        if verifying {
                            state.observations += 1;
                            (receipt, code_receipt, None, state.verify_gate.take(), None)
                        } else if planning {
                            state.plans.push(request.clone());
                            (receipt, code_receipt, None, state.plan_gate.take(), state.plan_patch.take())
                        } else {
                            state.requests.push(request.clone());
                            let fault = state.fault.filter(|(at, _)| *at == state.requests.len()).map(|(_, f)| f);
                            (receipt, code_receipt, fault, state.gate.take(), None)
                        }
                    };
                    if let Some(gate) = gate {
                        let _ = gate.started.send(());
                        let _ = gate.release.await;
                    }
                    let mut body = if verifying {
                        let mut response = description.clone();
                        response["receipt"] = json!(receipt);
                        if workload_description {
                            let observation = json!({"capabilities":response["capabilities"],
                                "space":response["space"],"receipt":response["receipt"]});
                            response = json!({"format":"hades-provider-description-v2", "model":response["model"],
                                "workloads":{"documents":null,"code":null}});
                            response["workloads"][workload_key] = observation;
                            if let Some(mut code) = secondary_description.clone() {
                                code["receipt"] = json!(code_receipt.unwrap());
                                response["workloads"]["code"] = code;
                            }
                        }
                        response
                    } else if planning {
                        let text = request["input"].as_str().unwrap();
                        let mut offset = 0;
                        let spans: Vec<_> = text.split_inclusive('\n').map(|part| {
                            let start = offset;
                            offset += part.len();
                            let count = part.chars().count();
                            json!({"start_byte":start,"end_byte":offset,
                                "body_tokens":count,"input_tokens":count+3,"overlap_tokens":0})
                        }).collect();
                        let mut plan = json!({"format":"hades-token-passages-v1",
                            "model":model,"task":passage_task,"receipt":receipt,
                            "source_sha256":Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>(),
                            "passage_tokens":request["passage_tokens"],"overlap_tokens":request["overlap_tokens"],
                            "max_input_tokens":request["max_input_tokens"],"spans":spans});
                        if late_planning {
                            let mut windows = Vec::new();
                            let mut previous_end = 0;
                            for start in 0..spans.len().saturating_sub(1).max(1) {
                                let left = spans[start]["start_byte"].as_u64().unwrap() as usize;
                                let right = spans[(start+1).min(spans.len()-1)]["end_byte"].as_u64().unwrap() as usize;
                                let count = text[left..right].chars().count();
                                windows.push(json!({"start_byte":left,"end_byte":right,"body_tokens":count,
                                    "input_tokens":count+3,"overlap_tokens":if previous_end > left {text[left..previous_end].chars().count()} else {0}}));
                                previous_end = right;
                            }
                            let assignments: Vec<_> = spans.iter().map(|span| windows.iter().position(|window|
                                window["start_byte"].as_u64().unwrap() <= span["start_byte"].as_u64().unwrap()
                                    && span["end_byte"].as_u64().unwrap() <= window["end_byte"].as_u64().unwrap()).unwrap()).collect();
                            plan["format"] = json!("hades-late-passages-v1");
                            plan.as_object_mut().unwrap().remove("spans");
                            plan["passages"] = json!(spans);
                            plan["windows"] = json!(windows);
                            plan["assignments"] = json!(assignments);
                            plan["context_cap_tokens"] = request["context_cap_tokens"].clone();
                            plan["max_input_tokens"] = request["context_cap_tokens"].clone();
                            plan["minimum_window_overlap_tokens"] = request["minimum_window_overlap_tokens"].clone();
                        }
                        if let Some(patch) = patch {
                            for (key,value) in patch.as_object().unwrap() { plan[key] = value.clone(); }
                        }
                        plan
                    } else if late_embedding {
                        let text = request["input"].as_str().unwrap();
                        let chars: Vec<_> = text.chars().collect();
                        let data: Vec<_> = request["boundaries"].as_array().unwrap().iter().enumerate().map(|(index, span)| {
                            let start = span[0].as_u64().unwrap() as usize;
                            let end = span[1].as_u64().unwrap() as usize;
                            let value: String = chars[start..end].iter().collect();
                            let mut vector = vec![0.0; dimension];
                            vector[usize::from(value.contains("banana"))] = 1.0;
                            match fault {
                                Some(Fault::Short) => { vector.pop(); }
                                Some(Fault::Zero) => vector.fill(0.0),
                                Some(Fault::NotNormalized) => vector.iter_mut().for_each(|v| *v *= 2.0),
                                _ => {}
                            }
                            json!({"index":index,"embedding":vector,"char_start":start,"char_end":end,
                                "token_start":start+3,"token_end":end+3})
                        }).collect();
                        json!({"format":"hades-late-embedding-v1","model":model,"receipt":receipt,
                            "source_sha256":Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>(),
                            "input_tokens":chars.len()+3,"data":data})
                    } else {
                    let data: Vec<_> = request["input"].as_array().unwrap().iter().enumerate()
                        .map(|(index, text)| {
                            let mut vector = vec![0.0; dimension];
                            vector[usize::from(text.as_str().unwrap().contains("banana"))] = 1.0;
                            match fault {
                                Some(Fault::Short) => { vector.pop(); }
                                Some(Fault::Zero) => vector.fill(0.0),
                                Some(Fault::NotNormalized) => vector.iter_mut().for_each(|v| *v *= 2.0),
                                _ => {}
                            }
                            json!({"index":index,"embedding":vector})
                        }).collect();
                    json!({"model":model,"receipt":receipt,"data":data})
                    };
                    let status = match fault {
                        Some(Fault::MissingReceipt) => { body.as_object_mut().unwrap().remove("receipt"); 200 }
                        Some(Fault::WrongModel) => { body["model"] = json!("other"); 200 }
                        Some(Fault::Reject) => { body = json!({"error":"fixture context overflow"}); 422 }
                        _ => 200,
                    };
                    let body = body.to_string();
                    let wire = format!("HTTP/1.1 {status} Response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                    let _ = stream.write_all(wire.as_bytes()).await;
                }).await.expect("fixture request must finish");
            }
        });
        Self {
            path,
            provider_id: fixture.profile.runtime_binding.provider_id.clone(),
            registry,
            state,
            server,
            _directory: directory,
        }
    }

    /// Resolve only this synthetic peer through its protected local registration.
    pub fn catalog(&self) -> ProviderCatalog {
        self.configuration().catalog().unwrap()
    }
    /// Explicit host configuration for shared CLI/service admission tests.
    pub fn configuration(&self) -> hades_core::config::RegisteredProvidersConfig {
        let connection = self.connection();
        let EmbeddingEndpoint::Unix(socket) = connection.endpoint else {
            unreachable!()
        };
        hades_core::config::RegisteredProvidersConfig {
            registry_root: self.registry.clone(),
            providers: vec![hades_core::config::RegisteredProviderConfig {
                id: connection.provider_id,
                socket,
                timeout_seconds: connection.timeout.as_secs(),
                max_response_bytes: connection.max_response_bytes,
            }],
        }
    }
    /// Count live verification separately from planning/inference.
    pub fn observations(&self) -> usize {
        self.state.lock().unwrap().observations
    }
    /// Hold model observation to test profile changes before client admission.
    pub fn hold_next_verification(&self) -> (oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (started, receive) = oneshot::channel();
        let (release, wait) = oneshot::channel();
        self.state.lock().unwrap().verify_gate = Some(Gate {
            started,
            release: wait,
        });
        (receive, release)
    }

    /// Supply transport settings without overriding any database-selected model fields.
    pub fn connection(&self) -> ProviderConnection {
        ProviderConnection {
            provider_id: self.provider_id.clone(),
            endpoint: EmbeddingEndpoint::Unix(self.path.clone()),
            timeout: Duration::from_secs(10),
            max_response_bytes: 8 * 1024 * 1024,
        }
    }
    /// Read the provider's current observation for initial connection or verified reload.
    pub fn receipt(&self) -> ProviderReceipt {
        self.state.lock().unwrap().receipt.clone()
    }
    /// Capture exact requests for task, model, batching, and receipt assertions.
    pub fn requests(&self) -> Vec<Value> {
        self.state.lock().unwrap().requests.clone()
    }
    /// Inspect planning separately so embedding batching assertions stay explicit.
    pub fn plans(&self) -> Vec<Value> {
        self.state.lock().unwrap().plans.clone()
    }
    /// Corrupt exactly one plan response before any embedding is requested.
    pub fn patch_next_plan(&self, patch: Value) {
        self.state.lock().unwrap().plan_patch = Some(patch);
    }
    /// Hold planning to test policy changes before the first embedding request.
    pub fn hold_next_plan(&self) -> (oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (started, receive) = oneshot::channel();
        let (release, wait) = oneshot::channel();
        self.state.lock().unwrap().plan_gate = Some(Gate {
            started,
            release: wait,
        });
        (receive, release)
    }
    /// Simulate a provider identity/load change independently of client expectations.
    #[allow(dead_code)]
    pub fn set_code_receipt(&self, receipt: ProviderReceipt) {
        self.state.lock().unwrap().code_receipt = Some(receipt);
    }

    pub fn set_receipt(&self, receipt: ProviderReceipt) {
        self.state.lock().unwrap().receipt = receipt;
    }
    /// Fail the Nth subsequent request, allowing earlier batches to finish successfully.
    pub fn fault_after(&self, requests: usize, fault: Fault) {
        let mut state = self.state.lock().unwrap();
        state.fault = Some((state.requests.len() + requests, fault));
    }
    /// Synchronize an in-flight request without using timing sleeps or polling.
    pub fn hold_next(&self) -> (oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (started, receive) = oneshot::channel();
        let (release, wait) = oneshot::channel();
        self.state.lock().unwrap().gate = Some(Gate {
            started,
            release: wait,
        });
        (receive, release)
    }
}

impl Drop for Peer {
    /// Stop fixture work on success or panic, including an unreleased request gate.
    fn drop(&mut self) {
        self.server.abort();
    }
}
