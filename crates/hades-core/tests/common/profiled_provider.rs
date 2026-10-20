//! Synthetic verified HTTP peer. Receipts are provider-owned observations, never
//! copied from the client's expected identity. All sockets are private and temporary.
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use hades_core::{
    db::{ArangoPool, crud},
    embedding_profile::{
        ArtifactManifest, EmbeddingProfile, ProfileState, ProviderReceipt, store::ProfileStore,
    },
    persephone::{bound_embedding::ProviderConnection, embedding::EmbeddingEndpoint},
    pipeline::profiled_documents::ProfiledDocuments,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::oneshot,
};

pub struct Fixture {
    pub profile: EmbeddingProfile,
    pub manifest: ArtifactManifest,
    pub receipt: ProviderReceipt,
}

impl Fixture {
    /// Derive a self-consistent model, provider, and space for a synthetic width.
    pub fn new(dimension: u32) -> Self {
        let value: Value =
            serde_json::from_str(include_str!("../fixtures/embedding_profile.json")).unwrap();
        let mut profile: EmbeddingProfile =
            serde_json::from_value(value["profile"].clone()).unwrap();
        profile.state = ProfileState::Initializing;
        profile.model.id = format!("fixture/model-{dimension}");
        profile.runtime_binding.provider_id = format!("fixture/provider-{dimension}");
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
        let store = ProfileStore::new(pool.clone()).unwrap();
        let initial = store
            .initialize(self.profile.clone(), self.manifest.clone())
            .await
            .unwrap();
        for collection in ["documents", "chunks", "embeddings"] {
            crud::create_collection(pool, collection, Some(2))
                .await
                .unwrap();
        }
        store.mark_ready(&initial).await.unwrap();
    }

    /// Resolve the saved profile using the peer's independently held observation.
    pub async fn connect(&self, pool: &ArangoPool, peer: &Peer) -> ProfiledDocuments {
        ProfiledDocuments::connect(pool.clone(), peer.connection(), &peer.receipt())
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
    requests: Vec<Value>,
    plans: Vec<Value>,
    plan_patch: Option<Value>,
    plan_gate: Option<Gate>,
    fault: Option<(usize, Fault)>,
    gate: Option<Gate>,
}

pub struct Peer {
    path: PathBuf,
    provider_id: String,
    state: Arc<Mutex<State>>,
    server: tokio::task::JoinHandle<()>,
    _directory: tempfile::TempDir,
}

impl Peer {
    /// Serve deterministic normalized vectors with injectable failures over a private socket.
    pub fn new(fixture: &Fixture) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("provider.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let model = fixture.profile.model.id.clone();
        let dimension = fixture
            .profile
            .workloads
            .documents
            .as_ref()
            .or(fixture.profile.workloads.code.as_ref())
            .unwrap()
            .space
            .representation
            .dimension as usize;
        let state = Arc::new(Mutex::new(State {
            receipt: fixture.receipt.clone(),
            requests: Vec::new(),
            plans: Vec::new(),
            plan_patch: None,
            plan_gate: None,
            fault: None,
            gate: None,
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
                    let planning = headers.starts_with("POST /v1/passages HTTP/1.1\r\n");
                    assert!(planning || headers.starts_with("POST /v1/embeddings HTTP/1.1\r\n"));
                    let length: usize = headers.lines().find_map(|line| {
                        line.to_lowercase().strip_prefix("content-length:")
                            .map(|n| n.trim().parse().unwrap())
                    }).unwrap();
                    assert!(length < 8 * 1024 * 1024);
                    let mut body = vec![0; length];
                    stream.read_exact(&mut body).await.unwrap();
                    let request: Value = serde_json::from_slice(&body).unwrap();
                    let (receipt, fault, gate, patch) = {
                        let mut state = observed.lock().unwrap();
                        if planning {
                            state.plans.push(request.clone());
                            (state.receipt.clone(), None, state.plan_gate.take(), state.plan_patch.take())
                        } else {
                            state.requests.push(request.clone());
                            let fault = state.fault.filter(|(at, _)| *at == state.requests.len()).map(|(_, f)| f);
                            (state.receipt.clone(), fault, state.gate.take(), None)
                        }
                    };
                    if let Some(gate) = gate {
                        let _ = gate.started.send(());
                        let _ = gate.release.await;
                    }
                    let mut body = if planning {
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
                            "model":model,"task":"retrieval.passage","receipt":receipt,
                            "source_sha256":Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>(),
                            "passage_tokens":request["passage_tokens"],"overlap_tokens":request["overlap_tokens"],
                            "max_input_tokens":request["max_input_tokens"],"spans":spans});
                        if let Some(patch) = patch {
                            for (key,value) in patch.as_object().unwrap() { plan[key] = value.clone(); }
                        }
                        plan
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
            state,
            server,
            _directory: directory,
        }
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
