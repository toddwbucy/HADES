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
                    assert!(headers.starts_with("POST /v1/embeddings HTTP/1.1\r\n"));
                    let length: usize = headers.lines().find_map(|line| {
                        line.to_lowercase().strip_prefix("content-length:")
                            .map(|n| n.trim().parse().unwrap())
                    }).unwrap();
                    assert!(length < 8 * 1024 * 1024);
                    let mut body = vec![0; length];
                    stream.read_exact(&mut body).await.unwrap();
                    let request: Value = serde_json::from_slice(&body).unwrap();
                    let (receipt, fault, gate) = {
                        let mut state = observed.lock().unwrap();
                        state.requests.push(request.clone());
                        let fault = state.fault.filter(|(at, _)| *at == state.requests.len()).map(|(_, f)| f);
                        (state.receipt.clone(), fault, state.gate.take())
                    };
                    if let Some(gate) = gate {
                        let _ = gate.started.send(());
                        let _ = gate.release.await;
                    }
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
                    let mut body = json!({"model":model,"receipt":receipt,"data":data});
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

    pub fn connection(&self) -> ProviderConnection {
        ProviderConnection {
            provider_id: self.provider_id.clone(),
            endpoint: EmbeddingEndpoint::Unix(self.path.clone()),
            timeout: Duration::from_secs(10),
            max_response_bytes: 8 * 1024 * 1024,
        }
    }
    pub fn receipt(&self) -> ProviderReceipt {
        self.state.lock().unwrap().receipt.clone()
    }
    pub fn requests(&self) -> Vec<Value> {
        self.state.lock().unwrap().requests.clone()
    }
    pub fn set_receipt(&self, receipt: ProviderReceipt) {
        self.state.lock().unwrap().receipt = receipt;
    }
    pub fn fault_after(&self, requests: usize, fault: Fault) {
        let mut state = self.state.lock().unwrap();
        state.fault = Some((state.requests.len() + requests, fault));
    }
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
    fn drop(&mut self) {
        self.server.abort();
    }
}
