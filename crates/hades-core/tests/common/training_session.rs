//! Private gRPC ownership contracts; no deployed trainer is contacted.
use hades_core::training::TrainingClient;
use hades_proto::training::training_service_server::{TrainingService, TrainingServiceServer};
use hades_proto::training::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tonic::{Request, Response, Status};

#[derive(Default)]
struct State {
    owner: Option<String>,
    generation: u32,
    calls: Vec<&'static str>,
    renewals: usize,
    reject_renewal: bool,
    legacy_provider: bool,
    operation_delay: Duration,
    stall_renewal: bool,
}
#[derive(Clone, Default)]
struct Service(Arc<Mutex<State>>, Arc<tokio::sync::Mutex<()>>);
impl Service {
    // Match the gRPC trait error type used by the mock handlers.
    #[allow(clippy::result_large_err)]
    fn owned<T>(&self, request: &Request<T>, name: &'static str) -> Result<(), Status> {
        let mut state = self.0.lock().unwrap();
        if request
            .metadata()
            .get("hades-training-session")
            .and_then(|v| v.to_str().ok())
            != state.owner.as_deref()
            || state.owner.is_none()
        {
            return Err(Status::failed_precondition("wrong owner"));
        }
        state.calls.push(name);
        Ok(())
    }
}
#[tonic::async_trait]
impl TrainingService for Service {
    async fn acquire_session(
        &self,
        _: Request<AcquireSessionRequest>,
    ) -> Result<Response<AcquireSessionResponse>, Status> {
        let mut state = self.0.lock().unwrap();
        if state.legacy_provider {
            return Err(Status::unimplemented("legacy provider"));
        }
        if state.owner.is_some() {
            return Err(Status::resource_exhausted("owned"));
        }
        state.generation += 1;
        let token = format!("test-session-{}", state.generation);
        state.owner = Some(token.clone());
        Ok(Response::new(AcquireSessionResponse {
            token,
            lease_seconds: 1,
        }))
    }
    async fn renew_session(
        &self,
        request: Request<SessionRequest>,
    ) -> Result<Response<SessionResponse>, Status> {
        let _operation = self.1.lock().await;
        self.owned(&request, "renew")?;
        if self.0.lock().unwrap().stall_renewal {
            std::future::pending::<()>().await;
        }
        let mut state = self.0.lock().unwrap();
        if state.reject_renewal {
            return Err(Status::failed_precondition("expired"));
        }
        state.renewals += 1;
        Ok(Response::new(SessionResponse {}))
    }
    async fn release_session(
        &self,
        request: Request<SessionRequest>,
    ) -> Result<Response<SessionResponse>, Status> {
        self.owned(&request, "release")?;
        self.0.lock().unwrap().owner = None;
        Ok(Response::new(SessionResponse {}))
    }
    async fn init_model(
        &self,
        request: Request<InitModelRequest>,
    ) -> Result<Response<InitModelResponse>, Status> {
        let _operation = self.1.lock().await;
        self.owned(&request, "init_model")?;
        let delay = self.0.lock().unwrap().operation_delay;
        tokio::time::sleep(delay).await;
        Ok(Response::new(InitModelResponse::default()))
    }
    async fn load_graph(
        &self,
        request: Request<LoadGraphRequest>,
    ) -> Result<Response<LoadGraphResponse>, Status> {
        self.owned(&request, "load_graph")?;
        Ok(Response::new(LoadGraphResponse::default()))
    }
    async fn train_step(
        &self,
        request: Request<TrainStepRequest>,
    ) -> Result<Response<TrainStepResponse>, Status> {
        self.owned(&request, "train_step")?;
        Ok(Response::new(TrainStepResponse::default()))
    }
    async fn evaluate(
        &self,
        request: Request<EvaluateRequest>,
    ) -> Result<Response<EvaluateResponse>, Status> {
        self.owned(&request, "evaluate")?;
        Ok(Response::new(EvaluateResponse::default()))
    }
    async fn get_embeddings(
        &self,
        request: Request<GetEmbeddingsRequest>,
    ) -> Result<Response<GetEmbeddingsResponse>, Status> {
        self.owned(&request, "get_embeddings")?;
        Ok(Response::new(GetEmbeddingsResponse::default()))
    }
    async fn checkpoint(
        &self,
        request: Request<CheckpointRequest>,
    ) -> Result<Response<CheckpointResponse>, Status> {
        self.owned(&request, "checkpoint")?;
        Ok(Response::new(CheckpointResponse::default()))
    }
    async fn load_checkpoint(
        &self,
        request: Request<LoadCheckpointRequest>,
    ) -> Result<Response<LoadCheckpointResponse>, Status> {
        self.owned(&request, "load_checkpoint")?;
        Ok(Response::new(LoadCheckpointResponse::default()))
    }
}

struct Server {
    service: Service,
    path: std::path::PathBuf,
    task: tokio::task::JoinHandle<()>,
    _dir: tempfile::TempDir,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl Server {
    fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("training.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(stream, _)| stream), listener))
        });
        let service = Service::default();
        let rpc = TrainingServiceServer::new(service.clone());
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(rpc)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        Self {
            service,
            path,
            task,
            _dir: dir,
        }
    }
    async fn wait_for(&self, predicate: impl Fn(&State) -> bool) {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if predicate(&self.service.0.lock().unwrap()) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }
}

#[tokio::test]
async fn cloned_client_owns_full_lifecycle_and_releases_only_after_last_drop() {
    let server = Server::start();
    let client = TrainingClient::connect_unix_at(&server.path).await.unwrap();
    let retained = client.clone();
    assert!(TrainingClient::connect_unix_at(&server.path).await.is_err());
    client
        .init_model(ModelConfig::default(), OptimizerConfig::default(), "cpu")
        .await
        .unwrap();
    client.load_graph("synthetic.safetensors").await.unwrap();
    client.train_step(vec![0], vec![1], vec![2]).await.unwrap();
    client.evaluate(vec![1], vec![2], vec![3]).await.unwrap();
    client.get_embeddings(None).await.unwrap();
    client.checkpoint("synthetic.pt").await.unwrap();
    client
        .load_checkpoint("synthetic.pt", Some("cpu"))
        .await
        .unwrap();
    for operation in [
        "init_model",
        "load_graph",
        "train_step",
        "evaluate",
        "get_embeddings",
        "checkpoint",
        "load_checkpoint",
    ] {
        assert!(server.service.0.lock().unwrap().calls.contains(&operation));
    }
    drop(client);
    server.wait_for(|state| state.renewals >= 2).await;
    assert!(TrainingClient::connect_unix_at(&server.path).await.is_err());
    retained.get_embeddings(None).await.unwrap();
    drop(retained);
    server.wait_for(|state| state.owner.is_none()).await;
    let successor = TrainingClient::connect_unix_at(&server.path).await.unwrap();
    successor
        .init_model(ModelConfig::default(), OptimizerConfig::default(), "cpu")
        .await
        .unwrap();
    drop(successor);
    server.wait_for(|state| state.owner.is_none()).await;
}

#[tokio::test]
async fn renewal_failure_fails_closed_without_automatic_reacquisition() {
    let server = Server::start();
    let client = TrainingClient::connect_unix_at(&server.path).await.unwrap();
    server.service.0.lock().unwrap().reject_renewal = true;
    server
        .wait_for(|state| state.calls.contains(&"renew"))
        .await;
    // The mock's recorded call precedes delivery of its failed response.
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if client.get_embeddings(None).await.is_err() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(server.service.0.lock().unwrap().generation, 1);
    drop(client);
    server.wait_for(|state| state.owner.is_none()).await;
}

#[tokio::test]
async fn cancelled_lifecycle_drops_its_last_guard_and_releases() {
    let server = Server::start();
    let client = TrainingClient::connect_unix_at(&server.path).await.unwrap();
    let (started, ready) = tokio::sync::oneshot::channel();
    let lifecycle = tokio::spawn(async move {
        let _client = client;
        started.send(()).unwrap();
        std::future::pending::<()>().await;
    });
    ready.await.unwrap();
    lifecycle.abort();
    assert!(lifecycle.await.unwrap_err().is_cancelled());
    server.wait_for(|state| state.owner.is_none()).await;
    let successor = TrainingClient::connect_unix_at(&server.path).await.unwrap();
    drop(successor);
    server.wait_for(|state| state.owner.is_none()).await;
}

#[tokio::test]
async fn legacy_provider_fails_without_unowned_fallback() {
    let server = Server::start();
    server.service.0.lock().unwrap().legacy_provider = true;
    let error = TrainingClient::connect_unix_at(&server.path)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("Unimplemented"));
    let state = server.service.0.lock().unwrap();
    assert!(state.owner.is_none());
    assert!(state.calls.is_empty());
    assert_eq!(state.generation, 0);
}

#[tokio::test]
async fn renewal_waiting_for_a_valid_operation_does_not_poison_the_session() {
    let server = Server::start();
    server.service.0.lock().unwrap().operation_delay = Duration::from_millis(900);
    let client = TrainingClient::connect(hades_core::training::TrainingClientConfig {
        endpoint: hades_core::training::TrainingEndpoint::Unix(server.path.clone()),
        slow_timeout: Duration::from_secs(2),
        ..Default::default()
    })
    .await
    .unwrap();
    // With a one-second lease the first renewal starts after 333ms. Like the
    // Python gateway it waits behind model work; a 333ms renewal cap would
    // falsely invalidate this successful 900ms initialization.
    client
        .init_model(ModelConfig::default(), OptimizerConfig::default(), "cpu")
        .await
        .unwrap();
    client.get_embeddings(None).await.unwrap();
    server.wait_for(|state| state.renewals >= 1).await;
    drop(client);
    server.wait_for(|state| state.owner.is_none()).await;
}

#[tokio::test]
async fn stalled_renewal_obeys_the_operation_deadline() {
    let server = Server::start();
    server.service.0.lock().unwrap().stall_renewal = true;
    let client = TrainingClient::connect(hades_core::training::TrainingClientConfig {
        endpoint: hades_core::training::TrainingEndpoint::Unix(server.path.clone()),
        timeout: Duration::from_millis(100),
        slow_timeout: Duration::from_millis(100),
        ..Default::default()
    })
    .await
    .unwrap();
    server
        .wait_for(|state| state.calls.contains(&"renew"))
        .await;
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if client.get_embeddings(None).await.is_err() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(server.service.0.lock().unwrap().generation, 1);
    drop(client);
    server.wait_for(|state| state.owner.is_none()).await;
}
