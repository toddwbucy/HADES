//! Private extractor receiving immutable file bytes; responses and races are explicit fixtures.
use hades_core::persephone::extraction::ExtractionClient;
use hades_proto::extraction::{
    CapabilitiesRequest, ExtractRequest, ExtractResponse, ExtractorInfo, SourceType,
    extraction_service_server::{ExtractionService, ExtractionServiceServer},
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

type Gate = (oneshot::Sender<()>, oneshot::Receiver<()>);
#[derive(Default)]
struct State {
    requests: Vec<ExtractRequest>,
    next: Option<Result<ExtractResponse, Status>>,
    gate: Option<Gate>,
}
struct Service(Arc<Mutex<State>>);
#[tonic::async_trait]
impl ExtractionService for Service {
    async fn extract(
        &self,
        request: Request<ExtractRequest>,
    ) -> Result<Response<ExtractResponse>, Status> {
        let request = request.into_inner();
        let (response, gate) = {
            let mut state = self.0.lock().unwrap();
            state.requests.push(request.clone());
            (state.next.take(), state.gate.take())
        };
        if let Some((entered, release)) = gate {
            let _ = entered.send(());
            let _ = tokio::time::timeout(Duration::from_secs(10), release)
                .await
                .unwrap();
        }
        let response = if let Some(response) = response {
            response?
        } else {
            if request.content.is_empty() || std::path::Path::new(&request.file_path).is_absolute()
            {
                return Err(Status::invalid_argument(
                    "fixture requires file bytes and a basename",
                ));
            }
            ExtractResponse {
                full_text: String::from_utf8(request.content).unwrap(),
                source_type: SourceType::Markdown.into(),
                metadata: [("extractor".into(), "fixture".into())].into(),
                ..Default::default()
            }
        };
        Ok(Response::new(response))
    }
    async fn capabilities(
        &self,
        _: Request<CapabilitiesRequest>,
    ) -> Result<Response<ExtractorInfo>, Status> {
        Ok(Response::new(ExtractorInfo::default()))
    }
}

pub struct Extractor {
    pub path: PathBuf,
    state: Arc<Mutex<State>>,
    task: tokio::task::JoinHandle<()>,
    _directory: tempfile::TempDir,
}
impl Extractor {
    /// Serve one synthetic extractor on an independently owned private Unix socket.
    pub fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("extractor.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let incoming = futures::stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(stream, _)| stream), listener))
        });
        let state = Arc::new(Mutex::new(State::default()));
        let service = Service(state.clone());
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(
                    ExtractionServiceServer::new(service)
                        .max_decoding_message_size(16 * 1024 * 1024),
                )
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        Self {
            path,
            state,
            task,
            _directory: directory,
        }
    }
    /// Connect using the same client as the production file path.
    pub async fn client(&self) -> ExtractionClient {
        ExtractionClient::connect_unix_at(&self.path).await.unwrap()
    }
    /// Observe the exact bytes and filename that crossed the process boundary.
    pub fn requests(&self) -> Vec<ExtractRequest> {
        self.state.lock().unwrap().requests.clone()
    }
    /// Inject a response without altering the saved source request.
    pub fn respond(&self, response: Result<ExtractResponse, Status>) {
        self.state.lock().unwrap().next = Some(response);
    }
    /// Hold a response after the immutable snapshot has been received.
    pub fn hold(&self) -> (oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (entered, started) = oneshot::channel();
        let (release, waiting) = oneshot::channel();
        self.state.lock().unwrap().gate = Some((entered, waiting));
        (started, release)
    }
}
impl Drop for Extractor {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert!(
                !self.task.is_finished(),
                "extractor fixture stopped unexpectedly"
            );
        }
        self.task.abort();
    }
}
