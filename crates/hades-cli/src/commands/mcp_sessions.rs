//! Bound MCP sessions and retain request admission through SDK cache cleanup.
use futures::{Stream, StreamExt};
use rmcp::{
    RoleServer,
    model::{ClientJsonRpcMessage, ClientNotification, RequestId, ServerJsonRpcMessage},
    transport::Transport,
    transport::streamable_http_server::session::{
        ServerSseMessage, SessionId, SessionManager,
        local::{
            LocalSessionManager, LocalSessionManagerError, LocalSessionWorkerError,
            SessionTransport, create_local_session,
        },
    },
};
use std::{
    collections::HashMap,
    pin::Pin,
    sync::{Arc, LazyLock},
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch};

const MAX_RETAINED_REQUESTS: usize = 64;
const MAX_SESSIONS: usize = 32;
static SESSIONS: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(MAX_SESSIONS)));
static REQUESTS: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(MAX_RETAINED_REQUESTS)));
const MAX_SSE_STREAMS: usize = 64;
static SSE_STREAMS: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(MAX_SSE_STREAMS)));
const MAX_MESSAGE_BYTES: usize = 2 * 1024 * 1024;
const FORWARD_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, thiserror::Error)]
pub(super) enum SessionError {
    #[error(
        "MCP session, request, or stream capacity is full; close unused streams and retry with backoff"
    )]
    Overloaded,
    #[error(transparent)]
    Inner(#[from] LocalSessionManagerError),
    #[error("MCP request owner stopped")]
    OwnerStopped,
}

pub(super) struct BoundedSessionManager {
    inner: Arc<LocalSessionManager>,
    sessions_budget: Arc<Semaphore>,
    requests_budget: Arc<Semaphore>,
    streams_budget: Arc<Semaphore>,
    retained: Arc<Mutex<HashMap<(SessionId, u64), RetainedRequest>>>,
    closed: Arc<std::sync::Mutex<HashMap<SessionId, watch::Receiver<bool>>>>,
}

impl Default for BoundedSessionManager {
    fn default() -> Self {
        Self {
            inner: Arc::default(),
            retained: Arc::default(),
            closed: Arc::default(),
            sessions_budget: SESSIONS.clone(),
            requests_budget: REQUESTS.clone(),
            streams_budget: SSE_STREAMS.clone(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Completion {
    Pending,
    Response,
    Cancelled,
}

#[derive(Clone)]
struct RetainedRequest {
    permit: Arc<OwnedSemaphorePermit>,
    rpc_id: Option<RequestId>,
    completed: watch::Sender<Completion>,
}

/// The output queue also retains the reservation if the HTTP reader stalls.
struct ReservedStream {
    receiver: mpsc::Receiver<ServerSseMessage>,
    _permit: Arc<OwnedSemaphorePermit>,
}
impl Stream for ReservedStream {
    type Item = ServerSseMessage;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

struct AdmittedStream {
    inner: Pin<Box<dyn Stream<Item = ServerSseMessage> + Send + Sync>>,
    _permit: OwnedSemaphorePermit,
}
impl Stream for AdmittedStream {
    type Item = ServerSseMessage;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum TransportError {
    #[error(transparent)]
    Worker(Box<LocalSessionWorkerError>),
    #[error("MCP message exceeds the 2 MiB serialized limit or cannot be serialized: {0}")]
    Message(#[from] serde_json::Error),
}

impl From<LocalSessionWorkerError> for TransportError {
    fn from(error: LocalSessionWorkerError) -> Self {
        Self::Worker(Box::new(error))
    }
}

/// Count serialized bytes without allocating another copy of a large message.
struct ByteBudget(usize);
impl std::io::Write for ByteBudget {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_sub(bytes.len())
            .ok_or_else(|| std::io::Error::other("message byte limit exceeded"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn bound_message(message: ServerJsonRpcMessage) -> Result<ServerJsonRpcMessage, serde_json::Error> {
    let error = match serde_json::to_writer(ByteBudget(MAX_MESSAGE_BYTES), &message) {
        Ok(()) => return Ok(message),
        Err(error) => error,
    };
    let id = match message {
        ServerJsonRpcMessage::Response(response) => Some(response.id),
        ServerJsonRpcMessage::Error(error) => error.id,
        _ => return Err(error),
    };
    let replacement = ServerJsonRpcMessage::error(
        rmcp::ErrorData::internal_error(
            "MCP_RESPONSE_TOO_LARGE: serialized response exceeds 2 MiB",
            None,
        ),
        id,
    );
    serde_json::to_writer(ByteBudget(MAX_MESSAGE_BYTES), &replacement)?;
    Ok(replacement)
}

pub(super) struct ObservedTransport {
    inner: SessionTransport,
    closed: watch::Sender<bool>,
}
impl Drop for ObservedTransport {
    fn drop(&mut self) {
        self.closed.send_replace(true);
    }
}
impl Transport<RoleServer> for ObservedTransport {
    type Error = TransportError;
    fn send(
        &mut self,
        message: ServerJsonRpcMessage,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
        let send = match bound_message(message) {
            Ok(message) => Ok(self.inner.send(message)),
            Err(error) => {
                // A non-response (or oversized request ID) cannot be replaced
                // safely. Terminate rather than leave an unanswered cache owner.
                self.inner.cancel_token().cancel();
                self.closed.send_replace(true);
                Err(error)
            }
        };
        async move {
            send?.await?;
            Ok(())
        }
    }
    async fn receive(&mut self) -> Option<ClientJsonRpcMessage> {
        let message = self.inner.receive().await;
        if message.is_none() {
            self.closed.send_replace(true);
        }
        message
    }
    async fn close(&mut self) -> Result<(), Self::Error> {
        let result = self.inner.close().await;
        self.closed.send_replace(true);
        Ok(result?)
    }
}

impl SessionManager for BoundedSessionManager {
    type Error = SessionError;
    type Transport = ObservedTransport;

    async fn create_session(&self) -> Result<(SessionId, Self::Transport), Self::Error> {
        let permit = self
            .sessions_budget
            .clone()
            .try_acquire_owned()
            .map_err(|_| SessionError::Overloaded)?;
        // Acquire the only async lock before creating the worker. The remaining
        // registration is synchronous, so caller cancellation cannot orphan it.
        let mut sessions = self.inner.sessions.write().await;
        let id = rmcp::transport::common::server_side_http::session_id();
        let (handle, worker) = create_local_session(id.clone(), self.inner.session_config.clone());
        let inner = SessionTransport::spawn(worker);
        let (closed, mut rx) = watch::channel(false);
        self.closed
            .lock()
            .expect("session state lock")
            .insert(id.clone(), rx.clone());
        sessions.insert(id.clone(), handle.clone());
        let manager = self.inner.clone();
        let states = self.closed.clone();
        let reaped_id = id.clone();
        tokio::spawn(async move {
            let _ = rx.wait_for(|closed| *closed).await;
            // Close is only an enqueue. A subsequent channel request is ordered
            // after it and fails once the worker has exited and dropped caches.
            let _ = handle.close().await;
            let _ = handle.establish_request_wise_channel().await;
            manager.sessions.write().await.remove(&reaped_id);
            states
                .lock()
                .expect("session state lock")
                .remove(&reaped_id);
            drop(permit);
        });
        Ok((id, ObservedTransport { inner, closed }))
    }

    async fn initialize_session(
        &self,
        id: &SessionId,
        message: ClientJsonRpcMessage,
    ) -> Result<ServerJsonRpcMessage, Self::Error> {
        Ok(self.inner.initialize_session(id, message).await?)
    }
    async fn has_session(&self, id: &SessionId) -> Result<bool, Self::Error> {
        Ok(self.inner.has_session(id).await?)
    }
    async fn close_session(&self, id: &SessionId) -> Result<(), Self::Error> {
        let inner = self.inner.clone();
        let id = id.clone();
        tokio::spawn(async move { inner.close_session(&id).await })
            .await
            .map_err(|_| SessionError::OwnerStopped)??;
        Ok(())
    }
    async fn accept_message(
        &self,
        id: &SessionId,
        message: ClientJsonRpcMessage,
    ) -> Result<(), Self::Error> {
        let cancelled = match &message {
            ClientJsonRpcMessage::Notification(notification) => match &notification.notification {
                ClientNotification::CancelledNotification(cancelled) => {
                    cancelled.params.request_id.clone()
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(rpc_id) = cancelled {
            let inner = self.inner.clone();
            let retained = self.retained.clone();
            let id = id.clone();
            // Retain ownership if the notifying HTTP caller disappears.
            tokio::spawn(async move {
                inner.accept_message(&id, message).await?;
                for ((session, _), request) in retained.lock().await.iter() {
                    if session == &id && request.rpc_id.as_ref() == Some(&rpc_id) {
                        request.completed.send_if_modified(|state| {
                            if *state == Completion::Pending {
                                *state = Completion::Cancelled;
                                true
                            } else {
                                false
                            }
                        });
                    }
                }
                Ok::<_, LocalSessionManagerError>(())
            })
            .await
            .map_err(|_| SessionError::OwnerStopped)??;
            Ok(())
        } else {
            Ok(self.inner.accept_message(id, message).await?)
        }
    }
    async fn create_standalone_stream(
        &self,
        id: &SessionId,
    ) -> Result<impl Stream<Item = ServerSseMessage> + Send + Sync + 'static, Self::Error> {
        let permit = self
            .streams_budget
            .clone()
            .try_acquire_owned()
            .map_err(|_| SessionError::Overloaded)?;
        let inner = Box::pin(self.inner.create_standalone_stream(id).await?);
        Ok(AdmittedStream {
            inner,
            _permit: permit,
        })
    }
    async fn resume(
        &self,
        id: &SessionId,
        last_event_id: String,
    ) -> Result<impl Stream<Item = ServerSseMessage> + Send + Sync + 'static, Self::Error> {
        let stream_permit = self
            .streams_budget
            .clone()
            .try_acquire_owned()
            .map_err(|_| SessionError::Overloaded)?;
        let request_id = last_event_id
            .split_once('/')
            .and_then(|(_, id)| id.parse::<u64>().ok());
        let retained = if let Some(request_id) = request_id {
            self.retained
                .lock()
                .await
                .get(&(id.clone(), request_id))
                .cloned()
        } else {
            None
        };
        let input = self.inner.resume(id, last_event_id).await?;
        let stream: Pin<Box<dyn Stream<Item = ServerSseMessage> + Send + Sync>> =
            if let Some(retained) = retained {
                let (tx, rx) = mpsc::channel(1);
                let stream = ReservedStream {
                    receiver: rx,
                    _permit: retained.permit.clone(),
                };
                tokio::spawn(async move {
                    let _permit = retained.permit;
                    forward(input, tx, &retained.completed).await;
                });
                Box::pin(stream)
            } else {
                Box::pin(input)
            };
        Ok(AdmittedStream {
            inner: stream,
            _permit: stream_permit,
        })
    }
    async fn create_stream(
        &self,
        id: &SessionId,
        message: ClientJsonRpcMessage,
    ) -> Result<impl Stream<Item = ServerSseMessage> + Send + Sync + 'static, Self::Error> {
        let permit = Arc::new(
            self.requests_budget
                .clone()
                .try_acquire_owned()
                .map_err(|_| SessionError::Overloaded)?,
        );
        let handle = self
            .inner
            .sessions
            .read()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| LocalSessionManagerError::SessionNotFound(id.clone()))?;
        let closed = self
            .closed
            .lock()
            .expect("session state lock")
            .get(id)
            .cloned()
            .ok_or(SessionError::OwnerStopped)?;
        let rpc_id = match &message {
            ClientJsonRpcMessage::Request(request) => Some(request.id.clone()),
            _ => None,
        };
        let retry = self.inner.session_config.sse_retry;
        let replay_ttl = self.inner.session_config.completed_cache_ttl;
        let retained = self.retained.clone();
        let session_id = id.clone();
        let (ready_tx, ready_rx) = oneshot::channel();
        // Independent ownership prevents HTTP cancellation during channel creation
        // from orphaning an unaccounted SDK cache.
        tokio::spawn(async move {
            let receiver = match handle.establish_request_wise_channel().await {
                Ok(receiver) => receiver,
                Err(error) => {
                    let _ = ready_tx.send(Err(SessionError::Inner(error.into())));
                    return;
                }
            };
            let request_id = receiver.http_request_id.expect("request-wise channel ID");
            let (completed, completion_rx) = watch::channel(Completion::Pending);
            let key = (session_id, request_id);
            retained.lock().await.insert(
                key.clone(),
                RetainedRequest {
                    permit: permit.clone(),
                    rpc_id,
                    completed: completed.clone(),
                },
            );
            let (tx, rx) = mpsc::channel(1);
            let stream = ReservedStream {
                receiver: rx,
                _permit: permit.clone(),
            };
            if let Err(error) = handle.push_message(message, Some(request_id)).await {
                let _ = ready_tx.send(Err(SessionError::Inner(error.into())));
                retained.lock().await.remove(&key);
                // A failed push means the worker has terminated; its cache is gone.
                return;
            }
            let _ = ready_tx.send(Ok(stream));
            let priming =
                retry.map(|retry| ServerSseMessage::priming(format!("0/{request_id}"), retry));
            let input = futures::stream::iter(priming)
                .chain(futures::stream::unfold(receiver.inner, |mut rx| async {
                    rx.recv().await.map(|message| (message, rx))
                }));
            forward(input, tx, &completed).await;
            // EOF can mean a resume replaced this receiver, not completion.
            // A resumed forwarder signals the same terminal-response watch.
            retain_until_cleanup(permit, completion_rx, closed, replay_ttl, async move {
                // An acknowledgement proves the SDK cache was removed. If the
                // worker exited, dropping it already released all its caches.
                let result = handle.close_request_wise_channel(request_id).await;
                retained.lock().await.remove(&key);
                result
            })
            .await;
        });
        ready_rx.await.map_err(|_| SessionError::OwnerStopped)?
    }
}

async fn forward<S>(
    input: S,
    output: mpsc::Sender<ServerSseMessage>,
    completed: &watch::Sender<Completion>,
) where
    S: Stream<Item = ServerSseMessage>,
{
    let mut output = Some(output);
    tokio::pin!(input);
    while let Some(message) = input.next().await {
        if matches!(
            message.message.as_deref(),
            Some(ServerJsonRpcMessage::Response(_) | ServerJsonRpcMessage::Error(_))
        ) {
            completed.send_replace(Completion::Response);
        }
        if let Some(tx) = &output
            && !matches!(
                tokio::time::timeout(FORWARD_TIMEOUT, tx.send(message)).await,
                Ok(Ok(()))
            )
        {
            output = None;
        }
    }
}

async fn retain_until_cleanup<F, E>(
    permit: Arc<OwnedSemaphorePermit>,
    mut completed: watch::Receiver<Completion>,
    mut closed: watch::Receiver<bool>,
    replay_ttl: Duration,
    cleanup: F,
) where
    F: Future<Output = Result<(), E>>,
{
    // A request remains accounted if no terminal response has been observed.
    // Session termination must signal completion before its owner can retire.
    tokio::select! {
        _ = completed.wait_for(|completed| *completed != Completion::Pending) => {},
        _ = closed.wait_for(|closed| *closed) => {},
    }
    if *completed.borrow() == Completion::Cancelled && !*closed.borrow() {
        tokio::time::sleep(replay_ttl).await;
    }
    let _ = cleanup.await;
    drop(permit);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone)]
    struct TestServer(Arc<tokio::sync::Notify>);
    impl rmcp::ServerHandler for TestServer {
        async fn ping(
            &self,
            _context: rmcp::service::RequestContext<RoleServer>,
        ) -> Result<(), rmcp::ErrorData> {
            self.0.notified().await;
            Ok(())
        }
    }

    async fn initialized_session(
        manager: &BoundedSessionManager,
    ) -> (
        SessionId,
        rmcp::service::RunningService<RoleServer, TestServer>,
        Arc<tokio::sync::Notify>,
    ) {
        use rmcp::ServiceExt;
        let (id, transport) = manager.create_session().await.unwrap();
        let release = Arc::new(tokio::sync::Notify::new());
        let server = TestServer(release.clone());
        let service = tokio::spawn(async move { server.serve(transport).await.unwrap() });
        let init = serde_json::from_value(serde_json::json!({
            "jsonrpc":"2.0", "id":0, "method":"initialize", "params": {
                "protocolVersion":"2025-03-26", "capabilities":{},
                "clientInfo":{"name":"isolated-test", "version":"1"}
            }
        }))
        .unwrap();
        manager.initialize_session(&id, init).await.unwrap();
        manager
            .accept_message(
                &id,
                serde_json::from_value(serde_json::json!({
                "jsonrpc":"2.0", "method":"notifications/initialized"
                }))
                .unwrap(),
            )
            .await
            .unwrap();
        let service = service.await.unwrap();
        (id, service, release)
    }

    #[tokio::test]
    async fn idle_initialized_session_expires_and_reaps() {
        tokio::time::timeout(Duration::from_secs(3), async {
            let mut manager = BoundedSessionManager {
                sessions_budget: Arc::new(Semaphore::new(1)),
                ..Default::default()
            };
            Arc::get_mut(&mut manager.inner)
                .unwrap()
                .session_config
                .keep_alive = Some(Duration::from_millis(30));
            let (id, service, _) = initialized_session(&manager).await;
            let _ = service.waiting().await;
            loop {
                if manager.sessions_budget.available_permits() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert!(!manager.has_session(&id).await.unwrap());
            assert!(manager.closed.lock().unwrap().is_empty());
        })
        .await
        .expect("idle initialized session is reaped");
    }

    #[test]
    fn oversized_response_becomes_small_terminal_error_with_original_id() {
        let message: ServerJsonRpcMessage = serde_json::from_value(serde_json::json!({
            "jsonrpc":"2.0", "id":17, "result":{"content":[{"type":"text", "text":"x".repeat(MAX_MESSAGE_BYTES)}]}
        })).unwrap();
        let bounded = bound_message(message).unwrap();
        let value = serde_json::to_value(bounded).unwrap();
        assert_eq!(value["id"], 17);
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("MCP_RESPONSE_TOO_LARGE")
        );
        assert!(serde_json::to_vec(&value).unwrap().len() < 256);
    }

    #[test]
    fn serialized_message_limit_includes_json_escaping() {
        let value = serde_json::json!({"text":"quote: \" and newline: \n"});
        let size = serde_json::to_vec(&value).unwrap().len();
        assert!(serde_json::to_writer(ByteBudget(size), &value).is_ok());
        assert!(serde_json::to_writer(ByteBudget(size - 1), &value).is_err());
    }

    #[tokio::test]
    async fn cancelled_creation_retains_request_until_independent_cleanup() {
        tokio::time::timeout(Duration::from_secs(3), async {
            let mut manager = BoundedSessionManager {
                requests_budget: Arc::new(Semaphore::new(1)),
                ..Default::default()
            };
            Arc::get_mut(&mut manager.inner)
                .unwrap()
                .session_config
                .completed_cache_ttl = Duration::from_millis(20);
            let (id, service, release) = initialized_session(&manager).await;
            let ping = || {
                serde_json::from_value(
                    serde_json::json!({"jsonrpc":"2.0", "id":1, "method":"ping"}),
                )
                .unwrap()
            };
            {
                let creation = manager.create_stream(&id, ping());
                tokio::pin!(creation);
                // Poll through spawning the independent owner, then cancel the
                // HTTP caller before the worker acknowledges channel creation.
                assert!(futures::poll!(creation.as_mut()).is_pending());
            }
            assert!(matches!(
                manager.create_stream(&id, ping()).await,
                Err(SessionError::Overloaded)
            ));
            loop {
                if !manager.retained.lock().await.is_empty() {
                    break;
                }
                tokio::task::yield_now().await;
            }
            release.notify_one();
            loop {
                if manager.requests_budget.available_permits() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert!(manager.retained.lock().await.is_empty());
            manager.close_session(&id).await.unwrap();
            let _ = service.cancel().await;
        })
        .await
        .expect("cancelled creation cleans up independently");
    }

    #[tokio::test]
    async fn cancelled_session_registration_does_not_spawn_worker() {
        let manager = BoundedSessionManager {
            sessions_budget: Arc::new(Semaphore::new(1)),
            ..Default::default()
        };
        let lock = manager.inner.sessions.read().await;
        {
            let creation = manager.create_session();
            tokio::pin!(creation);
            assert!(futures::poll!(creation.as_mut()).is_pending());
            assert_eq!(manager.sessions_budget.available_permits(), 0);
        }
        assert_eq!(manager.sessions_budget.available_permits(), 1);
        assert!(lock.is_empty());
        assert!(manager.closed.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn abandoned_initialization_expires_and_reaps_session() {
        use rmcp::ServiceExt;
        tokio::time::timeout(Duration::from_secs(3), async {
            let mut manager = BoundedSessionManager {
                sessions_budget: Arc::new(Semaphore::new(1)),
                ..Default::default()
            };
            Arc::get_mut(&mut manager.inner)
                .unwrap()
                .session_config
                .init_timeout = Some(Duration::from_millis(20));
            let (id, transport) = manager.create_session().await.unwrap();
            let server = TestServer(Arc::new(tokio::sync::Notify::new()));
            assert!(server.serve(transport).await.is_err());
            loop {
                if manager.sessions_budget.available_permits() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert!(!manager.has_session(&id).await.unwrap());
            assert!(manager.closed.lock().unwrap().is_empty());
        })
        .await
        .expect("abandoned init is reaped");
    }

    #[tokio::test]
    async fn session_capacity_recovers_only_after_worker_cleanup() {
        tokio::time::timeout(Duration::from_secs(3), async {
            let manager = BoundedSessionManager {
                sessions_budget: Arc::new(Semaphore::new(2)),
                ..Default::default()
            };
            let (first_id, first) = manager.create_session().await.unwrap();
            let (_, second) = manager.create_session().await.unwrap();
            assert!(matches!(
                manager.create_session().await,
                Err(SessionError::Overloaded)
            ));
            drop(first);
            loop {
                if manager.sessions_budget.available_permits() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert!(!manager.has_session(&first_id).await.unwrap());
            assert!(!manager.closed.lock().unwrap().contains_key(&first_id));
            let (_, replacement) = manager.create_session().await.unwrap();
            assert!(matches!(
                manager.create_session().await,
                Err(SessionError::Overloaded)
            ));
            drop((second, replacement));
            loop {
                if manager.sessions_budget.available_permits() == 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert!(manager.inner.sessions.read().await.is_empty());
            assert!(manager.closed.lock().unwrap().is_empty());
        })
        .await
        .expect("worker cleanup returns capacity");
    }

    #[tokio::test]
    async fn sdk_active_request_resumes_and_releases_reservation() {
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut manager = BoundedSessionManager {
                requests_budget: Arc::new(Semaphore::new(1)),
                streams_budget: Arc::new(Semaphore::new(1)),
                ..Default::default()
            };
            Arc::get_mut(&mut manager.inner)
                .unwrap()
                .session_config
                .completed_cache_ttl = Duration::from_millis(100);
            let (id, service, release) = initialized_session(&manager).await;
            let ping = serde_json::from_value(
                serde_json::json!({"jsonrpc":"2.0", "id":1, "method":"ping"}),
            )
            .unwrap();
            let mut stream = Box::pin(manager.create_stream(&id, ping).await.unwrap());
            let priming = stream.next().await.unwrap();
            let resume_id = priming.event_id.unwrap();
            let extra = serde_json::from_value(
                serde_json::json!({"jsonrpc":"2.0", "id":99, "method":"ping"}),
            )
            .unwrap();
            assert!(matches!(
                manager.create_stream(&id, extra).await,
                Err(SessionError::Overloaded)
            ));
            let standalone = manager.create_standalone_stream(&id).await.unwrap();
            assert!(matches!(
                manager.create_standalone_stream(&id).await,
                Err(SessionError::Overloaded)
            ));
            assert!(matches!(
                manager.resume(&id, resume_id.clone()).await,
                Err(SessionError::Overloaded)
            ));
            drop(standalone);
            drop(stream);
            assert_eq!(manager.retained.lock().await.len(), 1);
            let mut resumed = Box::pin(manager.resume(&id, resume_id.clone()).await.unwrap());
            release.notify_one();
            let response = resumed.next().await.unwrap();
            assert!(matches!(
                response.message.as_deref(),
                Some(ServerJsonRpcMessage::Response(_))
            ));
            assert!(resumed.next().await.is_none());
            drop(resumed);
            // rmcp 2.2 removes normal terminal-response caches immediately,
            // despite exposing a completed-cache TTL for other close paths.
            assert!(manager.resume(&id, resume_id.clone()).await.is_err());
            loop {
                if manager.retained.lock().await.is_empty() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(manager.resume(&id, resume_id).await.is_err());
            let ping = serde_json::from_value(
                serde_json::json!({"jsonrpc":"2.0", "id":2, "method":"ping"}),
            )
            .unwrap();
            let mut cancelled_stream = Box::pin(manager.create_stream(&id, ping).await.unwrap());
            cancelled_stream.next().await.unwrap(); // priming; handler still blocked
            manager
                .accept_message(
                    &id,
                    serde_json::from_value(serde_json::json!({
                        "jsonrpc":"2.0", "method":"notifications/cancelled",
                        "params":{"requestId":2, "reason":"isolated cancellation test"}
                    }))
                    .unwrap(),
                )
                .await
                .unwrap();
            assert!(cancelled_stream.next().await.is_none());
            drop(cancelled_stream);
            loop {
                if manager.retained.lock().await.is_empty() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(
                manager.has_session(&id).await.unwrap(),
                "cancellation must not close the session"
            );
            manager.close_session(&id).await.unwrap();
            // The deliberately gated handler ignores cancellation; release its
            // gate before joining the server task. Cache retirement above did
            // not depend on this handler returning.
            release.notify_one();
            let _ = service.cancel().await;
            loop {
                if manager.closed.lock().unwrap().is_empty() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(manager.inner.sessions.read().await.is_empty());
        })
        .await
        .expect("isolated SDK lifecycle completes");
    }

    #[tokio::test]
    async fn completion_and_cleanup_ack_both_precede_release() {
        let pool = Arc::new(Semaphore::new(1));
        let permit = Arc::new(pool.clone().try_acquire_owned().unwrap());
        let (done, rx) = watch::channel(Completion::Pending);
        let (_closed, closed_rx) = watch::channel(false);
        let (cleanup_started, started) = oneshot::channel();
        let (release, released) = oneshot::channel();
        let owner = tokio::spawn(retain_until_cleanup(
            permit,
            rx,
            closed_rx,
            Duration::from_secs(60),
            async move {
                cleanup_started.send(()).unwrap();
                released.await.unwrap();
                Ok::<_, ()>(())
            },
        ));
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(pool.available_permits(), 0);
        assert!(!owner.is_finished(), "EOF alone must not release a cache");
        done.send_replace(Completion::Response);
        tokio::time::timeout(Duration::from_secs(1), started)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            pool.available_permits(),
            0,
            "cleanup acknowledgement is still pending"
        );
        release.send(()).unwrap();
        owner.await.unwrap();
        assert_eq!(pool.available_permits(), 1);
    }

    #[tokio::test]
    async fn disconnected_http_reader_does_not_prevent_terminal_observation() {
        let (done, rx) = watch::channel(Completion::Pending);
        let (tx, output) = mpsc::channel(1);
        drop(output);
        let terminal: ServerJsonRpcMessage =
            serde_json::from_value(serde_json::json!({"jsonrpc":"2.0", "id":1, "result":{}}))
                .unwrap();
        let input = futures::stream::iter([
            ServerSseMessage::priming("0/1", Duration::from_secs(3)),
            ServerSseMessage::new("1/1", terminal),
        ]);
        forward(input, tx, &done).await;
        assert_eq!(*rx.borrow(), Completion::Response);
    }
}

#[cfg(test)]
#[path = "mcp_memory_benchmark.rs"]
mod memory_benchmark;
