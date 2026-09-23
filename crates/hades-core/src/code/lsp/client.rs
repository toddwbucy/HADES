//! JSON-RPC transport client for LSP servers.
//!
//! Manages a language server subprocess, framing messages with
//! `Content-Length` headers over stdin/stdout per the LSP base protocol.
//!
//! Uses an async reader task for response correlation and notification
//! buffering.

use std::collections::{HashMap, VecDeque};
use std::process::Stdio;
use std::sync::{
    Arc, Mutex as SyncMutex,
    atomic::{AtomicBool, AtomicI64, Ordering},
};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{Mutex, Notify, oneshot};
use tokio_util::sync::CancellationToken;

use super::LspError;
use super::process::ServerProcess;

/// A pending request awaiting its response.
type ResponseSender = oneshot::Sender<Result<Value, LspError>>;

const SEND_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_HEADER_BYTES: usize = 8 * 1024;
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const MAX_PENDING: usize = 128;
const MAX_NOTIFICATIONS: usize = 1024;
const MAX_NOTIFICATION_BYTES: usize = 8 * 1024 * 1024;

#[derive(Default)]
struct Notifications {
    // Wire byte accounting is a bound on input size, not exact parsed heap usage.
    entries: VecDeque<(usize, Value)>,
    bytes: usize,
}
impl Notifications {
    fn push(&mut self, bytes: usize, message: Value) -> bool {
        // Notifications are best-effort observations, not pending responses.
        // Keep the latest bounded window without killing unrelated requests.
        if bytes > MAX_NOTIFICATION_BYTES {
            return false;
        }
        while self.entries.len() >= MAX_NOTIFICATIONS
            || bytes > MAX_NOTIFICATION_BYTES.saturating_sub(self.bytes)
        {
            let (evicted_bytes, _) = self.entries.pop_front().unwrap();
            self.bytes -= evicted_bytes;
        }
        self.bytes += bytes;
        self.entries.push_back((bytes, message));
        true
    }
    fn drain(&mut self, method: Option<&str>) -> Vec<Value> {
        let mut matched = Vec::new();
        let mut retained = VecDeque::new();
        for (bytes, message) in self.entries.drain(..) {
            if method.is_none_or(|m| message.get("method").and_then(Value::as_str) == Some(m)) {
                self.bytes -= bytes;
                matched.push(message);
            } else {
                retained.push_back((bytes, message));
            }
        }
        self.entries = retained;
        matched
    }
}

struct FrameBuffer(Vec<u8>);
impl std::io::Write for FrameBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > MAX_FRAME_BYTES.saturating_sub(self.0.len()) {
            return Err(std::io::Error::other("outgoing LSP frame exceeds 16 MiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct Pending {
    closed: bool,
    senders: HashMap<i64, ResponseSender>,
}

#[derive(Default)]
struct Transport {
    pending: SyncMutex<Pending>,
    stop: CancellationToken,
}
impl Transport {
    fn close(&self, reason: &str) {
        if reason == "peer closed" {
            tracing::debug!(reason, "closing LSP transport");
        } else {
            tracing::warn!(reason, "closing LSP transport");
        }
        let mut pending = self.pending.lock().unwrap();
        pending.closed = true;
        for (_, sender) in pending.senders.drain() {
            let _ = sender.send(Err(LspError::Process(reason.into())));
        }
        self.stop.cancel();
    }
}
struct PendingGuard {
    transport: Arc<Transport>,
    id: i64,
}
impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.transport
            .pending
            .lock()
            .unwrap()
            .senders
            .remove(&self.id);
    }
}
struct FailureGuard {
    transport: Arc<Transport>,
    reason: &'static str,
    armed: bool,
}
impl Drop for FailureGuard {
    fn drop(&mut self) {
        if self.armed {
            self.transport.close(self.reason);
        }
    }
}

/// JSON-RPC LSP transport client.
///
/// Spawns a language server as a child process and communicates via
/// Content-Length framed JSON-RPC over stdin/stdout.
pub struct LspClient {
    /// Handle to the child process.
    owner: Option<tokio::task::JoinHandle<Result<(), LspError>>>,
    running: Arc<AtomicBool>,
    /// Stdin writer (wrapped in Mutex for exclusive access).
    stdin: Arc<Mutex<tokio::process::ChildStdin>>,
    /// Next request ID to allocate.
    next_id: AtomicI64,
    /// Pending request map: id → oneshot sender.
    transport: Arc<Transport>,
    /// Buffered server notifications.
    notifications: Arc<Mutex<Notifications>>,
    /// Signalled when any notification arrives.
    notification_signal: Arc<Notify>,
    /// Reader task handle.
    reader_handle: Option<tokio::task::JoinHandle<()>>,
}

impl LspClient {
    /// Spawn the language server and start the reader task.
    pub async fn start(
        command: &str,
        args: &[&str],
        cwd: &std::path::Path,
    ) -> Result<Self, LspError> {
        let child = Command::new(command)
            .args(args)
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    LspError::NotFound(format!("{command}: {e}"))
                } else {
                    LspError::Process(format!("failed to spawn {command}: {e}"))
                }
            })?;

        let mut process = ServerProcess::new(child);
        let stdin = process
            .child
            .stdin
            .take()
            .ok_or_else(|| LspError::Process("no stdin handle".into()))?;
        let stdout = process
            .child
            .stdout
            .take()
            .ok_or_else(|| LspError::Process("no stdout handle".into()))?;

        let transport = Arc::new(Transport::default());
        let notifications = Arc::new(Mutex::new(Notifications::default()));
        let notification_signal = Arc::new(Notify::new());
        let reader_stdin = Arc::new(Mutex::new(stdin));
        let reader_transport = transport.clone();
        let reader_notifs = notifications.clone();
        let reader_signal = notification_signal.clone();
        let reader_stdin_clone = reader_stdin.clone();
        let reader_handle = tokio::spawn(async move {
            let _closed = FailureGuard {
                transport: reader_transport.clone(),
                reason: "LSP reader terminated",
                armed: true,
            };
            tokio::select! {
                biased;
                _ = reader_transport.stop.cancelled() => {},
                _ = reader_loop(stdout, reader_transport.clone(), reader_notifs, reader_signal, reader_stdin_clone) => {},
            }
        });
        let running = Arc::new(AtomicBool::new(true));
        let owner_running = running.clone();
        let owner_transport = transport.clone();
        let owner = tokio::spawn(async move {
            let result = process.finish(owner_transport.stop.clone()).await;
            owner_running.store(false, Ordering::Release);
            if let Err(error) = &result {
                owner_transport.close(&format!("LSP process cleanup failed: {error}"));
            }
            // The group is stopped and direct child reaped. Let the reader drain
            // final buffered responses before EOF closes pending channels.
            result
        });

        Ok(Self {
            owner: Some(owner),
            running,
            stdin: reader_stdin,
            next_id: AtomicI64::new(1),
            transport,
            notifications,
            notification_signal,
            reader_handle: Some(reader_handle),
        })
    }

    /// Send a JSON-RPC request and await the response.
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: std::time::Duration,
    ) -> Result<Value, LspError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.transport.pending.lock().unwrap();
            if pending.closed {
                return Err(LspError::Process("LSP transport is closed".into()));
            }
            if pending.senders.len() >= MAX_PENDING {
                return Err(LspError::Process(
                    "LSP pending request limit (128) reached".into(),
                ));
            }
            pending.senders.insert(id, tx);
        }
        let _pending = PendingGuard {
            transport: self.transport.clone(),
            id,
        };
        let message = serde_json::json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        let operation = async {
            send_frame(&self.stdin, &self.transport, &message).await?;
            rx.await
                .map_err(|_| LspError::Process("LSP response channel closed".into()))?
        };
        tokio::time::timeout(timeout, operation)
            .await
            .map_err(|_| LspError::Timeout(format!("{method} (id={id}, timeout={timeout:?})")))?
    }

    /// Send a JSON-RPC notification (no response expected).
    pub async fn notify(&self, method: &str, params: Value) -> Result<(), LspError> {
        let message = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        bounded_send(&self.stdin, &self.transport, &message).await
    }

    /// Drain all buffered notifications, optionally filtered by method.
    pub async fn drain_notifications(&self, method: Option<&str>) -> Vec<Value> {
        self.notifications.lock().await.drain(method)
    }

    /// Wait for a notification to arrive (with timeout).
    pub async fn wait_for_notification(&self, timeout: std::time::Duration) -> bool {
        tokio::time::timeout(timeout, self.notification_signal.notified())
            .await
            .is_ok()
    }

    /// Graceful shutdown: send shutdown request, then exit notification.
    pub async fn shutdown(mut self) -> Result<(), LspError> {
        let _ = self.request("shutdown", Value::Null, SEND_TIMEOUT).await;
        let _ = self.notify("exit", Value::Null).await;
        if let Some(mut owner) = self.owner.take() {
            match tokio::time::timeout(SEND_TIMEOUT, &mut owner).await {
                Ok(result) => {
                    result.map_err(|e| LspError::Process(format!("LSP process owner: {e}")))??
                }
                Err(_) => {
                    self.transport.close("LSP shutdown grace expired");
                    tokio::time::timeout(SEND_TIMEOUT, owner)
                        .await
                        .map_err(|_| LspError::Timeout("language server reap".into()))?
                        .map_err(|e| LspError::Process(format!("LSP process owner: {e}")))??;
                }
            }
        }
        if let Some(reader) = self.reader_handle.take() {
            reader.abort();
            let _ = reader.await;
        }
        Ok(())
    }

    /// Whether the owned direct child is still running and transport usable.
    pub fn is_alive(&mut self) -> bool {
        self.running.load(Ordering::Acquire) && !self.transport.stop.is_cancelled()
    }
}
impl Drop for LspClient {
    fn drop(&mut self) {
        self.transport.close("LSP client dropped");
    }
}

async fn bounded_send(
    stdin: &Mutex<tokio::process::ChildStdin>,
    transport: &Arc<Transport>,
    message: &Value,
) -> Result<(), LspError> {
    tokio::time::timeout(SEND_TIMEOUT, send_frame(stdin, transport, message))
        .await
        .map_err(|_| LspError::Timeout("LSP message transmission".into()))?
}

async fn send_frame(
    stdin: &Mutex<tokio::process::ChildStdin>,
    transport: &Arc<Transport>,
    message: &Value,
) -> Result<(), LspError> {
    let mut writer = tokio::select! {
        biased;
        _ = transport.stop.cancelled() => return Err(LspError::Process("LSP transport is closed".into())),
        writer = stdin.lock() => writer,
    };
    // Only the admitted writer allocates a serialized frame. Oversize rejection
    // occurs before any bytes are sent and does not poison the transport.
    let mut body = FrameBuffer(Vec::new());
    serde_json::to_writer(&mut body, message)?;
    let header = format!("Content-Length: {}\r\n\r\n", body.0.len());
    // No guard while queued: cancelling before transmission leaves framing intact.
    let mut frame = FailureGuard {
        transport: transport.clone(),
        reason: "LSP frame transmission interrupted",
        armed: true,
    };
    tokio::select! {
        biased;
        _ = transport.stop.cancelled() => return Err(LspError::Process("LSP transport is closed".into())),
        result = async {
            writer.write_all(header.as_bytes()).await?;
            writer.write_all(&body.0).await?;
            writer.flush().await
        } => { result?; }
    }
    frame.armed = false;
    Ok(())
}

/// Background task: reads Content-Length framed messages from stdout
/// and dispatches them to the appropriate handler.
async fn reader_loop(
    stdout: tokio::process::ChildStdout,
    transport: Arc<Transport>,
    notifications: Arc<Mutex<Notifications>>,
    notification_signal: Arc<Notify>,
    stdin: Arc<Mutex<tokio::process::ChildStdin>>,
) {
    let mut reader = BufReader::new(stdout);

    loop {
        // Read headers.
        let content_length = match read_headers(&mut reader).await {
            Ok(Some(len)) => len,
            Ok(None) => {
                transport.close("peer closed");
                break;
            }
            Err(error) => {
                transport.close(&error.to_string());
                break;
            }
        };

        // Read body.
        let mut body = vec![0u8; content_length];
        if reader.read_exact(&mut body).await.is_err() {
            break; // EOF or error
        }

        // Parse JSON.
        let message: Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(_) => break,
        };

        // Dispatch.
        if message.get("id").is_some()
            && (message.get("result").is_some() || message.get("error").is_some())
        {
            // Response to a request we sent.
            if let Some(id) = message["id"].as_i64() {
                let sender = transport.pending.lock().unwrap().senders.remove(&id);
                if let Some(tx) = sender {
                    let result = if let Some(err) = message.get("error") {
                        Err(LspError::JsonRpc {
                            code: err["code"].as_i64().unwrap_or(-1),
                            message: err["message"].as_str().unwrap_or("unknown").to_string(),
                        })
                    } else {
                        Ok(message["result"].clone())
                    };
                    let _ = tx.send(result);
                }
            }
        } else if message.get("id").is_some() && message.get("method").is_some() {
            // Server-to-client request — auto-accept with null result.
            if let Some(id) = message.get("id") {
                let response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": null,
                });
                if bounded_send(&stdin, &transport, &response).await.is_err() {
                    break;
                }
            }
        } else {
            // Notification from server.
            if notifications.lock().await.push(content_length, message) {
                notification_signal.notify_waiters();
            }
        }
    }
}

/// Read LSP headers from the stream, return Content-Length.
async fn read_headers<R: AsyncBufReadExt + Unpin>(
    reader: &mut R,
) -> Result<Option<usize>, LspError> {
    let mut content_length = None;
    let mut used = 0;
    let mut line = Vec::new();
    loop {
        let remaining = MAX_HEADER_BYTES - used;
        if remaining == 0 {
            return Err(LspError::Process("LSP headers exceed 8 KiB".into()));
        }
        line.clear();
        let n = (&mut *reader)
            .take(remaining as u64)
            .read_until(b'\n', &mut line)
            .await?;
        if n == 0 && used == 0 {
            return Ok(None);
        }
        used += n;
        if n == 0 || !line.ends_with(b"\n") {
            return Err(LspError::Process(
                "incomplete or oversized LSP header".into(),
            ));
        }
        let text = std::str::from_utf8(&line)
            .map_err(|_| LspError::Process("invalid LSP header encoding".into()))?
            .trim();
        if text.is_empty() {
            return content_length
                .map(Some)
                .ok_or_else(|| LspError::Process("missing LSP Content-Length".into()));
        }
        let (name, value) = text
            .split_once(':')
            .ok_or_else(|| LspError::Process("malformed LSP header".into()))?;
        if name.eq_ignore_ascii_case("Content-Length") {
            if content_length.is_some() {
                return Err(LspError::Process("duplicate LSP Content-Length".into()));
            }
            let value = value.trim();
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err(LspError::Process("invalid LSP Content-Length".into()));
            }
            let length = value
                .parse::<usize>()
                .map_err(|_| LspError::Process("invalid LSP Content-Length".into()))?;
            if length > MAX_FRAME_BYTES {
                return Err(LspError::Process("LSP frame exceeds 16 MiB".into()));
            }
            content_length = Some(length);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn peer_response_then_eof_is_debug_but_truncated_header_warns() {
        use tracing::instrument::WithSubscriber;
        #[derive(Clone)]
        struct Capture(Arc<SyncMutex<String>>);
        impl tracing::Subscriber for Capture {
            fn register_callsite(
                &self,
                _: &'static tracing::Metadata<'static>,
            ) -> tracing::subscriber::Interest {
                tracing::subscriber::Interest::always()
            }
            fn max_level_hint(&self) -> Option<tracing::metadata::LevelFilter> {
                Some(tracing::metadata::LevelFilter::TRACE)
            }
            fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
                tracing::span::Id::from_u64(1)
            }
            fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
            fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
            fn enter(&self, _: &tracing::span::Id) {}
            fn exit(&self, _: &tracing::span::Id) {}
            fn event(&self, event: &tracing::Event<'_>) {
                struct Visitor(String);
                impl tracing::field::Visit for Visitor {
                    fn record_debug(
                        &mut self,
                        _: &tracing::field::Field,
                        value: &dyn std::fmt::Debug,
                    ) {
                        self.0.push_str(&format!("{value:?}"));
                    }
                }
                let mut visitor = Visitor(event.metadata().level().to_string());
                event.record(&mut visitor);
                self.0.lock().unwrap().push_str(&visitor.0);
            }
        }
        for truncated in [false, true] {
            let capture = Capture(Arc::new(SyncMutex::new(String::new())));
            let subscriber = capture.clone();
            let mut child = tokio::process::Command::new("/usr/bin/python3").args(["-c",
                if truncated { "import sys; sys.stdout.write('Content-Length:')" }
                else { "import sys,json; b=json.dumps({'jsonrpc':'2.0','id':1,'result':42}); sys.stdout.write('Content-Length: '+str(len(b))+'\\r\\n\\r\\n'+b)" }
            ]).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn().unwrap();
            let transport = Arc::new(Transport::default());
            let (tx, rx) = tokio::sync::oneshot::channel();
            transport.pending.lock().unwrap().senders.insert(1, tx);
            reader_loop(
                child.stdout.take().unwrap(),
                transport,
                Arc::new(Mutex::new(Notifications::default())),
                Arc::new(Notify::new()),
                Arc::new(Mutex::new(child.stdin.take().unwrap())),
            )
            .with_subscriber(subscriber)
            .await;
            child.wait().await.unwrap();
            let logs = capture.0.lock().unwrap().clone();
            if truncated {
                assert!(rx.await.unwrap().is_err());
                assert!(
                    logs.contains("WARN") && logs.contains("incomplete"),
                    "{logs}"
                );
            } else {
                assert_eq!(rx.await.unwrap().unwrap(), 42);
                assert!(
                    logs.contains("peer closed") && !logs.contains("WARN"),
                    "{logs}"
                );
            }
        }
    }

    #[test]
    fn test_read_headers_parsing() {
        // Verify the header parser in isolation using a mock stream.
        let header = b"Content-Length: 42\r\n\r\n";
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(async {
            let mut reader = BufReader::new(&header[..]);
            read_headers(&mut reader).await
        });
        assert_eq!(result.unwrap(), Some(42));
    }

    #[test]
    fn test_read_headers_eof() {
        let header = b"";
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(async {
            let mut reader = BufReader::new(&header[..]);
            read_headers(&mut reader).await
        });
        assert_eq!(result.unwrap(), None);
    }

    #[tokio::test]
    async fn queued_timeout_and_cancellation_remove_pending_without_closing_transport() {
        let directory = tempfile::tempdir().unwrap();
        let mut client = LspClient::start(
            "/usr/bin/python3",
            &["-c", "import time; time.sleep(30)"],
            directory.path(),
        )
        .await
        .unwrap();
        let writer = client.stdin.lock().await;
        let timeout = client
            .request("queued", Value::Null, Duration::from_millis(20))
            .await;
        assert!(matches!(timeout, Err(LspError::Timeout(_))));
        assert!(client.transport.pending.lock().unwrap().senders.is_empty());
        assert!(!client.transport.stop.is_cancelled());
        assert!(
            tokio::time::timeout(
                Duration::from_millis(20),
                client.request("cancelled", Value::Null, Duration::from_secs(30))
            )
            .await
            .is_err()
        );
        assert!(client.transport.pending.lock().unwrap().senders.is_empty());
        assert!(!client.transport.stop.is_cancelled());
        drop(writer);
        assert!(client.is_alive());
        // A cancelled queued request did not poison the writer or leave a frame.
        client.notify("small", Value::Null).await.unwrap();
        client.transport.close("fixture complete");
        tokio::time::timeout(Duration::from_secs(2), client.shutdown())
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn partial_write_cancellation_closes_transport_and_drains_pending() {
        let directory = tempfile::tempdir().unwrap();
        let client = LspClient::start(
            "/usr/bin/python3",
            &["-c", "import time; time.sleep(30)"],
            directory.path(),
        )
        .await
        .unwrap();
        // Poll the first request until its small frame is sent and it awaits a response.
        let mut waiting = Box::pin(client.request("waiting", Value::Null, Duration::from_secs(30)));
        tokio::select! {
            biased;
            result = &mut waiting => panic!("peer unexpectedly answered: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(20)) => {},
        }
        assert_eq!(client.transport.pending.lock().unwrap().senders.len(), 1);
        assert!(
            tokio::time::timeout(
                Duration::from_millis(30),
                client.request(
                    "blocked",
                    Value::String("x".repeat(1024 * 1024)),
                    Duration::from_secs(30)
                )
            )
            .await
            .is_err()
        );
        assert!(client.transport.stop.is_cancelled());
        assert!(client.transport.pending.lock().unwrap().senders.is_empty());
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), &mut waiting)
                .await
                .unwrap(),
            Err(LspError::Process(_))
        ));
        drop(waiting);
        assert!(client.notify("must not reuse", Value::Null).await.is_err());
        tokio::time::timeout(Duration::from_secs(2), client.shutdown())
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn header_limits_apply_before_body_allocation() {
        let cases: &[&[u8]] = &[
            b"Content-Length: 16777217\r\n\r\n",
            b"Content-Length: 184467440737095516160\r\n\r\n",
            b"Content-Length: -1\r\n\r\n",
            b"Content-Length: +1\r\n\r\n",
            b"Content-Length: 1\r\nContent-Length: 1\r\n\r\n",
            b"X: missing length\r\n\r\n",
            b"Content-Length: 1",
        ];
        for bytes in cases {
            assert!(read_headers(&mut BufReader::new(*bytes)).await.is_err());
        }
        let prefix = b"content-length: 16777216\r\nX: ";
        let mut exact = prefix.to_vec();
        exact.extend(vec![b'x'; MAX_HEADER_BYTES - prefix.len() - 4]);
        exact.extend(b"\r\n\r\n");
        assert_eq!(
            read_headers(&mut BufReader::new(&exact[..])).await.unwrap(),
            Some(MAX_FRAME_BYTES)
        );
        exact.insert(prefix.len(), b'x');
        assert!(read_headers(&mut BufReader::new(&exact[..])).await.is_err());
    }

    #[tokio::test]
    async fn notification_flood_does_not_fail_pending_requests() {
        let directory = tempfile::tempdir().unwrap();
        let server = r#"
import sys, json

def send(value):
    body = json.dumps(value).encode()
    sys.stdout.buffer.write(b'Content-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
    sys.stdout.buffer.flush()

while True:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            sys.exit(0)
        if line == b'\r\n':
            break
        name, value = line.decode().split(':', 1)
        headers[name.lower()] = value.strip()
    request = json.loads(sys.stdin.buffer.read(int(headers['content-length'])))
    method = request.get('method')
    if method == 'exit':
        break
    if method == 'flood':
        for i in range(1100):
            send({'jsonrpc': '2.0', 'method': '$/progress', 'params': {'sequence': i}})
    if method == 'oversized-notification':
        send({'jsonrpc': '2.0', 'method': 'textDocument/publishDiagnostics', 'params': {'text': 'x' * (8 * 1024 * 1024)}})
    if 'id' in request:
        send({'jsonrpc': '2.0', 'id': request['id'], 'result': 'answered'})
"#;
        let client = LspClient::start("/usr/bin/python3", &["-c", server], directory.path())
            .await
            .unwrap();
        for method in ["flood", "oversized-notification"] {
            assert_eq!(
                client
                    .request(method, Value::Null, Duration::from_secs(10))
                    .await
                    .unwrap(),
                "answered"
            );
            let queue = client.notifications.lock().await;
            assert!(queue.entries.len() <= MAX_NOTIFICATIONS);
            assert!(queue.bytes <= MAX_NOTIFICATION_BYTES);
            assert!(!client.transport.stop.is_cancelled());
        }
        let retained = client.drain_notifications(Some("$/progress")).await;
        assert_eq!(retained.last().unwrap()["params"]["sequence"], 1099);
        client.shutdown().await.unwrap();
    }

    #[test]
    fn notification_drains_release_both_count_and_wire_budget() {
        let mut queue = Notifications::default();
        let message = |method| serde_json::json!({"method": method});
        assert!(queue.push(MAX_NOTIFICATION_BYTES - 1, message("keep")));
        assert!(queue.push(1, message("release")));
        assert!(!queue.push(MAX_NOTIFICATION_BYTES + 1, message("oversized")));
        assert_eq!(queue.drain(Some("release")).len(), 1);
        assert_eq!(queue.bytes, MAX_NOTIFICATION_BYTES - 1);
        assert!(queue.push(1, message("refill")));
        assert_eq!(queue.drain(None).len(), 2);
        assert_eq!(queue.bytes, 0);
        for _ in 0..MAX_NOTIFICATIONS {
            assert!(queue.push(1, message("count")));
        }
        assert!(queue.push(1, message("overflow")));
        assert_eq!(queue.entries.len(), MAX_NOTIFICATIONS);
        assert_eq!(queue.drain(Some("count")).len(), MAX_NOTIFICATIONS - 1);
        assert_eq!(queue.bytes, 1);
        assert_eq!(queue.drain(None).len(), 1);
        assert!(queue.push(MAX_NOTIFICATION_BYTES, message("large")));
        assert!(queue.push(1, message("latest")));
        assert_eq!(queue.bytes, 1);
        assert!(queue.drain(Some("large")).is_empty());
        assert_eq!(queue.drain(Some("latest")).len(), 1);
        assert_eq!(queue.bytes, 0);
    }

    #[tokio::test]
    async fn pending_admission_is_bounded_and_cancelled_slots_are_reusable() {
        let directory = tempfile::tempdir().unwrap();
        let client = LspClient::start(
            "/usr/bin/python3",
            &["-c", "import time; time.sleep(30)"],
            directory.path(),
        )
        .await
        .unwrap();
        let writer = client.stdin.lock().await;
        let mut requests = Vec::new();
        for _ in 0..MAX_PENDING {
            let mut request =
                Box::pin(client.request("queued", Value::Null, Duration::from_secs(30)));
            assert!(futures::poll!(request.as_mut()).is_pending());
            requests.push(request);
        }
        assert_eq!(
            client.transport.pending.lock().unwrap().senders.len(),
            MAX_PENDING
        );
        let rejected = client
            .request("rejected", Value::Null, Duration::from_secs(30))
            .await;
        assert!(
            rejected
                .unwrap_err()
                .to_string()
                .contains("pending request limit")
        );
        requests.pop();
        let mut replacement =
            Box::pin(client.request("replacement", Value::Null, Duration::from_secs(30)));
        assert!(futures::poll!(replacement.as_mut()).is_pending());
        assert_eq!(
            client.transport.pending.lock().unwrap().senders.len(),
            MAX_PENDING
        );
        drop(replacement);
        drop(requests);
        drop(writer);
        assert!(client.transport.pending.lock().unwrap().senders.is_empty());
        assert!(!client.transport.stop.is_cancelled());
        client.transport.close("fixture complete");
        tokio::time::timeout(Duration::from_secs(2), client.shutdown())
            .await
            .unwrap()
            .unwrap();
    }
}
