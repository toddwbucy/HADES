//! Admission shared by the Unix daemon and MCP socket transports.
use std::sync::{Arc, LazyLock};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub(super) const MAX_CONNECTIONS: usize = 64;
static CONNECTIONS: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(MAX_CONNECTIONS)));

/// Keep the permit for the connection's entire lifetime, including response I/O.
pub(super) fn try_connection() -> Option<OwnedSemaphorePermit> {
    CONNECTIONS.clone().try_acquire_owned().ok()
}

/// The socket can reconnect without discarding its MCP session or replay cache.
const MCP_CONNECTION_LIFETIME: std::time::Duration = std::time::Duration::from_secs(300);

pub(super) struct AdmittedListener(pub tokio::net::TcpListener);

impl axum::serve::Listener for AdmittedListener {
    type Io = AdmittedIo<tokio::net::TcpStream>;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let (stream, addr) = axum::serve::Listener::accept(&mut self.0).await;
            if let Some(permit) = try_connection() {
                return (
                    AdmittedIo::new(stream, permit, MCP_CONNECTION_LIFETIME),
                    addr,
                );
            }
            // Refuse before HTTP parsing, task creation, or request allocation.
            drop(stream);
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.0.local_addr()
    }
}

/// Retains admission through Hyper's response I/O, including after a handler returns.
/// An absolute lifetime also bounds slow headers, bodies, and stalled SSE readers.
pub(super) struct AdmittedIo<T> {
    inner: T,
    _permit: OwnedSemaphorePermit,
    deadline: std::pin::Pin<Box<tokio::time::Sleep>>,
}

impl<T> AdmittedIo<T> {
    fn new(inner: T, permit: OwnedSemaphorePermit, lifetime: std::time::Duration) -> Self {
        Self {
            inner,
            _permit: permit,
            deadline: Box::pin(tokio::time::sleep(lifetime)),
        }
    }

    fn check_deadline(&mut self, cx: &mut std::task::Context<'_>) -> std::io::Result<()> {
        use std::future::Future;
        if self.deadline.as_mut().poll(cx).is_ready() {
            Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "MCP connection lifetime expired",
            ))
        } else {
            Ok(())
        }
    }
}

impl<T: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for AdmittedIo<T> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        self.check_deadline(cx)?;
        std::pin::Pin::new(&mut self.inner).poll_read(cx, buf)
    }
}

impl<T: tokio::io::AsyncWrite + Unpin> tokio::io::AsyncWrite for AdmittedIo<T> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        self.check_deadline(cx)?;
        std::pin::Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        self.check_deadline(cx)?;
        std::pin::Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        self.check_deadline(cx)?;
        std::pin::Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stalled_io_expires_and_holds_admission_until_drop() {
        use std::time::Duration;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for writing in [false, true] {
            let pool = Arc::new(Semaphore::new(1));
            let permit = pool.clone().try_acquire_owned().unwrap();
            let (server, _client) = tokio::io::duplex(1);
            let mut io = AdmittedIo::new(server, permit, Duration::from_millis(20));
            let error = tokio::time::timeout(Duration::from_secs(2), async {
                if writing {
                    io.write_all(b"too much for the buffer").await.unwrap_err()
                } else {
                    io.read_u8().await.unwrap_err()
                }
            })
            .await
            .expect("deadline wakes stalled I/O");
            assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
            assert_eq!(pool.available_permits(), 0);
            drop(io);
            assert_eq!(pool.available_permits(), 1);
        }
    }

    #[test]
    fn configured_connection_cap_refuses_then_readmits() {
        let mut permits: Vec<_> = (0..MAX_CONNECTIONS)
            .map(|_| try_connection().expect("connection slot"))
            .collect();
        assert!(try_connection().is_none());
        drop(permits.pop());
        let replacement = try_connection().expect("released connection slot");
        assert!(try_connection().is_none());
        drop((permits, replacement));
        assert_eq!(CONNECTIONS.available_permits(), MAX_CONNECTIONS);
    }
}
