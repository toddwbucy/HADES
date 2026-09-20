//! Admission for socket transports. The Unix daemon currently uses this pool.
use std::sync::{Arc, LazyLock};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub(super) const MAX_CONNECTIONS: usize = 64;
static CONNECTIONS: LazyLock<Arc<Semaphore>> =
    LazyLock::new(|| Arc::new(Semaphore::new(MAX_CONNECTIONS)));

/// Keep the permit for the connection's entire lifetime, including response I/O.
pub(super) fn try_connection() -> Option<OwnedSemaphorePermit> {
    CONNECTIONS.clone().try_acquire_owned().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

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
