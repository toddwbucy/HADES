//! Allocation limits applied during serialization, before response buffering.
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use std::io::{self, Write};

const MAX_RESPONSE: usize = 16 * 1024 * 1024;
pub struct BoundedJson<T>(pub T);
impl<T: Serialize> IntoResponse for BoundedJson<T> {
    fn into_response(self) -> Response {
        match encode(&self.0, MAX_RESPONSE) {
            Ok(bytes) => ([(header::CONTENT_TYPE, "application/json")], bytes).into_response(),
            Err(_) => (
                StatusCode::BAD_GATEWAY,
                "viewer response exceeds serialization limits",
            )
                .into_response(),
        }
    }
}
fn encode(value: &impl Serialize, cap: usize) -> serde_json::Result<Vec<u8>> {
    struct Limited {
        bytes: Vec<u8>,
        cap: usize,
    }
    impl Write for Limited {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            if data.len() > self.cap.saturating_sub(self.bytes.len()) {
                return Err(io::Error::other("response byte limit"));
            }
            self.bytes.extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = Limited {
        bytes: Vec::new(),
        cap,
    };
    serde_json::to_writer(&mut output, value)?;
    Ok(output.bytes)
}

/// Account serialized document bytes without allocating a second representation.
pub fn charge(value: &impl Serialize, remaining: &mut usize) -> serde_json::Result<()> {
    struct Counter<'a>(&'a mut usize);
    impl Write for Counter<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > *self.0 {
                return Err(io::Error::other("graph byte budget exceeded"));
            }
            *self.0 -= bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    serde_json::to_writer(Counter(remaining), value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn serialization_and_cumulative_budget_enforce_exact_boundaries() {
        assert_eq!(encode(&"abc", 5).unwrap(), b"\"abc\"");
        assert!(encode(&"abc", 4).is_err());
        let mut remaining = 7;
        charge(&"a", &mut remaining).unwrap();
        charge(&"b", &mut remaining).unwrap();
        assert_eq!(remaining, 1);
        assert!(charge(&"c", &mut remaining).is_err());
    }
}
