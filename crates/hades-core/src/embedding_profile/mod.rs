//! Portable embedding identity contracts for #225.
//!
//! These types do not register providers, verify a running model, or activate
//! database pairing. Callers must establish trusted artifact provenance and a
//! verified immutable serving snapshot before using a receipt. Production
//! entry points remain unchanged until enforcement is implemented together.

mod contract;
mod manifest;
#[cfg(test)]
mod tests;

pub use contract::*;
pub use manifest::*;

use thiserror::Error;

/// A malformed identity or an incompatible profile/provider binding.
#[derive(Debug, Error)]
#[error("embedding profile: {0}")]
pub struct ContractError(pub String);

pub(crate) fn require(condition: bool, message: &str) -> Result<(), ContractError> {
    if condition {
        Ok(())
    } else {
        Err(ContractError(message.into()))
    }
}

pub(crate) fn label(value: &str) -> Result<(), ContractError> {
    require(
        !value.trim().is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control),
        "identity labels must be nonempty, at most 4096 UTF-8 bytes, and contain no controls",
    )
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
