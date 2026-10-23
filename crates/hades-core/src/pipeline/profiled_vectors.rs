//! Common admission for vectors before either managed corpus can persist them.
use crate::{db::ArangoError, embedding_profile::Normalization};

pub(super) fn validate_vector(
    vector: &[f32],
    dimension: u32,
    normalization: Normalization,
) -> Result<(), ArangoError> {
    let invalid =
        |message: &str| ArangoError::Request(format!("profile-bound embedding: {message}"));
    if vector.len() != dimension as usize || !vector.iter().all(|v| v.is_finite()) {
        return Err(invalid("invalid vector dimension or finite values"));
    }
    let norm: f64 = vector.iter().map(|v| f64::from(*v).powi(2)).sum();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid("vector has zero or invalid norm"));
    }
    if normalization == Normalization::L2 && (norm.sqrt() - 1.0).abs() > 0.001 {
        return Err(invalid("vector violates the profile's L2 normalization"));
    }
    Ok(())
}
