//! HADES gRPC/protobuf definitions.
//!
//! Generated from `.proto` files under the `proto/` directory. Provides both
//! client stubs and server traits. The `persephone.*` packages (embedding,
//! extraction) are HADES-owned compute services that carry the legacy
//! provider-protocol brand; `hades.training` was decoupled from it (issue
//! #106) — none of these are the Persephone PM system.

// Every generated client and server method returns `Result<_, tonic::Status>`,
// and `tonic::Status` is 176 bytes, so `result_large_err` fires on all of
// them. Its suggested fix is to box the error, which is not available here:
// the signatures come from tonic-build and nothing in this crate is
// hand-written. Allowed at the crate root rather than per item for the same
// reason.
//
// It began failing when the toolchain reached 1.98, which widened where the
// lint fires. The hand-written error types that wrap `tonic::Status` were
// fixed properly instead, by boxing the variant. Generated code has no such
// option, so it is allowed here.
#![allow(clippy::result_large_err)]

/// Common types shared across Persephone services.
pub mod common {
    tonic::include_proto!("persephone.common");
}

/// Embedding service — vector embedding generation.
pub mod embedding {
    tonic::include_proto!("persephone.embedding");
}

/// Extraction service — document content extraction.
pub mod extraction {
    tonic::include_proto!("persephone.extraction");
}

/// Training service — RGCN link prediction on knowledge graphs (HADES-owned;
/// decoupled from the `persephone.*` provider brand — issue #106).
pub mod training {
    tonic::include_proto!("hades.training");
}
