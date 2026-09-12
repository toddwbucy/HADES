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
// This was already allowed at the two hand-written call sites that surface
// these errors (`hades-core::training`, `hades-frontend::server`). It began
// failing CI when the stable toolchain moved from 1.97 to 1.98, which is also
// why a local clippy run on 1.97 passes while CI does not. `ci.yml` tracks
// `stable`, so the two drift apart whenever a release lands.
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
