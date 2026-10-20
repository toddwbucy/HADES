//! Protected metadata fixtures; no real weights, model loader, or shared registry.
use hades_core::embedding_profile::{ArtifactManifest, EmbeddingProfile};
use serde_json::json;
use std::{
    fs::{DirBuilder, OpenOptions},
    io::Write,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::{Path, PathBuf},
};

/// Reproduce the registrar's on-disk envelope with synthetic independently declared identities.
pub fn create(parent: &Path, profile: &EmbeddingProfile, manifest: &ArtifactManifest) -> PathBuf {
    let root = parent.join("registry");
    DirBuilder::new().mode(0o700).create(&root).unwrap();
    for name in ["records", "artifacts", "sessions"] {
        DirBuilder::new()
            .mode(0o700)
            .create(root.join(name))
            .unwrap();
    }
    let snapshot = format!("model_{}", "0".repeat(32));
    DirBuilder::new()
        .mode(0o500)
        .create(root.join("artifacts").join(&snapshot))
        .unwrap();
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(root.join(".lock"))
        .unwrap();
    let record = json!({"format":"hades-model-registration-v1",
        "provider_id":profile.runtime_binding.provider_id,"adapter":"fixture",
        "model_id":profile.model.id,"revision":profile.model.revision,
        "manifest":manifest,"manifest_md5":manifest.digest().unwrap(),"snapshot":snapshot});
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(
            root.join("records")
                .join(format!("{}.json", profile.runtime_binding.provider_id)),
        )
        .unwrap();
    file.write_all(&serde_json::to_vec(&record).unwrap())
        .unwrap();
    root
}
