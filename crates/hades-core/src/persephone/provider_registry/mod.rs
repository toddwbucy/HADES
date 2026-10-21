//! Trusted local provider discovery for internal profile-bound operations.
//! The installation supplies a protected registration root and explicit Unix
//! connections. Callers select IDs, never endpoints, manifests, or observed receipts.

mod registration;
pub use registration::RegistrationSummary;
mod description;
pub use description::{
    DescriptionFormat, ObservedWorkload, ObservedWorkloads, ProviderDescription,
    WorkloadObservation,
};

use std::{
    collections::BTreeMap,
    os::unix::fs::{FileTypeExt, MetadataExt},
    path::{Component, Path, PathBuf},
    time::Duration,
};

use http::{Method, Request};
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Bytes;
use hyper_util::client::legacy::Client;
use hyperlocal::UnixClientExt;

use super::{
    bound_embedding::{BoundEmbeddingClient, ProviderConnection},
    embedding::{EmbeddingEndpoint, EmbeddingError},
};
use crate::embedding_profile::{
    ArtifactManifest, ChunkPolicy, EmbeddingProfile, EmbeddingRole, MAX_PROFILE_BYTES,
    ModelIdentity, ProfileFormat, ProfileState, RuntimeBinding, Workload, WorkloadKind, Workloads,
    require,
};
use registration::Registered;

/// An installation-owned allowlist. Construction does not contact providers or load models.
#[derive(Clone)]
pub struct ProviderCatalog {
    root: PathBuf,
    connections: BTreeMap<String, ProviderConnection>,
}

/// A registered artifact plus a checked live observation. No public constructor or deserializer.
#[derive(Clone)]
pub struct VerifiedProvider {
    registration: Registered,
    description: ProviderDescription,
    connection: ProviderConnection,
}

/// Require explicit normalized absolute paths rather than cwd-dependent resolution.
fn absolute(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
}

impl ProviderCatalog {
    /// Accept operational settings only from trusted installation configuration.
    /// Remote HTTP providers need a separate authenticated identity contract and are refused here.
    pub fn new(
        root: PathBuf,
        connections: Vec<ProviderConnection>,
    ) -> Result<Self, EmbeddingError> {
        require(
            absolute(&root),
            "registry root must be an absolute path without traversal",
        )?;
        require(
            connections.len() <= 128,
            "provider allowlist exceeds 128 entries",
        )?;
        let mut configured = BTreeMap::new();
        for connection in connections {
            registration::provider_id(&connection.provider_id)?;
            require(
                !connection.timeout.is_zero() && connection.timeout <= Duration::from_secs(600),
                "provider timeout must be positive and at most 600 seconds",
            )?;
            require(
                (1..=8 * 1024 * 1024).contains(&connection.max_response_bytes),
                "provider response limit must be 1..=8 MiB",
            )?;
            require(
                matches!(&connection.endpoint, EmbeddingEndpoint::Unix(path) if absolute(path)),
                "registered local providers require an explicit absolute Unix socket",
            )?;
            let key = connection.provider_id.clone();
            require(
                configured.insert(key, connection).is_none(),
                "duplicate configured provider ID",
            )?;
        }
        Ok(Self {
            root,
            connections: configured,
        })
    }

    /// List only configured IDs, without scanning unrelated registrations or loading providers.
    pub fn provider_ids(&self) -> impl Iterator<Item = &str> {
        self.connections.keys().map(String::as_str)
    }

    /// Inspect trusted local metadata; loaded_verified is always false until explicit observation.
    pub async fn inspect(&self, id: &str) -> Result<RegistrationSummary, EmbeddingError> {
        self.connection(id)?;
        Ok(self.registration(id).await?.summary())
    }

    /// Explicitly load/observe one configured provider and compare it with protected registration.
    pub async fn verify(&self, id: &str) -> Result<VerifiedProvider, EmbeddingError> {
        let connection = self.connection(id)?.clone();
        let registered = self.registration(id).await?;
        self.observe(registered, connection).await
    }

    /// Refuse an unknown/rebound registration before loading a model, then admit the saved policy.
    pub async fn verify_profile(
        &self,
        profile: &EmbeddingProfile,
        manifest: &ArtifactManifest,
    ) -> Result<VerifiedProvider, EmbeddingError> {
        profile.validate(manifest)?;
        let id = &profile.runtime_binding.provider_id;
        let connection = self.connection(id)?.clone();
        let registered = self.registration(id).await?;
        registered_model(&registered, &profile.model, manifest)?;
        let observed = self.observe(registered, connection).await?;
        observed.check_profile(profile, manifest)?;
        Ok(observed)
    }

    /// Resolve an ID without endpoint defaults, environment overrides, or fallback providers.
    fn connection(&self, id: &str) -> Result<&ProviderConnection, EmbeddingError> {
        registration::provider_id(id)?;
        self.connections.get(id).ok_or_else(|| {
            EmbeddingError::InvalidResponse("provider is not in the installation allowlist".into())
        })
    }

    /// Keep bounded filesystem/lock work off the async worker; no registration writes occur.
    async fn registration(&self, id: &str) -> Result<Registered, EmbeddingError> {
        let root = self.root.clone();
        let id = id.to_owned();
        tokio::task::spawn_blocking(move || registration::read(&root, &id))
            .await
            .map_err(|_| EmbeddingError::InvalidResponse("registration reader failed".into()))?
    }

    /// Do not hold a registration lock during model loading; recheck its exact file pin afterward.
    async fn observe(
        &self,
        registered: Registered,
        connection: ProviderConnection,
    ) -> Result<VerifiedProvider, EmbeddingError> {
        let description = describe(&connection).await?;
        let after = self.registration(&connection.provider_id).await?;
        require(
            registered == after,
            "registration changed during provider verification; retry explicitly",
        )?;
        registered_model(&registered, description.model(), &registered.value.manifest)?;
        let observed = VerifiedProvider {
            registration: registered,
            description,
            connection,
        };
        observed.validate_description()?;
        Ok(observed)
    }
}

/// Match approved artifact metadata; a network response cannot introduce a new trust root.
fn registered_model(
    registered: &Registered,
    model: &ModelIdentity,
    manifest: &ArtifactManifest,
) -> Result<(), EmbeddingError> {
    let record = &registered.value;
    require(
        model.id == record.model_id
            && model.revision == record.revision
            && model.manifest_md5 == record.manifest_md5
            && manifest.canonical_bytes()? == record.manifest.canonical_bytes()?,
        "model identity differs from protected registration",
    )?;
    Ok(())
}

impl VerifiedProvider {
    /// Borrow the checked observation; cloning its data cannot mint another verified handle.
    pub fn description(&self) -> &ProviderDescription {
        &self.description
    }

    /// Borrow the operator-approved manifest rather than one supplied over HTTP.
    pub fn manifest(&self) -> &ArtifactManifest {
        &self.registration.value.manifest
    }

    /// Identify the selected installation entry without exposing endpoint or filesystem paths.
    pub fn provider_id(&self) -> &str {
        &self.registration.value.provider_id
    }

    /// Build an initializing documents-only profile from the actual observation and explicit policy.
    /// This does not create a database, mark readiness, enable code, or adopt existing vectors.
    pub fn document_profile(
        &self,
        generation: String,
        chunking: ChunkPolicy,
    ) -> Result<EmbeddingProfile, EmbeddingError> {
        self.observed_profile(WorkloadKind::Documents, generation, chunking)
    }

    /// Reuse profile validation separately for every advertised space and capability set.
    fn observed_profile(
        &self,
        kind: WorkloadKind,
        generation: String,
        chunking: ChunkPolicy,
    ) -> Result<EmbeddingProfile, EmbeddingError> {
        let observed = self.workload(kind)?;
        let workload = Workload {
            space: observed.space.clone(),
            chunking,
            specialized_code: kind == WorkloadKind::Code && observed.capabilities.specialized_code,
        };
        let workloads = match kind {
            WorkloadKind::Documents => Workloads {
                documents: Some(workload),
                code: None,
            },
            WorkloadKind::Code => Workloads {
                documents: None,
                code: Some(workload),
            },
        };
        let profile = EmbeddingProfile {
            format: ProfileFormat::V1,
            revision: 1,
            state: ProfileState::Initializing,
            generation,
            model: self.description.model().clone(),
            capabilities: observed.capabilities.clone(),
            workloads,
            runtime_binding: RuntimeBinding {
                provider_id: self.provider_id().into(),
            },
        };
        profile.validate(self.manifest())?;
        Ok(profile)
    }

    /// Read an explicit observation; single-space policy compatibility is handled separately.
    fn workload(&self, kind: WorkloadKind) -> Result<WorkloadObservation<'_>, EmbeddingError> {
        self.description.workload(kind).ok_or_else(|| {
            EmbeddingError::InvalidResponse(format!(
                "provider does not advertise the {kind:?} workload"
            ))
        })
    }

    /// Preserve explicit ordinary-text code profiles on the original single-space protocol.
    /// The caller still checks the exact saved space; V2 omissions never imply support.
    fn workload_for_policy(
        &self,
        kind: WorkloadKind,
        policy: &Workload,
    ) -> Result<WorkloadObservation<'_>, EmbeddingError> {
        match (&self.description, kind, policy.specialized_code) {
            (ProviderDescription::Single { .. }, WorkloadKind::Code, false) => {
                self.workload(WorkloadKind::Documents)
            }
            _ => self.workload(kind),
        }
    }

    /// One observation describes a coherent load; validate even workloads this caller does not use.
    fn validate_description(&self) -> Result<(), EmbeddingError> {
        let mut generation = None;
        for kind in [WorkloadKind::Documents, WorkloadKind::Code] {
            let Some(observed) = self.description.workload(kind) else {
                continue;
            };
            observed.receipt.validate()?;
            require(
                observed.receipt.artifact_md5 == self.registration.value.manifest_md5
                    && observed.receipt.space_id == observed.space.id()?,
                "provider description and receipt identities disagree",
            )?;
            if let Some(previous) = generation {
                require(
                    previous == &observed.receipt.serving_generation,
                    "provider workloads describe different serving generations",
                )?;
            }
            generation = Some(&observed.receipt.serving_generation);
            self.observed_profile(
                kind,
                "provider-observation".into(),
                ChunkPolicy::Conventional {
                    passage_tokens: 1,
                    overlap_tokens: 0,
                },
            )?;
        }
        require(
            generation.is_some(),
            "provider must advertise at least one workload",
        )?;
        Ok(())
    }

    /// Construct a role-bound client with the checked receipt; later reloads never retarget it.
    pub async fn client(
        &self,
        profile: &EmbeddingProfile,
        manifest: &ArtifactManifest,
        kind: WorkloadKind,
        role: EmbeddingRole,
    ) -> Result<BoundEmbeddingClient, EmbeddingError> {
        self.check_profile(profile, manifest)?;
        let policy = match kind {
            WorkloadKind::Documents => profile.workloads.documents.as_ref(),
            WorkloadKind::Code => profile.workloads.code.as_ref(),
        }
        .ok_or_else(|| EmbeddingError::InvalidResponse("workload is disabled_by_profile".into()))?;
        BoundEmbeddingClient::connect(
            profile,
            manifest,
            kind,
            role,
            self.connection.clone(),
            self.workload_for_policy(kind, policy)?.receipt,
        )
        .await
    }

    /// Match all enabled workloads while allowing operational serving ceilings to change.
    fn check_profile(
        &self,
        profile: &EmbeddingProfile,
        manifest: &ArtifactManifest,
    ) -> Result<(), EmbeddingError> {
        profile.validate(manifest)?;
        registered_model(&self.registration, &profile.model, manifest)?;
        require(
            profile.runtime_binding.provider_id == self.provider_id(),
            "profile selects another provider",
        )?;
        let mut saved = profile.model.tokenizer_files.clone();
        let mut actual = self.description.model().tokenizer_files.clone();
        saved.sort();
        actual.sort();
        require(
            saved == actual,
            "provider tokenizer dependencies differ from the saved profile",
        )?;
        for (kind, workload) in [
            (WorkloadKind::Documents, &profile.workloads.documents),
            (WorkloadKind::Code, &profile.workloads.code),
        ] {
            let Some(workload) = workload else { continue };
            let observed = self.workload_for_policy(kind, workload)?;
            let capabilities = observed.capabilities;
            require(
                &workload.space == observed.space,
                "provider embedding behavior differs from the saved workload",
            )?;
            require(
                capabilities.text
                    && capabilities.tasks.contains(&workload.space.passage.task)
                    && capabilities.tasks.contains(&workload.space.query.task)
                    && (!workload.specialized_code || capabilities.specialized_code),
                "provider no longer supports an enabled workload",
            )?;
            if matches!(workload.chunking, ChunkPolicy::Late { .. }) {
                require(
                    capabilities.late_chunking && capabilities.tokenizer_offsets,
                    "provider no longer supports the saved late-chunk policy",
                )?;
            }
        }
        Ok(())
    }
}

/// Observe only a same-account Unix service behind a private directory, with bounded HTTP work.
async fn describe(connection: &ProviderConnection) -> Result<ProviderDescription, EmbeddingError> {
    let EmbeddingEndpoint::Unix(path) = &connection.endpoint else {
        return Err(EmbeddingError::InvalidResponse(
            "local provider requires a Unix socket".into(),
        ));
    };
    let socket = std::fs::symlink_metadata(path)?;
    let parent = std::fs::symlink_metadata(
        path.parent()
            .ok_or_else(|| EmbeddingError::InvalidResponse("socket has no parent".into()))?,
    )?;
    // SAFETY: geteuid has no arguments or memory ownership requirements.
    let owner = unsafe { libc::geteuid() };
    require(
        socket.file_type().is_socket()
            && socket.uid() == owner
            && parent.is_dir()
            && parent.uid() == owner
            && parent.mode() & 0o7777 == 0o700,
        "verified socket must belong to this account inside a private 0700 directory",
    )?;
    // Protect the complete pathname used by this request and subsequent bound clients.
    // Root-owned sticky /tmp is safe for an owned private child; ordinary shared
    // writable ancestors could let another account replace that child between checks.
    for ancestor in path.ancestors().skip(1) {
        let info = std::fs::symlink_metadata(ancestor)?;
        require(
            info.is_dir() && (info.uid() == 0 || info.uid() == owner),
            "socket ancestors must be directories owned by this account or root",
        )?;
        require(
            info.mode() & 0o022 == 0 || (info.uid() == 0 && info.mode() & 0o1000 != 0),
            "socket ancestors must not be writable by other accounts outside root-owned sticky directories",
        )?;
    }
    let request = Request::builder()
        .method(Method::POST)
        .uri(hyperlocal::Uri::new(path, "/v1/verify"))
        .body(Full::new(Bytes::new()))?;
    let client: Client<_, Full<Bytes>> = Client::unix();
    tokio::time::timeout(connection.timeout, async {
        let response = client.request(request).await?;
        let status = response.status();
        let body = Limited::new(
            response.into_body(),
            connection.max_response_bytes.min(MAX_PROFILE_BYTES),
        );
        let bytes = body
            .collect()
            .await
            .map_err(|e| {
                EmbeddingError::InvalidResponse(format!(
                    "provider description exceeds response bound: {e}"
                ))
            })?
            .to_bytes();
        if !status.is_success() {
            return Err(EmbeddingError::Http {
                status: status.as_u16(),
                message: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        serde_json::from_slice(&bytes).map_err(|e| {
            EmbeddingError::InvalidResponse(format!("invalid provider description: {e}"))
        })
    })
    .await
    .map_err(|_| EmbeddingError::Timeout(connection.timeout.as_secs()))?
}
