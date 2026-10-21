//! Internal, recoverable creation of conventional document workspaces.
//! Installation code supplies authority, ledger, database endpoint and provider catalog.
//! No CLI/MCP/frontend caller is enabled; existing databases are never adopted.

pub(crate) mod database;
mod ledger;

use crate::{
    db::{ArangoClient, ArangoError, ArangoPool},
    embedding_profile::{
        ArtifactManifest, ChunkPolicy, Distance, EmbeddingProfile, ProfileState, hex,
        store::{ProfileSnapshot, ProfileStore},
    },
    persephone::{embedding::EmbeddingError, provider_registry::ProviderCatalog},
};
use ledger::{Lease, Record, Scope};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Exact confirmed intent; runtime endpoints and authority are installation-owned settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetupRequest {
    pub operation_id: String,
    pub database: String,
    pub profile: EmbeddingProfile,
    pub manifest: ArtifactManifest,
}

/// Durable outcome, separate from the step to resume after a recoverable failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SetupStatus {
    Validating,
    Creating,
    Initializing,
    Ready,
    Failed,
    NeedsReconciliation,
}

/// Every side effect is preceded by a durable checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SetupStep {
    Validate,
    Create,
    Marker,
    Profile,
    Schema,
    Verify,
    Publish,
    Complete,
}

/// Portable operation status, without installation paths, endpoints, credentials or model weights.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetupReceipt {
    pub operation_id: String,
    pub database: String,
    pub request_sha256: String,
    pub status: SetupStatus,
    pub step: SetupStep,
    pub diagnostic: Option<String>,
}

/// Setup failures never imply rollback of database creation or permission to adopt a name.
#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    #[error("workspace setup: {0}")]
    Invalid(String),
    #[error("workspace setup requires reconciliation: {0}")]
    Reconciliation(String),
    #[error("workspace setup ledger is busy; retry explicitly")]
    Busy,
    #[error("workspace setup ledger: {0}")]
    Io(#[from] std::io::Error),
    #[error("workspace setup ledger JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Database(#[from] ArangoError),
    #[error(transparent)]
    Provider(#[from] EmbeddingError),
}

/// Trusted installation context. A principal is supplied by authorization code, never browser input.
#[derive(Clone)]
pub struct WorkspaceSetup {
    root: PathBuf,
    scope: Scope,
    owner: String,
    admin: ArangoClient,
    catalog: ProviderCatalog,
}

impl WorkspaceSetup {
    /// Bind one private existing ledger directory to an installation, endpoint and authorized owner.
    pub fn new(
        root: PathBuf,
        installation: String,
        owner: String,
        admin: ArangoClient,
        catalog: ProviderCatalog,
    ) -> Result<Self, SetupError> {
        token(&installation)?;
        token(&owner)?;
        ensure(
            admin.database() == "_system",
            "setup requires an explicit _system client",
        )?;
        ledger::absolute(&root)?;
        let endpoint = serde_json::to_vec(&(admin.socket_path(), admin.base_url()))?;
        Ok(Self {
            root,
            scope: Scope {
                format: "hades-setup-scope-v1".into(),
                installation,
                endpoint_sha256: hex(&Sha256::digest(endpoint)),
            },
            owner,
            admin: admin.with_response_limit(2 * 1024 * 1024)?,
            catalog,
        })
    }

    /// Inspect only this authorized owner's operation; no model or database request is made.
    pub async fn inspect(&self, operation_id: &str) -> Result<Option<SetupReceipt>, SetupError> {
        token(operation_id)?;
        let lease = Lease::open(self.root.clone(), self.scope.clone()).await?;
        let record = lease.read(operation_id).await?;
        if let Some(record) = &record {
            ensure(
                record.owner == self.owner,
                "operation belongs to another owner",
            )?;
        }
        Ok(record.map(|r| r.receipt))
    }

    /// Create or resume identical confirmed intent. The ledger lock spans all external work.
    /// Cancellation leaves the preceding checkpoint; it never deletes a database or profile.
    pub async fn run(&self, request: SetupRequest) -> Result<SetupReceipt, SetupError> {
        request.validate()?;
        let lease = Lease::open(self.root.clone(), self.scope.clone()).await?;
        let mut record = lease.claim(request, &self.owner).await?;
        if record.receipt.status == SetupStatus::NeedsReconciliation {
            return Err(SetupError::Reconciliation(
                "operation is held for explicit administrative reconciliation".into(),
            ));
        }
        let result = self.drive(&lease, &mut record).await;
        if let Err(error) = &result
            && record.receipt.status != SetupStatus::Ready
        {
            record.receipt.status = if record.receipt.step == SetupStep::Create
                || matches!(error, SetupError::Reconciliation(_))
            {
                SetupStatus::NeedsReconciliation
            } else {
                SetupStatus::Failed
            };
            record.receipt.diagnostic = Some(
                match error {
                    SetupError::Provider(_) => "provider_verification_failed",
                    SetupError::Database(_) => "database_operation_failed",
                    SetupError::Reconciliation(_) => "ownership_or_metadata_uncertain",
                    _ => "setup_invariant_or_ledger_failed",
                }
                .into(),
            );
            lease.save(&record).await?;
        }
        result?;
        Ok(record.receipt)
    }

    /// Resume only acknowledged ownership; a create checkpoint without a durable database ID is ambiguous.
    async fn drive(&self, lease: &Lease, record: &mut Record) -> Result<(), SetupError> {
        let client = self.admin.for_database(&record.request.database)?;
        let pool = ArangoPool::new(client.clone(), client.clone());
        if record.receipt.step == SetupStep::Create {
            return Err(SetupError::Reconciliation(
                "database creation acknowledgement was not recorded".into(),
            ));
        }
        if record.receipt.step == SetupStep::Validate {
            if database::identity(&client).await?.is_some() {
                return Err(SetupError::Reconciliation(
                    "database already exists; setup cannot adopt it".into(),
                ));
            }
            self.catalog
                .verify_profile(&record.request.profile, &record.request.manifest)
                .await?;
            checkpoint(lease, record, SetupStep::Create, SetupStatus::Creating).await?;
            let response = self
                .admin
                .post(
                    "database",
                    &serde_json::json!({"name": record.request.database}),
                )
                .await?;
            ensure(
                response["result"] == true,
                "database creation was not acknowledged",
            )?;
            record.database_id = Some(database::identity(&client).await?.ok_or_else(|| {
                SetupError::Reconciliation("created database identity unavailable".into())
            })?);
            checkpoint(lease, record, SetupStep::Marker, SetupStatus::Initializing).await?;
        }
        let ownership = database::Ownership::new(record, &self.scope)?;
        ownership.check_database(&client).await?;
        if record.receipt.step == SetupStep::Marker {
            ownership.install(&client).await?;
            checkpoint(lease, record, SetupStep::Profile, SetupStatus::Initializing).await?;
        }
        ownership.check(&client).await?;
        database::inventory(&client).await?;
        let store = ProfileStore::new(pool.clone())?;
        if record.receipt.step == SetupStep::Profile {
            database::collection(&client, crate::embedding_profile::store::COLLECTION, true)
                .await?;
            store
                .initialize_for_setup(
                    record.request.profile.clone(),
                    record.request.manifest.clone(),
                    ownership.clone(),
                )
                .await?;
            checkpoint(lease, record, SetupStep::Schema, SetupStatus::Initializing).await?;
        }
        if record.receipt.step == SetupStep::Schema {
            database::schema(&client, true, true).await?;
            checkpoint(lease, record, SetupStep::Verify, SetupStatus::Initializing).await?;
        }
        let snapshot = store.load().await?;
        matching_profile(&snapshot, &record.request)?;
        if snapshot.profile().state == ProfileState::Ready
            && !matches!(
                record.receipt.step,
                SetupStep::Publish | SetupStep::Complete
            )
        {
            return Err(SetupError::Reconciliation(
                "readiness appeared before its publication checkpoint".into(),
            ));
        }
        let expected_records = if snapshot.profile().state == ProfileState::Ready {
            5
        } else {
            4
        };
        if database::count(&client, crate::embedding_profile::store::COLLECTION).await?
            != expected_records
        {
            return Err(SetupError::Reconciliation(
                "unexpected workspace metadata records".into(),
            ));
        }
        database::schema(
            &client,
            false,
            snapshot.profile().state != ProfileState::Ready,
        )
        .await?;
        if snapshot.profile().state != ProfileState::Ready {
            self.catalog
                .verify_profile(&record.request.profile, &record.request.manifest)
                .await?;
            checkpoint(lease, record, SetupStep::Publish, SetupStatus::Initializing).await?;
            store.mark_ready_for_setup(&snapshot, ownership).await?;
        }
        // Reconcile a lost ready commit acknowledgement without rewriting immutable metadata.
        let published = store.load().await?;
        matching_profile(&published, &record.request)?;
        ensure(
            published.profile().state == ProfileState::Ready,
            "readiness was not published",
        )?;
        checkpoint(lease, record, SetupStep::Complete, SetupStatus::Ready).await
    }
}

impl SetupRequest {
    /// Bound the exact intent and refuse unsupported policies before reserving a database name.
    fn validate(&self) -> Result<(), SetupError> {
        token(&self.operation_id)?;
        ensure(
            !self.database.is_empty()
                && self.database.len() <= 64
                && self.database.as_bytes()[0].is_ascii_alphabetic()
                && self
                    .database
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')),
            "database must be 1..64 ASCII letters/digits/underscore/hyphen, starting with a letter",
        )?;
        self.profile
            .validate(&self.manifest)
            .map_err(|e| SetupError::Invalid(e.to_string()))?;
        let documents = self.profile.workloads.documents.as_ref();
        ensure(
            self.profile.revision == 1
                && self.profile.state == ProfileState::Initializing
                && self.profile.workloads.code.is_none()
                && documents.is_some_and(|w| {
                    matches!(w.chunking, ChunkPolicy::Conventional { .. })
                        && w.space.representation.distance == Distance::Cosine
                        && !w.specialized_code
                }),
            "setup requires initializing revision one with conventional cosine documents only",
        )
    }
    /// Hash typed intent with canonical artifact order, including the selected database and operation.
    fn digest(&self) -> Result<String, SetupError> {
        let mut request = self.clone();
        request.manifest = ArtifactManifest::from_json(
            &self
                .manifest
                .canonical_bytes()
                .map_err(|e| SetupError::Invalid(e.to_string()))?,
        )
        .map_err(|e| SetupError::Invalid(e.to_string()))?;
        Ok(hex(&Sha256::digest(serde_json::to_vec(&request)?)))
    }
}

/// Only the intended initializing profile or its exact ready successor is reconcilable.
pub(crate) fn matching_profile(
    snapshot: &ProfileSnapshot,
    request: &SetupRequest,
) -> Result<(), SetupError> {
    let mut expected = request.profile.clone();
    if snapshot.profile().state == ProfileState::Ready {
        expected.state = ProfileState::Ready;
        expected.revision = 2;
    }
    ensure(
        snapshot.profile() == &expected
            && snapshot
                .manifest()
                .canonical_bytes()
                .map_err(|e| SetupError::Invalid(e.to_string()))?
                == request
                    .manifest
                    .canonical_bytes()
                    .map_err(|e| SetupError::Invalid(e.to_string()))?,
        "stored profile differs from confirmed setup intent",
    )
    .map_err(|_| {
        SetupError::Reconciliation("stored profile differs from confirmed setup intent".into())
    })
}

/// Persist a step before sending any side effect belonging to it.
async fn checkpoint(
    lease: &Lease,
    record: &mut Record,
    step: SetupStep,
    status: SetupStatus,
) -> Result<(), SetupError> {
    record.receipt.step = step;
    record.receipt.status = status;
    record.receipt.diagnostic = None;
    lease.save(record).await
}

/// Limit operation and authority identifiers to bounded path-free tokens.
fn token(value: &str) -> Result<(), SetupError> {
    ensure(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')),
        "setup identifiers must be 1..128 ASCII letters/digits/underscore/hyphen",
    )
}
/// Return an invariant error without interpolating request bodies or secrets.
fn ensure(value: bool, message: &str) -> Result<(), SetupError> {
    if value {
        Ok(())
    } else {
        Err(SetupError::Invalid(message.into()))
    }
}
