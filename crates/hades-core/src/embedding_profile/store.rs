//! Internal persistence primitives for new, exclusively owned workspaces.
//!
//! No CLI, service, or frontend calls these yet. The provisioning owner must
//! establish database ownership and schema/provider readiness; storage alone
//! cannot do that. Existing corpora cannot be adopted through initialization.

use std::{future::Future, path::PathBuf};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::db::{ArangoClient, ArangoError, ArangoPool, transaction};

use super::{
    ArtifactManifest, EmbeddingProfile, MAX_MANIFEST_BYTES, MAX_PROFILE_BYTES, Md5Digest,
    ProfileState, SpaceId, hex,
};

/// Separate from graph schema metadata, which can be independently reseeded.
pub const COLLECTION: &str = "hades_workspace";
const POINTER_KEY: &str = "embedding_profile";
const RECORD_FORMAT: &str = "hades-workspace-record-v1";
// Allow ArangoDB's document envelope without relaxing the portable payload limit.
const ENVELOPE_BYTES: usize = 8192;

/// A database-local store. Every operation uses the writer for authoritative reads.
#[derive(Clone)]
pub struct ProfileStore {
    pool: ArangoPool,
}

/// Validated immutable data and opaque database/document revision pins.
/// Not serializable: this is an in-process admission snapshot, not a job receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileSnapshot {
    scope: Scope,
    pointer: Record<Pointer>,
    profile: Record<EmbeddingProfile>,
    manifest: Record<ArtifactManifest>,
    generation: Record<Generation>,
}

impl ProfileSnapshot {
    /// Borrow the validated profile without allowing mutation of its admission pin.
    pub fn profile(&self) -> &EmbeddingProfile {
        &self.profile.payload
    }

    /// Borrow the validated manifest referenced by this exact profile revision.
    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest.payload
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Scope {
    database: String,
    database_id: String,
    socket: Option<PathBuf>,
    url: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record<T> {
    #[serde(rename = "_key")]
    key: String,
    #[serde(rename = "_id")]
    id: String,
    #[serde(rename = "_rev")]
    rev: String,
    format: String,
    payload: T,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pointer {
    revision: u64,
    generation: String,
    manifest_md5: Md5Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Generation {
    generation: String,
    manifest_md5: Md5Digest,
    documents: Option<SpaceId>,
    code: Option<SpaceId>,
}

impl Generation {
    /// Derive the immutable corpus identity from every enabled workload.
    fn from_profile(profile: &EmbeddingProfile) -> Result<Self, ArangoError> {
        Ok(Self {
            generation: profile.generation.clone(),
            manifest_md5: profile.model.manifest_md5.clone(),
            documents: profile
                .workloads
                .documents
                .as_ref()
                .map(|w| w.space.id())
                .transpose()
                .map_err(invalid)?,
            code: profile
                .workloads
                .code
                .as_ref()
                .map(|w| w.space.id())
                .transpose()
                .map_err(invalid)?,
        })
    }
}

impl ProfileStore {
    /// Select an explicit non-system database; no default database is inferred.
    pub fn new(pool: ArangoPool) -> Result<Self, ArangoError> {
        ensure(
            !pool.writer().database().is_empty()
                && pool.writer().database() != "_system"
                && pool.writer().database() == pool.reader().database(),
            "select one explicit non-system database",
        )?;
        Ok(Self { pool })
    }

    /// Initialize revision one only in a newly owned database with no user collections.
    /// The caller must serialize provisioning; collection inspection is not an
    /// ownership ledger or a lock against unrelated administrative DDL.
    ///
    /// Collection creation is outside the metadata transaction. Failure may leave
    /// an empty collection; retry never adopts or deletes it. Lost commit responses
    /// require explicit reconciliation with `load`, not automatic reinitialization.
    pub async fn initialize(
        &self,
        profile: EmbeddingProfile,
        manifest: ArtifactManifest,
    ) -> Result<ProfileSnapshot, ArangoError> {
        profile.validate(&manifest).map_err(invalid)?;
        ensure(
            profile.revision == 1 && profile.state == ProfileState::Initializing,
            "initial profile must be initializing revision one",
        )?;
        // Store canonical manifest ordering, so equivalent registrations read identically.
        let manifest = ArtifactManifest::from_json(&manifest.canonical_bytes().map_err(invalid)?)
            .map_err(invalid)?;
        let client = self
            .pool
            .writer()
            .clone()
            .with_response_limit(MAX_MANIFEST_BYTES)?;
        let collections = client.get("collection?excludeSystem=true").await?;
        ensure(
            collections["result"].as_array().is_some_and(Vec::is_empty),
            "initialization requires a new empty database; legacy adoption is unavailable",
        )?;
        // Do not use create-if-absent: a losing concurrent initializer must fail.
        client
            .post(
                "collection",
                &json!({
                    "name": COLLECTION, "type": 2, "waitForSync": true
                }),
            )
            .await?;
        transaction::run(
            &self.pool,
            vec![COLLECTION.into()],
            move |client| async move {
                let generation = Generation::from_profile(&profile)?;
                let pointer = Pointer {
                    revision: profile.revision,
                    generation: profile.generation.clone(),
                    manifest_md5: profile.model.manifest_md5.clone(),
                };
                insert(&client, &manifest_key(&pointer), &manifest).await?;
                insert(&client, &generation_key(&pointer), &generation).await?;
                insert(&client, &profile_key(&pointer), &profile).await?;
                insert(&client, POINTER_KEY, &pointer).await?;
                load_locked(&client).await
            },
        )
        .await
    }

    /// Resolve one coherent metadata snapshot. Missing, unknown, or inconsistent
    /// records fail closed; no legacy model or process configuration is substituted.
    pub async fn load(&self) -> Result<ProfileSnapshot, ArangoError> {
        transaction::run(&self.pool, vec![COLLECTION.into()], |client| async move {
            load_locked(&client).await
        })
        .await
    }

    /// Publish readiness after the provisioning owner verifies schema and provider.
    /// Only initializing revision one -> ready revision two is supported. Identity,
    /// workloads, and generation stay fixed; model migration is deliberately absent.
    /// A stale or repeated request fails and must be reconciled by reloading.
    pub async fn mark_ready(
        &self,
        expected: &ProfileSnapshot,
    ) -> Result<ProfileSnapshot, ArangoError> {
        ensure(
            expected.profile().state == ProfileState::Initializing
                && expected.profile().revision == 1,
            "expected initializing revision one",
        )?;
        let expected = expected.clone();
        transaction::run(
            &self.pool,
            vec![COLLECTION.into()],
            move |client| async move {
                check_locked(&client, &expected).await?;
                let mut profile = expected.profile().clone();
                profile.revision = 2;
                profile.state = ProfileState::Ready;
                profile.validate(expected.manifest()).map_err(invalid)?;
                let mut pointer = expected.pointer.payload.clone();
                pointer.revision = 2;
                insert(&client, &profile_key(&pointer), &profile).await?;
                client
                    .put(
                        &format!("document/{COLLECTION}/{POINTER_KEY}?ignoreRevs=false"),
                        &json!({
                            "_key": POINTER_KEY, "_rev": expected.pointer.rev,
                            "format": RECORD_FORMAT, "payload": pointer
                        }),
                    )
                    .await?;
                load_locked(&client).await
            },
        )
        .await
    }

    /// Recheck a ready admission snapshot and execute prepared writes under the
    /// same exclusive transaction. Stale work never reaches the callback.
    /// Callback error/cancellation rolls back via the shared transaction owner.
    ///
    /// Callers are trusted core code: use only the provided transaction client,
    /// declare every target collection, and perform model/network work beforehand.
    /// This checks database policy freshness, not vector shape/provider provenance.
    /// A lost commit acknowledgement remains ambiguous and requires reconciliation.
    pub async fn guarded_write<T, F, Fut>(
        &self,
        expected: &ProfileSnapshot,
        mut collections: Vec<String>,
        operation: F,
    ) -> Result<T, ArangoError>
    where
        T: Send + 'static,
        F: FnOnce(ArangoClient) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, ArangoError>> + Send + 'static,
    {
        ensure(
            expected.profile().state == ProfileState::Ready,
            "workspace profile is not ready",
        )?;
        ensure(
            !collections.is_empty()
                && collections.iter().all(|name| {
                    name != COLLECTION
                        && !name.is_empty()
                        && !name.starts_with('_')
                        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
                }),
            "declare non-system data collections, excluding workspace metadata",
        )?;
        collections.push(COLLECTION.into());
        collections.sort();
        collections.dedup();
        let expected = expected.clone();
        transaction::run(&self.pool, collections, move |client| async move {
            check_locked(&client, &expected).await?;
            let result = operation(client.clone()).await?;
            // Also reject accidental metadata mutation by the trusted callback.
            check_locked(&client, &expected).await?;
            Ok(result)
        })
        .await
    }
}

/// Require exact content and revision pins, including immutable record revisions.
async fn check_locked(
    client: &ArangoClient,
    expected: &ProfileSnapshot,
) -> Result<(), ArangoError> {
    ensure(
        &load_locked(client).await? == expected,
        "stale database profile; discard prepared work and resolve a new binding",
    )
}

/// Read and cross-check all four records while the workspace collection is locked.
async fn load_locked(client: &ArangoClient) -> Result<ProfileSnapshot, ArangoError> {
    let pointer: Record<Pointer> = read(client, POINTER_KEY, MAX_PROFILE_BYTES).await?;
    let p = &pointer.payload;
    ensure(p.revision > 0, "invalid active revision")?;
    super::label(&p.generation).map_err(invalid)?;
    let manifest: Record<ArtifactManifest> =
        read(client, &manifest_key(p), MAX_MANIFEST_BYTES).await?;
    let profile: Record<EmbeddingProfile> =
        read(client, &profile_key(p), MAX_PROFILE_BYTES).await?;
    let generation: Record<Generation> =
        read(client, &generation_key(p), MAX_PROFILE_BYTES).await?;
    profile
        .payload
        .validate(&manifest.payload)
        .map_err(invalid)?;
    ensure(
        profile.payload.revision == p.revision
            && profile.payload.generation == p.generation
            && profile.payload.model.manifest_md5 == p.manifest_md5,
        "active pointer differs from its profile revision",
    )?;
    ensure(
        generation.payload == Generation::from_profile(&profile.payload)?,
        "generation identity differs from its profile revision",
    )?;
    let info = client
        .clone()
        .with_response_limit(ENVELOPE_BYTES)?
        .get("database/current")
        .await?;
    let id = info["result"]["id"]
        .as_str()
        .filter(|id| !id.is_empty() && id.bytes().all(|c| c.is_ascii_digit()))
        .ok_or_else(|| invalid("missing database identity"))?;
    ensure(
        info["result"]["name"].as_str() == Some(client.database()),
        "database response does not match selected database",
    )?;
    Ok(ProfileSnapshot {
        scope: Scope {
            database: client.database().into(),
            database_id: id.into(),
            socket: client.socket_path().map(PathBuf::from),
            url: client.base_url().into(),
        },
        pointer,
        profile,
        manifest,
        generation,
    })
}

/// Read a bounded, versioned document and validate its ArangoDB envelope.
async fn read<T: DeserializeOwned + Serialize>(
    client: &ArangoClient,
    key: &str,
    limit: usize,
) -> Result<Record<T>, ArangoError> {
    let raw = client
        .clone()
        .with_response_limit(limit + ENVELOPE_BYTES)?
        .get(&format!("document/{COLLECTION}/{key}"))
        .await?;
    let record: Record<T> = serde_json::from_value(raw)?;
    ensure(
        record.format == RECORD_FORMAT
            && record.key == key
            && record.id == format!("{COLLECTION}/{key}"),
        "unknown workspace record format or identity",
    )?;
    ensure(
        !record.rev.is_empty()
            && record.rev.len() <= 128
            && record
                .rev
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
        "invalid workspace document revision",
    )?;
    ensure(
        serde_json::to_vec(&record.payload)?.len() <= limit,
        "workspace payload exceeds its size limit",
    )?;
    Ok(record)
}

/// Insert immutable records without any overwrite or upsert behavior.
async fn insert(
    client: &ArangoClient,
    key: &str,
    payload: &impl Serialize,
) -> Result<(), ArangoError> {
    let response = client
        .post(
            &format!("document/{COLLECTION}"),
            &json!({"_key": key, "format": RECORD_FORMAT, "payload": payload}),
        )
        .await?;
    ensure(
        response["_key"].as_str() == Some(key)
            && response["_rev"].as_str().is_some_and(|rev| !rev.is_empty()),
        "invalid workspace insert acknowledgement",
    )
}

/// Generate a safe revision key without interpolating arbitrary labels.
fn profile_key(pointer: &Pointer) -> String {
    format!("profile_{}", pointer.revision)
}

/// Address manifests by their already validated MD5 identity.
fn manifest_key(pointer: &Pointer) -> String {
    format!("manifest_{}", pointer.manifest_md5.as_str())
}

/// Encode arbitrary generation labels into a fixed safe document key.
fn generation_key(pointer: &Pointer) -> String {
    format!(
        "generation_{}",
        hex(&Sha256::digest(pointer.generation.as_bytes()))
    )
}

/// Attach the storage boundary to validation diagnostics without echoing payloads.
fn invalid(error: impl std::fmt::Display) -> ArangoError {
    ArangoError::Request(format!("embedding profile store: {error}"))
}

/// Refuse failed invariants rather than guessing or repairing stored metadata.
fn ensure(condition: bool, message: &str) -> Result<(), ArangoError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
