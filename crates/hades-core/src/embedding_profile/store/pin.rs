//! Compact, portable evidence of one validated database admission snapshot.
use super::*;

const FORMAT: &str = "hades-ingest-profile-pin-v1";

/// A job carries identities and a digest, never the potentially large manifest.
/// This is comparison evidence, not authority to initialize or change a database.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilePin {
    format: String,
    database: String,
    database_id: String,
    endpoint_sha256: String,
    snapshot_sha256: String,
    profile_revision: u64,
    generation: String,
    manifest_md5: Md5Digest,
    documents_space: Option<SpaceId>,
    code_space: Option<SpaceId>,
}

fn digest(value: &impl Serialize) -> Result<String, ArangoError> {
    Ok(hex(&Sha256::digest(serde_json::to_vec(value)?)))
}

impl ProfileSnapshot {
    /// Pin all payloads and Arango revisions, including database incarnation.
    /// Changing and restoring an identical profile still invalidates this pin.
    pub fn job_pin(&self) -> Result<ProfilePin, ArangoError> {
        ensure(
            self.profile().state == ProfileState::Ready,
            "job profile is not ready",
        )?;
        let pin = ProfilePin {
            format: FORMAT.into(),
            database: self.scope.database.clone(),
            database_id: self.scope.database_id.clone(),
            endpoint_sha256: digest(&(&self.scope.socket, &self.scope.url))?,
            snapshot_sha256: digest(&(
                &self.scope,
                &self.pointer,
                &self.profile,
                &self.manifest,
                &self.generation,
            ))?,
            profile_revision: self.profile().revision,
            generation: self.profile().generation.clone(),
            manifest_md5: self.profile().model.manifest_md5.clone(),
            documents_space: self.generation.payload.documents.clone(),
            code_space: self.generation.payload.code.clone(),
        };
        pin.validate()?;
        Ok(pin)
    }
}

impl ProfilePin {
    /// Reject malformed/oversized handoffs before contacting a provider.
    pub fn validate(&self) -> Result<(), ArangoError> {
        let sha256 = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
        ensure(
            self.format == FORMAT
                && self.profile_revision > 0
                && sha256(&self.endpoint_sha256)
                && sha256(&self.snapshot_sha256)
                && !self.database_id.is_empty()
                && self.database_id.len() <= 128
                && self.database_id.bytes().all(|b| b.is_ascii_digit()),
            "invalid ingestion profile pin",
        )?;
        super::super::label(&self.database).map_err(invalid)?;
        super::super::label(&self.generation).map_err(invalid)?;
        ensure(
            serde_json::to_vec(self)?.len() <= 16 * 1024,
            "ingestion profile pin exceeds its byte limit",
        )
    }

    /// Match the complete freshly validated snapshot, not just model name or width.
    pub fn check(&self, snapshot: &ProfileSnapshot) -> Result<(), ArangoError> {
        self.validate()?;
        ensure(
            *self == snapshot.job_pin()?,
            "ingestion job profile changed: database, revision, artifact, space or generation differs from admission; start a new job after review",
        )
    }
}
