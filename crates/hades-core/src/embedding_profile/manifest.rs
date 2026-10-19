use std::collections::BTreeSet;
use std::io::Read;

use md5::{Digest, Md5};
use serde::{Deserialize, Serialize};

use super::{ContractError, hex, require};

/// Maximum serialized manifest size, including noncanonical input whitespace.
pub const MAX_MANIFEST_BYTES: usize = 1024 * 1024;
pub const MAX_MANIFEST_FILES: usize = 4096;

/// Lowercase MD5 of artifact bytes; drift detection, not source authentication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Md5Digest(String);

impl TryFrom<String> for Md5Digest {
    type Error = ContractError;

    /// Validate the wire digest without accepting uppercase or shortened forms.
    fn try_from(value: String) -> Result<Self, Self::Error> {
        require(
            value.len() == 32
                && value
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "MD5 must contain exactly 32 lowercase hexadecimal digits",
        )?;
        Ok(Self(value))
    }
}

impl From<Md5Digest> for String {
    /// Transfer the validated lowercase encoding to a serialized string.
    fn from(value: Md5Digest) -> Self {
        value.0
    }
}

impl Md5Digest {
    /// Hash exact bytes for drift detection; this does not authenticate their origin.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self(hex(&Md5::digest(bytes)))
    }

    /// Borrow the validated lowercase digest encoding.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestFormat {
    #[serde(rename = "hades-model-manifest-v1")]
    V1,
}

/// Logical POSIX path and checksum of one explicitly selected required asset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFile {
    pub path: String,
    pub bytes: u64,
    pub md5: Md5Digest,
}

impl ArtifactFile {
    /// Hash an already opened asset in bounded memory. Selecting all required
    /// assets and preventing mutation during loading belong to the registrar.
    pub fn from_reader(path: String, mut source: impl Read) -> Result<Self, ContractError> {
        validate_path(&path)?;
        let mut digest = Md5::new();
        let mut bytes = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let count = match source.read(&mut buffer) {
                Ok(count) => count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(ContractError(format!("cannot hash asset: {error}"))),
            };
            if count == 0 {
                break;
            }
            bytes = bytes
                .checked_add(count as u64)
                .ok_or_else(|| ContractError("asset exceeds u64 byte count".into()))?;
            digest.update(&buffer[..count]);
        }
        Ok(Self {
            path,
            bytes,
            md5: Md5Digest(hex(&digest.finalize())),
        })
    }
}

/// Exact v1 manifest shape; the aggregate digest is stored outside this object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub format: ManifestFormat,
    pub files: Vec<ArtifactFile>,
}

impl ArtifactManifest {
    /// Validate an explicit asset set, including path conflicts and encoded size.
    pub fn new(files: Vec<ArtifactFile>) -> Result<Self, ContractError> {
        let manifest = Self {
            format: ManifestFormat::V1,
            files,
        };
        manifest.canonical_bytes()?;
        Ok(manifest)
    }

    /// Bounded parsing for persisted/wire manifests. Deserializing a struct
    /// alone is not validation; identity methods also validate before hashing.
    pub fn from_json(bytes: &[u8]) -> Result<Self, ContractError> {
        require(bytes.len() <= MAX_MANIFEST_BYTES, "manifest exceeds 1 MiB")?;
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|error| ContractError(format!("invalid manifest: {error}")))?;
        manifest.canonical_bytes()?;
        Ok(manifest)
    }

    /// Validate and emit exact v1 UTF-8 bytes with sorted paths and object keys.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, ContractError> {
        require(
            !self.files.is_empty() && self.files.len() <= MAX_MANIFEST_FILES,
            "manifest must contain 1..=4096 required assets",
        )?;
        let mut paths = BTreeSet::new();
        for file in &self.files {
            validate_path(&file.path)?;
            require(
                paths.insert(file.path.as_str()),
                "manifest contains duplicate asset paths",
            )?;
        }
        for path in &paths {
            for (index, _) in path.match_indices('/') {
                require(
                    !paths.contains(&path[..index]),
                    "asset path is also used as a directory",
                )?;
            }
        }
        let mut files = self.files.clone();
        files.sort_unstable_by(|a, b| a.path.as_bytes().cmp(b.path.as_bytes()));
        let value = serde_json::json!({ "format": self.format, "files": files });
        let bytes = crate::canonical_json::to_vec(&value);
        require(
            bytes.len() <= MAX_MANIFEST_BYTES,
            "canonical manifest exceeds 1 MiB",
        )?;
        Ok(bytes)
    }

    /// Hash the canonical manifest, excluding the aggregate digest from its input.
    pub fn digest(&self) -> Result<Md5Digest, ContractError> {
        Ok(Md5Digest::of_bytes(&self.canonical_bytes()?))
    }

    /// Check against a previously trusted registration, not a provider's own
    /// freshly supplied checksum. This does not read or verify filesystem bytes.
    pub fn check_digest(&self, expected: &Md5Digest) -> Result<(), ContractError> {
        require(
            &self.digest()? == expected,
            "artifact manifest MD5 mismatch; reverify registration",
        )
    }
}

/// Reject ambiguous relative POSIX paths while preserving Unicode and backslashes.
pub(crate) fn validate_path(path: &str) -> Result<(), ContractError> {
    require(
        !path.is_empty()
            && path.len() <= 4096
            && !path.contains('\0')
            && path
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != ".."),
        "asset paths must be unambiguous relative POSIX paths of 1..=4096 UTF-8 bytes",
    )
}
