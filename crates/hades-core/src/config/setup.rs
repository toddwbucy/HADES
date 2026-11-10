//! Installation-owned setup ledger location for managed databases (#225).
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Where `hades db setup` keeps its operation ledger, and which installation it serves.
///
/// There is no default: a managed database is created only against an explicitly
/// configured ledger, so a missing section is a refusal, never a guessed path. The
/// coordinator applies the same identifier rule and demands an existing private root.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetupConfig {
    pub ledger_root: PathBuf,
    pub installation_id: String,
}

impl SetupConfig {
    /// Refuse a relative ledger root or an identifier the coordinator would refuse later.
    pub fn validate(&self) -> Result<(), String> {
        if !self.ledger_root.is_absolute() {
            return Err("setup.ledger_root must be an absolute path".into());
        }
        let id = &self.installation_id;
        if id.is_empty()
            || id.len() > 128
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
        {
            return Err(
                "setup.installation_id must be 1..128 ASCII letters/digits/underscore/hyphen"
                    .into(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates() {
        let cfg: SetupConfig = serde_yaml::from_str(
            "ledger_root: /absolute/private/ledger\ninstallation_id: laptop-1\n",
        )
        .unwrap();
        assert!(cfg.validate().is_ok());
        assert!(
            SetupConfig {
                ledger_root: "relative".into(),
                installation_id: "x".into()
            }
            .validate()
            .is_err()
        );
        assert!(
            SetupConfig {
                ledger_root: "/a".into(),
                installation_id: "bad id".into()
            }
            .validate()
            .is_err()
        );
        assert!(
            serde_yaml::from_str::<SetupConfig>("ledger_root: /a\ninstallation_id: x\nextra: 1\n")
                .is_err()
        );
    }
}
