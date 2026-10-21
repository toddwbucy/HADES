//! Host-local provider routes, separate from portable database/model identity.
use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use crate::persephone::{
    bound_embedding::ProviderConnection,
    embedding::{EmbeddingEndpoint, EmbeddingError},
    provider_registry::ProviderCatalog,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredProvidersConfig {
    pub registry_root: PathBuf,
    pub providers: Vec<RegisteredProviderConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredProviderConfig {
    pub id: String,
    pub socket: PathBuf,
    #[serde(default = "timeout_seconds")]
    pub timeout_seconds: u64,
    #[serde(default = "response_bytes")]
    pub max_response_bytes: usize,
}

fn timeout_seconds() -> u64 {
    120
}

fn response_bytes() -> usize {
    8 * 1024 * 1024
}

impl RegisteredProvidersConfig {
    /// Validate explicit routes without reading registrations or loading a model.
    /// The catalog checks protected registrations and sockets when resolving an ID.
    pub fn catalog(&self) -> Result<ProviderCatalog, EmbeddingError> {
        ProviderCatalog::new(
            self.registry_root.clone(),
            self.providers
                .iter()
                .map(|provider| ProviderConnection {
                    provider_id: provider.id.clone(),
                    endpoint: EmbeddingEndpoint::Unix(provider.socket.clone()),
                    timeout: Duration::from_secs(provider.timeout_seconds),
                    max_response_bytes: provider.max_response_bytes,
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HadesConfig, snapshot::Snapshot};

    const YAML: &str = "registered_providers:\n  registry_root: /private/registry\n  providers:\n    - id: local-jina\n      socket: /private/embed.sock\n";

    #[test]
    fn explicit_routes_survive_config_serialization_without_legacy_defaults() {
        assert!(HadesConfig::default().registered_providers.is_none());
        let mut config: HadesConfig = serde_yaml::from_str(YAML).unwrap();
        config.embedding.service.socket = "/wrong/legacy.sock".into();
        config.embedding.model.name = "wrong/model".into();
        let routes = config.registered_providers.as_ref().unwrap();
        assert_eq!(routes.providers[0].timeout_seconds, 120);
        assert_eq!(routes.providers[0].max_response_bytes, 8 * 1024 * 1024);
        assert_eq!(
            routes.catalog().unwrap().provider_ids().collect::<Vec<_>>(),
            ["local-jina"]
        );
        let restored: HadesConfig =
            serde_json::from_value(serde_json::to_value(&config).unwrap()).unwrap();
        assert_eq!(
            restored.registered_providers.unwrap().providers[0].socket,
            PathBuf::from("/private/embed.sock")
        );
        // The existing sealed handoff remains bounded even with installation routes.
        Snapshot::new(&config).unwrap();
        config.registered_providers.as_mut().unwrap().registry_root = "x".repeat(64 * 1024).into();
        assert!(Snapshot::new(&config).is_err());
    }

    #[test]
    fn malformed_routes_never_become_default_connections() {
        for extra in [
            "  endpoint: http://example.invalid\n",
            "  registry_rooot: /wrong\n",
        ] {
            assert!(serde_yaml::from_str::<HadesConfig>(&format!("{YAML}{extra}")).is_err());
        }
        for (field, value) in [
            ("id", serde_json::json!("bad/id")),
            ("socket", serde_json::json!("relative.sock")),
            ("socket", serde_json::json!("http://example.invalid")),
            ("timeout_seconds", serde_json::json!(0)),
            ("timeout_seconds", serde_json::json!(601)),
            ("max_response_bytes", serde_json::json!(0)),
            ("max_response_bytes", serde_json::json!(8 * 1024 * 1024 + 1)),
        ] {
            let config: HadesConfig = serde_yaml::from_str(YAML).unwrap();
            let mut routes = serde_json::to_value(config.registered_providers.unwrap()).unwrap();
            routes["providers"][0][field] = value;
            let routes: RegisteredProvidersConfig = serde_json::from_value(routes).unwrap();
            assert!(routes.catalog().is_err(), "{field}");
        }
        let config: HadesConfig = serde_yaml::from_str(YAML).unwrap();
        let mut routes = config.registered_providers.unwrap();
        routes.providers.push(routes.providers[0].clone());
        assert!(routes.catalog().is_err());
        routes.providers.pop();
        routes.registry_root = "/private/../registry".into();
        assert!(routes.catalog().is_err());
    }
}
