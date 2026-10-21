//! Database ownership and the minimal model-bound document schema.
//! No graph seeds, vector-index defaults, collection deletion, or legacy adoption.
use super::{
    SetupError,
    ledger::{Record, Scope},
};
use crate::db::{ArangoClient, ArangoError};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub(crate) const MARKERS: &str = "hades_setup";
pub(crate) const DATA: [&str; 3] = ["documents", "chunks", "embeddings"];

/// Internal proof minted only from an acknowledged, durable setup ledger record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Ownership {
    format: String,
    installation: String,
    endpoint_sha256: String,
    owner: String,
    operation_id: String,
    request_sha256: String,
    database: String,
    database_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Marker {
    #[serde(rename = "_key")]
    key: String,
    #[serde(rename = "_id")]
    id: String,
    #[serde(rename = "_rev")]
    rev: String,
    payload: Ownership,
}

impl Ownership {
    /// Require the database incarnation recorded after a positively acknowledged create.
    pub(super) fn new(record: &Record, scope: &Scope) -> Result<Self, SetupError> {
        Ok(Self {
            format: "hades-setup-owner-v1".into(),
            installation: scope.installation.clone(),
            endpoint_sha256: scope.endpoint_sha256.clone(),
            owner: record.owner.clone(),
            operation_id: record.request.operation_id.clone(),
            request_sha256: record.receipt.request_sha256.clone(),
            database: record.request.database.clone(),
            database_id: record.database_id.clone().ok_or_else(|| {
                SetupError::Reconciliation("no acknowledged database incarnation".into())
            })?,
        })
    }

    /// A matching name cannot substitute for the acknowledged database incarnation.
    pub(crate) async fn check_database(&self, client: &ArangoClient) -> Result<(), SetupError> {
        verify(
            client.database() == self.database
                && identity(client).await?.as_ref() == Some(&self.database_id),
            "database incarnation differs from the acknowledged setup operation",
        )
    }

    /// Reconcile only an exact marker, refusing unknown records or changed ownership.
    pub(crate) async fn check(&self, client: &ArangoClient) -> Result<(), SetupError> {
        self.check_database(client).await?;
        collection(client, MARKERS, false).await?;
        let marker: Marker =
            serde_json::from_value(client.get(&format!("document/{MARKERS}/operation")).await?)?;
        verify(
            marker.key == "operation"
                && marker.id == format!("{MARKERS}/operation")
                && !marker.rev.is_empty()
                && marker.rev.len() <= 128
                && marker.payload == *self,
            "database setup marker does not match its durable owner",
        )?;
        verify(
            count(client, MARKERS).await? == 1,
            "unexpected setup marker records",
        )
    }

    /// Publish the ownership marker without overwriting or adopting existing content.
    pub(super) async fn install(&self, client: &ArangoClient) -> Result<(), SetupError> {
        self.check_database(client).await?;
        match client.get(&format!("document/{MARKERS}/operation")).await {
            Ok(_) => return self.check(client).await,
            Err(e) if e.is_not_found() => {}
            Err(e) => return Err(e.into()),
        }
        let names = inventory(client).await?;
        verify(
            names.iter().all(|name| name == MARKERS),
            "unmarked database contains user collections",
        )?;
        collection(client, MARKERS, true).await?;
        verify(
            count(client, MARKERS).await? == 0,
            "unmarked setup collection is not empty",
        )?;
        client
            .post(
                &format!("document/{MARKERS}?waitForSync=true"),
                &json!({"_key":"operation", "payload":self}),
            )
            .await?;
        self.check(client).await
    }
}

/// Recognize only the exact empty schema of an already-acknowledged marker checkpoint.
pub(super) async fn marker_pending(client: &ArangoClient) -> Result<bool, SetupError> {
    let names = inventory(client).await?;
    if names.is_empty() {
        return Ok(true);
    }
    if names.len() != 1 || names[0] != MARKERS {
        return Ok(false);
    }
    collection(client, MARKERS, false).await?;
    Ok(count(client, MARKERS).await? == 0)
}

/// Resolve the selected database's identity without enumerating or adopting other databases.
pub(super) async fn identity(client: &ArangoClient) -> Result<Option<String>, SetupError> {
    let response = match client.get("database/current").await {
        Ok(value) => value,
        Err(e) if e.is_not_found() => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let result = &response["result"];
    let id = result["id"].as_str().unwrap_or("");
    verify(
        result["name"].as_str() == Some(client.database())
            && !id.is_empty()
            && id.len() <= 64
            && id.bytes().all(|b| b.is_ascii_digit()),
        "invalid database identity response",
    )?;
    Ok(Some(id.into()))
}

/// Refuse unknown user collections without deleting or reseeding anything.
pub(crate) async fn inventory(client: &ArangoClient) -> Result<Vec<String>, SetupError> {
    inventory_with_jobs(client, false).await
}

/// Ready workspaces may have the separately validated ingestion history collection.
pub(super) async fn inventory_with_jobs(
    client: &ArangoClient,
    jobs: bool,
) -> Result<Vec<String>, SetupError> {
    let value = client.get("collection?excludeSystem=true").await?;
    let entries = value["result"]
        .as_array()
        .ok_or_else(|| SetupError::Reconciliation("invalid collection inventory".into()))?;
    let mut names = Vec::new();
    for entry in entries {
        let name = entry["name"].as_str().unwrap_or("");
        verify(
            [MARKERS, crate::embedding_profile::store::COLLECTION].contains(&name)
                || DATA.contains(&name)
                || (jobs && name == crate::ingest_jobs::COLLECTION),
            "workspace contains collections outside the confirmed setup schema",
        )?;
        names.push(name.into());
    }
    Ok(names)
}

/// Create only a missing collection in an already-owned database, then validate its exact purpose.
pub(crate) async fn collection(
    client: &ArangoClient,
    name: &str,
    create: bool,
) -> Result<(), SetupError> {
    let path = format!("collection/{name}/properties");
    let properties = match client.get(&path).await {
        Ok(value) => value,
        Err(e) if e.is_not_found() && create => {
            client
                .post(
                    "collection",
                    &json!({"name":name, "type":2,"waitForSync":true}),
                )
                .await?;
            client.get(&path).await?
        }
        Err(e) => return Err(e.into()),
    };
    verify(
        properties["name"] == name
            && properties["type"] == 2
            && properties["waitForSync"] == true
            && properties["isSystem"] == false,
        "collection differs from the confirmed document schema",
    )?;
    let indexes = client.get(&format!("index?collection={name}")).await?;
    let indexes = indexes["indexes"]
        .as_array()
        .ok_or_else(|| SetupError::Reconciliation("invalid index inventory".into()))?;
    verify(
        indexes.len() == 1
            && indexes[0]["type"] == "primary"
            && indexes[0]["unique"] == true
            && indexes[0]["fields"] == json!(["_key"]),
        "unexpected indexes in setup schema",
    )
}

/// Require metadata and validate existing empty data collections without creating missing ones.
pub(super) async fn partial_schema(client: &ArangoClient) -> Result<(), SetupError> {
    let names = inventory(client).await?;
    collection(client, crate::embedding_profile::store::COLLECTION, false).await?;
    for name in DATA {
        if names.iter().any(|existing| existing == name) {
            collection(client, name, false).await?;
            verify(
                count(client, name).await? == 0,
                "initializing workspace contains data; setup cannot adopt it",
            )?;
        }
    }
    Ok(())
}

/// The exact-scan document path needs primary keys, not an unqualified ANN vector index.
pub(crate) async fn schema(
    client: &ArangoClient,
    create: bool,
    empty: bool,
) -> Result<(), SetupError> {
    let names = inventory_with_jobs(client, !empty).await?;
    if names
        .iter()
        .any(|name| name == crate::ingest_jobs::COLLECTION)
    {
        collection(client, crate::ingest_jobs::COLLECTION, false).await?;
    }
    collection(client, crate::embedding_profile::store::COLLECTION, false).await?;
    for name in DATA {
        collection(client, name, create).await?;
        if empty {
            verify(
                count(client, name).await? == 0,
                "initializing workspace contains data; setup cannot adopt it",
            )?;
        }
    }
    Ok(())
}

/// Require a real collection count rather than defaulting malformed responses to zero.
pub(crate) async fn count(client: &ArangoClient, name: &str) -> Result<u64, SetupError> {
    client.get(&format!("collection/{name}/count")).await?["count"]
        .as_u64()
        .ok_or_else(|| SetupError::Reconciliation("invalid collection count".into()))
}

/// Preserve a recoverable database error while refusing mismatched ownership or stored schema.
fn verify(value: bool, message: &str) -> Result<(), SetupError> {
    if value {
        Ok(())
    } else {
        Err(SetupError::Reconciliation(message.into()))
    }
}

/// Adapt setup checks to the shared transactional storage error boundary.
pub(crate) fn storage_error(error: SetupError) -> ArangoError {
    ArangoError::Request(format!("workspace setup: {error}"))
}
