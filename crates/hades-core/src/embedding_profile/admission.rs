//! Public-operation boundary while legacy ingestion is being replaced.
//!
//! Absence is established from authoritative collection responses, never from
//! a missing profile document or an arbitrary HTTP/permission/transport failure.
use crate::db::{ArangoError, ArangoPool};

/// A metadata or setup collection reserves this database for profile-aware paths,
/// even when initialization failed or its active pointer is missing/corrupt.
pub async fn is_managed(pool: &ArangoPool) -> Result<bool, ArangoError> {
    let client = pool.writer().clone().with_response_limit(64 * 1024)?;
    for collection in [
        super::store::COLLECTION,
        crate::workspace_setup::database::MARKERS,
    ] {
        match client
            .get(&format!("collection/{collection}/properties"))
            .await
        {
            Ok(properties) => {
                if properties["name"] != collection
                    || !matches!(properties["type"].as_u64(), Some(2 | 3))
                {
                    return Err(ArangoError::Request(
                        "invalid workspace collection response".into(),
                    ));
                }
                return Ok(true);
            }
            // 1203 is specifically collection/data-source absence. A missing
            // database, missing document, proxy 404, or denied access is an error.
            Err(ArangoError::Api {
                status: 404,
                error_num: 1203,
                ..
            }) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(false)
}

/// Refuse legacy vector operations before analyzers, providers, or writes start.
/// This is admission, not a DDL lock: privileged database replacement/migration
/// requires quiescing work; existing corpora cannot be adopted by setup.
pub async fn require_legacy(pool: &ArangoPool, operation: &str) -> Result<(), ArangoError> {
    if is_managed(pool).await? {
        return Err(ArangoError::Request(format!(
            "{operation} is unavailable for a model-bound or initializing workspace; \
             use a profile-aware operation. Keep its saved model profile intact"
        )));
    }
    Ok(())
}

/// Whether code is enabled by the admitted workspace policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeAdmission {
    Enabled,
    Disabled,
}

/// Keep routing policy explicit rather than deriving it from the presence of a pin.
pub struct TreeAdmission {
    pub pin: Option<super::store::ProfilePin>,
    pub code: CodeAdmission,
}

/// Admit a document tree before any pipeline starts, including no-op trees.
/// Managed, enabled code still requires the future profile-aware code pipeline.
pub async fn document_tree_admission(
    pool: &ArangoPool,
    has_code: bool,
) -> Result<TreeAdmission, ArangoError> {
    if !is_managed(pool).await? {
        return Ok(TreeAdmission {
            pin: None,
            code: CodeAdmission::Enabled,
        });
    }
    let snapshot = super::store::ProfileStore::new(pool.clone())?
        .load()
        .await?;
    validate_documents(pool, &snapshot).await?;
    let pin = snapshot.job_pin()?;
    let workloads = &snapshot.profile().workloads;
    if has_code && workloads.code.is_some() {
        return Err(ArangoError::Request(
            "model-bound code ingestion is not yet supported; an enabled code workload cannot be skipped as disabled_by_profile".into(),
        ));
    }
    Ok(TreeAdmission {
        pin: Some(pin),
        code: if workloads.code.is_some() {
            CodeAdmission::Enabled
        } else {
            CodeAdmission::Disabled
        },
    })
}

/// Shared read-only policy/storage checks; no provider or extractor is contacted.
pub(crate) async fn validate_documents(
    pool: &ArangoPool,
    snapshot: &super::store::ProfileSnapshot,
) -> Result<(), ArangoError> {
    use super::{ChunkPolicy, Distance, ProfileState};
    let invalid =
        |message: &str| ArangoError::Request(format!("profile-bound documents: {message}"));
    let profile = snapshot.profile();
    if profile.state != ProfileState::Ready {
        return Err(invalid("workspace profile is not ready"));
    }
    let workload = profile
        .workloads
        .documents
        .as_ref()
        .ok_or_else(|| invalid("documents are disabled_by_profile"))?;
    if !matches!(
        workload.chunking,
        ChunkPolicy::Conventional { .. }
            | ChunkPolicy::Late {
                passages: Some(_),
                ..
            }
    ) {
        return Err(invalid(
            "late document chunking requires an explicit passage policy",
        ));
    }
    if workload.space.representation.distance != Distance::Cosine {
        return Err(invalid("this document path supports cosine distance only"));
    }
    // Schema is provisioned separately; ingestion must never repair it as a side effect.
    let client = pool.writer().clone().with_response_limit(64 * 1024)?;
    for collection in ["documents", "chunks", "embeddings"] {
        let properties = client
            .get(&format!("collection/{collection}/properties"))
            .await?;
        if properties["type"].as_u64() != Some(2) {
            return Err(invalid(
                "document storage requires three document collections",
            ));
        }
        if collection != "documents" {
            crate::db::index::require_owner_index(&client, collection, "parent_key").await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor_mock::{Mock, Reply};
    use serde_json::json;

    fn missing() -> Reply {
        Reply {
            status: 404,
            body: json!({"error":true,"code":404,"errorNum":1203,"errorMessage":"collection not found"}),
            gate: None,
        }
    }

    #[tokio::test]
    async fn only_explicit_absence_of_both_collections_allows_legacy_operations() {
        let mut mock = Mock::with_policy_responses(vec![missing(), missing()]).await;
        require_legacy(&mock.pool, "ingestion").await.unwrap();
        mock.event("GET collection/hades_workspace/properties")
            .await;
        mock.event("GET collection/hades_setup/properties").await;
        for stage in ["hades_workspace", "hades_setup"] {
            let mut replies = vec![];
            if stage == "hades_setup" {
                replies.push(missing());
            }
            replies.push(Reply::page(json!({"name":stage,"type":2})));
            let mock = Mock::with_policy_responses(replies).await;
            assert!(
                require_legacy(&mock.pool, "ingestion")
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("model-bound or initializing")
            );
        }
    }

    #[tokio::test]
    async fn metadata_permission_transport_and_malformed_responses_fail_closed() {
        for (status, code) in [(401, 11), (403, 11), (404, 1202), (404, 1228), (503, 1203)] {
            let mock = Mock::with_policy_responses(vec![Reply {
                status, body: json!({"error":true,"code":status,"errorNum":code,"errorMessage":"fixture failure"}), gate:None,
            }]).await;
            assert!(is_managed(&mock.pool).await.is_err(), "{status}/{code}");
        }
        for body in [
            json!({}),
            json!({"name":"other","type":2}),
            json!({"name":"hades_workspace","type":7}),
        ] {
            let mock = Mock::with_policy_responses(vec![Reply::page(body)]).await;
            assert!(is_managed(&mock.pool).await.is_err());
        }
        let mock = Mock::with_policy_responses(vec![Reply {
            status: 404,
            body: json!({"proxy":"not found"}),
            gate: None,
        }])
        .await;
        assert!(is_managed(&mock.pool).await.is_err());
        let pool = mock.pool.clone();
        drop(mock);
        assert!(is_managed(&pool).await.is_err());
    }

    #[tokio::test]
    async fn policy_is_read_from_writer_even_when_reader_claims_legacy() {
        let reader = Mock::with_policy_responses(vec![missing(), missing()]).await;
        let writer = Mock::with_policy_responses(vec![Reply::page(
            json!({"name":"hades_workspace","type":2}),
        )])
        .await;
        let pool = ArangoPool::new(reader.pool.reader().clone(), writer.pool.writer().clone());
        assert!(is_managed(&pool).await.unwrap());
    }
}
