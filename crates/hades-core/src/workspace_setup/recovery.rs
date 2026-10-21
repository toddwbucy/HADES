//! Advisory recovery checks. Never publish readiness, edit ownership, or resume an operation.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryAction {
    StartNewOperation,
    RetryIdenticalRequest,
    InspectRequired,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryReason {
    OperationMissing,
    ReservationNotVerified,
    AwaitingValidation,
    CreationUnacknowledged,
    DatabaseAlreadyExists,
    DatabaseMissing,
    DatabaseIncarnationChanged,
    OwnershipNotVerified,
    InitializationIncomplete,
    MetadataNotVerified,
    SchemaNotVerified,
    ProviderVerificationPending,
    PublicationIncomplete,
    Ready,
    HeldForReconciliation,
}

/// Portable advisory result; not a permit, provider verification or corpus integrity report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryReport {
    pub format: String,
    pub operation_id: String,
    pub receipt: Option<SetupReceipt>,
    pub database_exists: Option<bool>,
    pub database_id_matches: Option<bool>,
    pub ownership_verified: bool,
    pub profile_state: Option<ProfileState>,
    pub action: RecoveryAction,
    pub reason: RecoveryReason,
}

impl RecoveryReport {
    fn conclude(mut self, action: RecoveryAction, reason: RecoveryReason) -> Self {
        // A diagnostic observation can never release an administrative hold.
        if self
            .receipt
            .as_ref()
            .is_some_and(|r| r.status == SetupStatus::NeedsReconciliation)
            && action != RecoveryAction::InspectRequired
        {
            self.action = RecoveryAction::InspectRequired;
            self.reason = RecoveryReason::HeldForReconciliation;
        } else {
            self.action = action;
            self.reason = reason;
        }
        self
    }
}

impl WorkspaceSetup {
    /// Inspect one owner's durable intent and current DB invariants under the setup lease.
    /// No provider is loaded and no database/operation record is changed. Retry runs
    /// its normal checks again; this report cannot authorize or resume work by itself.
    pub async fn diagnose(&self, operation_id: &str) -> Result<RecoveryReport, SetupError> {
        token(operation_id)?;
        let lease = Lease::open_existing(self.root.clone(), self.scope.clone()).await?;
        let record = lease.read(operation_id).await?;
        let mut report = RecoveryReport {
            format: "hades-setup-recovery-v1".into(),
            operation_id: operation_id.into(),
            receipt: None,
            database_exists: None,
            database_id_matches: None,
            ownership_verified: false,
            profile_state: None,
            action: RecoveryAction::StartNewOperation,
            reason: RecoveryReason::OperationMissing,
        };
        let Some(record) = record else {
            return Ok(report);
        };
        ensure(
            record.owner == self.owner,
            "operation belongs to another owner",
        )?;
        report.receipt = Some(record.receipt.clone());
        if lease.check_claim(&record).await.is_err() {
            return Ok(report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::ReservationNotVerified,
            ));
        }
        let client = self.admin.for_database(&record.request.database)?;
        let actual = database::identity(&client).await?;
        report.database_exists = Some(actual.is_some());
        if record.receipt.step == SetupStep::Create {
            return Ok(report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::CreationUnacknowledged,
            ));
        }
        if record.receipt.step == SetupStep::Validate {
            return Ok(if actual.is_some() {
                report.conclude(
                    RecoveryAction::InspectRequired,
                    RecoveryReason::DatabaseAlreadyExists,
                )
            } else {
                report.conclude(
                    RecoveryAction::RetryIdenticalRequest,
                    RecoveryReason::AwaitingValidation,
                )
            });
        }
        report.database_id_matches = Some(actual == record.database_id);
        if actual.is_none() {
            return Ok(report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::DatabaseMissing,
            ));
        }
        if actual != record.database_id {
            return Ok(report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::DatabaseIncarnationChanged,
            ));
        }
        let ownership = database::Ownership::new(&record, &self.scope)?;
        if ownership.check(&client).await.is_err() {
            // The acknowledged incarnation can still be at the exact empty
            // marker checkpoint. Inspect the same strict pending schema that
            // normal retry accepts; never infer ownership from a name alone.
            if record.receipt.step == SetupStep::Marker
                && database::marker_pending(&client).await.unwrap_or(false)
            {
                return Ok(report.conclude(
                    RecoveryAction::RetryIdenticalRequest,
                    RecoveryReason::InitializationIncomplete,
                ));
            }
            return Ok(report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::OwnershipNotVerified,
            ));
        }
        report.ownership_verified = true;
        let permits_jobs = matches!(
            record.receipt.step,
            SetupStep::Publish | SetupStep::Complete
        );
        if database::inventory_with_jobs(&client, permits_jobs)
            .await
            .is_err()
        {
            return Ok(report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::SchemaNotVerified,
            ));
        }
        if matches!(
            record.receipt.step,
            SetupStep::Marker | SetupStep::Profile | SetupStep::Schema
        ) {
            return Ok(report.conclude(
                RecoveryAction::RetryIdenticalRequest,
                RecoveryReason::InitializationIncomplete,
            ));
        }
        let pool = ArangoPool::new(client.clone(), client.clone());
        let store = ProfileStore::new(pool)?;
        let snapshot = match store.load().await {
            Ok(snapshot) => snapshot,
            Err(_) => {
                return Ok(report.conclude(
                    RecoveryAction::InspectRequired,
                    RecoveryReason::MetadataNotVerified,
                ));
            }
        };
        report.profile_state = Some(snapshot.profile().state);
        let ready = snapshot.profile().state == ProfileState::Ready;
        if matching_profile(&snapshot, &record.request).is_err()
            || (ready && !permits_jobs)
            || database::count(&client, crate::embedding_profile::store::COLLECTION).await?
                != if ready { 5 } else { 4 }
        {
            return Ok(report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::MetadataNotVerified,
            ));
        }
        if database::schema(&client, false, !ready).await.is_err() {
            return Ok(report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::SchemaNotVerified,
            ));
        }
        // Recheck the identity, marker and complete profile after the individual
        // reads. The report is still advisory: legitimate jobs can continue later.
        if ownership.check(&client).await.is_err() || store.load().await? != snapshot {
            return Ok(report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::MetadataNotVerified,
            ));
        }
        Ok(match (record.receipt.step, ready) {
            (SetupStep::Complete, true) => {
                report.conclude(RecoveryAction::None, RecoveryReason::Ready)
            }
            (SetupStep::Publish, true) => report.conclude(
                RecoveryAction::RetryIdenticalRequest,
                RecoveryReason::PublicationIncomplete,
            ),
            (_, false) if record.receipt.step != SetupStep::Complete => report.conclude(
                RecoveryAction::RetryIdenticalRequest,
                RecoveryReason::ProviderVerificationPending,
            ),
            _ => report.conclude(
                RecoveryAction::InspectRequired,
                RecoveryReason::MetadataNotVerified,
            ),
        })
    }
}
