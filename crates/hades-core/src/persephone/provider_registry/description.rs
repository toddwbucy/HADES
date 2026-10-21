//! Strict observations for one registered artifact and its independently served workloads.

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::embedding_profile::{
    Capabilities, ChunkPolicy, ContractError, EmbeddingSpace, ModelIdentity, ProviderReceipt,
    Workload, WorkloadKind, require,
};

/// The original unversioned description retains its single text-space observation.
/// V2 explicitly advertises each workload; neither form implies database intent.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ProviderDescription {
    Single {
        model: ModelIdentity,
        capabilities: Capabilities,
        space: Box<EmbeddingSpace>,
        receipt: ProviderReceipt,
    },
    Workloads {
        format: DescriptionFormat,
        model: ModelIdentity,
        workloads: Box<ObservedWorkloads>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DescriptionFormat {
    #[serde(rename = "hades-provider-description-v2")]
    V2,
}

/// Capabilities belong to this workload, not every task sharing the loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedWorkload {
    pub capabilities: Capabilities,
    pub space: EmbeddingSpace,
    pub receipt: ProviderReceipt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedWorkloads {
    pub documents: Option<ObservedWorkload>,
    pub code: Option<ObservedWorkload>,
}

/// Borrow a workload uniformly without reserializing or inventing a second observation.
#[derive(Debug, Clone, Copy)]
pub struct WorkloadObservation<'a> {
    pub capabilities: &'a Capabilities,
    pub space: &'a EmbeddingSpace,
    pub receipt: &'a ProviderReceipt,
}

impl ProviderDescription {
    /// All advertised workloads use this single registered model and tokenizer dependency set.
    pub fn model(&self) -> &ModelIdentity {
        match self {
            Self::Single { model, .. } | Self::Workloads { model, .. } => model,
        }
    }

    /// Inspect the actual wire declaration, independently of any database policy.
    /// The original single-space format has one Documents observation; use workload()
    /// to resolve its compatibility with an explicit ordinary-text code policy.
    pub fn advertised_workload(&self, kind: WorkloadKind) -> Option<WorkloadObservation<'_>> {
        match self {
            Self::Single {
                capabilities,
                space,
                receipt,
                ..
            } if kind == WorkloadKind::Documents => Some(WorkloadObservation {
                capabilities,
                space,
                receipt,
            }),
            Self::Single { .. } => None,
            Self::Workloads { workloads, .. } => {
                let observed = match kind {
                    WorkloadKind::Documents => workloads.documents.as_ref(),
                    WorkloadKind::Code => workloads.code.as_ref(),
                }?;
                Some(WorkloadObservation {
                    capabilities: &observed.capabilities,
                    space: &observed.space,
                    receipt: &observed.receipt,
                })
            }
        }
    }

    /// Resolve and check an enabled policy using the same compatibility rules as client binding.
    /// V2 omissions never fall back; original text observations may serve ordinary-text code
    /// only when its saved space and required capabilities match exactly.
    pub fn workload(
        &self,
        kind: WorkloadKind,
        policy: &Workload,
    ) -> Result<WorkloadObservation<'_>, ContractError> {
        let advertised_kind = match (self, kind, policy.specialized_code) {
            (Self::Single { .. }, WorkloadKind::Code, false) => WorkloadKind::Documents,
            _ => kind,
        };
        let observed = self.advertised_workload(advertised_kind).ok_or_else(|| {
            ContractError(format!("provider does not advertise the {kind:?} workload"))
        })?;
        require(
            &policy.space == observed.space,
            "provider embedding behavior differs from the saved workload",
        )?;
        let capabilities = observed.capabilities;
        require(
            capabilities.text
                && capabilities.tasks.contains(&policy.space.passage.task)
                && capabilities.tasks.contains(&policy.space.query.task)
                && (!policy.specialized_code
                    || (kind == WorkloadKind::Code && capabilities.specialized_code)),
            "provider no longer supports an enabled workload",
        )?;
        if matches!(policy.chunking, ChunkPolicy::Late { .. }) {
            require(
                capabilities.late_chunking && capabilities.tokenizer_offsets,
                "provider no longer supports the saved late-chunk policy",
            )?;
        }
        Ok(observed)
    }
}

/// Deserialize directly into typed fields, retaining duplicate/unknown-field checks.
/// An untagged deserialize or an intermediate JSON Value could obscure wire errors.
impl<'de> Deserialize<'de> for ProviderDescription {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            model: ModelIdentity,
            #[serde(default, deserialize_with = "present")]
            format: Option<DescriptionFormat>,
            #[serde(default, deserialize_with = "present")]
            capabilities: Option<Capabilities>,
            #[serde(default, deserialize_with = "present")]
            space: Option<EmbeddingSpace>,
            #[serde(default, deserialize_with = "present")]
            receipt: Option<ProviderReceipt>,
            #[serde(default, deserialize_with = "present")]
            workloads: Option<ObservedWorkloads>,
        }
        let wire = Wire::deserialize(deserializer)?;
        match (
            wire.format,
            wire.capabilities,
            wire.space,
            wire.receipt,
            wire.workloads,
        ) {
            (None, Some(capabilities), Some(space), Some(receipt), None) => Ok(Self::Single {
                model: wire.model,
                capabilities,
                space: Box::new(space),
                receipt,
            }),
            (Some(format), None, None, None, Some(workloads)) => Ok(Self::Workloads {
                format,
                model: wire.model,
                workloads: Box::new(workloads),
            }),
            _ => Err(de::Error::custom(
                "provider description must use exactly one complete wire format",
            )),
        }
    }
}

/// Optional format-specific fields may be absent, but explicit null is not a valid field value.
fn present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}
