//! Versioned guided-reading contracts. Participant labels are attribution, not credentials.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const FORMAT: &str = "hades-guided-reading/v1";
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Anchor {
    pub source_hash: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub passage_revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_session: Option<String>,
    /// Normalized left/top/width/height on the physical PDF page, for a figure.
    #[serde(default)]
    pub figure_box: Option<[f64; 4]>,
    pub representation: String,
    pub page: Option<u32>,
    pub block: Option<String>,
    pub exact: String,
    pub prefix: String,
    pub suffix: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Highlight {
    pub id: String,
    pub anchor: Anchor,
    pub label: String,
    pub removed: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Draft {
    pub title: String,
    pub note: String,
    pub is: Option<String>,
    pub is_not: Option<String>,
    pub standing: String,
    pub highlight: String,
    pub basis: String,
    /// Relationship endpoints must exist in the session's allowed graph collections.
    /// At least one endpoint must belong to this reading session.
    pub from: Option<String>,
    pub to: Option<String>,
    pub relation: Option<String>,
    pub inferred: bool,
    pub rationale: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub id: String,
    pub revision: u64,
    pub origin: String,
    pub editor: String,
    pub draft: Draft,
    pub status: String,
    pub created_ids: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Bound {
    pub is: String,
    pub is_not: String,
    pub revision: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub source_hash: String,
    pub graph: String,
    #[serde(default)]
    pub graph_collections: Vec<String>,
    pub revision: u64,
    pub stage: String,
    pub bound: Bound,
    pub highlights: BTreeMap<String, Highlight>,
    pub proposals: BTreeMap<String, Proposal>,
    pub focus: Vec<String>,
    pub focus_actor: String,
    pub pins: BTreeMap<String, Vec<Pin>>,
    pub last_event_hash: String,
    #[serde(default)]
    pub outline_order: Vec<String>,
    #[serde(default)]
    pub history_bytes: usize,
    #[serde(default)]
    pub corrections: BTreeMap<String, Vec<PassageRevision>>,
}
fn is_zero(value: &u64) -> bool {
    *value == 0
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassageRevision {
    pub revision: u64,
    pub text: String,
    pub actor: String,
    pub at: String,
    pub restored_from: Option<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Pin {
    pub reference: String,
    pub note: String,
    pub seen: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    CorrectPassage {
        block: String,
        passage_revision: u64,
        text: String,
        #[serde(default)]
        restore_revision: Option<u64>,
    },
    Bound {
        is: String,
        is_not: String,
    },
    Highlight {
        id: String,
        anchor: Anchor,
        label: String,
    },
    Recolor {
        id: String,
        label: String,
    },
    RemoveHighlight {
        id: String,
    },
    Focus {
        references: Vec<String>,
    },
    Pins {
        name: String,
        pins: Vec<Pin>,
    },
    Seen {
        name: String,
        index: usize,
    },
    Propose {
        id: String,
        draft: Draft,
    },
    Edit {
        id: String,
        proposal_revision: u64,
        draft: Draft,
    },
    Accept {
        id: String,
        proposal_revision: u64,
    },
    Reject {
        id: String,
        proposal_revision: u64,
    },
    OrderOutline {
        proposals: Vec<String>,
    },
    Advance,
    Revisit {
        stage: String,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mutation {
    pub session: String,
    pub request_id: String,
    pub expected_revision: u64,
    pub actor: String,
    pub action: Action,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadRequest {
    pub session: Option<String>,
    #[serde(default)]
    pub view: String,
    pub source_hash: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRequest {
    pub id: String,
    pub title: String,
    pub source_hash: String,
    pub actor: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceRequest {
    pub filename: String,
    pub content_base64: String,
}

pub fn token(s: &str) -> Result<(), String> {
    if s.is_empty()
        || s.len() > 96
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(
            "identifier must contain 1..96 ASCII letters, digits, underscores or hyphens".into(),
        );
    }
    Ok(())
}
fn text(s: &str, max: usize) -> Result<(), String> {
    if s.trim().is_empty() || s.len() > max {
        return Err(format!("text must contain 1..{max} bytes"));
    }
    Ok(())
}
fn label(s: &str) -> Result<(), String> {
    if !["node", "is", "is_not", "question"].contains(&s) {
        return Err("unknown reading/v1 palette label".into());
    }
    Ok(())
}
fn draft_valid(d: &Draft, s: &Session) -> Result<(), String> {
    text(&d.title, 256)?;
    text(&d.note, 12000)?;
    text(&d.basis, 2000)?;
    text(&d.standing, 256)?;
    for v in [&d.is, &d.is_not, &d.rationale].into_iter().flatten() {
        text(v, 4000)?;
    }
    if d.inferred && d.rationale.as_deref().is_none_or(|s| s.trim().is_empty()) {
        return Err("inferred proposals require a rationale".into());
    }
    if !s.highlights.get(&d.highlight).is_some_and(|h| !h.removed) {
        return Err("proposal needs an active human highlight".into());
    }
    match (&d.from, &d.to, &d.relation) {
        (None, None, None) => {}
        (Some(from), Some(to), Some(rel)) => {
            text(rel, 128)?;
            let local = format!("{}_nodes/", s.graph);
            if !from.starts_with(&local) && !to.starts_with(&local) {
                return Err("connect a note in this reading graph to an existing object".into());
            }
            for endpoint in [from, to] {
                let (collection, key) =
                    endpoint.split_once('/').ok_or("invalid graph object ID")?;
                token(collection)?;
                token(key)?;
                if collection != format!("{}_nodes", s.graph)
                    && !s.graph_collections.iter().any(|c| c == collection)
                {
                    return Err(
                        "endpoint collection was not included in this reading workspace".into(),
                    );
                }
            }
        }
        _ => return Err("a relationship requires from, to, and relation together".into()),
    }
    Ok(())
}

impl Session {
    pub fn new(p: &OpenRequest) -> Result<Self, String> {
        token(&p.id)?;
        text(&p.title, 256)?;
        text(&p.actor, 256)?;
        if p.id.len() > 32 {
            return Err("session ID maximum is 32 characters".into());
        }
        Ok(Self {
            id: p.id.clone(),
            title: p.title.clone(),
            source_hash: p.source_hash.clone(),
            graph: format!("reading_{}", p.id),
            graph_collections: vec![],
            revision: 0,
            stage: "bound".into(),
            bound: Bound::default(),
            highlights: BTreeMap::new(),
            proposals: BTreeMap::new(),
            focus: vec![],
            focus_actor: String::new(),
            pins: BTreeMap::new(),
            last_event_hash: String::new(),
            outline_order: vec![],
            history_bytes: 0,
            corrections: BTreeMap::new(),
        })
    }
    /// Apply to a private copy. The store atomically publishes it and the event.
    pub fn apply(&mut self, p: &Mutation, agent: bool) -> Result<Vec<Value>, String> {
        token(&p.request_id)?;
        text(&p.actor, 256)?;
        if self.revision != p.expected_revision {
            return Err("stale session revision; reload before changing it".into());
        }
        if self.revision >= 2000 {
            return Err(
                "session event limit reached; export this work before starting another session"
                    .into(),
            );
        }
        if agent && !matches!(p.action, Action::Propose { .. } | Action::Pins { .. }) {
            return Err("agents may propose or post pins, not accept or classify".into());
        }
        let mut writes = vec![];
        match &p.action {
            Action::CorrectPassage {
                block,
                passage_revision,
                text: corrected,
                restore_revision,
            } => {
                if block.is_empty() || block.len() > 512 || corrected.len() > 65536 {
                    return Err("passage ID or correction exceeds its size limit".into());
                }
                if self.passage_revision(block) != *passage_revision {
                    return Err("passage changed; reload before correcting it".into());
                }
                let versions = self.corrections.entry(block.clone()).or_default();
                if versions.len() >= 100 {
                    return Err("passage revision limit reached".into());
                }
                versions.push(PassageRevision {
                    revision: passage_revision + 1,
                    text: corrected.clone(),
                    actor: p.actor.clone(),
                    at: chrono::Utc::now().to_rfc3339(),
                    restored_from: *restore_revision,
                });
            }
            Action::Bound { is, is_not } => {
                text(is, 4000)?;
                text(is_not, 4000)?;
                self.bound = Bound {
                    is: is.clone(),
                    is_not: is_not.clone(),
                    revision: self.bound.revision + 1,
                };
                // Bounds are immutable graph revisions. Earlier notes retain their original basis.
                writes.push(json!({"_key":format!("bound_{}", self.bound.revision), "type":"document", "label":self.title, "text":format!("IS: {is}\nIS NOT: {is_not}"), "is":is,"is_not":is_not,"source_hash":self.source_hash,"bound_revision":self.bound.revision}));
            }
            Action::Highlight {
                id,
                anchor,
                label: l,
            } => {
                if self.stage == "bound" {
                    return Err("complete Bound before highlighting".into());
                }
                token(id)?;
                label(l)?;
                if let Some([x, y, w, h]) = anchor.figure_box {
                    if anchor.representation != "pdf-figure-v1"
                        || anchor.page.is_none_or(|p| p == 0)
                        || [x, y, w, h].iter().any(|v| !v.is_finite() || *v < 0.0)
                        || w <= 0.0
                        || h <= 0.0
                        || x + w > 1.000001
                        || y + h > 1.000001
                        || anchor.exact.len() > 12000
                    {
                        return Err("invalid figure page/box/caption".into());
                    }
                } else {
                    text(&anchor.exact, 12000)?;
                }
                if anchor.source_hash != self.source_hash
                    || anchor.prefix.len() > 500
                    || anchor.suffix.len() > 500
                {
                    return Err("invalid source anchor".into());
                }
                if anchor.passage_revision > 0 {
                    if anchor.representation != "docling-blocks-v1"
                        || anchor.correction_session.as_deref() != Some(self.id.as_str())
                    {
                        return Err(
                            "corrected passage anchor must identify this reading session".into(),
                        );
                    }
                } else if anchor.correction_session.is_some() {
                    return Err("original extraction anchors have no correction session".into());
                }
                if !["pdfjs-text-v1", "docling-blocks-v1", "pdf-figure-v1"]
                    .contains(&anchor.representation.as_str())
                {
                    return Err("unknown anchor representation".into());
                }
                if anchor.representation == "pdf-figure-v1" && anchor.figure_box.is_none() {
                    return Err("figure anchor needs a box".into());
                }
                if anchor.representation == "pdfjs-text-v1" && anchor.page.is_none_or(|v| v == 0) {
                    return Err("PDF anchor requires a physical page".into());
                }
                if anchor.representation == "docling-blocks-v1" && anchor.block.is_none() {
                    return Err("structured anchor requires a block ID".into());
                }
                if self.highlights.contains_key(id) {
                    return Err("highlight ID already exists".into());
                }
                self.highlights.insert(
                    id.clone(),
                    Highlight {
                        id: id.clone(),
                        anchor: anchor.clone(),
                        label: l.clone(),
                        removed: false,
                    },
                );
            }
            Action::Recolor { id, label: l } => {
                label(l)?;
                let h = self.highlights.get_mut(id).ok_or("highlight missing")?;
                if h.removed {
                    return Err("highlight removed".into());
                }
                h.label = l.clone();
            }
            Action::RemoveHighlight { id } => {
                self.highlights
                    .get_mut(id)
                    .ok_or("highlight missing")?
                    .removed = true;
            }
            Action::Focus { references } => {
                if references.len() > 6 {
                    return Err("focus accepts at most six references".into());
                }
                for r in references {
                    self.reference(r)?;
                }
                self.focus = references.clone();
                self.focus_actor = p.actor.clone();
            }
            Action::Pins { name, pins } => {
                token(name)?;
                if pins.len() > 50 || (self.pins.len() >= 20 && !self.pins.contains_key(name)) {
                    return Err("pin set limit reached".into());
                }
                for pin in pins {
                    self.reference(&pin.reference)?;
                    text(&pin.note, 1200)?;
                }
                self.pins.insert(name.clone(), pins.clone());
            }
            Action::Seen { name, index } => {
                self.pins
                    .get_mut(name)
                    .and_then(|ps| ps.get_mut(*index))
                    .ok_or("pin missing")?
                    .seen = true;
            }
            Action::Propose { id, draft } => {
                token(id)?;
                if id.len() > 64 {
                    return Err("proposal ID must fit 64 bytes for derived graph keys".into());
                }
                draft_valid(draft, self)?;
                if self.stage != "read" && self.stage != "connect" {
                    return Err("proposals require Read or Connect stage".into());
                }
                if self.proposals.contains_key(id) {
                    return Err("proposal ID already exists".into());
                }
                self.proposals.insert(
                    id.clone(),
                    Proposal {
                        id: id.clone(),
                        revision: 1,
                        origin: p.actor.clone(),
                        editor: p.actor.clone(),
                        draft: draft.clone(),
                        status: "pending".into(),
                        created_ids: vec![],
                    },
                );
            }
            Action::Edit {
                id,
                proposal_revision,
                draft,
            } => {
                draft_valid(draft, self)?;
                let proposal = self.pending(id, *proposal_revision)?;
                proposal.revision += 1;
                proposal.draft = draft.clone();
                proposal.editor = p.actor.clone();
            }
            Action::Accept {
                id,
                proposal_revision,
            }
            | Action::Reject {
                id,
                proposal_revision,
            } => {
                let accepting = matches!(p.action, Action::Accept { .. });
                let graph = self.graph.clone();
                let bound = self.bound.revision;
                let proposal = self.proposals.get(id).ok_or("proposal missing")?;
                if accepting {
                    draft_valid(&proposal.draft, self)?;
                }
                let anchor = self
                    .highlights
                    .get(&proposal.draft.highlight)
                    .ok_or("highlight missing")?
                    .anchor
                    .clone();
                let proposal = self.pending(id, *proposal_revision)?;
                if accepting {
                    let d = &proposal.draft;
                    let key = format!("note_{id}");
                    let evidence = json!({"anchor":anchor,"source_standing":d.standing,"inferred":d.inferred,"rationale":d.rationale});
                    let attrs = json!({"proposal_id":id,"proposal_revision":proposal_revision,"model_origin":proposal.origin,"editor":proposal.editor,"reviewer":p.actor,"evidence":evidence,"is":d.is,"is_not":d.is_not,"boundaries_unresolved":d.is.is_none() || d.is_not.is_none()});
                    if let (Some(from), Some(to), Some(relation)) = (&d.from, &d.to, &d.relation) {
                        writes.push(json!({"_key":key,"_from":from,"_to":to,"relation":relation,"label":relation,"text":d.note,"attrs":attrs}));
                        proposal.created_ids.push(format!("{graph}_edges/{key}"));
                    } else {
                        writes.push(json!({"_key":key,"type":"note","label":d.title,"text":d.note,"attrs":attrs}));
                        let node = format!("{graph}_nodes/{key}");
                        writes.push(json!({"_key":format!("basis_{id}"),"_from":node,"_to":format!("{graph}_nodes/bound_{bound}"),"relation":"basis","label":"basis","text":d.basis,"attrs":attrs}));
                        proposal
                            .created_ids
                            .extend([node, format!("{graph}_edges/basis_{id}")]);
                    }
                    proposal.status = "accepted".into();
                } else {
                    proposal.status = "rejected".into();
                }
            }
            Action::OrderOutline { proposals } => {
                if self.stage != "outline" {
                    return Err("outline ordering requires Outline stage".into());
                }
                let expected: std::collections::BTreeSet<_> = self
                    .proposals
                    .values()
                    .filter(|p| p.status == "accepted" && p.draft.from.is_none())
                    .map(|p| p.id.clone())
                    .collect();
                let supplied: std::collections::BTreeSet<_> = proposals.iter().cloned().collect();
                if expected != supplied || supplied.len() != proposals.len() {
                    return Err("outline must list each accepted note exactly once".into());
                }
                self.outline_order = proposals.clone();
            }
            Action::Revisit { stage } => {
                let stages = ["bound", "read", "connect", "outline"];
                let current = stages.iter().position(|s| *s == self.stage).unwrap();
                let target = stages
                    .iter()
                    .position(|s| *s == stage)
                    .ok_or("unknown stage")?;
                if target >= current {
                    return Err("revisit may only return to an earlier completed stage".into());
                }
                self.stage = stage.clone();
            }
            Action::Advance => {
                self.stage = match self.stage.as_str() {
                    "bound" if self.bound.revision > 0 => "read",
                    "read"
                        if !self.highlights.is_empty()
                            && self.proposals.values().all(|p| p.status != "pending") =>
                    {
                        "connect"
                    }
                    "connect"
                        if self.proposals.values().any(|p| p.status == "accepted")
                            && self.proposals.values().all(|p| p.status != "pending") =>
                    {
                        "outline"
                    }
                    "bound" => {
                        return Err(
                            "Save both IS and IS NOT boundaries before starting Read.".into()
                        );
                    }
                    "read" if self.highlights.is_empty() => {
                        return Err("Add a source highlight before completing Read.".into());
                    }
                    "read" => {
                        return Err(
                            "Accept or reject pending proposals before completing Read.".into()
                        );
                    }
                    "connect" if self.proposals.values().any(|p| p.status == "pending") => {
                        return Err(
                            "Accept or reject pending proposals before completing Connect.".into(),
                        );
                    }
                    "connect" => {
                        return Err(
                            "Accept at least one knowledge proposal before completing Connect."
                                .into(),
                        );
                    }
                    _ => return Err("Outline is the final reading stage.".into()),
                }
                .into();
            }
        }
        self.revision += 1;
        Ok(writes)
    }
    pub fn passage_revision(&self, block: &str) -> u64 {
        self.corrections
            .get(block)
            .and_then(|v| v.last())
            .map_or(0, |v| v.revision)
    }
    pub fn passage_text<'a>(
        &'a self,
        block: &str,
        revision: u64,
        original: &'a str,
    ) -> Result<&'a str, String> {
        if revision == 0 {
            return Ok(original);
        }
        self.corrections
            .get(block)
            .and_then(|v| v.iter().find(|v| v.revision == revision))
            .map(|v| v.text.as_str())
            .ok_or_else(|| "passage revision does not exist".into())
    }
    pub fn reference_warnings(&self) -> Vec<Value> {
        self.highlights.values().filter_map(|h|{
            let block=h.anchor.block.as_deref()?;
            let current=self.passage_revision(block);
            (h.anchor.representation=="docling-blocks-v1" && current!=h.anchor.passage_revision).then(||json!({"highlight":h.id,"block":block,"anchored_revision":h.anchor.passage_revision,"current_revision":current}))
        }).collect()
    }
    fn pending(&mut self, id: &str, revision: u64) -> Result<&mut Proposal, String> {
        let p = self.proposals.get_mut(id).ok_or("proposal missing")?;
        if p.status != "pending" || p.revision != revision {
            return Err("proposal already decided or revision is stale".into());
        }
        Ok(p)
    }
    fn reference(&self, r: &str) -> Result<(), String> {
        if self.highlights.contains_key(r)
            || self.proposals.contains_key(r)
            || self
                .proposals
                .values()
                .any(|p| p.created_ids.iter().any(|id| id == r))
        {
            Ok(())
        } else {
            Err(
                "reference is not a highlight, proposal, or accepted graph object in this session"
                    .into(),
            )
        }
    }
}
