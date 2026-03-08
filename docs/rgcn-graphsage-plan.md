# RGCN / GraphSAGE Implementation Plan

**Status**: Pre-implementation — schema audit required first
**Bident task**: `task_graphsage` (blocked → ready after snapshot + audit)
**Last updated**: 2026-02-26

---

## Decision Summary

| Question | Decision |
|----------|----------|
| Architecture | RGCN with basis decomposition |
| Training objective | Link prediction (self-supervised) |
| Role in compliance workflow | Fourth signal — augments Jina, does not replace it |
| Learning mode | Inductive (new nodes embeddable without retraining) |
| Node features | Jina V4 embeddings (2048-dim, shared across all node types) |
| Development database | `NL_graph_v0` snapshot — NL is never blocked |

---

## Why RGCN Over HGT

The NL graph is a **knowledge graph** (typed relations between entities). RGCN was
designed for this topology. HGT is better suited for graphs with fundamentally
different per-node-type feature spaces (images + text + tabular mixed). Here, all
node types share the same Jina 2048-dim embedding space — type differentiation is
already encoded in content. RGCN adds edge-type-aware aggregation on top of that.

At ~1767 nodes, HGT's attention mechanism is not cost-justified. RGCN is simpler,
more debuggable, and a better fit.

**HGT is the right upgrade path** when the graph exceeds ~50k nodes or when node
types develop genuinely different feature spaces.

---

## Schema Stability Rule

`num_relations` is a fixed architectural parameter. Adding a new **edge type**
changes tensor shapes and requires a full retrain. Adding new **nodes or edges of
existing types** is free — this is the inductive property.

**Node attributes** stored in ArangoDB documents are invisible to the model (Jina
embeddings are the features). New document fields never require retraining.

**Implication**: freeze the relation set before training. Define empty collections
for planned future edge types now so they get a relation slot without requiring a
rebuild when the first edge is added.

---

## Snapshot Database Workflow

NL is live and actively growing. The RGCN is trained against versioned snapshots.

```
NL (live — NL-Hecate keeps working)
    │
    └─► arangodump → NL_graph_v0  (first snapshot, created 2026-02-26)
                         │
                         ├─ fix schema issues (see audit below)
                         ├─ define empty future relation collections
                         ├─ freeze num_relations = N
                         ├─ train RGCN v0
                         ├─ evaluate link prediction quality
                         └─ iterate until stable
                                  │
                                  └─► promote schema decisions back to NL
                                               │
                                               └─► NL_graph_v1 → train production RGCN
```

### Creating a snapshot

```bash
# Snapshot NL → NL_graph_v0
arangodump --server.database NL \
           --server.password "$ARANGO_PASSWORD" \
           --output-directory /tmp/NL_snapshot_$(date +%Y%m%d)

arangorestore --server.database NL_graph_v0 \
              --server.password "$ARANGO_PASSWORD" \
              --create-database true \
              --input-directory /tmp/NL_snapshot_$(date +%Y%m%d)

# Or via hades (slower but auditable)
poetry run hades db create-database NL_graph_v0
```

---

## Schema Audit Findings (2026-02-26)

NL has **18 edge collections** with **3523 total edges** across **~1767 nodes**.
Three categories of issues were found.

### Category 1 — Fix before training (broken schema)

**`paper_edges` (109 edges)**
`_from` and `_to` are `null`. Actual references stored in custom fields `from_node`
and `to_node`, pointing to `arxiv_papers/...` — a collection that does not exist in
NL (correct name is `arxiv_metadata`). These documents are not connected to the
ArangoDB graph and cannot be used by RGCN or AQL graph traversal.

Fix: rewrite `_from`/`_to` using `from_node`/`to_node` values, mapping
`arxiv_papers/` → `arxiv_metadata/` where needed.

### Category 2 — Decision needed

| Edge collection | Count | Issue | Recommendation |
|-----------------|-------|-------|----------------|
| `persephone_edges` | 358 | Sessions → tasks audit trail | **Include** — provenance is a real signal: code files created during specific tasks, connected to compliance motivation |
| `nl_migration_edges` | 85 | `nl_axioms` → `hope_axioms` lineage | **Include** — encodes theoretical evolution; may be weak signal but costs nothing |
| `nl_build_path_edges` | 2 | `NL_PATH_HYBRID ↔ NL_PATH_PURE` only | **Exclude or merge** — 2 edges cannot meaningfully train a relation type |
| `nl_smell_compliance_edges` | 7 of 60 | 7 edges from `nl_build_paths`, 53 from `arxiv_metadata` | **Decision needed**: split into two relation types, or leave mixed and accept that `nl_build_paths` → `nl_code_smells` is a valid compliance relationship |

### Category 3 — Semantically sound, ready to include

| Edge collection | Count | Semantic role |
|-----------------|-------|---------------|
| `nl_equation_source_edges` | 629 | Equation derived from paper section |
| `nl_axiom_basis_edges` | 607 | Abstraction IS the basis for an axiom |
| `nl_validated_against_edges` | 606 | Abstraction validates against IS NOT axiom |
| `nl_hecate_trace_edges` | 355 | Spec traces to equation |
| `nl_equation_depends_edges` | 227 | Equation depends on equation |
| `nl_lineage_chain_edges` | 174 | Theoretical lineage across papers |
| `nl_smell_source_edges` | 163 | Smell grounded in paper text |
| `nl_cross_paper_edges` | 71 | Cross-paper concept relationship |
| `nl_smell_compliance_edges` | 60 | Source code complies with smell |
| `nl_signature_equation_edges` | 26 | Function signature implements equation |
| `nl_axiom_inherits_edges` | 16 | Axiom inherits from axiom |
| `nl_reframing_link_edges` | 16 | Concept reframing link |
| `nl_structural_embodiment_edges` | 13 | Code structurally embodies concept |
| `nl_definition_source_edges` | 6 | Definition sourced from paper |

Note: `nl_axiom_basis_edges` and `nl_validated_against_edges` look structurally
identical but encode opposite semantics (IS vs IS NOT). They must remain **separate
relation types**.

---

## Future Edge Types — Define Now, Populate Later

These are planned but not yet created. Defining empty collections before the first
training run locks in their relation slot at zero cost.

| Proposed collection | Semantic role | Priority |
|--------------------|---------------|----------|
| `nl_code_callgraph_edges` | File A calls function from file B | High — needed for method-level ingestion phase |
| `nl_code_equation_edges` | Function implements Eq-N | High — direct spec→code traceability |
| `nl_code_test_edges` | Test file covers this module | Medium |
| `nl_axiom_violation_edges` | Code file violates an axiom (negative signal) | High — needed for supervised fine-tuning later |

---

## Target Architecture

```
Input: Jina embeddings (2048-dim) — all node types share this space

Linear projection:  2048 → 256    (shared, trainable)
        │
RGCNConv(256, 256, num_relations=N, num_bases=30)
        │
ReLU + Dropout(0.2)
        │
RGCNConv(256, 128, num_relations=N, num_bases=30)
        │
Output: structural embedding (128-dim)   ← the fourth signal
```

Basis decomposition: `W_r = Σ_b a_{rb} · V_b` with B=30 shared bases.
Prevents parameter explosion: `30 × 256 × 256 + N × 30` vs `N × 256 × 256`.

Training: **link prediction** (self-supervised).
- Positive samples: existing edges
- Negative samples: randomly sampled non-edges
- Loss: binary cross-entropy on dot-product similarity between endpoint embeddings

The 128-dim structural embedding is stored back in ArangoDB alongside the Jina
embedding. In the compliance report it appears as a distinct fourth signal — not a
replacement for Jina.

---

## Implementation Phases

### Phase 0 — Snapshot + Schema Audit (prerequisite)
- [ ] Create `NL_graph_v0` via `arangodump`/`arangorestore`
- [ ] Fix `paper_edges` `_from`/`_to` null issue in snapshot
- [ ] Decide on `nl_build_path_edges` (exclude or keep)
- [ ] Decide on `nl_smell_compliance_edges` mixed `_from` types
- [ ] Create empty future edge collections in snapshot
- [ ] Freeze `num_relations = N` and document the relation index mapping
- [ ] Verify all edges resolve (no dangling references in training graph)

### Phase 1 — Data Pipeline
- [ ] ArangoDB → PyG `HeteroData` loader
  - Node feature extraction: Jina embeddings per collection
  - Edge index construction: one tensor per relation type
  - Node ID mapping: ArangoDB `_id` → integer index (bidirectional)
- [ ] Train/val/test edge split (80/10/10 on existing edges)
- [ ] Negative sampling strategy

### Phase 2 — Model Implementation
- [ ] `RGCNEncoder`: projection + 2× RGCNConv layers (PyG `RGCNConv`)
- [ ] Link prediction decoder: dot product + sigmoid
- [ ] Training loop with early stopping
- [ ] Embedding export: write 128-dim vectors back to ArangoDB

### Phase 3 — HADES Integration
- [ ] `hades graph train --database NL_graph_v0` — train and store embeddings
- [ ] `hades graph embed <node_id>` — get structural embedding for a node
- [ ] `hades graph neighbors <node_id> --k 10` — k-nearest by structural embedding
- [ ] Compliance report integration: add structural similarity as fourth signal

### Phase 4 — Production
- [ ] Create `NL_graph_v1` snapshot from stable NL
- [ ] Apply frozen schema to NL proper
- [ ] Train production RGCN on NL_graph_v1
- [ ] Periodic retraining schedule (suggested: on each major NL graph milestone)

---

## What Does NOT Require Retraining

| Change | Requires retrain? |
|--------|-----------------|
| New code file ingested (existing edge types) | No |
| New compliance edge added | No |
| New equation node added | No |
| New metadata field on existing document | No |
| New **edge type** (new collection) | **Yes** |
| Node feature dimension changes | **Yes** |
| Jina model upgraded (embedding space shifts) | **Yes** |

---

## Open Decisions (NL-Hecate input needed)

1. Should `nl_smell_compliance_edges` from `nl_build_paths` be a separate relation
   type (`nl_build_path_compliance_edges`) or merged with code file compliance?

2. Are `nl_code_callgraph_edges` and `nl_code_equation_edges` planned work within
   the next 6 months? If yes, define the empty collections before Phase 1 training.

3. Should `nl_migration_edges` (nl_axioms → hope_axioms) be included as a
   "theoretical evolution" relation, or excluded as historical provenance noise?

4. Periodic retraining trigger: on-demand only, or on a schedule, or on graph
   milestone events (e.g., every 500 new nodes)?
