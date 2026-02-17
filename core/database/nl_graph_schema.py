"""Named graph schema for the NL (New Logic) knowledge base.

Defines ArangoDB edge collections and named graphs that materialize
the cross-reference relationships embedded in NL document fields.

The NL database contains 84 collections organized by paper and concept type.
Documents reference each other via fields like ``axiom_basis``, ``depends_on``,
``inherits_from``, etc. This module maps those implicit relationships to
explicit ArangoDB edge collections suitable for graph traversal.

Vertex collections (already exist as document collections):
    - Paper-scoped: {paper}_{type} (e.g., hope_equations, atlas_definitions)
    - NL-global: nl_axioms, nl_reframings, nl_build_paths, nl_code_smells, ...
    - Cross-paper: paper_edges, hecate_specs, builds, build_runs

Edge collections (to be materialized):
    - One edge collection per relationship type
    - Edges carry ``_from``, ``_to``, and optional metadata

Named graphs compose edge collections into traversable units:
    - nl_core: Universal axiom backbone (axiom_basis + validated_against + inherits_from)
    - nl_equations: Equation dependency network
    - nl_hierarchy: Concept hierarchy (structural embodiments + lineage chains)
    - nl_hecate: Build spec traceability
    - nl_cross_paper: Cross-paper relationships
    - nl_concept_map: Full knowledge graph (all edges)

Usage:
    from core.database.nl_graph_schema import NL_GRAPH_SCHEMA, get_edge_definitions

    # Get edge collection definitions for materialization
    for edge_def in NL_GRAPH_SCHEMA.edge_collections:
        print(edge_def.name, edge_def.source_field)

    # Get named graph payloads for ArangoDB /_api/gharial
    for graph in NL_GRAPH_SCHEMA.named_graphs:
        payload = graph.to_gharial_payload()
        client.request("POST", f"/_db/NL/_api/gharial", json=payload)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Paper prefixes — each paper contributes a family of typed collections
# ---------------------------------------------------------------------------

PAPER_PREFIXES = [
    "hope",
    "atlas",
    "lattice",
    "miras",
    "titans",
    "titans_revisited",
    "tnt",
    "trellis",
]

CONCEPT_TYPES = [
    "abstractions",
    "algorithms",
    "axioms",
    "definitions",
    "equations",
    "lineage",
]

# All paper-scoped collections that exist in the NL database.
# Not every paper has every type (e.g., only hope has algorithms).
# The materialization step validates existence before creating edges.
PAPER_COLLECTIONS: list[str] = []
for _prefix in PAPER_PREFIXES:
    for _ctype in CONCEPT_TYPES:
        PAPER_COLLECTIONS.append(f"{_prefix}_{_ctype}")

# Additional paper-scoped collections not covered by the matrix
EXTRA_PAPER_COLLECTIONS = [
    "hope_assumptions",
    "hope_blockers",
    "hope_code_smells",
    "hope_context_sources",
    "hope_examples",
    "hope_extensions",
    "hope_nl_reframings",
    "hope_optimizers",
    "hope_probes",
    "hope_python_signatures",
    "hope_triton_specs",
    "hope_validation_reports",
    "conveyance_definitions",
    "conveyance_equations",
    "conveyance_hypotheses",
    "conveyance_philosophy",
    "conveyance_protocols",
    "community_implementations",
]

# NL-global collections (not paper-scoped)
NL_GLOBAL_COLLECTIONS = [
    "nl_axioms",
    "nl_reframings",
    "nl_build_paths",
    "nl_code_smells",
    "nl_ethnographic_notes",
    "nl_ethnography",
    "nl_optimizers",
    "nl_probe_patterns",
    "nl_roadmap",
    "nl_system",
    "nl_toolchain",
    "nl_articles",
]

# Infrastructure collections
INFRA_COLLECTIONS = [
    "paper_edges",
    "hecate_specs",
    "builds",
    "build_runs",
]

# All vertex collections (union of all groups)
ALL_VERTEX_COLLECTIONS = sorted(
    set(PAPER_COLLECTIONS + EXTRA_PAPER_COLLECTIONS + NL_GLOBAL_COLLECTIONS + INFRA_COLLECTIONS)
)

# Collections that contain axioms (sources for axiom_basis edges)
AXIOM_COLLECTIONS = ["nl_axioms"] + [f"{p}_axioms" for p in PAPER_PREFIXES]

# Collections that contain equations
EQUATION_COLLECTIONS = [f"{p}_equations" for p in PAPER_PREFIXES] + ["conveyance_equations"]

# Collections that contain definitions
DEFINITION_COLLECTIONS = [f"{p}_definitions" for p in PAPER_PREFIXES] + ["conveyance_definitions"]

# Collections that can be axiom_basis targets (any doc can reference axioms)
AXIOM_COMPLIANT_COLLECTIONS = sorted(set(ALL_VERTEX_COLLECTIONS) - {"paper_edges", "builds", "build_runs"})


# ---------------------------------------------------------------------------
# Edge collection definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeCollectionDef:
    """Definition of an edge collection to be materialized.

    Attributes:
        name: ArangoDB edge collection name.
        source_field: Document field that holds the reference(s).
            Can be a string field (single ref) or list field (multiple refs).
        from_collections: Vertex collections that can appear as _from.
        to_collections: Vertex collections that can appear as _to.
        description: Human-readable description of this relationship.
        is_array: Whether source_field holds a list of references.
        edge_attributes: Additional fields to copy from the source doc onto edges.
        extraction_note: How to resolve the reference (full ID vs bare key).
    """

    name: str
    source_field: str
    from_collections: list[str]
    to_collections: list[str]
    description: str
    is_array: bool = False
    edge_attributes: list[str] = field(default_factory=list)
    extraction_note: str = "full_id"


@dataclass(frozen=True)
class NamedGraphDef:
    """Definition of an ArangoDB named graph.

    Attributes:
        name: Graph name (used in GRAPH "name" AQL clauses).
        edge_definitions: List of edge definitions composing this graph.
        description: Human-readable purpose.
    """

    name: str
    edge_definitions: list[EdgeCollectionDef]
    description: str

    def to_gharial_payload(self) -> dict[str, Any]:
        """Build the JSON payload for POST /_api/gharial.

        Returns:
            Dict with ``name`` and ``edgeDefinitions`` keys.
        """
        seen: dict[str, dict[str, set[str]]] = {}
        for edef in self.edge_definitions:
            if edef.name not in seen:
                seen[edef.name] = {"from": set(), "to": set()}
            seen[edef.name]["from"].update(edef.from_collections)
            seen[edef.name]["to"].update(edef.to_collections)

        edge_defs = []
        for coll_name, directions in seen.items():
            edge_defs.append(
                {
                    "collection": coll_name,
                    "from": sorted(directions["from"]),
                    "to": sorted(directions["to"]),
                }
            )

        return {
            "name": self.name,
            "edgeDefinitions": edge_defs,
        }


@dataclass(frozen=True)
class NLGraphSchema:
    """Complete graph schema for the NL knowledge base."""

    edge_collections: list[EdgeCollectionDef]
    named_graphs: list[NamedGraphDef]

    def get_edge_collection(self, name: str) -> EdgeCollectionDef:
        """Look up an edge collection definition by name."""
        for ec in self.edge_collections:
            if ec.name == name:
                return ec
        available = ", ".join(ec.name for ec in self.edge_collections)
        raise KeyError(f"Unknown edge collection: {name!r}. Available: {available}")

    def get_named_graph(self, name: str) -> NamedGraphDef:
        """Look up a named graph definition by name."""
        for ng in self.named_graphs:
            if ng.name == name:
                return ng
        available = ", ".join(ng.name for ng in self.named_graphs)
        raise KeyError(f"Unknown named graph: {name!r}. Available: {available}")

    def all_edge_collection_names(self) -> list[str]:
        """Return sorted list of all edge collection names."""
        return sorted({ec.name for ec in self.edge_collections})

    def all_named_graph_names(self) -> list[str]:
        """Return sorted list of all named graph names."""
        return sorted(ng.name for ng in self.named_graphs)


# ---------------------------------------------------------------------------
# Edge collection instances
# ---------------------------------------------------------------------------

# 1. axiom_basis — Nearly every doc references its IS axiom
AXIOM_BASIS = EdgeCollectionDef(
    name="nl_axiom_basis_edges",
    source_field="axiom_basis",
    from_collections=AXIOM_COMPLIANT_COLLECTIONS,
    to_collections=AXIOM_COLLECTIONS,
    description=(
        "Links any NL concept to the axiom it embodies. "
        "The most universal edge in the graph — nearly every document has one. "
        "Values are full ArangoDB IDs like 'hope_axioms/container-is'."
    ),
    extraction_note="full_id",
)

# 2. validated_against — Paired with axiom_basis, references IS_NOT axiom
VALIDATED_AGAINST = EdgeCollectionDef(
    name="nl_validated_against_edges",
    source_field="validated_against",
    from_collections=AXIOM_COMPLIANT_COLLECTIONS,
    to_collections=AXIOM_COLLECTIONS,
    description=(
        "Links any NL concept to the anti-axiom (IS_NOT) it was validated against. "
        "Nearly always paired with axiom_basis. "
        "Values are full ArangoDB IDs like 'hope_axioms/container-is-not'."
    ),
    extraction_note="full_id",
)

# 3. inherits_from — Paper axioms inherit from NL root axioms
AXIOM_INHERITS = EdgeCollectionDef(
    name="nl_axiom_inherits_edges",
    source_field="inherits_from",
    from_collections=[f"{p}_axioms" for p in PAPER_PREFIXES],
    to_collections=["nl_axioms"],
    description=(
        "Paper-level axiom inheritance. Each {paper}_axioms doc inherits from "
        "nl_axioms/NL_IS or nl_axioms/NL_IS_NOT, forming a three-tier hierarchy: "
        "nl_axioms → {paper}_axioms → all paper docs."
    ),
    extraction_note="full_id",
)

# 4. structural_embodiments — Axiom IS containers point to definitions
STRUCTURAL_EMBODIMENT = EdgeCollectionDef(
    name="nl_structural_embodiment_edges",
    source_field="structural_embodiments",
    from_collections=[f"{p}_axioms" for p in PAPER_PREFIXES],
    to_collections=DEFINITION_COLLECTIONS,
    description=(
        "Links an axiom IS container to the definitions that structurally embody it. "
        "Array field — one axiom may have many embodiments. "
        "Values are full IDs like 'atlas_definitions/def-associative-memory'."
    ),
    is_array=True,
    extraction_note="full_id",
)

# 5. depends_on — Equation-to-equation dependencies
EQUATION_DEPENDS = EdgeCollectionDef(
    name="nl_equation_depends_edges",
    source_field="depends_on",
    from_collections=EQUATION_COLLECTIONS,
    to_collections=EQUATION_COLLECTIONS,
    description=(
        "Equation dependency graph. An equation depends_on other equations. "
        "Array field with full IDs like 'hope_equations/eq-006-assoc-mem-obj'. "
        "Forms the computational dependency DAG within each paper."
    ),
    is_array=True,
    extraction_note="full_id",
)

# 6. source_equation — Definition derived from an equation
DEFINITION_SOURCE_EQ = EdgeCollectionDef(
    name="nl_definition_source_edges",
    source_field="source_equation",
    from_collections=DEFINITION_COLLECTIONS,
    to_collections=EQUATION_COLLECTIONS,
    description=(
        "Links a definition to the equation it derives from. "
        "Single reference field with full ID like 'hope_equations/eq-006-assoc-mem-obj'."
    ),
    extraction_note="full_id",
)

# 7. linked_equation — Python signatures linked to equations
SIGNATURE_EQUATION = EdgeCollectionDef(
    name="nl_signature_equation_edges",
    source_field="linked_equation",
    from_collections=["hope_python_signatures", "hope_algorithms"],
    to_collections=EQUATION_COLLECTIONS,
    description=(
        "Links a Python signature or algorithm to the equation it implements. " "Single reference field with full ID."
    ),
    extraction_note="full_id",
)

# 8. nl_reframing_link — Concepts linked to their NL reframings
NL_REFRAMING = EdgeCollectionDef(
    name="nl_reframing_link_edges",
    source_field="nl_reframing_link",
    from_collections=AXIOM_COMPLIANT_COLLECTIONS,
    to_collections=["nl_reframings", "hope_nl_reframings"],
    description=(
        "Links any concept to its NL reframing — the philosophical lens "
        "through which the concept is understood in the NL paradigm."
    ),
    extraction_note="full_id",
)

# 9. migrated_from — Provenance from hope_* to nl_* promotion
MIGRATION_PROVENANCE = EdgeCollectionDef(
    name="nl_migration_edges",
    source_field="migrated_from",
    from_collections=NL_GLOBAL_COLLECTIONS,
    to_collections=EXTRA_PAPER_COLLECTIONS + PAPER_COLLECTIONS,
    description=(
        "1:1 provenance edge from promoted nl_* docs back to their original "
        "hope_* source. E.g., nl_reframings/x → hope_nl_reframings/y. "
        "Preserves migration history."
    ),
    extraction_note="full_id",
)

# 10. lineage chain — Ordered concept chains (lineage collections)
LINEAGE_CHAIN = EdgeCollectionDef(
    name="nl_lineage_chain_edges",
    source_field="chain",
    from_collections=[f"{p}_lineage" for p in PAPER_PREFIXES],
    to_collections=sorted(
        set([f"{p}_abstractions" for p in PAPER_PREFIXES] + EQUATION_COLLECTIONS + DEFINITION_COLLECTIONS)
    ),
    description=(
        "Ordered chain of concepts forming a lineage. Each lineage doc has a "
        "'chain' array of full IDs. Materialized as sequential edges: "
        "chain[0]→chain[1]→chain[2]... with the lineage doc as edge metadata. "
        "Also creates edges from the lineage doc to each chain member."
    ),
    is_array=True,
    extraction_note="full_id",
    edge_attributes=["name", "type", "description"],
)

# 11. traced_to_equations — Hecate spec traceability to equations
HECATE_TRACE_EQ = EdgeCollectionDef(
    name="nl_hecate_trace_edges",
    source_field="traced_to_equations",
    from_collections=["hecate_specs"],
    to_collections=EQUATION_COLLECTIONS,
    description=(
        "Hecate build spec traceability to equations. Array of full IDs. "
        "Links implementation specs to the mathematical foundations they implement."
    ),
    is_array=True,
    extraction_note="full_id",
)

# 12. traced_to_algorithms — Hecate spec traceability to algorithms
HECATE_TRACE_ALG = EdgeCollectionDef(
    name="nl_hecate_trace_edges",
    source_field="traced_to_algorithms",
    from_collections=["hecate_specs"],
    to_collections=["hope_algorithms"],
    description=("Hecate build spec traceability to algorithms. Array of full IDs."),
    is_array=True,
    extraction_note="full_id",
)

# 13. traced_to_axioms — Hecate spec traceability to axioms
HECATE_TRACE_AXIOM = EdgeCollectionDef(
    name="nl_hecate_trace_edges",
    source_field="traced_to_axioms",
    from_collections=["hecate_specs"],
    to_collections=AXIOM_COLLECTIONS,
    description=("Hecate build spec traceability to axioms. Array of full IDs."),
    is_array=True,
    extraction_note="full_id",
)

# 14. paper_edges — Cross-paper relationships (from_node / to_node)
CROSS_PAPER = EdgeCollectionDef(
    name="nl_cross_paper_edges",
    source_field="from_node",
    from_collections=ALL_VERTEX_COLLECTIONS,
    to_collections=ALL_VERTEX_COLLECTIONS,
    description=(
        "Cross-paper concept relationships from the paper_edges collection. "
        "Each doc has from_node and to_node fields with full IDs. "
        "These are already explicit edges but stored as documents — "
        "materialization converts them to native ArangoDB edges."
    ),
    edge_attributes=["name", "description", "relationship", "source_paper", "target_paper"],
    extraction_note="full_id",
)

# 15. smell_compliance — Code smell compliance references
SMELL_COMPLIANCE = EdgeCollectionDef(
    name="nl_smell_compliance_edges",
    source_field="smell_compliance",
    from_collections=AXIOM_COMPLIANT_COLLECTIONS,
    to_collections=["nl_code_smells", "hope_code_smells"],
    description=(
        "Links concepts to code smell compliance records. " "Can be a single ID or a dict with smell IDs as values."
    ),
    extraction_note="full_id",
)

# 16. counterpart — Build path cross-references
BUILD_PATH_COUNTERPART = EdgeCollectionDef(
    name="nl_build_path_edges",
    source_field="counterpart",
    from_collections=["nl_build_paths"],
    to_collections=["nl_reframings"],
    description=("Links a build path to its counterpart NL reframing."),
    extraction_note="full_id",
)


# ---------------------------------------------------------------------------
# Collect all edge definitions
# ---------------------------------------------------------------------------

ALL_EDGE_COLLECTIONS: list[EdgeCollectionDef] = [
    AXIOM_BASIS,
    VALIDATED_AGAINST,
    AXIOM_INHERITS,
    STRUCTURAL_EMBODIMENT,
    EQUATION_DEPENDS,
    DEFINITION_SOURCE_EQ,
    SIGNATURE_EQUATION,
    NL_REFRAMING,
    MIGRATION_PROVENANCE,
    LINEAGE_CHAIN,
    HECATE_TRACE_EQ,
    HECATE_TRACE_ALG,
    HECATE_TRACE_AXIOM,
    CROSS_PAPER,
    SMELL_COMPLIANCE,
    BUILD_PATH_COUNTERPART,
]


# ---------------------------------------------------------------------------
# Named graph definitions
# ---------------------------------------------------------------------------

# Core axiom backbone — the universal IS / IS_NOT compliance graph
NL_CORE_GRAPH = NamedGraphDef(
    name="nl_core",
    edge_definitions=[AXIOM_BASIS, VALIDATED_AGAINST, AXIOM_INHERITS],
    description=(
        "The axiom compliance backbone. Every NL concept links to its IS axiom "
        "(axiom_basis) and IS_NOT anti-axiom (validated_against). Paper axioms "
        "inherit from nl_axioms. This is the most connected subgraph."
    ),
)

# Equation dependency network
NL_EQUATIONS_GRAPH = NamedGraphDef(
    name="nl_equations",
    edge_definitions=[EQUATION_DEPENDS, DEFINITION_SOURCE_EQ, SIGNATURE_EQUATION],
    description=(
        "The equation dependency DAG. Equations depend on other equations, "
        "definitions derive from equations, and Python signatures implement equations."
    ),
)

# Concept hierarchy — structural embodiments, lineage chains
NL_HIERARCHY_GRAPH = NamedGraphDef(
    name="nl_hierarchy",
    edge_definitions=[STRUCTURAL_EMBODIMENT, LINEAGE_CHAIN],
    description=(
        "The concept hierarchy. Axiom IS containers embody definitions, "
        "and lineage chains trace ordered concept progressions."
    ),
)

# Hecate traceability — build specs to math foundations
NL_HECATE_GRAPH = NamedGraphDef(
    name="nl_hecate",
    edge_definitions=[HECATE_TRACE_EQ, HECATE_TRACE_ALG, HECATE_TRACE_AXIOM],
    description=(
        "Hecate build spec traceability. Links implementation specifications "
        "to the equations, algorithms, and axioms they implement."
    ),
)

# Cross-paper relationships
NL_CROSS_PAPER_GRAPH = NamedGraphDef(
    name="nl_cross_paper",
    edge_definitions=[CROSS_PAPER],
    description=(
        "Cross-paper concept relationships. Maps how ideas from one paper "
        "(e.g., HOPE) connect to concepts in another (e.g., Titans, ATLAS)."
    ),
)

# Full knowledge graph — everything
NL_CONCEPT_MAP = NamedGraphDef(
    name="nl_concept_map",
    edge_definitions=ALL_EDGE_COLLECTIONS,
    description=(
        "The complete NL knowledge graph. Composes all edge collections into "
        "a single traversable graph. Use for broad exploration queries like "
        "'show me everything related to this concept'."
    ),
)

ALL_NAMED_GRAPHS: list[NamedGraphDef] = [
    NL_CORE_GRAPH,
    NL_EQUATIONS_GRAPH,
    NL_HIERARCHY_GRAPH,
    NL_HECATE_GRAPH,
    NL_CROSS_PAPER_GRAPH,
    NL_CONCEPT_MAP,
]


# ---------------------------------------------------------------------------
# Schema singleton
# ---------------------------------------------------------------------------

NL_GRAPH_SCHEMA = NLGraphSchema(
    edge_collections=ALL_EDGE_COLLECTIONS,
    named_graphs=ALL_NAMED_GRAPHS,
)


def get_edge_definitions() -> list[EdgeCollectionDef]:
    """Return all edge collection definitions."""
    return list(NL_GRAPH_SCHEMA.edge_collections)


def get_named_graphs() -> list[NamedGraphDef]:
    """Return all named graph definitions."""
    return list(NL_GRAPH_SCHEMA.named_graphs)
