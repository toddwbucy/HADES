"""Tests for core.database.nl_graph_schema — NL named graph schema definitions."""

import pytest

from core.database.nl_graph_schema import (
    ALL_EDGE_COLLECTIONS,
    ALL_NAMED_GRAPHS,
    ALL_VERTEX_COLLECTIONS,
    AXIOM_COLLECTIONS,
    EQUATION_COLLECTIONS,
    NL_GRAPH_SCHEMA,
    PAPER_PREFIXES,
    EdgeCollectionDef,
    NamedGraphDef,
    get_edge_definitions,
    get_named_graphs,
)


class TestPaperPrefixes:
    def test_known_papers_present(self):
        assert "hope" in PAPER_PREFIXES
        assert "atlas" in PAPER_PREFIXES
        assert "titans" in PAPER_PREFIXES
        assert "tnt" in PAPER_PREFIXES

    def test_no_duplicates(self):
        assert len(PAPER_PREFIXES) == len(set(PAPER_PREFIXES))


class TestCollectionLists:
    def test_vertex_collections_not_empty(self):
        assert len(ALL_VERTEX_COLLECTIONS) > 50

    def test_vertex_collections_sorted(self):
        assert ALL_VERTEX_COLLECTIONS == sorted(ALL_VERTEX_COLLECTIONS)

    def test_vertex_collections_no_duplicates(self):
        assert len(ALL_VERTEX_COLLECTIONS) == len(set(ALL_VERTEX_COLLECTIONS))

    def test_axiom_collections_include_nl(self):
        assert "nl_axioms" in AXIOM_COLLECTIONS

    def test_axiom_collections_include_papers(self):
        for prefix in PAPER_PREFIXES:
            assert f"{prefix}_axioms" in AXIOM_COLLECTIONS

    def test_equation_collections_include_papers(self):
        for prefix in PAPER_PREFIXES:
            assert f"{prefix}_equations" in EQUATION_COLLECTIONS


class TestEdgeCollectionDef:
    def test_frozen(self):
        edef = EdgeCollectionDef(
            name="test_edges",
            source_field="ref",
            from_collections=["a"],
            to_collections=["b"],
            description="test",
        )
        with pytest.raises(AttributeError):
            edef.name = "other"

    def test_default_values(self):
        edef = EdgeCollectionDef(
            name="test_edges",
            source_field="ref",
            from_collections=["a"],
            to_collections=["b"],
            description="test",
        )
        assert edef.is_array is False
        assert edef.edge_attributes == []
        assert edef.extraction_note == "full_id"


class TestNamedGraphDef:
    def test_to_gharial_payload_structure(self):
        edef = EdgeCollectionDef(
            name="test_edges",
            source_field="ref",
            from_collections=["src_a", "src_b"],
            to_collections=["tgt_a"],
            description="test",
        )
        graph = NamedGraphDef(
            name="test_graph",
            edge_definitions=[edef],
            description="test graph",
        )
        payload = graph.to_gharial_payload()
        assert payload["name"] == "test_graph"
        assert len(payload["edgeDefinitions"]) == 1
        assert payload["edgeDefinitions"][0]["collection"] == "test_edges"
        assert sorted(payload["edgeDefinitions"][0]["from"]) == ["src_a", "src_b"]
        assert payload["edgeDefinitions"][0]["to"] == ["tgt_a"]

    def test_shared_edge_collection_merges(self):
        """Edge definitions sharing a collection name should merge from/to sets."""
        edef1 = EdgeCollectionDef(
            name="shared_edges",
            source_field="field_a",
            from_collections=["src_a"],
            to_collections=["tgt_a"],
            description="first",
        )
        edef2 = EdgeCollectionDef(
            name="shared_edges",
            source_field="field_b",
            from_collections=["src_b"],
            to_collections=["tgt_b"],
            description="second",
        )
        graph = NamedGraphDef(
            name="merged_graph",
            edge_definitions=[edef1, edef2],
            description="test",
        )
        payload = graph.to_gharial_payload()
        assert len(payload["edgeDefinitions"]) == 1
        edge_def = payload["edgeDefinitions"][0]
        assert edge_def["collection"] == "shared_edges"
        assert "src_a" in edge_def["from"]
        assert "src_b" in edge_def["from"]
        assert "tgt_a" in edge_def["to"]
        assert "tgt_b" in edge_def["to"]


class TestNLGraphSchema:
    def test_singleton_has_edge_collections(self):
        assert len(NL_GRAPH_SCHEMA.edge_collections) == 16

    def test_singleton_has_named_graphs(self):
        assert len(NL_GRAPH_SCHEMA.named_graphs) == 6

    def test_unique_edge_collection_names(self):
        names = NL_GRAPH_SCHEMA.all_edge_collection_names()
        assert len(names) == 14  # 16 defs but some share names (hecate)

    def test_named_graph_names(self):
        names = NL_GRAPH_SCHEMA.all_named_graph_names()
        assert "nl_core" in names
        assert "nl_equations" in names
        assert "nl_hierarchy" in names
        assert "nl_hecate" in names
        assert "nl_cross_paper" in names
        assert "nl_concept_map" in names

    def test_get_edge_collection_found(self):
        ec = NL_GRAPH_SCHEMA.get_edge_collection("nl_axiom_basis_edges")
        assert ec.source_field == "axiom_basis"

    def test_get_edge_collection_not_found(self):
        with pytest.raises(KeyError, match="nonexistent"):
            NL_GRAPH_SCHEMA.get_edge_collection("nonexistent")

    def test_get_named_graph_found(self):
        ng = NL_GRAPH_SCHEMA.get_named_graph("nl_core")
        assert len(ng.edge_definitions) == 3

    def test_get_named_graph_not_found(self):
        with pytest.raises(KeyError, match="nonexistent"):
            NL_GRAPH_SCHEMA.get_named_graph("nonexistent")

    def test_concept_map_includes_all_edges(self):
        concept_map = NL_GRAPH_SCHEMA.get_named_graph("nl_concept_map")
        assert len(concept_map.edge_definitions) == len(ALL_EDGE_COLLECTIONS)

    def test_all_edge_defs_have_from_and_to(self):
        for edef in NL_GRAPH_SCHEMA.edge_collections:
            assert len(edef.from_collections) > 0, f"{edef.name} has no from_collections"
            assert len(edef.to_collections) > 0, f"{edef.name} has no to_collections"

    def test_all_edge_defs_have_description(self):
        for edef in NL_GRAPH_SCHEMA.edge_collections:
            assert len(edef.description) > 10, f"{edef.name} has short description"


class TestHelperFunctions:
    def test_get_edge_definitions(self):
        defs = get_edge_definitions()
        assert len(defs) == len(ALL_EDGE_COLLECTIONS)
        assert defs is not NL_GRAPH_SCHEMA.edge_collections  # returns copy

    def test_get_named_graphs(self):
        graphs = get_named_graphs()
        assert len(graphs) == len(ALL_NAMED_GRAPHS)
        assert graphs is not NL_GRAPH_SCHEMA.named_graphs  # returns copy


class TestGharialPayloads:
    def test_all_named_graphs_produce_valid_payloads(self):
        for graph in NL_GRAPH_SCHEMA.named_graphs:
            payload = graph.to_gharial_payload()
            assert "name" in payload
            assert "edgeDefinitions" in payload
            assert len(payload["edgeDefinitions"]) > 0
            for edef in payload["edgeDefinitions"]:
                assert "collection" in edef
                assert "from" in edef
                assert "to" in edef
                assert isinstance(edef["from"], list)
                assert isinstance(edef["to"], list)
                assert len(edef["from"]) > 0
                assert len(edef["to"]) > 0

    def test_hecate_graph_merges_three_traced_fields(self):
        hecate = NL_GRAPH_SCHEMA.get_named_graph("nl_hecate")
        payload = hecate.to_gharial_payload()
        assert len(payload["edgeDefinitions"]) == 1  # all share nl_hecate_trace_edges
        edef = payload["edgeDefinitions"][0]
        assert edef["collection"] == "nl_hecate_trace_edges"
        assert "hecate_specs" in edef["from"]

    def test_edge_collection_names_follow_convention(self):
        for name in NL_GRAPH_SCHEMA.all_edge_collection_names():
            assert name.startswith("nl_"), f"{name} doesn't start with nl_"
            assert name.endswith("_edges"), f"{name} doesn't end with _edges"
