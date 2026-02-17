"""Tests for core.database.nl_graph_materialize — NL graph edge materialization."""

from unittest.mock import MagicMock, patch

from core.database.nl_graph_materialize import MaterializeStats, NLGraphMaterializer
from core.database.nl_graph_schema import (
    AXIOM_BASIS,
    CROSS_PAPER,
    LINEAGE_CHAIN,
    EdgeCollectionDef,
)


class TestMaterializeStats:
    def test_defaults(self):
        stats = MaterializeStats()
        assert stats.edges_created == 0
        assert stats.edges_skipped == 0
        assert stats.collections_scanned == 0
        assert stats.collections_missing == 0
        assert stats.errors == []
        assert stats.duration_ms == 0.0

    def test_to_dict(self):
        stats = MaterializeStats(edges_created=10, edges_skipped=2, duration_ms=123.456)
        d = stats.to_dict()
        assert d["edges_created"] == 10
        assert d["edges_skipped"] == 2
        assert d["duration_ms"] == 123.5  # rounded

    def test_errors_are_independent(self):
        """Each Stats instance should have its own errors list."""
        s1 = MaterializeStats()
        s2 = MaterializeStats()
        s1.errors.append("err1")
        assert s2.errors == []


class TestResolveRef:
    def setup_method(self):
        self.client = MagicMock()
        self.materializer = NLGraphMaterializer(self.client, database="test_db")

    def test_full_id_passes_through(self):
        assert self.materializer._resolve_ref("collection/key") == "collection/key"

    def test_bare_key_returns_none(self):
        assert self.materializer._resolve_ref("bare_key") is None

    def test_empty_string_returns_none(self):
        assert self.materializer._resolve_ref("") is None

    def test_none_returns_none(self):
        assert self.materializer._resolve_ref(None) is None

    def test_non_string_returns_none(self):
        assert self.materializer._resolve_ref(42) is None

    def test_nested_slash(self):
        assert self.materializer._resolve_ref("a/b/c") == "a/b/c"


class TestBuildEdgesStandard:
    def setup_method(self):
        self.client = MagicMock()
        self.materializer = NLGraphMaterializer(self.client, database="test_db")

    def test_single_ref_creates_edge(self):
        """A doc with a string source_field should produce one edge."""
        existing = {"src_coll", "tgt_coll"}
        edge_def = EdgeCollectionDef(
            name="test_edges",
            source_field="ref_field",
            from_collections=["src_coll"],
            to_collections=["tgt_coll"],
            description="test",
        )
        self.client.query.return_value = [{"_id": "src_coll/doc1", "_key": "doc1", "ref_field": "tgt_coll/target1"}]

        stats = self.materializer._build_edges_standard(edge_def, existing, dry_run=True)
        assert stats.edges_created == 1
        assert stats.collections_scanned == 1

    def test_array_ref_creates_multiple_edges(self):
        """A doc with a list source_field should produce multiple edges."""
        existing = {"src_coll", "tgt_coll"}
        edge_def = EdgeCollectionDef(
            name="test_edges",
            source_field="refs",
            from_collections=["src_coll"],
            to_collections=["tgt_coll"],
            description="test",
            is_array=True,
        )
        self.client.query.return_value = [
            {"_id": "src_coll/doc1", "_key": "doc1", "refs": ["tgt_coll/a", "tgt_coll/b", "tgt_coll/c"]}
        ]

        stats = self.materializer._build_edges_standard(edge_def, existing, dry_run=True)
        assert stats.edges_created == 3

    def test_missing_collection_skipped(self):
        """Collections not in existing set should be skipped."""
        existing = {"tgt_coll"}  # src_coll missing
        edge_def = EdgeCollectionDef(
            name="test_edges",
            source_field="ref",
            from_collections=["src_coll"],
            to_collections=["tgt_coll"],
            description="test",
        )

        stats = self.materializer._build_edges_standard(edge_def, existing, dry_run=True)
        assert stats.collections_missing == 1
        assert stats.edges_created == 0

    def test_bare_key_ref_skipped(self):
        """References without collection prefix should be skipped."""
        existing = {"src_coll", "tgt_coll"}
        edge_def = EdgeCollectionDef(
            name="test_edges",
            source_field="ref",
            from_collections=["src_coll"],
            to_collections=["tgt_coll"],
            description="test",
        )
        self.client.query.return_value = [{"_id": "src_coll/doc1", "_key": "doc1", "ref": "bare_key_no_slash"}]

        stats = self.materializer._build_edges_standard(edge_def, existing, dry_run=True)
        assert stats.edges_created == 0
        assert stats.edges_skipped == 1

    def test_nonexistent_target_collection_skipped(self):
        """Edges pointing to collections not in existing set should be skipped."""
        existing = {"src_coll"}  # tgt_coll missing
        edge_def = EdgeCollectionDef(
            name="test_edges",
            source_field="ref",
            from_collections=["src_coll"],
            to_collections=["tgt_coll"],
            description="test",
        )
        self.client.query.return_value = [{"_id": "src_coll/doc1", "_key": "doc1", "ref": "tgt_coll/target1"}]

        stats = self.materializer._build_edges_standard(edge_def, existing, dry_run=True)
        assert stats.edges_created == 0
        assert stats.edges_skipped == 1

    def test_edge_attributes_copied(self):
        """edge_attributes from the source doc should be copied onto the edge."""
        existing = {"src_coll", "tgt_coll", "test_edges"}
        edge_def = EdgeCollectionDef(
            name="test_edges",
            source_field="ref",
            from_collections=["src_coll"],
            to_collections=["tgt_coll"],
            description="test",
            edge_attributes=["name", "description"],
        )
        self.client.query.return_value = [
            {
                "_id": "src_coll/doc1",
                "_key": "doc1",
                "ref": "tgt_coll/target1",
                "name": "test_name",
                "description": "test_desc",
            }
        ]

        # Mock _insert_edges to capture what was passed
        inserted = []

        def capture_insert(collection, edges):
            inserted.extend(edges)
            return {"created": len(edges), "errors": 0}

        self.materializer._insert_edges = capture_insert
        stats = self.materializer._build_edges_standard(edge_def, existing, dry_run=False)

        assert stats.edges_created == 1
        assert len(inserted) == 1
        assert inserted[0]["name"] == "test_name"
        assert inserted[0]["description"] == "test_desc"

    def test_null_ref_field_skipped(self):
        """Docs where the ref field is None should be skipped silently."""
        existing = {"src_coll", "tgt_coll"}
        edge_def = EdgeCollectionDef(
            name="test_edges",
            source_field="ref",
            from_collections=["src_coll"],
            to_collections=["tgt_coll"],
            description="test",
        )
        self.client.query.return_value = [{"_id": "src_coll/doc1", "_key": "doc1", "ref": None}]

        stats = self.materializer._build_edges_standard(edge_def, existing, dry_run=True)
        assert stats.edges_created == 0
        assert stats.edges_skipped == 0


class TestBuildEdgesCrossPaper:
    def setup_method(self):
        self.client = MagicMock()
        self.materializer = NLGraphMaterializer(self.client, database="test_db")

    def test_creates_edges_from_paired_fields(self):
        existing = {"paper_edges", "src_coll", "tgt_coll"}
        self.client.query.return_value = [
            {
                "_id": "paper_edges/edge1",
                "_key": "edge1",
                "from_node": "src_coll/a",
                "to_node": "tgt_coll/b",
                "name": "test relation",
                "relationship": "extends",
            }
        ]

        stats = self.materializer._build_edges_cross_paper(CROSS_PAPER, existing, dry_run=True)
        assert stats.edges_created == 1
        assert stats.collections_scanned == 1

    def test_missing_paper_edges_collection(self):
        existing = {"other_coll"}
        stats = self.materializer._build_edges_cross_paper(CROSS_PAPER, existing, dry_run=True)
        assert stats.collections_missing == 1
        assert stats.edges_created == 0


class TestBuildEdgesLineage:
    def setup_method(self):
        self.client = MagicMock()
        self.materializer = NLGraphMaterializer(self.client, database="test_db")

    def test_chain_produces_sequential_and_membership_edges(self):
        """A chain [A, B, C] should produce: A→B, B→C (sequential) + lineage→A, lineage→B, lineage→C (membership)."""
        existing = {"atlas_lineage", "atlas_abstractions"}
        lineage_def = EdgeCollectionDef(
            name="nl_lineage_chain_edges",
            source_field="chain",
            from_collections=["atlas_lineage"],
            to_collections=["atlas_abstractions"],
            description="test",
            is_array=True,
            edge_attributes=["name", "type"],
        )
        self.client.query.return_value = [
            {
                "_id": "atlas_lineage/lineage1",
                "_key": "lineage1",
                "chain": [
                    "atlas_abstractions/a",
                    "atlas_abstractions/b",
                    "atlas_abstractions/c",
                ],
                "name": "test lineage",
                "type": "theoretical",
            }
        ]

        stats = self.materializer._build_edges_lineage(lineage_def, existing, dry_run=True)
        # 2 sequential (a→b, b→c) + 3 membership (lineage→a, lineage→b, lineage→c) = 5
        assert stats.edges_created == 5

    def test_short_chain_skipped(self):
        """Chains with fewer than 2 items should be skipped."""
        existing = {"atlas_lineage", "atlas_abstractions"}
        lineage_def = EdgeCollectionDef(
            name="nl_lineage_chain_edges",
            source_field="chain",
            from_collections=["atlas_lineage"],
            to_collections=["atlas_abstractions"],
            description="test",
            is_array=True,
        )
        self.client.query.return_value = [{"_id": "atlas_lineage/l1", "_key": "l1", "chain": ["atlas_abstractions/a"]}]

        stats = self.materializer._build_edges_lineage(lineage_def, existing, dry_run=True)
        assert stats.edges_created == 0


class TestMaterializeEdgeRouting:
    def setup_method(self):
        self.client = MagicMock()
        self.materializer = NLGraphMaterializer(self.client, database="test_db")

    def test_cross_paper_routed_correctly(self):
        with patch.object(self.materializer, "_build_edges_cross_paper") as mock:
            mock.return_value = MaterializeStats()
            self.materializer.materialize_edge(CROSS_PAPER, set())
            mock.assert_called_once()

    def test_lineage_routed_correctly(self):
        with patch.object(self.materializer, "_build_edges_lineage") as mock:
            mock.return_value = MaterializeStats()
            self.materializer.materialize_edge(LINEAGE_CHAIN, set())
            mock.assert_called_once()

    def test_standard_routed_for_axiom_basis(self):
        with patch.object(self.materializer, "_build_edges_standard") as mock:
            mock.return_value = MaterializeStats()
            self.materializer.materialize_edge(AXIOM_BASIS, set())
            mock.assert_called_once()


class TestMaterializeAll:
    def setup_method(self):
        self.client = MagicMock()
        self.materializer = NLGraphMaterializer(self.client, database="test_db")

    def test_dry_run_flag_passed_through(self):
        self.client.request.return_value = {"result": []}
        with patch.object(self.materializer, "materialize_edge") as mock:
            mock.return_value = MaterializeStats()
            self.materializer.materialize_all(dry_run=True)
            for call in mock.call_args_list:
                assert call.kwargs.get("dry_run") is True or call.args[2] is True

    def test_edge_filter_limits_processing(self):
        self.client.request.return_value = {"result": []}
        with patch.object(self.materializer, "materialize_edge") as mock:
            mock.return_value = MaterializeStats()
            self.materializer.materialize_all(edge_filter="nl_axiom_basis_edges")
            # Only axiom_basis should be processed (1 definition)
            assert mock.call_count == 1

    def test_result_structure(self):
        self.client.request.return_value = {"result": []}
        with patch.object(self.materializer, "materialize_edge") as mock:
            mock.return_value = MaterializeStats(edges_created=5)
            result = self.materializer.materialize_all(dry_run=True)
            assert "edge_collections" in result
            assert "totals" in result
            assert "dry_run" in result
            assert "edge_collections_processed" in result
            assert result["dry_run"] is True
