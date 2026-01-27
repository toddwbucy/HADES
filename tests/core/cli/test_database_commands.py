"""Tests for database CLI commands (create, delete, aql, graph)."""

from unittest.mock import MagicMock, patch

from core.cli.commands.database import (
    create_collection,
    delete_document,
    execute_aql,
    graph_create,
    graph_drop,
    graph_list,
    graph_neighbors,
    graph_shortest_path,
    graph_traverse,
)
from core.cli.output import ErrorCode

# =============================================================================
# Fixtures / Helpers
# =============================================================================

_ARANGO_CFG = {
    "database": "test_db",
    "host": "localhost",
    "port": 8529,
    "username": "root",
    "password": "test",
}


def _patch_stack(mock_get_config, mock_get_arango, mock_client_class, *, read_only=False):
    """Set up the standard 3-mock patch stack and return the mock client."""
    mock_get_config.return_value = MagicMock()
    mock_get_arango.return_value = _ARANGO_CFG
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    return mock_client


_DB_PATCH = (
    "core.database.arango.optimized_client.ArangoHttp2Client",
    "core.cli.commands.database.get_arango_config",
    "core.cli.commands.database.get_config",
)


class TestCreateCollection:
    """Tests for create_collection command."""

    @patch(*_DB_PATCH[:1])
    @patch(_DB_PATCH[1])
    @patch(_DB_PATCH[2])
    def test_create_success(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)

        response = create_collection(name="my_collection", start_time=0)

        assert response.success is True
        assert response.command == "database.create"
        assert response.data["collection"] == "my_collection"
        assert response.data["created"] is True
        mock_client.request.assert_called_once_with(
            "POST",
            "/_db/test_db/_api/collection",
            json={"name": "my_collection"},
        )
        mock_client.close.assert_called_once()

    @patch("core.cli.commands.database.get_config")
    def test_create_handles_config_error(self, mock_get_config):
        mock_get_config.side_effect = ValueError("Missing ARANGO_PASSWORD")

        response = create_collection(name="test", start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_create_handles_db_error(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.request.side_effect = Exception("Collection already exists")

        response = create_collection(name="existing", start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.DATABASE_ERROR.value
        mock_client.close.assert_called_once()

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_create_uses_read_write_socket(self, mock_get_config, mock_get_arango, _mock_client):
        mock_get_config.return_value = MagicMock()
        mock_get_arango.return_value = _ARANGO_CFG

        create_collection(name="test", start_time=0)

        call_kwargs = mock_get_arango.call_args[1]
        assert call_kwargs["read_only"] is False


class TestDeleteDocument:
    """Tests for delete_document command."""

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_delete_success(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)

        response = delete_document(collection="papers", key="2401.12345", start_time=0)

        assert response.success is True
        assert response.data["deleted"] is True
        mock_client.request.assert_called_once_with(
            "DELETE",
            "/_db/test_db/_api/document/papers/2401.12345",
        )
        mock_client.close.assert_called_once()

    @patch("core.cli.commands.database.get_config")
    def test_delete_handles_config_error(self, mock_get_config):
        mock_get_config.side_effect = ValueError("Missing ARANGO_PASSWORD")

        response = delete_document(collection="papers", key="123", start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_delete_handles_db_error(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.request.side_effect = Exception("Document not found")

        response = delete_document(collection="papers", key="nonexistent", start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.DATABASE_ERROR.value
        mock_client.close.assert_called_once()

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_delete_uses_read_write_socket(self, mock_get_config, mock_get_arango, _mock_client):
        mock_get_config.return_value = MagicMock()
        mock_get_arango.return_value = _ARANGO_CFG

        delete_document(collection="papers", key="123", start_time=0)

        call_kwargs = mock_get_arango.call_args[1]
        assert call_kwargs["read_only"] is False


# =============================================================================
# AQL Tests
# =============================================================================


class TestExecuteAql:
    """Tests for execute_aql command."""

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_success(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.query.return_value = [{"title": "Paper A"}, {"title": "Paper B"}]

        response = execute_aql("FOR d IN col RETURN d", None, start_time=0)

        assert response.success is True
        assert response.command == "database.aql"
        assert response.data["count"] == 2
        assert len(response.data["results"]) == 2
        mock_client.query.assert_called_once_with("FOR d IN col RETURN d", bind_vars=None)
        mock_client.close.assert_called_once()

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_with_bind_vars(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.query.return_value = [{"x": 1}]

        response = execute_aql("FOR d IN col FILTER d.x == @x RETURN d", '{"x": 1}', start_time=0)

        assert response.success is True
        mock_client.query.assert_called_once_with("FOR d IN col FILTER d.x == @x RETURN d", bind_vars={"x": 1})

    def test_invalid_bind_vars_json(self):
        response = execute_aql("RETURN 1", "not-json", start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "Invalid bind vars JSON" in response.error["message"]

    @patch("core.cli.commands.database.get_config")
    def test_config_error(self, mock_get_config):
        mock_get_config.side_effect = ValueError("Missing password")

        response = execute_aql("RETURN 1", None, start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_query_error(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.query.side_effect = Exception("AQL syntax error")

        response = execute_aql("INVALID AQL", None, start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.QUERY_FAILED.value
        mock_client.close.assert_called_once()


# =============================================================================
# Graph Tests
# =============================================================================


class TestGraphCreate:
    """Tests for graph_create command."""

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_success(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)

        edge_defs = '[{"collection":"edges","from":["A"],"to":["B"]}]'
        response = graph_create("my_graph", edge_defs, start_time=0)

        assert response.success is True
        assert response.command == "database.graph.create"
        assert response.data["graph"] == "my_graph"
        assert response.data["created"] is True
        mock_client.request.assert_called_once_with(
            "POST",
            "/_db/test_db/_api/gharial",
            json={
                "name": "my_graph",
                "edgeDefinitions": [{"collection": "edges", "from": ["A"], "to": ["B"]}],
            },
        )
        mock_client.close.assert_called_once()

    def test_invalid_json_edge_defs(self):
        response = graph_create("g", "not-json", start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "Invalid edge definitions JSON" in response.error["message"]

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_db_error(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.request.side_effect = Exception("Graph already exists")

        response = graph_create("g", "[]", start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.GRAPH_ERROR.value
        mock_client.close.assert_called_once()


class TestGraphList:
    """Tests for graph_list command."""

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_success(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.request.return_value = {
            "graphs": [
                {"_key": "g1", "edgeDefinitions": [{"collection": "e1", "from": ["A"], "to": ["B"]}]},
                {"_key": "g2", "edgeDefinitions": []},
            ]
        }

        response = graph_list(start_time=0)

        assert response.success is True
        assert len(response.data["graphs"]) == 2
        assert response.data["graphs"][0]["name"] == "g1"

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_empty(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.request.return_value = {"graphs": []}

        response = graph_list(start_time=0)

        assert response.success is True
        assert response.data["graphs"] == []


class TestGraphDrop:
    """Tests for graph_drop command."""

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_success(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)

        response = graph_drop("my_graph", drop_collections=True, start_time=0)

        assert response.success is True
        assert response.data["dropped"] is True
        assert response.data["collections_dropped"] is True
        mock_client.request.assert_called_once_with(
            "DELETE",
            "/_db/test_db/_api/gharial/my_graph?dropCollections=true",
        )
        mock_client.close.assert_called_once()

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_error(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.request.side_effect = Exception("Graph not found")

        response = graph_drop("missing", drop_collections=False, start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.GRAPH_ERROR.value


class TestGraphTraverse:
    """Tests for graph_traverse command."""

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_success(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.query.return_value = [
            {"vertex": {"_id": "nodes/2"}, "edge": {"_id": "edges/1"}},
        ]

        response = graph_traverse(
            start_vertex="nodes/1",
            graph="g",
            direction="outbound",
            min_depth=1,
            max_depth=3,
            limit=100,
            start_time=0,
        )

        assert response.success is True
        assert response.data["start"] == "nodes/1"
        assert response.data["graph"] == "g"
        assert len(response.data["results"]) == 1
        mock_client.close.assert_called_once()

    def test_invalid_direction(self):
        response = graph_traverse(
            start_vertex="nodes/1",
            graph="g",
            direction="sideways",
            min_depth=1,
            max_depth=3,
            limit=100,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "Invalid direction" in response.error["message"]

    def test_negative_min_depth(self):
        response = graph_traverse(
            start_vertex="nodes/1",
            graph="g",
            direction="outbound",
            min_depth=-1,
            max_depth=3,
            limit=100,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "min_depth must be >= 0" in response.error["message"]

    def test_max_depth_less_than_min_depth(self):
        response = graph_traverse(
            start_vertex="nodes/1",
            graph="g",
            direction="outbound",
            min_depth=3,
            max_depth=1,
            limit=100,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "max_depth must be >= min_depth" in response.error["message"]

    def test_zero_limit(self):
        response = graph_traverse(
            start_vertex="nodes/1",
            graph="g",
            direction="outbound",
            min_depth=1,
            max_depth=3,
            limit=0,
            start_time=0,
        )

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value
        assert "limit must be > 0" in response.error["message"]


class TestGraphShortestPath:
    """Tests for graph_shortest_path command."""

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_success(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.query.return_value = [
            {"vertex": {"_id": "nodes/1"}, "edge": None},
            {"vertex": {"_id": "nodes/3"}, "edge": {"_id": "edges/2"}},
            {"vertex": {"_id": "nodes/5"}, "edge": {"_id": "edges/4"}},
        ]

        response = graph_shortest_path(
            from_vertex="nodes/1",
            to_vertex="nodes/5",
            graph="g",
            direction="any",
            start_time=0,
        )

        assert response.success is True
        assert response.data["from"] == "nodes/1"
        assert response.data["to"] == "nodes/5"
        assert len(response.data["results"]) == 3
        mock_client.close.assert_called_once()


class TestGraphNeighbors:
    """Tests for graph_neighbors command."""

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.database.get_arango_config")
    @patch("core.cli.commands.database.get_config")
    def test_success(self, mock_get_config, mock_get_arango, mock_client_class):
        mock_client = _patch_stack(mock_get_config, mock_get_arango, mock_client_class)
        mock_client.query.return_value = [
            {"vertex": {"_id": "nodes/2"}, "edge": {"_id": "edges/1"}},
        ]

        response = graph_neighbors(
            start_vertex="nodes/1",
            graph="g",
            direction="outbound",
            limit=100,
            start_time=0,
        )

        assert response.success is True
        assert response.command == "database.graph.traverse"  # delegates to traverse
        assert len(response.data["results"]) == 1

        # Verify it used min_depth=1, max_depth=1
        call_args = mock_client.query.call_args
        bind_vars = call_args[1]["bind_vars"]
        assert bind_vars["min_depth"] == 1
        assert bind_vars["max_depth"] == 1
