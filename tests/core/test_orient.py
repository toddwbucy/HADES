"""Tests for the orient command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.cli.commands.orient import _safe_count, orient


class TestSafeCount:
    def test_returns_count(self):
        client = MagicMock()
        client.query.return_value = [42]
        assert _safe_count(client, "my_col") == 42

    def test_returns_none_on_error(self):
        client = MagicMock()
        client.query.side_effect = RuntimeError("not found")
        assert _safe_count(client, "missing_col") is None

    def test_returns_zero_for_empty_result(self):
        client = MagicMock()
        client.query.return_value = []
        assert _safe_count(client, "empty_col") == 0


class TestOrient:
    @patch("core.cli.commands.orient._make_client")
    def test_returns_database_name(self, mock_make_client):
        client = MagicMock()
        client.query.return_value = []
        mock_make_client.return_value = (client, None, "test_db")

        result = orient(0.0)

        assert result.success is True
        assert result.data["database"] == "test_db"
        assert "profiles" in result.data
        assert "total_collections" in result.data
        assert "total_documents" in result.data
        # all_collections only in verbose mode
        assert "all_collections" not in result.data

    @patch("core.cli.commands.orient._make_client")
    def test_handles_connection_error(self, mock_make_client):
        mock_make_client.side_effect = RuntimeError("Connection refused")

        result = orient(0.0)

        assert result.success is False
        assert "Connection refused" in result.error["message"]

    @patch("core.cli.commands.orient._make_client")
    def test_includes_persephone_when_present(self, mock_make_client):
        client = MagicMock()

        def query_side_effect(aql, bind_vars=None):
            if "COLLECTIONS" in aql:
                return ["persephone_tasks", "persephone_sessions", "persephone_logs"]
            if bind_vars and bind_vars.get("@col") == "persephone_tasks":
                return [5]
            if bind_vars and bind_vars.get("@col") == "persephone_sessions":
                return [3]
            if bind_vars and bind_vars.get("@col") == "persephone_logs":
                return [12]
            return [0]

        client.query.side_effect = query_side_effect
        mock_make_client.return_value = (client, None, "bident")

        result = orient(0.0)

        assert result.success is True
        assert "persephone" in result.data
        assert result.data["persephone"]["tasks"] == 5
        assert result.data["persephone"]["sessions"] == 3
