"""Tests for Persephone activity logging (Phase 7)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from core.persephone.collections import PersephoneCollections  # noqa: F401
from core.persephone.logging import create_log, list_logs  # noqa: F401
from core.persephone.models import LogCreate

# ── Pydantic model tests ─────────────────────────────────────────


class TestLogCreate:
    """Test LogCreate schema validation."""

    def test_valid_minimal(self):
        log = LogCreate(action="task.created", created_at="2026-01-01T00:00:00+00:00")
        assert log.action == "task.created"
        assert log.task_key is None
        assert log.session_key is None
        assert log.details is None

    def test_valid_full(self):
        log = LogCreate(
            action="task.transitioned",
            task_key="task_abc123",
            session_key="ses_def456",
            details={"from_status": "open", "to_status": "in_progress"},
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert log.action == "task.transitioned"
        assert log.task_key == "task_abc123"
        assert log.details["from_status"] == "open"

    def test_missing_action_fails(self):
        with pytest.raises(ValidationError):
            LogCreate(created_at="2026-01-01T00:00:00+00:00")

    def test_empty_action_fails(self):
        with pytest.raises(ValidationError):
            LogCreate(action="", created_at="2026-01-01T00:00:00+00:00")

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            LogCreate(
                action="task.created",
                created_at="2026-01-01T00:00:00+00:00",
                unknown_field="value",
            )


# ── create_log tests ─────────────────────────────────────────────


class TestCreateLog:
    """Test create_log() function."""

    def _make_mock_client(self):
        client = MagicMock()
        client.request.return_value = {"_id": "persephone_logs/log_abc123", "_rev": "1"}
        return client

    def test_creates_log_document(self):
        client = self._make_mock_client()
        cols = PersephoneCollections()

        result = create_log(
            client, "test_db",
            action="task.created",
            task_key="task_abc",
            details={"title": "Test"},
            collections=cols,
        )

        # Verify document was POSTed
        client.request.assert_called_once()
        call_args = client.request.call_args
        assert call_args[0] == ("POST", "/_db/test_db/_api/document/persephone_logs")

        # Verify document shape
        doc = call_args[1]["json"]
        assert doc["_key"].startswith("log_")
        assert doc["action"] == "task.created"
        assert doc["task_key"] == "task_abc"
        assert doc["details"] == {"title": "Test"}
        assert "created_at" in doc

        # Verify return value
        assert result["action"] == "task.created"
        assert result["_id"] == "persephone_logs/log_abc123"

    def test_creates_log_without_optional_fields(self):
        client = self._make_mock_client()

        create_log(
            client, "test_db",
            action="session.started",
        )

        doc = client.request.call_args[1]["json"]
        assert doc["task_key"] is None
        assert doc["session_key"] is None
        assert doc["details"] is None

    def test_raises_on_empty_action(self):
        client = self._make_mock_client()

        with pytest.raises(ValueError):
            create_log(client, "test_db", action="")


# ── list_logs tests ──────────────────────────────────────────────


class TestListLogs:
    """Test list_logs() function."""

    def _make_mock_client(self, results=None):
        client = MagicMock()
        client.query.return_value = results or []
        return client

    def test_no_filters(self):
        client = self._make_mock_client([{"action": "task.created"}])

        result = list_logs(client, "test_db")

        client.query.assert_called_once()
        bind_vars = client.query.call_args[1]["bind_vars"]
        assert bind_vars["@col"] == "persephone_logs"
        assert bind_vars["limit"] == 50
        assert "task_key" not in bind_vars
        assert result == [{"action": "task.created"}]

    def test_filter_by_task(self):
        client = self._make_mock_client()

        list_logs(client, "test_db", task_key="task_abc")

        aql = client.query.call_args[0][0]
        bind_vars = client.query.call_args[1]["bind_vars"]
        assert "FILTER doc.task_key == @task_key" in aql
        assert bind_vars["task_key"] == "task_abc"

    def test_filter_by_session(self):
        client = self._make_mock_client()

        list_logs(client, "test_db", session_key="ses_def")

        aql = client.query.call_args[0][0]
        bind_vars = client.query.call_args[1]["bind_vars"]
        assert "FILTER doc.session_key == @session_key" in aql
        assert bind_vars["session_key"] == "ses_def"

    def test_filter_by_action(self):
        client = self._make_mock_client()

        list_logs(client, "test_db", action="task.transitioned")

        aql = client.query.call_args[0][0]
        bind_vars = client.query.call_args[1]["bind_vars"]
        assert "FILTER doc.action == @action" in aql
        assert bind_vars["action"] == "task.transitioned"

    def test_custom_limit(self):
        client = self._make_mock_client()

        list_logs(client, "test_db", limit=10)

        bind_vars = client.query.call_args[1]["bind_vars"]
        assert bind_vars["limit"] == 10

    def test_multiple_filters(self):
        client = self._make_mock_client()

        list_logs(client, "test_db", task_key="task_abc", action="task.transitioned")

        aql = client.query.call_args[0][0]
        assert "FILTER doc.task_key == @task_key" in aql
        assert "FILTER doc.action == @action" in aql


# ── task_sessions CLI tests ──────────────────────────────────────


class TestTaskSessionsCLI:
    """Test the task_sessions CLI command function."""

    @patch("core.cli.commands.persephone._make_client")
    def test_returns_sessions(self, mock_make_client):
        mock_client = MagicMock()
        mock_client.query.return_value = [
            {"_key": "ses_abc", "agent_type": "claude_code", "edge_type": "implements"},
        ]
        mock_make_client.return_value = (mock_client, None, "test_db")

        from core.cli.commands.persephone import task_sessions

        result = task_sessions("task_xyz", 0.0)

        assert result.success is True
        assert result.data["task_key"] == "task_xyz"
        assert result.data["count"] == 1
        assert result.data["sessions"][0]["agent_type"] == "claude_code"

    @patch("core.cli.commands.persephone._make_client")
    def test_handles_db_error(self, mock_make_client):
        mock_make_client.side_effect = RuntimeError("Connection refused")

        from core.cli.commands.persephone import task_sessions

        result = task_sessions("task_xyz", 0.0)

        assert result.success is False
        assert "Connection refused" in result.error["message"]


# ── Best-effort logging integration ─────────────────────────────


class TestBestEffortLogging:
    """Verify that logging failures don't break core operations."""

    @patch("core.persephone.logging.create_log", side_effect=RuntimeError("DB down"))
    def test_create_task_succeeds_when_logging_fails(self, mock_log):
        """create_task should succeed even if create_log raises."""
        from core.persephone.tasks import create_task

        client = MagicMock()
        client.request.return_value = {"_id": "persephone_tasks/task_abc", "_rev": "1"}

        result = create_task(client, "test_db", "Test task")

        assert result["title"] == "Test task"
        assert result["_key"].startswith("task_")
        mock_log.assert_called_once()
