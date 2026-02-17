"""Tests for core.persephone.sessions — agent fingerprinting and session management."""

import os
from unittest.mock import MagicMock, patch

import pytest

from core.persephone.sessions import (
    AgentFingerprint,
    _create_edge,
    _generate_session_key,
    _get_current_branch,
    build_usage_briefing,
    create_session_task_edge,
    detect_agent,
    end_session,
    force_new_session,
    get_or_create_session,
    get_session,
    heartbeat,
)


class TestAgentFingerprint:
    def test_frozen(self):
        fp = AgentFingerprint(agent_type="claude_code", agent_pid=123, context_id="abc")
        with pytest.raises(AttributeError):
            fp.agent_type = "other"

    def test_fields(self):
        fp = AgentFingerprint(agent_type="cursor", agent_pid=456, context_id="xyz")
        assert fp.agent_type == "cursor"
        assert fp.agent_pid == 456
        assert fp.context_id == "xyz"


class TestGenerateSessionKey:
    def test_format(self):
        key = _generate_session_key()
        assert key.startswith("ses_")
        assert len(key) == 10  # ses_ + 6 hex chars

    def test_unique(self):
        keys = {_generate_session_key() for _ in range(100)}
        assert len(keys) == 100


class TestDetectAgent:
    def setup_method(self):
        # Clear the lru_cache between tests
        detect_agent.cache_clear()

    def test_explicit_session_id(self):
        with patch.dict(os.environ, {"PERSEPHONE_SESSION_ID": "my-session-123"}):
            detect_agent.cache_clear()
            fp = detect_agent()
            assert fp.agent_type == "explicit"
            assert fp.context_id == "my-session-123"

    def test_claude_code_env_var(self):
        with patch.dict(os.environ, {"CLAUDE_CODE": "1"}, clear=False):
            detect_agent.cache_clear()
            # Remove PERSEPHONE_SESSION_ID if set
            env = os.environ.copy()
            env.pop("PERSEPHONE_SESSION_ID", None)
            with patch.dict(os.environ, env, clear=True):
                with patch.dict(os.environ, {"CLAUDE_CODE": "1"}):
                    detect_agent.cache_clear()
                    fp = detect_agent()
                    assert fp.agent_type == "claude_code"

    def test_terminal_fallback(self):
        env = {
            "TERM_SESSION_ID": "term-abc",
        }
        # Clear all agent env vars
        clear_vars = {k: "" for k in ["PERSEPHONE_SESSION_ID", "CLAUDE_CODE", "CLAUDE_CODE_AGENT", "CURSOR_AGENT", "CODEX_AGENT"]}
        with patch.dict(os.environ, {**clear_vars, **env}, clear=False):
            detect_agent.cache_clear()
            with patch("core.persephone.sessions._get_parent_processes", return_value=[]):
                fp = detect_agent()
                assert fp.agent_type == "terminal"
                assert fp.context_id == "term-abc"

    def test_unknown_fallback(self):
        clear_vars = {k: "" for k in [
            "PERSEPHONE_SESSION_ID", "CLAUDE_CODE", "CLAUDE_CODE_AGENT",
            "CURSOR_AGENT", "CODEX_AGENT", "TERM_SESSION_ID", "TMUX_PANE",
        ]}
        with patch.dict(os.environ, clear_vars, clear=False):
            detect_agent.cache_clear()
            with patch("core.persephone.sessions._get_parent_processes", return_value=[]):
                fp = detect_agent()
                assert fp.agent_type == "unknown"

    def test_process_tree_detection(self):
        clear_vars = {k: "" for k in [
            "PERSEPHONE_SESSION_ID", "CLAUDE_CODE", "CLAUDE_CODE_AGENT",
            "CURSOR_AGENT", "CODEX_AGENT",
        ]}
        with patch.dict(os.environ, clear_vars, clear=False):
            detect_agent.cache_clear()
            with patch(
                "core.persephone.sessions._get_parent_processes",
                return_value=[(100, "python"), (99, "claude"), (1, "init")],
            ):
                fp = detect_agent()
                assert fp.agent_type == "claude_code"
                assert fp.agent_pid == 99


class TestGetCurrentBranch:
    def test_returns_branch_name(self):
        # This test runs in a git repo, so it should return something
        branch = _get_current_branch()
        assert isinstance(branch, str)
        assert len(branch) > 0

    def test_fallback_on_error(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _get_current_branch() == "unknown"


class TestGetOrCreateSession:
    def setup_method(self):
        self.client = MagicMock()
        self.fp = AgentFingerprint(agent_type="claude_code", agent_pid=123, context_id="ctx-1")

    def test_returns_existing_session(self):
        existing = {
            "_key": "ses_abc123",
            "_id": "persephone_sessions/ses_abc123",
            "agent_type": "claude_code",
            "branch": "main",
            "context_id": "ctx-1",
            "ended_at": None,
        }
        self.client.query.return_value = [existing]

        with patch("core.persephone.sessions._get_current_branch", return_value="main"):
            session = get_or_create_session(self.client, "bident", fingerprint=self.fp)

        assert session["_key"] == "ses_abc123"
        # Heartbeat should have been called
        self.client.request.assert_called()

    def test_creates_new_when_none_exists(self):
        self.client.query.return_value = []
        self.client.request.return_value = {"_id": "persephone_sessions/ses_new123", "_rev": "1"}

        with patch("core.persephone.sessions._get_current_branch", return_value="main"):
            session = get_or_create_session(self.client, "bident", fingerprint=self.fp)

        assert session["agent_type"] == "claude_code"
        assert session["branch"] == "main"
        assert session["ended_at"] is None


class TestForceNewSession:
    def setup_method(self):
        self.client = MagicMock()
        self.fp = AgentFingerprint(agent_type="claude_code", agent_pid=123, context_id="ctx-1")

    def test_ends_previous_and_creates_new(self):
        previous = {
            "_key": "ses_old111",
            "_id": "persephone_sessions/ses_old111",
            "agent_type": "claude_code",
            "branch": "main",
            "ended_at": None,
        }
        self.client.query.return_value = [previous]
        self.client.request.return_value = {"_id": "persephone_sessions/ses_new222", "_rev": "1"}

        with patch("core.persephone.sessions._get_current_branch", return_value="main"):
            session = force_new_session(self.client, "bident", fingerprint=self.fp)

        assert session["previous_session_key"] == "ses_old111"
        assert session["agent_type"] == "claude_code"

    def test_creates_new_without_previous(self):
        self.client.query.return_value = []
        self.client.request.return_value = {"_id": "persephone_sessions/ses_new333", "_rev": "1"}

        with patch("core.persephone.sessions._get_current_branch", return_value="main"):
            session = force_new_session(self.client, "bident", fingerprint=self.fp)

        assert session["previous_session_key"] is None


class TestHeartbeat:
    def test_updates_last_activity(self):
        client = MagicMock()
        heartbeat(client, "bident", "ses_abc123")
        client.request.assert_called_once()
        args = client.request.call_args
        assert args[0][0] == "PATCH"
        assert "last_activity" in args[1]["json"]

    def test_handles_missing_session(self):
        from core.database.arango.optimized_client import ArangoHttpError

        client = MagicMock()
        client.request.side_effect = ArangoHttpError(404, "not found")
        # Should not raise
        heartbeat(client, "bident", "ses_missing")


class TestEndSession:
    def test_sets_ended_at(self):
        client = MagicMock()
        end_session(client, "bident", "ses_abc123")
        args = client.request.call_args
        assert "ended_at" in args[1]["json"]
        assert args[1]["json"]["ended_at"] is not None


class TestGetSession:
    def test_returns_session(self):
        client = MagicMock()
        client.request.return_value = {"_key": "ses_abc123", "agent_type": "claude_code"}
        session = get_session(client, "bident", "ses_abc123")
        assert session["_key"] == "ses_abc123"

    def test_returns_none_when_not_found(self):
        from core.database.arango.optimized_client import ArangoHttpError

        client = MagicMock()
        client.request.side_effect = ArangoHttpError(404, "not found")
        assert get_session(client, "bident", "ses_missing") is None


class TestCreateSessionTaskEdge:
    def test_creates_edge(self):
        client = MagicMock()
        client.request.return_value = {"_id": "persephone_edges/e1", "_rev": "1"}

        edge = create_session_task_edge(client, "bident", "ses_abc", "task_xyz")
        assert edge["_from"] == "persephone_sessions/ses_abc"
        assert edge["_to"] == "persephone_tasks/task_xyz"
        assert edge["type"] == "implements"

    def test_deterministic_key(self):
        client = MagicMock()
        client.request.return_value = {"_id": "persephone_edges/e1", "_rev": "1"}

        edge = create_session_task_edge(client, "bident", "ses_abc", "task_xyz", "reviews")
        assert edge["_key"] == "ses_abc__task_xyz__reviews"


class TestCreateEdge:
    def test_edge_structure(self):
        client = MagicMock()
        client.request.return_value = {"_id": "persephone_edges/e1", "_rev": "1"}

        edge = _create_edge(
            client, "bident",
            _from="persephone_sessions/ses_a",
            _to="persephone_sessions/ses_b",
            edge_type="continues",
        )
        assert edge["_from"] == "persephone_sessions/ses_a"
        assert edge["_to"] == "persephone_sessions/ses_b"
        assert edge["type"] == "continues"
        assert "created_at" in edge


class TestBuildUsageBriefing:
    def setup_method(self):
        self.client = MagicMock()
        self.session = {
            "_key": "ses_abc123",
            "agent_type": "claude_code",
            "branch": "main",
            "started_at": "2026-02-17T00:00:00+00:00",
        }

    def test_briefing_structure(self):
        self.client.query.side_effect = [
            [],  # in_progress
            [],  # reviewable
            [{"_key": "task_1", "title": "Do something", "status": "open", "priority": "high"}],  # ready
        ]

        briefing = build_usage_briefing(self.client, "bident", self.session)
        assert "session" in briefing
        assert "in_progress" in briefing
        assert "reviewable" in briefing
        assert "ready" in briefing
        assert briefing["session"]["key"] == "ses_abc123"
        assert len(briefing["ready"]) == 1

    def test_returns_in_progress_tasks(self):
        in_progress_task = {"_key": "task_2", "title": "Working on it", "status": "in_progress"}
        self.client.query.side_effect = [
            [in_progress_task],  # in_progress
            [],  # reviewable
            [],  # ready
        ]

        briefing = build_usage_briefing(self.client, "bident", self.session)
        assert len(briefing["in_progress"]) == 1
        assert briefing["in_progress"][0]["_key"] == "task_2"
