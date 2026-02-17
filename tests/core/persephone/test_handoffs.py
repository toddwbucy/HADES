"""Tests for core.persephone.handoffs — structured handoff management."""

from unittest.mock import MagicMock, patch

import pytest

from core.persephone.handoffs import (
    _capture_git_state,
    _generate_handoff_key,
    create_handoff,
    get_latest_handoff,
    list_handoffs,
)
from core.persephone.sessions import build_usage_briefing


class TestGenerateHandoffKey:
    def test_format(self):
        key = _generate_handoff_key()
        assert key.startswith("hnd_")
        assert len(key) == 10  # hnd_ + 6 hex chars

    def test_unique(self):
        keys = {_generate_handoff_key() for _ in range(100)}
        assert len(keys) == 100


class TestCaptureGitState:
    def test_captures_all_fields(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123def456\n"),  # rev-parse HEAD
                MagicMock(returncode=0, stdout="feat/phase4\n"),   # branch --show-current
                MagicMock(returncode=0, stdout="M file1.py\nA file2.py\n"),  # status --porcelain
                MagicMock(returncode=0, stdout="file1.py\n"),      # git diff --name-only HEAD
                MagicMock(returncode=0, stdout="file2.py\n"),      # git diff --name-only --cached
            ]
            result = _capture_git_state()

        assert result["git_sha"] == "abc123def456"
        assert result["git_branch"] == "feat/phase4"
        assert result["git_dirty_files"] == 2

    def test_handles_git_unavailable(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _capture_git_state()

        assert result["git_sha"] is None
        assert result["git_branch"] is None
        assert result["git_dirty_files"] is None

    def test_handles_timeout(self):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)):
            result = _capture_git_state()

        assert result["git_sha"] is None

    def test_handles_detached_head(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123\n"),  # rev-parse HEAD
                MagicMock(returncode=0, stdout="\n"),          # branch --show-current (empty in detached HEAD)
                MagicMock(returncode=0, stdout=""),            # status --porcelain (clean)
                MagicMock(returncode=0, stdout=""),            # git diff --name-only HEAD
                MagicMock(returncode=0, stdout=""),            # git diff --name-only --cached
            ]
            result = _capture_git_state()

        assert result["git_sha"] == "abc123"
        assert result["git_branch"] is None  # empty string → None
        assert result["git_dirty_files"] == 0

    def test_clean_repo(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="abc123\n"),
                MagicMock(returncode=0, stdout="main\n"),
                MagicMock(returncode=0, stdout=""),
                MagicMock(returncode=0, stdout=""),            # git diff --name-only HEAD
                MagicMock(returncode=0, stdout=""),            # git diff --name-only --cached
            ]
            result = _capture_git_state()

        assert result["git_dirty_files"] == 0


class TestCreateHandoff:
    def setup_method(self):
        self.client = MagicMock()
        self.client.request.return_value = {"_id": "persephone_handoffs/hnd_abc123", "_rev": "1"}
        self.session = {
            "_key": "ses_abc123",
            "_id": "persephone_sessions/ses_abc123",
            "agent_type": "claude_code",
        }

    @patch("core.persephone.handoffs._capture_git_state")
    def test_creates_handoff_with_all_fields(self, mock_git):
        mock_git.return_value = {"git_branch": "main", "git_sha": "abc", "git_dirty_files": 0}

        doc = create_handoff(
            self.client,
            "bident",
            "task_xyz",
            done=["Implemented feature"],
            remaining=["Add tests"],
            decisions=["Used graph edges"],
            uncertain=["Performance impact?"],
            note="Looking good",
            session=self.session,
        )

        assert doc["task_key"] == "task_xyz"
        assert doc["session_key"] == "ses_abc123"
        assert doc["done"] == ["Implemented feature"]
        assert doc["remaining"] == ["Add tests"]
        assert doc["decisions"] == ["Used graph edges"]
        assert doc["uncertain"] == ["Performance impact?"]
        assert doc["note"] == "Looking good"
        assert doc["git_branch"] == "main"
        assert doc["git_sha"] == "abc"
        assert doc["git_dirty_files"] == 0
        assert "created_at" in doc

    @patch("core.persephone.handoffs._capture_git_state")
    def test_creates_edges(self, mock_git):
        mock_git.return_value = {"git_branch": None, "git_sha": None, "git_dirty_files": None}

        create_handoff(
            self.client,
            "bident",
            "task_xyz",
            done=["Something"],
            session=self.session,
        )

        # Should create: 1 handoff doc + 2 edges + 1 activity log = 4 POST requests
        post_calls = [c for c in self.client.request.call_args_list if c[0][0] == "POST"]
        assert len(post_calls) == 4

        # Check edge types from the JSON payloads
        edge_types = [c[1]["json"]["type"] for c in post_calls if "type" in c[1].get("json", {})]
        assert "authored_handoff" in edge_types
        assert "handoff_for" in edge_types

    @patch("core.persephone.handoffs._capture_git_state")
    def test_auto_detects_session(self, mock_git):
        mock_git.return_value = {"git_branch": None, "git_sha": None, "git_dirty_files": None}

        with patch("core.persephone.handoffs.get_or_create_session") as mock_session:
            mock_session.return_value = self.session
            create_handoff(
                self.client,
                "bident",
                "task_xyz",
                done=["Something"],
                # session=None → auto-detect
            )
            mock_session.assert_called_once()

    def test_raises_if_no_content(self):
        with pytest.raises(ValueError, match="At least one content field required"):
            create_handoff(
                self.client,
                "bident",
                "task_xyz",
                session=self.session,
            )

    @patch("core.persephone.handoffs._capture_git_state")
    def test_defaults_empty_lists(self, mock_git):
        mock_git.return_value = {"git_branch": None, "git_sha": None, "git_dirty_files": None}

        doc = create_handoff(
            self.client,
            "bident",
            "task_xyz",
            note="Just a note",
            session=self.session,
        )

        assert doc["done"] == []
        assert doc["remaining"] == []
        assert doc["decisions"] == []
        assert doc["uncertain"] == []
        assert doc["note"] == "Just a note"


class TestGetLatestHandoff:
    def test_returns_newest(self):
        client = MagicMock()
        handoff = {
            "_key": "hnd_latest",
            "task_key": "task_xyz",
            "done": ["Latest work"],
            "created_at": "2026-02-17T12:00:00+00:00",
        }
        client.query.return_value = [handoff]

        result = get_latest_handoff(client, "bident", "task_xyz")
        assert result["_key"] == "hnd_latest"

        # Verify AQL uses correct task ID
        bind_vars = client.query.call_args[1]["bind_vars"]
        assert bind_vars["task_id"] == "persephone_tasks/task_xyz"

    def test_returns_none_when_empty(self):
        client = MagicMock()
        client.query.return_value = []

        result = get_latest_handoff(client, "bident", "task_xyz")
        assert result is None


class TestListHandoffs:
    def test_returns_sorted_list(self):
        client = MagicMock()
        handoffs = [
            {"_key": "hnd_2", "created_at": "2026-02-17T12:00:00+00:00"},
            {"_key": "hnd_1", "created_at": "2026-02-17T11:00:00+00:00"},
        ]
        client.query.return_value = handoffs

        result = list_handoffs(client, "bident", "task_xyz")
        assert len(result) == 2
        assert result[0]["_key"] == "hnd_2"

    def test_respects_limit(self):
        client = MagicMock()
        client.query.return_value = []

        list_handoffs(client, "bident", "task_xyz", limit=5)

        bind_vars = client.query.call_args[1]["bind_vars"]
        assert bind_vars["limit"] == 5

    def test_returns_empty_list(self):
        client = MagicMock()
        client.query.return_value = []

        result = list_handoffs(client, "bident", "task_xyz")
        assert result == []


class TestBriefingWithHandoffs:
    def setup_method(self):
        self.client = MagicMock()
        self.session = {
            "_key": "ses_abc123",
            "agent_type": "claude_code",
            "branch": "main",
            "started_at": "2026-02-17T00:00:00+00:00",
        }

    def test_includes_handoffs_for_in_progress_tasks(self):
        in_progress_task = {
            "_key": "task_2",
            "_id": "persephone_tasks/task_2",
            "title": "Working on it",
            "status": "in_progress",
        }
        handoff = {
            "_key": "hnd_abc",
            "task_key": "task_2",
            "done": ["Setup complete"],
            "remaining": ["Write tests"],
        }

        self.client.query.side_effect = [
            [in_progress_task],  # in_progress query
            [],                  # reviewable query
            [],                  # ready query
        ]

        with patch("core.persephone.handoffs.get_latest_handoff", return_value=handoff):
            briefing = build_usage_briefing(self.client, "bident", self.session)

        assert "handoffs" in briefing
        assert "task_2" in briefing["handoffs"]
        assert briefing["handoffs"]["task_2"]["done"] == ["Setup complete"]

    def test_no_handoffs_when_no_in_progress(self):
        self.client.query.side_effect = [
            [],  # in_progress
            [],  # reviewable
            [],  # ready
        ]

        briefing = build_usage_briefing(self.client, "bident", self.session)
        assert "handoffs" in briefing
        assert briefing["handoffs"] == {}

    def test_skips_tasks_without_handoffs(self):
        task = {
            "_key": "task_3",
            "_id": "persephone_tasks/task_3",
            "title": "No handoff yet",
            "status": "in_progress",
        }

        self.client.query.side_effect = [
            [task],  # in_progress
            [],      # reviewable
            [],      # ready
        ]

        with patch("core.persephone.handoffs.get_latest_handoff", return_value=None):
            briefing = build_usage_briefing(self.client, "bident", self.session)

        assert briefing["handoffs"] == {}
