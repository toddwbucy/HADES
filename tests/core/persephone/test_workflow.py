"""Tests for core.persephone.workflow — state machine and guards."""

from unittest.mock import MagicMock, patch

import pytest

from core.persephone.collections import PERSEPHONE_COLLECTIONS
from core.persephone.workflow import (
    VALID_TRANSITIONS,
    GuardContext,
    TransitionError,
    _edge_type_for_transition,
    _guard_block_reason,
    _guard_dependency,
    _guard_different_reviewer,
    _guard_session_required,
    add_dependency,
    check_blocked,
    remove_dependency,
    transition,
)


def _make_ctx(**overrides):
    """Build a GuardContext with sensible defaults."""
    defaults = {
        "task": {"_key": "task_abc", "status": "open", "minor": False},
        "from_status": "open",
        "to_status": "in_progress",
        "session": {"_key": "ses_123", "agent_type": "claude_code"},
        "client": MagicMock(),
        "db_name": "bident",
        "collections": PERSEPHONE_COLLECTIONS,
        "human_override": False,
        "block_reason": None,
    }
    defaults.update(overrides)
    return GuardContext(**defaults)


class TestValidTransitions:
    def test_all_transitions_defined(self):
        assert len(VALID_TRANSITIONS) == 9

    def test_open_to_in_progress(self):
        assert ("open", "in_progress") in VALID_TRANSITIONS

    def test_in_review_to_closed(self):
        assert ("in_review", "closed") in VALID_TRANSITIONS

    def test_invalid_transition_not_in_set(self):
        assert ("open", "closed") not in VALID_TRANSITIONS
        assert ("closed", "in_progress") not in VALID_TRANSITIONS


class TestGuardSessionRequired:
    def test_allows_with_session(self):
        ctx = _make_ctx(to_status="in_progress")
        _guard_session_required(ctx)  # should not raise

    def test_blocks_without_session(self):
        ctx = _make_ctx(to_status="in_progress", session={})
        with pytest.raises(TransitionError, match="active session"):
            _guard_session_required(ctx)

    def test_allows_non_ownership_transitions(self):
        ctx = _make_ctx(to_status="open", session={})
        _guard_session_required(ctx)  # should not raise


class TestGuardDependency:
    def test_allows_when_no_blockers(self):
        ctx = _make_ctx(to_status="in_progress")
        ctx.client.query.return_value = []
        _guard_dependency(ctx)  # should not raise

    def test_blocks_when_dependencies_exist(self):
        ctx = _make_ctx(to_status="in_progress")
        ctx.client.query.return_value = [{"_key": "task_blocker", "title": "Blocker task"}]
        with pytest.raises(TransitionError, match="blocked by"):
            _guard_dependency(ctx)

    def test_skips_for_non_in_progress(self):
        ctx = _make_ctx(to_status="in_review")
        _guard_dependency(ctx)  # should not raise, no query made


class TestGuardBlockReason:
    def test_requires_reason_for_blocked(self):
        ctx = _make_ctx(to_status="blocked", block_reason=None)
        with pytest.raises(TransitionError, match="reason"):
            _guard_block_reason(ctx)

    def test_allows_with_reason(self):
        ctx = _make_ctx(to_status="blocked", block_reason="waiting on fix")
        _guard_block_reason(ctx)  # should not raise

    def test_skips_for_non_blocked(self):
        ctx = _make_ctx(to_status="in_progress", block_reason=None)
        _guard_block_reason(ctx)  # should not raise


class TestGuardDifferentReviewer:
    def test_blocks_same_implementer(self):
        ctx = _make_ctx(
            from_status="in_review",
            to_status="closed",
            task={"_key": "task_abc", "status": "in_review", "minor": False},
        )
        # Session is the implementer (implements edge)
        ctx.client.query.return_value = ["persephone_sessions/ses_123"]
        with pytest.raises(TransitionError, match="Cannot approve own work"):
            _guard_different_reviewer(ctx)

    def test_blocks_same_reviewer(self):
        ctx = _make_ctx(
            from_status="in_review",
            to_status="closed",
            task={"_key": "task_abc", "status": "in_review", "minor": False},
        )
        # Session submitted this for review (submitted_review edge)
        ctx.client.query.return_value = ["persephone_sessions/ses_123"]
        with pytest.raises(TransitionError, match="Cannot approve own work"):
            _guard_different_reviewer(ctx)

    def test_allows_different_reviewer(self):
        ctx = _make_ctx(
            from_status="in_review",
            to_status="closed",
            task={"_key": "task_abc", "status": "in_review", "minor": False},
        )
        ctx.client.query.return_value = ["persephone_sessions/ses_other"]
        _guard_different_reviewer(ctx)  # should not raise

    def test_allows_minor_task(self):
        ctx = _make_ctx(
            from_status="in_review",
            to_status="closed",
            task={"_key": "task_abc", "status": "in_review", "minor": True},
        )
        _guard_different_reviewer(ctx)  # should not raise, no query needed

    def test_allows_human_override(self):
        ctx = _make_ctx(
            from_status="in_review",
            to_status="closed",
            human_override=True,
            task={"_key": "task_abc", "status": "in_review", "minor": False},
        )
        _guard_different_reviewer(ctx)  # should not raise

    def test_skips_non_approval_transitions(self):
        ctx = _make_ctx(from_status="open", to_status="in_progress")
        _guard_different_reviewer(ctx)  # should not raise


class TestEdgeTypeMapping:
    def test_all_transitions_have_types(self):
        for from_s, to_s in VALID_TRANSITIONS:
            edge_type = _edge_type_for_transition(from_s, to_s)
            assert edge_type != "transitioned", f"Missing mapping for {from_s} → {to_s}"

    def test_specific_types(self):
        assert _edge_type_for_transition("open", "in_progress") == "implements"
        assert _edge_type_for_transition("in_review", "closed") == "approved"
        assert _edge_type_for_transition("in_review", "in_progress") == "rejected"
        assert _edge_type_for_transition("closed", "open") == "reopened"


class TestTransition:
    def setup_method(self):
        self.client = MagicMock()

    def test_invalid_transition_raises(self):
        self.client.request.return_value = {
            "_key": "task_abc", "status": "open", "minor": False,
            "_id": "persephone_tasks/task_abc", "_rev": "1",
        }
        # get_task uses client.request
        with patch("core.persephone.workflow.get_task") as mock_get:
            mock_get.return_value = {"_key": "task_abc", "status": "open", "minor": False}
            with pytest.raises(TransitionError, match="Invalid transition"):
                transition(self.client, "bident", "task_abc", "closed")

    def test_task_not_found_raises(self):
        with patch("core.persephone.workflow.get_task") as mock_get:
            mock_get.return_value = None
            with pytest.raises(TransitionError, match="not found"):
                transition(self.client, "bident", "task_missing", "in_progress")

    def test_successful_transition(self):
        session = {"_key": "ses_123", "agent_type": "claude_code"}
        updated_task = {"_key": "task_abc", "status": "in_progress"}

        with patch("core.persephone.workflow.get_task") as mock_get, \
             patch("core.persephone.workflow.update_task") as mock_update, \
             patch("core.persephone.workflow.get_or_create_session") as mock_session, \
             patch("core.persephone.workflow.create_session_task_edge") as mock_edge, \
             patch("core.persephone.workflow.check_blocked") as mock_blocked:

            mock_get.return_value = {"_key": "task_abc", "status": "open", "minor": False}
            mock_update.return_value = updated_task
            mock_session.return_value = session
            mock_blocked.return_value = []

            result = transition(self.client, "bident", "task_abc", "in_progress")
            assert result["status"] == "in_progress"
            mock_edge.assert_called_once()

    def test_block_reason_stored(self):
        session = {"_key": "ses_123", "agent_type": "claude_code"}

        with patch("core.persephone.workflow.get_task") as mock_get, \
             patch("core.persephone.workflow.update_task") as mock_update, \
             patch("core.persephone.workflow.get_or_create_session") as mock_session, \
             patch("core.persephone.workflow.create_session_task_edge"):

            mock_get.return_value = {"_key": "task_abc", "status": "in_progress", "minor": False}
            mock_update.return_value = {"_key": "task_abc", "status": "blocked", "block_reason": "waiting"}
            mock_session.return_value = session

            transition(
                self.client, "bident", "task_abc", "blocked",
                block_reason="waiting"
            )
            # Verify update_task was called with block_reason
            call_kwargs = mock_update.call_args
            assert call_kwargs[1]["block_reason"] == "waiting"

    def test_rollback_on_edge_failure(self):
        session = {"_key": "ses_123", "agent_type": "claude_code"}

        with patch("core.persephone.workflow.get_task") as mock_get, \
             patch("core.persephone.workflow.update_task") as mock_update, \
             patch("core.persephone.workflow.get_or_create_session") as mock_session, \
             patch("core.persephone.workflow.create_session_task_edge") as mock_edge, \
             patch("core.persephone.workflow.check_blocked") as mock_blocked:

            mock_get.return_value = {"_key": "task_abc", "status": "open", "minor": False}
            mock_update.return_value = {"_key": "task_abc", "status": "in_progress"}
            mock_session.return_value = session
            mock_blocked.return_value = []
            mock_edge.side_effect = RuntimeError("edge creation failed")

            with pytest.raises(RuntimeError, match="edge creation failed"):
                transition(self.client, "bident", "task_abc", "in_progress")

            # Verify rollback: update_task called twice (forward + rollback)
            assert mock_update.call_count == 2
            rollback_call = mock_update.call_args_list[1]
            assert rollback_call[1]["status"] == "open"


class TestAddDependency:
    def setup_method(self):
        self.client = MagicMock()

    def test_creates_edge(self):
        with patch("core.persephone.workflow.get_task") as mock_get:
            mock_get.side_effect = [
                {"_key": "task_a"},  # task exists
                {"_key": "task_b"},  # blocker exists
            ]
            self.client.request.return_value = {"_id": "persephone_edges/e1", "_rev": "1"}

            edge = add_dependency(self.client, "bident", "task_a", "task_b")
            assert edge["type"] == "blocked_by"
            assert edge["_from"] == "persephone_tasks/task_a"
            assert edge["_to"] == "persephone_tasks/task_b"

    def test_self_dependency_raises(self):
        with patch("core.persephone.workflow.get_task") as mock_get:
            mock_get.return_value = {"_key": "task_a"}
            with pytest.raises(TransitionError, match="cannot depend on itself"):
                add_dependency(self.client, "bident", "task_a", "task_a")

    def test_missing_task_raises(self):
        with patch("core.persephone.workflow.get_task") as mock_get:
            mock_get.return_value = None
            with pytest.raises(TransitionError, match="not found"):
                add_dependency(self.client, "bident", "task_missing", "task_b")


class TestRemoveDependency:
    def test_removes_existing(self):
        client = MagicMock()
        assert remove_dependency(client, "bident", "task_a", "task_b") is True

    def test_returns_false_when_missing(self):
        from core.database.arango.optimized_client import ArangoHttpError

        client = MagicMock()
        client.request.side_effect = ArangoHttpError(404, "not found")
        assert remove_dependency(client, "bident", "task_a", "task_b") is False


class TestCheckBlocked:
    def test_returns_blockers(self):
        client = MagicMock()
        client.query.return_value = [
            {"_key": "task_blocker", "title": "Blocker", "status": "open"}
        ]
        blockers = check_blocked(client, "bident", "task_a")
        assert len(blockers) == 1
        assert blockers[0]["_key"] == "task_blocker"

    def test_returns_empty_when_unblocked(self):
        client = MagicMock()
        client.query.return_value = []
        assert check_blocked(client, "bident", "task_a") == []
