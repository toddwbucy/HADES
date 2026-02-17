"""Tests for Persephone Pydantic schema models."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from core.persephone.models import (
    HandoffCreate,
    SessionCreate,
    TaskCreate,
    TaskUpdate,
    _NODE_REGISTRY,
    get_node_model,
    register_node_type,
    validate_node_edges,
)


# ── Fixtures ──────────────────────────────────────────────────────

NOW = "2025-01-15T12:00:00+00:00"


# ── TaskCreate tests ──────────────────────────────────────────────


class TestTaskCreate:
    def test_valid_minimal(self):
        t = TaskCreate(title="Fix bug", created_at=NOW, updated_at=NOW)
        assert t.title == "Fix bug"
        assert t.status == "open"
        assert t.priority == "medium"
        assert t.type == "task"
        assert t.labels == []
        assert t.minor is False

    def test_valid_all_fields(self):
        t = TaskCreate(
            title="Epic task",
            description="Full desc",
            status="in_progress",
            priority="critical",
            type="epic",
            labels=["backend", "urgent"],
            parent_key="task_abc123",
            acceptance="All tests pass",
            minor=True,
            block_reason="Waiting on deploy",
            created_at=NOW,
            updated_at=NOW,
        )
        assert t.priority == "critical"
        assert t.type == "epic"
        assert t.labels == ["backend", "urgent"]
        assert t.block_reason == "Waiting on deploy"

    def test_invalid_priority_raises(self):
        with pytest.raises(ValidationError, match="priority"):
            TaskCreate(title="X", priority="urgent", created_at=NOW, updated_at=NOW)

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError, match="status"):
            TaskCreate(title="X", status="done", created_at=NOW, updated_at=NOW)

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError, match="type"):
            TaskCreate(title="X", type="feature", created_at=NOW, updated_at=NOW)

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError, match="title"):
            TaskCreate(title="", created_at=NOW, updated_at=NOW)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            TaskCreate(title="X", created_at=NOW, updated_at=NOW, bogus="nope")

    def test_defaults_correct(self):
        t = TaskCreate(title="T", created_at=NOW, updated_at=NOW)
        d = t.model_dump()
        assert d["description"] is None
        assert d["parent_key"] is None
        assert d["acceptance"] is None
        assert d["block_reason"] is None


# ── TaskUpdate tests ──────────────────────────────────────────────


class TestTaskUpdate:
    def test_partial_update(self):
        u = TaskUpdate(status="closed")
        d = u.model_dump(exclude_unset=True)
        assert d == {"status": "closed"}

    def test_exclude_unset(self):
        u = TaskUpdate(priority="high", title="New title")
        d = u.model_dump(exclude_unset=True)
        assert set(d.keys()) == {"priority", "title"}

    def test_invalid_enum_raises(self):
        with pytest.raises(ValidationError, match="priority"):
            TaskUpdate(priority="ultra")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            TaskUpdate(nonexistent="val")

    def test_empty_update_allowed(self):
        u = TaskUpdate()
        d = u.model_dump(exclude_unset=True)
        assert d == {}


# ── HandoffCreate tests ──────────────────────────────────────────


class TestHandoffCreate:
    def test_valid_with_done_only(self):
        h = HandoffCreate(
            task_key="task_abc",
            session_key="ses_xyz",
            done=["Fixed the bug"],
            created_at=NOW,
        )
        assert h.done == ["Fixed the bug"]
        assert h.remaining == []

    def test_valid_with_note_only(self):
        h = HandoffCreate(
            task_key="task_abc",
            session_key="ses_xyz",
            note="Some context",
            created_at=NOW,
        )
        assert h.note == "Some context"

    def test_no_content_raises(self):
        with pytest.raises(ValidationError, match="content field"):
            HandoffCreate(
                task_key="task_abc",
                session_key="ses_xyz",
                created_at=NOW,
            )

    def test_all_fields(self):
        h = HandoffCreate(
            task_key="task_abc",
            session_key="ses_xyz",
            done=["a"],
            remaining=["b"],
            decisions=["c"],
            uncertain=["d"],
            note="e",
            git_branch="main",
            git_sha="abc123",
            git_dirty_files=3,
            created_at=NOW,
        )
        assert h.git_dirty_files == 3

    def test_empty_task_key_raises(self):
        with pytest.raises(ValidationError, match="task_key"):
            HandoffCreate(
                task_key="",
                session_key="ses_xyz",
                done=["x"],
                created_at=NOW,
            )

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            HandoffCreate(
                task_key="task_abc",
                session_key="ses_xyz",
                done=["x"],
                created_at=NOW,
                mystery="field",
            )


# ── SessionCreate tests ──────────────────────────────────────────


class TestSessionCreate:
    def test_valid_session(self):
        s = SessionCreate(
            agent_type="claude_code",
            agent_pid=12345,
            context_id="ctx_abc",
            branch="main",
            started_at=NOW,
            last_activity=NOW,
        )
        assert s.agent_type == "claude_code"
        assert s.ended_at is None

    def test_negative_pid_raises(self):
        with pytest.raises(ValidationError, match="agent_pid"):
            SessionCreate(
                agent_type="claude_code",
                agent_pid=-1,
                context_id="ctx_abc",
                branch="main",
                started_at=NOW,
                last_activity=NOW,
            )

    def test_zero_pid_raises(self):
        with pytest.raises(ValidationError, match="agent_pid"):
            SessionCreate(
                agent_type="claude_code",
                agent_pid=0,
                context_id="ctx_abc",
                branch="main",
                started_at=NOW,
                last_activity=NOW,
            )

    def test_empty_agent_type_raises(self):
        with pytest.raises(ValidationError, match="agent_type"):
            SessionCreate(
                agent_type="",
                agent_pid=100,
                context_id="ctx_abc",
                branch="main",
                started_at=NOW,
                last_activity=NOW,
            )

    def test_with_previous_session(self):
        s = SessionCreate(
            agent_type="cursor",
            agent_pid=999,
            context_id="ctx_abc",
            branch="feat/x",
            previous_session_key="ses_prev",
            started_at=NOW,
            last_activity=NOW,
            ended_at=NOW,
        )
        assert s.previous_session_key == "ses_prev"
        assert s.ended_at == NOW


# ── Node registry tests ──────────────────────────────────────────


class TestNodeRegistry:
    def test_builtins_registered(self):
        for name in ("task", "handoff", "session"):
            create_model, _ = get_node_model(name)
            assert issubclass(create_model, BaseModel)

    def test_task_has_update_model(self):
        _, update_model = get_node_model("task")
        assert update_model is TaskUpdate

    def test_handoff_no_update_model(self):
        _, update_model = get_node_model("handoff")
        assert update_model is None

    def test_custom_registration(self):
        class CustomCreate(BaseModel):
            name: str

        # Clean up after test
        try:
            register_node_type("custom_test_node", CustomCreate)
            create_model, update_model = get_node_model("custom_test_node")
            assert create_model is CustomCreate
            assert update_model is None
        finally:
            _NODE_REGISTRY.pop("custom_test_node", None)

    def test_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            register_node_type("task", TaskCreate)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown node type"):
            get_node_model("nonexistent")


# ── Edge validation tests ─────────────────────────────────────────


class TestValidateNodeEdges:
    def test_all_present(self):
        required = frozenset({"implements", "authored_handoff"})
        existing = {"implements", "authored_handoff", "extra"}
        assert validate_node_edges(required, existing) == []

    def test_missing_returned(self):
        required = frozenset({"implements", "reviews", "authored_handoff"})
        existing = {"implements"}
        missing = validate_node_edges(required, existing)
        assert missing == ["authored_handoff", "reviews"]

    def test_empty_requirements_passes(self):
        assert validate_node_edges(frozenset(), {"anything"}) == []

    def test_empty_existing_returns_all(self):
        required = frozenset({"a", "b"})
        assert validate_node_edges(required, set()) == ["a", "b"]
