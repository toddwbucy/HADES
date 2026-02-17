"""Pydantic schema models for Persephone node types.

Enforces field constraints, enum values, and structural invariants
at document creation/update time. Replaces hand-rolled validation
scattered across tasks.py, handoffs.py, and sessions.py.

Each node type has a Create model (all required fields) and optionally
an Update model (all fields optional, uses exclude_unset for partial patches).
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── Enum types ────────────────────────────────────────────────────

TaskStatus = Literal["open", "in_progress", "in_review", "closed", "blocked"]
TaskPriority = Literal["critical", "high", "medium", "low"]
TaskType = Literal["task", "bug", "epic"]

# ── Task models ───────────────────────────────────────────────────


class TaskCreate(BaseModel):
    """Schema for creating a new task document."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    title: str = Field(..., min_length=1)
    description: str | None = None
    status: TaskStatus = "open"
    priority: TaskPriority = "medium"
    type: TaskType = "task"
    labels: list[str] = Field(default_factory=list)
    parent_key: str | None = None
    acceptance: str | None = None
    minor: bool = False
    block_reason: str | None = None
    created_at: str
    updated_at: str


class TaskUpdate(BaseModel):
    """Schema for updating an existing task (partial patch).

    All fields optional — call .model_dump(exclude_unset=True)
    to get only the fields the caller actually provided.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    title: str | None = Field(default=None, min_length=1)
    description: str | None = None
    status: TaskStatus | None = None
    priority: TaskPriority | None = None
    type: TaskType | None = None
    labels: list[str] | None = None
    parent_key: str | None = None
    acceptance: str | None = None
    minor: bool | None = None
    block_reason: str | None = None


# ── Handoff models ────────────────────────────────────────────────


class HandoffCreate(BaseModel):
    """Schema for creating a handoff document."""

    model_config = ConfigDict(extra="forbid")

    task_key: str = Field(..., min_length=1)
    session_key: str = Field(..., min_length=1)
    done: list[str] = Field(default_factory=list)
    remaining: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    uncertain: list[str] = Field(default_factory=list)
    note: str | None = None
    git_branch: str | None = None
    git_sha: str | None = None
    git_dirty_files: int | None = None
    created_at: str

    @model_validator(mode="after")
    def require_content(self) -> Self:
        """At least one content field must be non-empty."""
        if not any([self.done, self.remaining, self.decisions, self.uncertain, self.note]):
            raise ValueError(
                "At least one content field required (done, remaining, decisions, uncertain, or note)"
            )
        return self


# ── Session models ────────────────────────────────────────────────


class SessionCreate(BaseModel):
    """Schema for creating a session document."""

    model_config = ConfigDict(extra="forbid")

    agent_type: str = Field(..., min_length=1)
    agent_pid: int = Field(..., ge=1)
    context_id: str = Field(..., min_length=1)
    branch: str = Field(..., min_length=1)
    previous_session_key: str | None = None
    started_at: str
    last_activity: str
    ended_at: str | None = None


# ── Node type registry ────────────────────────────────────────────

_NODE_REGISTRY: dict[str, tuple[type[BaseModel], type[BaseModel] | None]] = {
    "task": (TaskCreate, TaskUpdate),
    "handoff": (HandoffCreate, None),
    "session": (SessionCreate, None),
}


def register_node_type(
    name: str,
    create_model: type[BaseModel],
    update_model: type[BaseModel] | None = None,
) -> None:
    """Register a custom node type for schema enforcement.

    Raises:
        ValueError: If the node type name is already registered.
    """
    if name in _NODE_REGISTRY:
        raise ValueError(f"Node type '{name}' is already registered")
    _NODE_REGISTRY[name] = (create_model, update_model)


def get_node_model(name: str) -> tuple[type[BaseModel], type[BaseModel] | None]:
    """Look up (CreateModel, UpdateModel | None) for a node type.

    Raises:
        KeyError: If the node type is not registered.
    """
    if name not in _NODE_REGISTRY:
        raise KeyError(f"Unknown node type '{name}'")
    return _NODE_REGISTRY[name]


# ── Edge requirement validation ───────────────────────────────────


def validate_node_edges(required: frozenset[str], existing: set[str]) -> list[str]:
    """Check that all required edge types are present.

    Args:
        required: Edge types that must exist.
        existing: Edge types that actually exist.

    Returns:
        List of missing edge types. Empty list means all satisfied.
    """
    return sorted(required - existing)
