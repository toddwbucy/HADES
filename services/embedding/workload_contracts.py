"""Explicit conventional Jina task contracts shared by serving and qualification.

These constants describe behavior, not database intent or capabilities inferred
from a model name. Contextual pooling still requires its own implementation.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class TextContract:
    passage_task: str
    query_task: str
    passage_prompt: str
    query_prompt: str
    adapter: str
    specialized_code: bool = False

    @property
    def prompts(self):
        """Keep a symmetric task once, preserving passage/query order otherwise."""
        return {self.passage_task: self.passage_prompt, self.query_task: self.query_prompt}

    def behavior(self, role):
        """Return a detached behavior record used in the canonical space descriptor."""
        if role not in ("passage", "query"):
            raise ValueError("unsupported embedding role")
        return {"task": getattr(self, role + "_task"), "prompt": getattr(self, role + "_prompt"),
                "adapter": self.adapter, "special_tokens": "qwen2-add-special-tokens-v1"}


WORKLOAD_CONTRACTS = MappingProxyType({
    "documents": TextContract("retrieval.passage", "retrieval.query", "Passage: ", "Query: ", "retrieval"),
    # HADES explicitly uses a symmetric passage prompt for its code/code pair.
    "code": TextContract("code", "code", "Passage: ", "Passage: ", "code", specialized_code=True),
})
