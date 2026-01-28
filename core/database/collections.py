"""Collection name registry for HADES database profiles.

Maps logical collection roles (metadata, chunks, embeddings) to physical
ArangoDB collection names. Each profile defines a self-contained set of
collection names — no profile shares collections with another.

Built-in profiles:
    arxiv       — ingested full papers (arxiv_metadata, arxiv_abstract_chunks, arxiv_abstract_embeddings)
    sync        — synced abstracts (arxiv_papers, arxiv_abstracts, arxiv_embeddings)
    default     — generic names (documents, chunks, embeddings)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollectionProfile:
    """Maps logical collection roles to physical ArangoDB collection names."""

    metadata: str
    chunks: str
    embeddings: str


# Built-in profiles
PROFILES: dict[str, CollectionProfile] = {
    "arxiv": CollectionProfile(
        metadata="arxiv_metadata",
        chunks="arxiv_abstract_chunks",
        embeddings="arxiv_abstract_embeddings",
    ),
    "sync": CollectionProfile(
        metadata="arxiv_papers",
        chunks="arxiv_abstracts",
        embeddings="arxiv_embeddings",
    ),
    "default": CollectionProfile(
        metadata="documents",
        chunks="chunks",
        embeddings="embeddings",
    ),
}


def get_profile(name: str) -> CollectionProfile:
    """Get a collection profile by name.

    Args:
        name: Profile name (arxiv, sync, default)

    Returns:
        CollectionProfile with physical collection names

    Raises:
        KeyError: If profile name is not registered
    """
    try:
        return PROFILES[name]
    except KeyError:
        available = ", ".join(sorted(PROFILES))
        raise KeyError(f"Unknown collection profile: {name!r}. Available: {available}") from None
