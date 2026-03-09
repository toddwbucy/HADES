"""Structural embedding fusion for semantic search.

Three composable re-ranking strategies that fuse RGCN structural
embeddings with Jina V4 text similarity scores:

1. structural_rerank  — centroid-based: top-3 results define an implicit
                        structural query; boost results whose graph neighbors
                        also scored well.
2. centrality_boost   — connectivity-based: nodes with more diverse edges
                        get a small authority bonus.
3. anchor_rerank      — node-anchored: boost results structurally similar
                        to a user-specified anchor node.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Allowlist of collection name characters to prevent AQL injection.
# ArangoDB collection names: letters, digits, underscores, hyphens.
_SAFE_COL_RE = None


def _validate_collection_name(name: str) -> str:
    """Validate collection name against AQL injection."""
    import re

    global _SAFE_COL_RE
    if _SAFE_COL_RE is None:
        _SAFE_COL_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
    if not _SAFE_COL_RE.match(name):
        raise ValueError(f"Invalid collection name: {name!r}")
    return name


def _get_arango_config() -> dict:
    """Get ArangoDB connection config from HADES config."""
    from core.cli.config import get_arango_config, get_config

    config = get_config()
    return get_arango_config(config, read_only=True)


def _aql(database: str, query: str, bind_vars: dict | None = None) -> list:
    """Execute AQL using the configured ArangoDB connection."""
    import base64
    import json
    import urllib.request

    arango_cfg = _get_arango_config()
    host = arango_cfg.get("host", "127.0.0.1")
    port = arango_cfg.get("port", 8529)
    username = arango_cfg.get("username", "root")
    password = arango_cfg["password"]

    auth = base64.b64encode(f"{username}:{password}".encode()).decode()
    base_url = f"http://{host}:{port}/_db/{database}"

    payload: dict[str, Any] = {"query": query}
    if bind_vars:
        payload["bindVars"] = bind_vars
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/_api/cursor",
        data=data,
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    if resp.get("error"):
        logger.warning("AQL error: %s", resp.get("errorMessage"))
        return []
    return resp.get("result", [])


def _result_key(r: dict[str, Any]) -> str | None:
    """Extract the best available key for a search result."""
    return r.get("paper_key") or r.get("arxiv_id")


def _current_score(r: dict[str, Any]) -> float:
    """Get the best current score from a result, respecting composition order.

    Each fusion stage should build on the previous stage's output score.
    This looks for scores in reverse-composition order (latest first).
    """
    for key in (
        "boosted_score",
        "fused_score",
        "anchor_fused_score",
        "cross_encoder_score",
        "combined_score",
        "similarity",
    ):
        value = r.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _fetch_structural_embeddings(
    database: str, paper_keys: list[str], metadata_collection: str = "arxiv_metadata"
) -> dict[str, np.ndarray]:
    """Fetch structural embeddings for a list of paper keys.

    Searches the metadata collection first. If a key is not found there
    (common in multi-collection graphs), falls back to scanning all
    document collections for nodes with matching _key.

    Returns dict mapping paper_key -> embedding vector (numpy).
    Nodes without structural_embedding are omitted.
    """
    if not paper_keys:
        return {}

    _validate_collection_name(metadata_collection)

    # Primary lookup in the metadata collection
    results = _aql(
        database,
        f"""
        FOR d IN {metadata_collection}
          FILTER d._key IN @keys AND d.structural_embedding != null
          RETURN [d._key, d.structural_embedding]
        """,
        bind_vars={"keys": paper_keys},
    )

    found = {
        key: np.array(emb, dtype=np.float32)
        for key, emb in results
        if emb is not None
    }

    # If some keys weren't found, search all document collections
    missing = [k for k in paper_keys if k not in found]
    if missing:
        fallback_results = _aql(
            database,
            """
            FOR c IN COLLECTIONS()
              FILTER c.type == 2 AND !STARTS_WITH(c.name, '_')
              RETURN c.name
            """,
        )
        for col_name in fallback_results:
            if col_name == metadata_collection:
                continue
            _validate_collection_name(col_name)
            hits = _aql(
                database,
                f"""
                FOR d IN {col_name}
                  FILTER d._key IN @keys AND d.structural_embedding != null
                  RETURN [d._key, d.structural_embedding]
                """,
                bind_vars={"keys": missing},
            )
            for key, emb in hits:
                if emb is not None and key not in found:
                    found[key] = np.array(emb, dtype=np.float32)
            # Stop early if all found
            missing = [k for k in missing if k not in found]
            if not missing:
                break

    return found


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Strategy 1: Structural Re-ranking (centroid-based)
# ---------------------------------------------------------------------------


def structural_rerank(
    results: list[dict[str, Any]],
    database: str,
    alpha: float = 0.7,
    centroid_k: int = 3,
    metadata_collection: str = "arxiv_metadata",
) -> list[dict[str, Any]]:
    """Re-rank results by structural similarity to top-k centroid.

    Takes the structural embeddings of the top centroid_k results,
    computes their centroid, and scores all results by structural
    similarity to that centroid. Final score blends the current best
    score with the structural signal.

    Args:
        results: Search results (builds on previous stage scores)
        database: ArangoDB database name
        alpha: Weight for current score (1-alpha = structural weight)
        centroid_k: Number of top results to use for centroid computation
        metadata_collection: Collection containing structural_embedding field

    Returns:
        Re-ranked results with structural_score and fused_score fields
    """
    paper_keys = list({_result_key(r) for r in results if _result_key(r)})
    embeddings = _fetch_structural_embeddings(database, paper_keys, metadata_collection)

    if len(embeddings) < 2:
        logger.info("Structural rerank: <2 nodes have structural embeddings, skipping")
        return results

    # Build centroid from top-k results that have embeddings
    centroid_vecs = []
    for r in results[:centroid_k]:
        key = _result_key(r)
        if key and key in embeddings:
            centroid_vecs.append(embeddings[key])

    if not centroid_vecs:
        logger.info("Structural rerank: no top results have structural embeddings, skipping")
        return results

    centroid = np.mean(centroid_vecs, axis=0)

    # Score all results
    reranked = []
    for r in results:
        base_score = _current_score(r)
        key = _result_key(r)

        if key and key in embeddings:
            struct_score = _cosine_similarity(embeddings[key], centroid)
        else:
            struct_score = 0.0

        fused = alpha * base_score + (1 - alpha) * max(0, struct_score)

        reranked.append({
            **r,
            "structural_score": round(struct_score, 4),
            "fused_score": round(fused, 4),
        })

    reranked.sort(key=lambda x: x["fused_score"], reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# Strategy 2: Centrality Boost
# ---------------------------------------------------------------------------


def centrality_boost(
    results: list[dict[str, Any]],
    database: str,
    boost_weight: float = 0.1,
    metadata_collection: str = "arxiv_metadata",
) -> list[dict[str, Any]]:
    """Boost results from nodes with high graph centrality.

    Computes centrality from structural embeddings: a node is "central"
    in context if its structural embedding is similar to many other result
    nodes' embeddings. This captures graph-position authority without
    requiring raw edge count queries.

    Args:
        results: Search results (builds on previous stage scores)
        database: ArangoDB database name
        boost_weight: Maximum score boost for the most-central node
        metadata_collection: Collection containing structural_embedding field

    Returns:
        Results with centrality_score and boosted_score fields
    """
    paper_keys = list({_result_key(r) for r in results if _result_key(r)})
    embeddings = _fetch_structural_embeddings(database, paper_keys, metadata_collection)

    if len(embeddings) < 2:
        logger.info("Centrality boost: <2 nodes have structural embeddings, skipping")
        return results

    # Compute mean structural similarity of each node to all other embedded nodes.
    # High mean similarity = central position in the graph's embedding space.
    emb_keys = list(embeddings.keys())
    emb_matrix = np.stack([embeddings[k] for k in emb_keys])
    # Normalize rows for cosine similarity via matrix multiply
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normed = emb_matrix / norms
    sim_matrix = emb_normed @ emb_normed.T  # [n, n] cosine similarities

    # Mean similarity excluding self (diagonal)
    n = len(emb_keys)
    np.fill_diagonal(sim_matrix, 0.0)
    mean_sims = sim_matrix.sum(axis=1) / max(n - 1, 1)

    centrality_map = {k: float(mean_sims[i]) for i, k in enumerate(emb_keys)}
    max_centrality = max(centrality_map.values()) if centrality_map else 1.0
    if max_centrality <= 0:
        max_centrality = 1.0

    boosted = []
    for r in results:
        key = _result_key(r)
        base_score = _current_score(r)
        raw_centrality = centrality_map.get(key, 0.0) if key else 0.0

        # Normalize to [0, 1] and scale by boost_weight
        centrality = raw_centrality / max_centrality
        boosted_score = base_score + boost_weight * centrality

        boosted.append({
            **r,
            "centrality_score": round(centrality, 4),
            "boosted_score": round(boosted_score, 4),
        })

    boosted.sort(key=lambda x: x["boosted_score"], reverse=True)
    return boosted


# ---------------------------------------------------------------------------
# Strategy 3: Graph-Anchored Search
# ---------------------------------------------------------------------------


def anchor_rerank(
    results: list[dict[str, Any]],
    anchor_node: str,
    database: str,
    alpha: float = 0.6,
    metadata_collection: str = "arxiv_metadata",
) -> list[dict[str, Any]]:
    """Re-rank results by structural similarity to an anchor node.

    Instead of computing a centroid from top results, uses a specific
    node's structural embedding as the reference point. Useful when the
    user knows a relevant node and wants to find text near that graph region.

    Args:
        results: Search results (builds on previous stage scores)
        anchor_node: Node ID in format 'collection/key' or just 'key'
        database: ArangoDB database name
        alpha: Weight for current score (1-alpha = structural weight)
        metadata_collection: Default collection if anchor_node has no slash

    Returns:
        Re-ranked results with anchor_similarity and anchor_fused_score fields
    """
    # Resolve anchor node
    if "/" in anchor_node:
        col, key = anchor_node.split("/", 1)
    else:
        col, key = metadata_collection, anchor_node

    _validate_collection_name(col)

    anchor_results = _aql(
        database,
        f"FOR d IN {col} FILTER d._key == @key RETURN d.structural_embedding",
        bind_vars={"key": key},
    )

    if not anchor_results or anchor_results[0] is None:
        logger.warning("Anchor node %s/%s has no structural embedding", col, key)
        return results

    anchor_emb = np.array(anchor_results[0], dtype=np.float32)

    # Fetch structural embeddings for result nodes
    paper_keys = list({_result_key(r) for r in results if _result_key(r)})
    embeddings = _fetch_structural_embeddings(database, paper_keys, metadata_collection)

    reranked = []
    for r in results:
        base_score = _current_score(r)
        key = _result_key(r)

        if key and key in embeddings:
            anchor_sim = _cosine_similarity(embeddings[key], anchor_emb)
        else:
            anchor_sim = 0.0

        fused = alpha * base_score + (1 - alpha) * max(0, anchor_sim)

        reranked.append({
            **r,
            "anchor_similarity": round(anchor_sim, 4),
            "anchor_fused_score": round(fused, 4),
        })

    reranked.sort(key=lambda x: x["anchor_fused_score"], reverse=True)
    return reranked
