"""ArangoDB → PyG HeteroData loader for RGCN training.

Loads the NL knowledge graph from ArangoDB into a PyTorch Geometric
HeteroData object. Handles:
  - Node feature extraction (Jina V4 embeddings where available)
  - Learnable type embeddings for nodes without Jina features
  - Edge index construction per relation type
  - Bidirectional ID mapping (ArangoDB _id ↔ integer index)

Usage:
    from core.graph.loader import GraphLoader

    loader = GraphLoader(database="NL_graph_v0")
    data, id_map = loader.load()
    # data is a PyG HeteroData object
    # id_map maps ArangoDB _id → (node_type, int_index) and back
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Edge collections to include in the graph, with their relation index.
# nl_build_path_edges excluded (only 2 edges, not trainable).
# 4 future empty collections included for slot reservation.
EDGE_COLLECTIONS: list[str] = [
    "nl_axiom_basis_edges",  # 0
    "nl_axiom_inherits_edges",  # 1
    "nl_axiom_violation_edges",  # 2  (future, empty)
    "nl_cross_paper_edges",  # 3
    "nl_code_callgraph_edges",  # 4  (future, empty)
    "nl_code_equation_edges",  # 5  (future, empty)
    "nl_code_test_edges",  # 6  (future, empty)
    "nl_definition_source_edges",  # 7
    "nl_equation_depends_edges",  # 8
    "nl_equation_source_edges",  # 9
    "nl_hecate_trace_edges",  # 10
    "nl_lineage_chain_edges",  # 11
    "nl_migration_edges",  # 12
    "nl_paper_cross_reference_edges",  # 13
    "nl_reframing_link_edges",  # 14
    "nl_signature_equation_edges",  # 15
    "nl_smell_compliance_edges",  # 16
    "nl_smell_source_edges",  # 17
    "nl_structural_embodiment_edges",  # 18
    "nl_validated_against_edges",  # 19
    "persephone_edges",  # 20
]

NUM_RELATIONS = len(EDGE_COLLECTIONS)  # 21 — frozen architectural constant

# Jina V4 embedding dimension
JINA_DIM = 2048


@dataclass
class IDMap:
    """Bidirectional mapping between ArangoDB _id and integer node index."""

    arango_to_idx: dict[str, int] = field(default_factory=dict)
    idx_to_arango: dict[int, str] = field(default_factory=dict)
    collection_of: dict[str, str] = field(default_factory=dict)  # _id → collection name
    _next_idx: int = 0

    def get_or_create(self, arango_id: str) -> int:
        """Get or create integer index for an ArangoDB _id."""
        if arango_id in self.arango_to_idx:
            return self.arango_to_idx[arango_id]
        idx = self._next_idx
        self.arango_to_idx[arango_id] = idx
        self.idx_to_arango[idx] = arango_id
        self.collection_of[arango_id] = arango_id.split("/")[0]
        self._next_idx += 1
        return idx

    def __len__(self) -> int:
        return self._next_idx


class GraphLoader:
    """Load ArangoDB graph into PyG-compatible tensors.

    Args:
        database: ArangoDB database name (e.g., "NL_graph_v0")
        password: ArangoDB password (reads ARANGO_PASSWORD env var if not provided)
        host: ArangoDB host
        port: ArangoDB port
    """

    def __init__(
        self,
        database: str = "NL_graph_v0",
        password: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8529,
    ) -> None:
        import os

        self.database = database
        self.password = password or os.environ["ARANGO_PASSWORD"]
        self.host = host
        self.port = port
        self._base_url = f"http://{host}:{port}/_db/{database}"

    def _aql(self, query: str, batch_size: int = 10000) -> list[Any]:
        """Execute AQL query and return results."""
        import base64
        import json
        import urllib.request

        auth = base64.b64encode(f"root:{self.password}".encode()).decode()
        data = json.dumps({"query": query, "batchSize": batch_size}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/_api/cursor",
            data=data,
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
            },
        )
        resp = json.loads(urllib.request.urlopen(req).read())
        if resp.get("error"):
            raise RuntimeError(f"AQL error: {resp.get('errorMessage', resp)}")

        results = resp["result"]
        # Handle cursor pagination
        while resp.get("hasMore"):
            cursor_id = resp["id"]
            req = urllib.request.Request(
                f"{self._base_url}/_api/cursor/{cursor_id}",
                method="PUT",
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/json",
                },
            )
            resp = json.loads(urllib.request.urlopen(req).read())
            results.extend(resp["result"])

        return results

    def load(self) -> tuple[dict, IDMap]:
        """Load graph from ArangoDB.

        Returns:
            Tuple of (graph_data, id_map) where:
            - graph_data is a dict with:
                - "node_features": Tensor [N, JINA_DIM] — Jina embeddings (zeros if missing)
                - "has_embedding": Tensor [N] — bool mask, True if node has Jina embedding
                - "node_collections": Tensor [N] — integer collection type index
                - "edge_index": Tensor [2, E] — all edges (source, target)
                - "edge_type": Tensor [E] — relation type index per edge
                - "num_nodes": int
                - "num_edges": int
                - "num_relations": int
                - "collection_names": list[str] — collection name for each type index
            - id_map is the bidirectional ID mapping
        """
        id_map = IDMap()

        logger.info("Loading edges from %d collections...", len(EDGE_COLLECTIONS))
        all_src: list[int] = []
        all_dst: list[int] = []
        all_rel: list[int] = []

        for rel_idx, col_name in enumerate(EDGE_COLLECTIONS):
            edges = self._aql(f"FOR e IN {col_name} RETURN [e._from, e._to]")
            if not edges:
                logger.debug("  %s: empty (future slot)", col_name)
                continue
            for src_id, dst_id in edges:
                src_idx = id_map.get_or_create(src_id)
                dst_idx = id_map.get_or_create(dst_id)
                all_src.append(src_idx)
                all_dst.append(dst_idx)
                all_rel.append(rel_idx)

            logger.info("  %s: %d edges", col_name, len(edges))

        num_nodes = len(id_map)
        num_edges = len(all_src)
        logger.info("Graph: %d nodes, %d edges, %d relation types", num_nodes, num_edges, NUM_RELATIONS)

        # Build edge tensors
        edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
        edge_type = torch.tensor(all_rel, dtype=torch.long)

        # Load node features — batch by collection for efficiency
        logger.info("Loading node features...")
        node_features = torch.zeros(num_nodes, JINA_DIM, dtype=torch.float32)
        has_embedding = torch.zeros(num_nodes, dtype=torch.bool)

        # Group nodes by collection
        nodes_by_collection: dict[str, list[tuple[str, int]]] = {}
        for arango_id, idx in id_map.arango_to_idx.items():
            col = arango_id.split("/")[0]
            nodes_by_collection.setdefault(col, []).append((arango_id, idx))

        # Map collection names to integer indices
        collection_names = sorted(nodes_by_collection.keys())
        col_to_idx = {name: i for i, name in enumerate(collection_names)}
        node_collections = torch.zeros(num_nodes, dtype=torch.long)

        for col_name, node_list in nodes_by_collection.items():
            col_type_idx = col_to_idx[col_name]
            keys = [aid.split("/")[1] for aid, _ in node_list]
            idx_list = [idx for _, idx in node_list]

            # Set collection type for all nodes in this collection
            for idx in idx_list:
                node_collections[idx] = col_type_idx

            # Batch-fetch embeddings
            # AQL with IN filter for all keys at once
            keys_str = ", ".join(f'"{k}"' for k in keys)
            results = self._aql(f"FOR d IN {col_name} FILTER d._key IN [{keys_str}] " f"RETURN [d._key, d.embedding]")

            key_to_embedding = {}
            for key, emb in results:
                if emb is not None:
                    key_to_embedding[key] = emb

            embedded_count = 0
            for (_arango_id, idx), key in zip(node_list, keys, strict=False):
                if key in key_to_embedding:
                    emb = key_to_embedding[key]
                    if len(emb) == JINA_DIM:
                        node_features[idx] = torch.tensor(emb, dtype=torch.float32)
                        has_embedding[idx] = True
                        embedded_count += 1

            logger.info(
                "  %s: %d/%d nodes with embeddings",
                col_name,
                embedded_count,
                len(node_list),
            )

        total_embedded = has_embedding.sum().item()
        logger.info(
            "Features: %d/%d nodes have Jina embeddings (%.0f%%)",
            total_embedded,
            num_nodes,
            total_embedded / num_nodes * 100,
        )

        graph_data = {
            "node_features": node_features,
            "has_embedding": has_embedding,
            "node_collections": node_collections,
            "edge_index": edge_index,
            "edge_type": edge_type,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_relations": NUM_RELATIONS,
            "collection_names": collection_names,
        }

        return graph_data, id_map
