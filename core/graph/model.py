"""RGCN model for structural node embeddings.

Two-layer RGCN with basis decomposition for parameter-efficient
multi-relation modeling. Trained via link prediction (self-supervised).

Architecture:
    Input: Jina V4 embeddings (2048-dim) or learned type embeddings
    Linear projection: 2048 → hidden_dim
    RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases)
    ReLU + Dropout
    RGCNConv(hidden_dim, embed_dim, num_relations, num_bases)
    Output: structural embedding (embed_dim)

Usage:
    from core.graph.model import RGCNEncoder, LinkPredictor

    encoder = RGCNEncoder(num_relations=21, num_collection_types=62)
    predictor = LinkPredictor(embed_dim=128)

    z = encoder(x, edge_index, edge_type, has_embedding, node_collections)
    score = predictor(z[src], z[dst])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from core.graph.loader import JINA_DIM, NUM_RELATIONS


class RGCNEncoder(nn.Module):
    """Two-layer RGCN encoder with basis decomposition.

    Handles mixed node features: nodes with Jina embeddings use
    a linear projection, nodes without use a learned per-collection-type
    embedding.

    Args:
        num_relations: Number of edge relation types (frozen at NUM_RELATIONS)
        num_collection_types: Number of distinct node collection types
        hidden_dim: Hidden layer dimension
        embed_dim: Output structural embedding dimension
        num_bases: Number of basis matrices for decomposition
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_relations: int = NUM_RELATIONS,
        num_collection_types: int = 62,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        num_bases: int = 30,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Project Jina 2048-dim → hidden_dim for embedded nodes
        self.jina_proj = nn.Linear(JINA_DIM, hidden_dim)

        # Learned embeddings for nodes without Jina features
        self.type_embeddings = nn.Embedding(num_collection_types, hidden_dim)

        # Two RGCN convolution layers with basis decomposition
        self.conv1 = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations=num_relations,
            num_bases=min(num_bases, num_relations),
        )
        self.conv2 = RGCNConv(
            hidden_dim,
            embed_dim,
            num_relations=num_relations,
            num_bases=min(num_bases, num_relations),
        )

        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        has_embedding: torch.Tensor,
        node_collections: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass — produces structural embeddings for all nodes.

        Args:
            x: Node features [N, JINA_DIM] (zeros for unembedded nodes)
            edge_index: Edge indices [2, E]
            edge_type: Relation type per edge [E]
            has_embedding: Bool mask [N] — True if node has Jina embedding
            node_collections: Collection type index per node [N]

        Returns:
            Structural embeddings [N, embed_dim]
        """
        # L2-normalize Jina embeddings before projection (prevents gradient explosion)
        x_norm = F.normalize(x, p=2, dim=-1)

        # Project normalized Jina embeddings to hidden dim
        h = self.jina_proj(x_norm)

        # Replace unembedded nodes with learned type embedding
        unembedded = ~has_embedding
        if unembedded.any():
            h[unembedded] = self.type_embeddings(node_collections[unembedded])

        # RGCN layers
        h = self.conv1(h, edge_index, edge_type)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index, edge_type)

        # L2-normalize output embeddings for stable dot-product scoring
        h = F.normalize(h, p=2, dim=-1)

        return h


class LinkPredictor(nn.Module):
    """Link prediction decoder — dot product + sigmoid.

    Args:
        embed_dim: Dimension of structural embeddings
    """

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """Predict link probability between source and destination nodes.

        Args:
            z_src: Source node embeddings [B, embed_dim]
            z_dst: Destination node embeddings [B, embed_dim]

        Returns:
            Link probabilities [B] (after sigmoid)
        """
        return torch.sigmoid((z_src * z_dst).sum(dim=-1))
