"""RGCN training pipeline — link prediction on knowledge graph.

Handles:
  - Train/val/test edge split (80/10/10)
  - Negative sampling
  - Training with early stopping
  - Embedding export back to ArangoDB

Usage:
    from core.graph.train import RGCNTrainer

    trainer = RGCNTrainer(database="NL_graph_v0", device="cuda:2")
    trainer.load_data()
    metrics = trainer.train(epochs=200, patience=20)
    trainer.export_embeddings()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from core.graph.loader import NUM_RELATIONS, GraphLoader, IDMap
from core.graph.model import LinkPredictor, RGCNEncoder

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    hidden_dim: int = 256
    embed_dim: int = 128
    num_bases: int = 21  # Match num_relations for small graphs
    dropout: float = 0.2
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    neg_sampling_ratio: float = 1.0  # 1:1 positive:negative


class RGCNTrainer:
    """End-to-end RGCN training pipeline.

    Args:
        database: ArangoDB database name
        device: torch device string
        config: Training configuration
        model_dir: Directory to save trained models
    """

    def __init__(
        self,
        database: str = "NL_graph_v0",
        device: str = "cuda:2",
        config: TrainConfig | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.database = database
        self.device = torch.device(device)
        self.config = config or TrainConfig()
        self.model_dir = model_dir or Path("/home/todd/git/HADES/models/rgcn")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.graph_data: dict | None = None
        self.id_map: IDMap | None = None
        self.encoder: RGCNEncoder | None = None
        self.predictor: LinkPredictor | None = None

    def load_data(self) -> dict:
        """Load graph from ArangoDB and prepare train/val/test splits."""
        loader = GraphLoader(database=self.database)
        self.graph_data, self.id_map = loader.load()

        # Move tensors to device
        for key in ["node_features", "has_embedding", "node_collections", "edge_index", "edge_type"]:
            self.graph_data[key] = self.graph_data[key].to(self.device)

        # Split edges into train/val/test
        self._split_edges()

        return self.graph_data

    def _split_edges(self) -> None:
        """Split edges into train/val/test sets (80/10/10)."""
        num_edges = self.graph_data["num_edges"]
        perm = torch.randperm(num_edges)

        val_size = int(num_edges * self.config.val_ratio)
        test_size = int(num_edges * self.config.test_ratio)
        train_size = num_edges - val_size - test_size

        self.graph_data["train_mask"] = perm[:train_size]
        self.graph_data["val_mask"] = perm[train_size : train_size + val_size]
        self.graph_data["test_mask"] = perm[train_size + val_size :]

        logger.info(
            "Edge split: %d train, %d val, %d test",
            train_size,
            val_size,
            test_size,
        )

    def _negative_sample(self, edge_index: torch.Tensor, num_neg: int) -> torch.Tensor:
        """Sample negative edges (non-existing node pairs).

        Args:
            edge_index: Positive edge indices [2, E_pos]
            num_neg: Number of negative samples to generate

        Returns:
            Negative edge indices [2, num_neg]
        """
        num_nodes = self.graph_data["num_nodes"]

        # Build set of existing edges for fast lookup
        existing = set()
        src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        for s, d in zip(src, dst, strict=False):
            existing.add((int(s), int(d)))

        neg_src, neg_dst = [], []
        attempts = 0
        max_attempts = num_neg * 10

        while len(neg_src) < num_neg and attempts < max_attempts:
            s = torch.randint(0, num_nodes, (num_neg,))
            d = torch.randint(0, num_nodes, (num_neg,))

            for si, di in zip(s.tolist(), d.tolist(), strict=False):
                if si != di and (si, di) not in existing:
                    neg_src.append(si)
                    neg_dst.append(di)
                    if len(neg_src) >= num_neg:
                        break
            attempts += num_neg

        neg_src = neg_src[:num_neg]
        neg_dst = neg_dst[:num_neg]

        return torch.tensor([neg_src, neg_dst], dtype=torch.long, device=self.device)

    def train(self, epochs: int | None = None, patience: int | None = None) -> dict:
        """Train the RGCN via link prediction.

        Args:
            epochs: Max training epochs (overrides config)
            patience: Early stopping patience (overrides config)

        Returns:
            Dict with training metrics
        """
        if self.graph_data is None:
            raise RuntimeError("Call load_data() before train()")

        epochs = epochs or self.config.epochs
        patience = patience or self.config.patience

        cfg = self.config
        num_collection_types = len(self.graph_data["collection_names"])

        self.encoder = RGCNEncoder(
            num_relations=NUM_RELATIONS,
            num_collection_types=num_collection_types,
            hidden_dim=cfg.hidden_dim,
            embed_dim=cfg.embed_dim,
            num_bases=cfg.num_bases,
            dropout=cfg.dropout,
        ).to(self.device)

        self.predictor = LinkPredictor(embed_dim=cfg.embed_dim).to(self.device)

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Full graph data for message passing
        x = self.graph_data["node_features"]
        edge_index = self.graph_data["edge_index"]
        edge_type = self.graph_data["edge_type"]
        has_emb = self.graph_data["has_embedding"]
        node_cols = self.graph_data["node_collections"]

        # Use train edges for supervision
        train_mask = self.graph_data["train_mask"]
        val_mask = self.graph_data["val_mask"]

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        train_losses = []
        val_losses = []

        start_time = time.time()
        logger.info("Starting RGCN training (%d epochs, patience=%d)...", epochs, patience)

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.encoder.train()
            self.predictor.train()
            optimizer.zero_grad()

            # Encode all nodes using full graph structure
            z = self.encoder(x, edge_index, edge_type, has_emb, node_cols)

            # Positive edges (train split)
            pos_edge = edge_index[:, train_mask]
            pos_score = self.predictor(z[pos_edge[0]], z[pos_edge[1]])
            pos_label = torch.ones(pos_edge.size(1), device=self.device)

            # Negative sampling
            num_neg = int(pos_edge.size(1) * cfg.neg_sampling_ratio)
            neg_edge = self._negative_sample(edge_index, num_neg)
            neg_score = self.predictor(z[neg_edge[0]], z[neg_edge[1]])
            neg_label = torch.zeros(neg_edge.size(1), device=self.device)

            # Binary cross-entropy loss
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy(scores, labels)

            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            train_losses.append(train_loss)

            # --- Validate ---
            self.encoder.eval()
            self.predictor.eval()
            with torch.no_grad():
                z = self.encoder(x, edge_index, edge_type, has_emb, node_cols)

                val_pos_edge = edge_index[:, val_mask]
                val_pos_score = self.predictor(z[val_pos_edge[0]], z[val_pos_edge[1]])
                val_pos_label = torch.ones(val_pos_edge.size(1), device=self.device)

                val_neg_edge = self._negative_sample(edge_index, val_pos_edge.size(1))
                val_neg_score = self.predictor(z[val_neg_edge[0]], z[val_neg_edge[1]])
                val_neg_label = torch.zeros(val_neg_edge.size(1), device=self.device)

                val_scores = torch.cat([val_pos_score, val_neg_score])
                val_labels = torch.cat([val_pos_label, val_neg_label])
                val_loss = F.binary_cross_entropy(val_scores, val_labels).item()
                val_losses.append(val_loss)

                # Accuracy
                val_preds = (val_scores > 0.5).float()
                val_acc = (val_preds == val_labels).float().mean().item()

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f",
                    epoch,
                    train_loss,
                    val_loss,
                    val_acc,
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {
                    "encoder": {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()},
                    "predictor": {k: v.cpu().clone() for k, v in self.predictor.state_dict().items()},
                }
            elif epoch - best_epoch >= patience:
                logger.info("Early stopping at epoch %d (best: %d)", epoch, best_epoch)
                break

        # Restore best model
        if best_state:
            self.encoder.load_state_dict({k: v.to(self.device) for k, v in best_state["encoder"].items()})
            self.predictor.load_state_dict({k: v.to(self.device) for k, v in best_state["predictor"].items()})

        elapsed = time.time() - start_time

        # Test assessment
        test_metrics = self._assess_test()

        # Save model
        model_path = self.model_dir / f"rgcn_{self.database}.pt"
        torch.save(
            {
                "encoder_state": best_state["encoder"] if best_state else self.encoder.state_dict(),
                "predictor_state": best_state["predictor"] if best_state else self.predictor.state_dict(),
                "config": {
                    "hidden_dim": cfg.hidden_dim,
                    "embed_dim": cfg.embed_dim,
                    "num_bases": cfg.num_bases,
                    "num_relations": NUM_RELATIONS,
                    "num_collection_types": len(self.graph_data["collection_names"]),
                    "collection_names": self.graph_data["collection_names"],
                },
                "database": self.database,
                "num_nodes": self.graph_data["num_nodes"],
                "num_edges": self.graph_data["num_edges"],
            },
            model_path,
        )
        logger.info("Model saved to %s", model_path)

        metrics = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "elapsed_seconds": elapsed,
            "model_path": str(model_path),
            **test_metrics,
        }

        return metrics

    def _assess_test(self) -> dict:
        """Assess on test set."""
        self.encoder.eval()
        self.predictor.eval()

        with torch.no_grad():
            x = self.graph_data["node_features"]
            edge_index = self.graph_data["edge_index"]
            edge_type = self.graph_data["edge_type"]
            has_emb = self.graph_data["has_embedding"]
            node_cols = self.graph_data["node_collections"]
            test_mask = self.graph_data["test_mask"]

            z = self.encoder(x, edge_index, edge_type, has_emb, node_cols)

            # Positive test edges
            test_pos_edge = edge_index[:, test_mask]
            test_pos_score = self.predictor(z[test_pos_edge[0]], z[test_pos_edge[1]])

            # Negative test edges
            test_neg_edge = self._negative_sample(edge_index, test_pos_edge.size(1))
            test_neg_score = self.predictor(z[test_neg_edge[0]], z[test_neg_edge[1]])

            # Metrics
            scores = torch.cat([test_pos_score, test_neg_score])
            labels = torch.cat(
                [
                    torch.ones(test_pos_edge.size(1), device=self.device),
                    torch.zeros(test_neg_edge.size(1), device=self.device),
                ]
            )

            preds = (scores > 0.5).float()
            acc = (preds == labels).float().mean().item()

            # AUC via sorting
            sorted_indices = torch.argsort(scores, descending=True)
            sorted_labels = labels[sorted_indices]
            num_pos = sorted_labels.sum().item()
            num_neg_total = len(sorted_labels) - num_pos

            if num_pos > 0 and num_neg_total > 0:
                tps = torch.cumsum(sorted_labels, dim=0)
                fps = torch.cumsum(1 - sorted_labels, dim=0)
                tpr = tps / num_pos
                fpr = fps / num_neg_total
                auc = torch.trapezoid(tpr, fpr).item()
            else:
                auc = 0.0

        logger.info("Test: acc=%.3f, auc=%.3f", acc, auc)
        return {"test_acc": acc, "test_auc": auc}

    def load_model(self, model_path: str | Path | None = None) -> dict:
        """Load a trained RGCN model from checkpoint.

        Args:
            model_path: Path to .pt checkpoint. Defaults to models/rgcn/rgcn_{database}.pt

        Returns:
            Checkpoint config dict
        """
        path = Path(model_path) if model_path else self.model_dir / f"rgcn_{self.database}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run 'hades graph-embed train' first.")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]

        self.encoder = RGCNEncoder(
            num_relations=cfg["num_relations"],
            num_collection_types=cfg["num_collection_types"],
            hidden_dim=cfg["hidden_dim"],
            embed_dim=cfg["embed_dim"],
            num_bases=cfg["num_bases"],
        ).to(self.device)

        self.predictor = LinkPredictor(embed_dim=cfg["embed_dim"]).to(self.device)

        self.encoder.load_state_dict(
            {k: v.to(self.device) for k, v in checkpoint["encoder_state"].items()}
        )
        self.predictor.load_state_dict(
            {k: v.to(self.device) for k, v in checkpoint["predictor_state"].items()}
        )

        logger.info(
            "Loaded model from %s (trained on %d nodes, %d edges)",
            path,
            checkpoint.get("num_nodes", 0),
            checkpoint.get("num_edges", 0),
        )
        return cfg

    def update_embeddings(self, target_database: str | None = None) -> dict:
        """Incremental update: reload graph, re-embed with trained model, export.

        This is the inductive update path — no retraining needed. The trained
        RGCN model produces embeddings for all nodes (including newly added ones)
        by running a single forward pass over the current graph.

        Args:
            target_database: Database to write embeddings to (defaults to source)

        Returns:
            Dict with update metrics
        """
        if self.encoder is None:
            raise RuntimeError("Load a model first via load_model()")
        if self.graph_data is None:
            raise RuntimeError("Load graph data first via load_data()")

        old_nodes = self.graph_data.get("_prev_num_nodes", self.graph_data["num_nodes"])
        num_nodes = self.graph_data["num_nodes"]
        num_edges = self.graph_data["num_edges"]

        # export_embeddings() calls get_embeddings() internally
        n_exported = self.export_embeddings(target_database=target_database)

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "new_nodes": max(0, num_nodes - old_nodes),
            "nodes_exported": n_exported,
            "target_database": target_database or self.database,
        }

    def get_embeddings(self) -> torch.Tensor:
        """Get structural embeddings for all nodes.

        Returns:
            Tensor [N, embed_dim] of structural embeddings
        """
        if self.encoder is None:
            raise RuntimeError("Train or load a model first")

        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(
                self.graph_data["node_features"],
                self.graph_data["edge_index"],
                self.graph_data["edge_type"],
                self.graph_data["has_embedding"],
                self.graph_data["node_collections"],
            )
        return z

    def export_embeddings(self, target_database: str | None = None) -> int:
        """Export structural embeddings back to ArangoDB.

        Writes a `structural_embedding` field to each document.

        Args:
            target_database: Database to write to (defaults to source database)

        Returns:
            Number of nodes updated
        """
        import base64
        import json
        import os
        import urllib.request

        target_db = target_database or self.database
        pw = os.environ["ARANGO_PASSWORD"]
        auth = base64.b64encode(f"root:{pw}".encode()).decode()
        base_url = f"http://127.0.0.1:8529/_db/{target_db}"

        z = self.get_embeddings().cpu()
        updated = 0

        # Group by collection for batch updates
        nodes_by_col: dict[str, list[tuple[str, list[float]]]] = {}
        for idx in range(len(self.id_map)):
            arango_id = self.id_map.idx_to_arango[idx]
            col, key = arango_id.split("/", 1)
            embedding = z[idx].tolist()
            nodes_by_col.setdefault(col, []).append((key, embedding))

        for col, nodes in nodes_by_col.items():
            # Batch update via AQL
            keys = [k for k, _ in nodes]
            embeddings = [e for _, e in nodes]

            # Chunk to avoid huge AQL queries
            chunk_size = 100
            for i in range(0, len(keys), chunk_size):
                chunk_keys = keys[i : i + chunk_size]
                chunk_embs = embeddings[i : i + chunk_size]

                updates = [{"_key": k, "structural_embedding": e} for k, e in zip(chunk_keys, chunk_embs, strict=False)]
                query = f"""
                FOR u IN @updates
                  UPDATE u._key WITH {{ structural_embedding: u.structural_embedding }}
                  IN {col}
                  RETURN 1
                """
                data = json.dumps(
                    {
                        "query": query,
                        "bindVars": {"updates": updates},
                    }
                ).encode()
                req = urllib.request.Request(
                    f"{base_url}/_api/cursor",
                    data=data,
                    headers={
                        "Authorization": f"Basic {auth}",
                        "Content-Type": "application/json",
                    },
                )
                resp = json.loads(urllib.request.urlopen(req).read())
                if not resp.get("error"):
                    updated += len(resp.get("result", []))
                else:
                    logger.warning("Error updating %s: %s", col, resp.get("errorMessage"))

            logger.info("  %s: updated %d nodes", col, len(nodes))

        logger.info("Exported %d structural embeddings to %s", updated, target_db)
        return updated
