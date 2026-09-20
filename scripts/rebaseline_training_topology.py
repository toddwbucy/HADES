#!/usr/bin/env python3
"""Bounded CPU-only synthetic comparison for #14; no database or model service."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "services"), str(ROOT / "services/generated")]
from hades.training import training_pb2 as pb
from training.config import TrainingConfig
from training.server import TrainingServicer


class Context:
    async def abort(self, code, details):
        raise RuntimeError(f"{code}: {details}")


async def measure(architecture, seed, legacy):
    torch.manual_seed(seed)
    service = TrainingServicer(TrainingConfig())
    service.device = torch.device("cpu")
    service.model_config = pb.ModelConfig(architecture=architecture, num_relations=1,
        num_collection_types=1, hidden_dim=8, embed_dim=4, num_bases=1)
    service.opt_config = pb.OptimizerConfig(learning_rate=0.01)
    service._build_model(6)
    tensors = {"node_features": torch.randn(8, 6),
        "node_collections": torch.zeros(8, dtype=torch.long),
        "edge_src": torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        "edge_dst": torch.tensor([1, 2, 3, 4, 5, 6, 7, 0]),
        "edge_type": torch.zeros(8, dtype=torch.long),
        "train_idx": torch.tensor([0, 1, 2, 3]), "val_idx": torch.tensor([4, 5]),
        "test_idx": torch.tensor([6, 7])}
    context = Context()
    with tempfile.TemporaryDirectory(prefix="hades-topology-baseline-") as directory:
        path = str(Path(directory) / "graph.safetensors")
        save_file(tensors, path)
        await service.LoadGraph(pb.LoadGraphRequest(safetensors_path=path), context)
        if legacy:
            service._adjacency = lambda: (service.edge_src, service.edge_dst, service.edge_type)
        for _ in range(20):
            await service.TrainStep(pb.TrainStepRequest(train_edge_indices=[0, 1, 2, 3],
                neg_src=[0, 1, 2, 3], neg_dst=[4, 5, 6, 7]), context)
        result = await service.Evaluate(pb.EvaluateRequest(edge_indices=[6, 7],
            neg_src=[6, 7], neg_dst=[2, 3]), context)
    return {"architecture": architecture, "seed": seed,
        "adjacency": "legacy_leaked" if legacy else "train_only",
        "epochs": 20, "test_loss": result.loss, "test_accuracy": result.accuracy,
        "test_auc": result.auc}


async def main():
    torch.set_num_threads(1)
    results = [await measure(architecture, seed, legacy)
        for architecture in ("rgcn", "hetero_sage")
        for seed in (7, 29, 41) for legacy in (True, False)]
    print(json.dumps({"fixture": "synthetic directed 8-node ring; 4 train/2 val/2 test edges",
        "device": "cpu", "torch": torch.__version__,
        "limitations": "Synthetic regression baseline only; does not establish production quality. AUC correction tracked in #17.",
        "results": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
