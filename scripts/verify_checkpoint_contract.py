#!/usr/bin/env python3
"""Load a Rust-generated fixture and verify CPU checkpoint interoperability."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "services"), str(ROOT / "services/generated")]
from hades.training import training_pb2 as pb
from training.config import TrainingConfig
from training.server import TrainingServicer


class Context:
    async def abort(self, code, details):
        raise RuntimeError(f"{code}: {details}")


async def main(path):
    torch.set_num_threads(1)
    torch.manual_seed(29)
    with safe_open(path, framework="pt", device="cpu") as file:
        contract = json.loads(file.metadata()["graph_contract"])
        indices = file.get_tensor("train_idx").long().tolist()
        src = file.get_tensor("neg_src").long().tolist()
        dst = file.get_tensor("neg_dst").long().tolist()
    context = Context()
    service = TrainingServicer(TrainingConfig())
    model = pb.ModelConfig(architecture=contract["architecture"],
        num_relations=len(contract["relation_order"]),
        num_collection_types=len(contract["collection_names"]), hidden_dim=8, embed_dim=4, num_bases=2)
    await service.InitModel(pb.InitModelRequest(device="cpu", model=model), context)
    await service.LoadGraph(pb.LoadGraphRequest(safetensors_path=path), context)
    await service.TrainStep(pb.TrainStepRequest(train_edge_indices=indices, neg_src=src, neg_dst=dst), context)
    expected = await service.GetEmbeddings(pb.GetEmbeddingsRequest(), context)
    with tempfile.TemporaryDirectory(prefix="hades-contract-interop-") as directory:
        checkpoint = str(Path(directory) / "checkpoint.pt")
        await service.Checkpoint(pb.CheckpointRequest(path=checkpoint), context)
        restored = TrainingServicer(TrainingConfig())
        await restored.LoadCheckpoint(pb.LoadCheckpointRequest(path=checkpoint, device="cpu"), context)
        before = {key: value.clone() for key, value in restored.model.state_dict().items()}
        await restored.LoadGraph(pb.LoadGraphRequest(safetensors_path=path), context)
        for key, value in restored.model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        actual = await restored.GetEmbeddings(pb.GetEmbeddingsRequest(), context)
        assert actual.embeddings == expected.embeddings
        assert restored.model_contract == contract
    print(json.dumps({"status": "passed", "device": "cpu", "num_nodes": actual.num_nodes,
        "feature_dim": contract["feature_dim"], "contract_version": contract["version"]}))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
