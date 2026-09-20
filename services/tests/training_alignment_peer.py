"""CPU peer for the opt-in Rust/database alignment contract (not a pytest test)."""
import json
from pathlib import Path
import sys

import torch

from test_training_rpc_validation import invoke, pb
from training.config import TrainingConfig
from training.server import TrainingServicer


def main(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    service = TrainingServicer(TrainingConfig())
    invoke(service, "InitModel", pb.InitModelRequest(device="cpu", model=pb.ModelConfig(
        architecture="hetero_sage", num_relations=1, num_collection_types=2,
        hidden_dim=8, embed_dim=4, num_bases=1, dropout=0.0)))
    invoke(service, "LoadGraph", pb.LoadGraphRequest(
        safetensors_path=str(root / "graph.safetensors")))
    # The expected features come from named fixture documents, not a copy of
    # the serialized tensor. Check both collection identity and every row.
    torch.testing.assert_close(service.x.cpu(), torch.tensor(manifest["features"]))
    assert service.node_collections.tolist() == manifest["collections"]
    assert list(zip(service.edge_src.tolist(), service.edge_dst.tolist())) == [
        tuple(edge) for edge in manifest["edges"]]
    first = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest(
        output_path=str(root / "full.bin")))
    assert (first.num_nodes, first.embed_dim) == (3, 4)
    before = torch.frombuffer(bytearray((root / "full.bin").read_bytes()),
                              dtype=torch.float32).reshape(3, 4).clone()
    # Exercise a real optimizer step, then request a deliberately reversed,
    # non-contiguous subset. No stubbed encoder or hand-made output vectors.
    invoke(service, "TrainStep", pb.TrainStepRequest(
        train_edge_indices=[0, 1], neg_src=[0, 2], neg_dst=[0, 2]))
    complete = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest())
    after = torch.frombuffer(bytearray(complete.embeddings), dtype=torch.float32).reshape(3, 4)
    assert not torch.equal(before, after), "optimizer step did not change embeddings"
    subset = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest(
        node_indices=[2, 0], output_path=str(root / "subset.bin")))
    assert (subset.num_nodes, subset.embed_dim) == (2, 4)
    compact = torch.frombuffer(bytearray((root / "subset.bin").read_bytes()),
                              dtype=torch.float32).reshape(2, 4)
    torch.testing.assert_close(compact, after[[2, 0]])
    (root / "expected.json").write_text(json.dumps({
        "before": dict(zip(manifest["ids"], before.tolist())),
        "after": dict(zip(manifest["ids"], after.tolist())),
    }))


if __name__ == "__main__":
    main(sys.argv[1])
