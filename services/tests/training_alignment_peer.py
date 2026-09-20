"""CPU peer for the opt-in Rust/database alignment contract (not a pytest test)."""
import json
import asyncio
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
    invoke(service, "Checkpoint", pb.CheckpointRequest(path=str(root / "before.pt")))
    # Exercise a real optimizer step, then request a deliberately reversed,
    # non-contiguous subset. No stubbed encoder or hand-made output vectors.
    invoke(service, "TrainStep", pb.TrainStepRequest(
        train_edge_indices=[0, 1], neg_src=[0, 2], neg_dst=[0, 2]))
    complete = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest())
    after = torch.frombuffer(bytearray(complete.embeddings), dtype=torch.float32).reshape(3, 4)
    assert not torch.equal(before, after), "optimizer step did not change embeddings"
    invoke(service, "Checkpoint", pb.CheckpointRequest(path=str(root / "after.pt")))
    saved_before = torch.load(root / "before.pt", map_location="cpu", weights_only=True)
    saved_after = torch.load(root / "after.pt", map_location="cpu", weights_only=True)
    assert saved_before["graph_contract"] == saved_after["graph_contract"]
    assert any(not torch.equal(weight, saved_after["model"][key])
               for key, weight in saved_before["model"].items())
    for name, expected in (("before", before), ("after", after)):
        invoke(service, "LoadCheckpoint", pb.LoadCheckpointRequest(
            path=str(root / f"{name}.pt"), device="cpu"))
        restored = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest())
        expected_bytes = (root / "full.bin").read_bytes() if name == "before" else complete.embeddings
        assert restored.embeddings == expected_bytes, "checkpoint replay changed output bytes"
        actual = torch.frombuffer(bytearray(restored.embeddings), dtype=torch.float32).reshape(3, 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    subset = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest(
        node_indices=[2, 0], output_path=str(root / "subset.bin")))
    assert (subset.num_nodes, subset.embed_dim) == (2, 4)
    compact = torch.frombuffer(bytearray((root / "subset.bin").read_bytes()),
                              dtype=torch.float32).reshape(2, 4)
    torch.testing.assert_close(compact, after[[2, 0]])
    # Changed features and adjacency retain the same semantic graph contract.
    # Both are real Rust-serialized variants, using the original ID ordering.
    changes = {}
    for variant in ("features", "neighbors"):
        invoke(service, "LoadGraph", pb.LoadGraphRequest(
            safetensors_path=str(root / f"{variant}.safetensors")))
        assert service.graph_contract == saved_after["graph_contract"]
        changed = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest())
        vectors = torch.frombuffer(bytearray(changed.embeddings), dtype=torch.float32).reshape(3, 4)
        changed_ids = [ident for ident, old, new in zip(manifest["ids"], after, vectors)
                       if not torch.equal(old, new)]
        assert changed_ids, f"{variant} mutation did not change any output"
        changes[variant] = changed_ids
    print(json.dumps({"compatible_checkpoints_with_distinct_weights": True,
                      "exact_checkpoint_replay": True, "changed_output_ids": changes}), flush=True)
    from cli_training_peer import exercise
    asyncio.run(exercise(root, manifest, after))
    (root / "expected.json").write_text(json.dumps({
        "before": dict(zip(manifest["ids"], before.tolist())),
        "after": dict(zip(manifest["ids"], after.tolist())),
    }))


if __name__ == "__main__":
    main(sys.argv[1])
