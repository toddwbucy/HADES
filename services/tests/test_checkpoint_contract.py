"""Checkpoint compatibility must preserve weights and active state on rejection."""
from copy import deepcopy
import json

import grpc
import pytest
import torch
from safetensors.torch import save_file

from test_training_rpc_validation import loaded_service, invoke, rejected, fixture_contract, pb
from training.config import TrainingConfig
from training.server import TrainingServicer


def write_graph(path, contract, *, metadata=True):
    tensors = {
        "node_features": torch.arange(4 * contract["feature_dim"], dtype=torch.float32).reshape(4, -1) / 10,
        "node_collections": torch.tensor([0, 1, 0, 1]),
        "edge_src": torch.tensor([0, 1, 2]), "edge_dst": torch.tensor([1, 2, 3]),
        "edge_type": torch.tensor([0, 1, 0]),
        "train_idx": torch.tensor([0]), "val_idx": torch.tensor([1]), "test_idx": torch.tensor([2]),
    }
    save_file(tensors, str(path), metadata={"graph_contract": json.dumps(contract)} if metadata else None)


def reject_graph_preserving_everything(service, path):
    optimizer = service.optimizer
    optimizer_state = deepcopy(optimizer.state_dict())
    model_contract, graph_contract = service.model_contract, service.graph_contract
    indices, width = service.train_idx, service.in_dim
    rejected(service, "LoadGraph", pb.LoadGraphRequest(safetensors_path=str(path)), grpc.StatusCode.INVALID_ARGUMENT)
    assert service.optimizer is optimizer
    torch.testing.assert_close(optimizer.state_dict(), optimizer_state)
    assert service.model_contract is model_contract and service.graph_contract is graph_contract
    assert service.train_idx is indices and service.in_dim == width


@pytest.mark.parametrize("architecture", ["rgcn", "hetero_sage"])
def test_checkpoint_roundtrip_reuses_weights_and_compatible_graph(tmp_path, architecture):
    service = loaded_service(architecture, relations=2, collections=2)
    graph = tmp_path / "graph.safetensors"
    write_graph(graph, service.model_contract)
    invoke(service, "LoadGraph", pb.LoadGraphRequest(safetensors_path=str(graph)))
    invoke(service, "TrainStep", pb.TrainStepRequest(train_edge_indices=[0], neg_src=[3], neg_dst=[0]))
    expected = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest())
    checkpoint = str(tmp_path / "model.pt")
    invoke(service, "Checkpoint", pb.CheckpointRequest(path=checkpoint))
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["graph_contract"] == service.model_contract
    restored = TrainingServicer(TrainingConfig())
    invoke(restored, "LoadCheckpoint", pb.LoadCheckpointRequest(path=checkpoint, device="cpu"))
    assert restored.x is None
    weights = {k: v.clone() for k, v in restored.model.state_dict().items()}
    invoke(restored, "LoadGraph", pb.LoadGraphRequest(safetensors_path=str(graph)))
    for key, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, weights[key])
    actual = invoke(restored, "GetEmbeddings", pb.GetEmbeddingsRequest())
    assert actual.embeddings == expected.embeddings
    # Restoring the best snapshot of the same contract keeps its loaded graph.
    old_graph = restored.x
    invoke(restored, "LoadCheckpoint", pb.LoadCheckpointRequest(path=checkpoint, device="cpu"))
    assert restored.x is old_graph


@pytest.mark.parametrize("change", ["dimension", "relations", "collections", "model", "architecture",
    "policy", "version", "missing_model", "missing_contract", "empty_models", "mixed_models"])
def test_incompatible_graph_preserves_restored_weights_and_graph(tmp_path, change):
    service = loaded_service(relations=2, collections=2)
    checkpoint = str(tmp_path / "checkpoint.pt")
    invoke(service, "Checkpoint", pb.CheckpointRequest(path=checkpoint))
    invoke(service, "LoadCheckpoint", pb.LoadCheckpointRequest(path=checkpoint, device="cpu"))
    contract = deepcopy(service.model_contract)
    if change == "dimension":
        contract["feature_dim"] = 7
    elif change == "relations":
        contract["relation_order"].reverse()
    elif change == "collections":
        contract["collection_names"].reverse()
    elif change == "model":
        contract["feature_models"]["collection_0"] = ["different-model:v2"]
    elif change == "architecture":
        contract["architecture"] = "rgcn"
    elif change == "policy":
        contract["feature_policy"] = "different-pooling"
    elif change == "version":
        contract["version"] = 2
    elif change == "missing_model":
        del contract["feature_models"]["collection_0"]
    elif change == "empty_models":
        contract["feature_models"]["collection_0"] = []
    elif change == "mixed_models":
        contract["feature_models"]["collection_0"] = ["a", "b"]
    path = tmp_path / "incompatible.safetensors"
    write_graph(path, contract, metadata=change != "missing_contract")
    reject_graph_preserving_everything(service, path)


def test_fresh_dimension_setup_is_allowed_once(tmp_path):
    service = TrainingServicer(TrainingConfig())
    model = pb.ModelConfig(architecture="hetero_sage", num_relations=2,
        num_collection_types=2, hidden_dim=8, embed_dim=4)
    invoke(service, "InitModel", pb.InitModelRequest(device="cpu", model=model))
    path = tmp_path / "first.safetensors"
    contract = fixture_contract(dimension=7, relations=2, collections=2)
    write_graph(path, contract)
    invoke(service, "LoadGraph", pb.LoadGraphRequest(safetensors_path=str(path)))
    assert service.in_dim == 7
    contract["feature_dim"] = 8
    write_graph(path, contract)
    reject_graph_preserving_everything(service, path)


def test_legacy_checkpoint_is_rejected_without_replacing_active_state(tmp_path):
    service = loaded_service()
    path = tmp_path / "legacy.pt"
    torch.save({"in_dim": 6, "model": service.model.state_dict(), "model_config": {
        "architecture": "hetero_sage", "num_relations": 1, "num_collection_types": 1,
        "hidden_dim": 8, "embed_dim": 4, "num_bases": 1}}, path)
    rejected(service, "LoadCheckpoint", pb.LoadCheckpointRequest(path=str(path), device="cpu"),
             grpc.StatusCode.INVALID_ARGUMENT)


def test_new_checkpoint_cannot_bind_an_old_graph_with_same_type_counts(tmp_path):
    service = loaded_service()
    other = loaded_service()
    other.model_contract = deepcopy(other.model_contract)
    other.model_contract["relation_order"] = ["different-relation"]
    path = str(tmp_path / "other.pt")
    invoke(other, "Checkpoint", pb.CheckpointRequest(path=path))
    invoke(service, "LoadCheckpoint", pb.LoadCheckpointRequest(path=path, device="cpu"))
    assert service.x is None and service.graph_contract is None


def test_unbound_checkpoint_does_not_create_file(tmp_path):
    service = TrainingServicer(TrainingConfig())
    model = pb.ModelConfig(architecture="hetero_sage", num_relations=1, num_collection_types=1,
                          hidden_dim=8, embed_dim=4)
    invoke(service, "InitModel", pb.InitModelRequest(device="cpu", model=model))
    path = tmp_path / "unbound.pt"
    rejected(service, "Checkpoint", pb.CheckpointRequest(path=str(path)), grpc.StatusCode.FAILED_PRECONDITION)
    assert not path.exists()
