"""Held-out topology must be absent in both training and evaluation (#14)."""
from unittest.mock import patch

import grpc
import pytest
import torch
from safetensors.torch import save_file

from test_training_rpc_validation import loaded_service, invoke, rejected, pb


def graph():
    return {
        "node_features": torch.arange(24, dtype=torch.float32).reshape(4, 6) / 24,
        "node_collections": torch.zeros(4, dtype=torch.long),
        "edge_src": torch.tensor([0, 1, 2]), "edge_dst": torch.tensor([1, 2, 3]),
        "edge_type": torch.zeros(3, dtype=torch.long),
        "train_idx": torch.tensor([0]), "val_idx": torch.tensor([1]), "test_idx": torch.tensor([2]),
    }


def load(service, tmp_path, tensors):
    path = tmp_path / "graph.safetensors"
    save_file(tensors, str(path))
    return invoke(service, "LoadGraph", pb.LoadGraphRequest(safetensors_path=str(path)))


@pytest.mark.parametrize("architecture", ["rgcn", "hetero_sage"])
def test_encoder_uses_only_training_adjacency_for_training_and_evaluation(tmp_path, architecture):
    service = loaded_service()
    service.model_config.architecture = architecture
    service._build_model(6)
    load(service, tmp_path, graph())
    with patch.object(service.model, "encode", wraps=service.model.encode) as spy:
        invoke(service, "TrainStep", pb.TrainStepRequest(train_edge_indices=[0], neg_src=[3], neg_dst=[0]))
        invoke(service, "Evaluate", pb.EvaluateRequest(edge_indices=[1, 2], neg_src=[3], neg_dst=[0]))
        invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest())
        assert spy.call_count == 3
        for call in spy.call_args_list:
            assert call.args[2].tolist() == [0]
            assert call.args[3].tolist() == [1]
    if architecture == "hetero_sage":
        with torch.no_grad():
            torch.testing.assert_close(service._encode_subset([2, 3]), service._encode()[[2, 3]])


def test_heldout_target_cannot_be_used_for_training(tmp_path):
    service = loaded_service()
    load(service, tmp_path, graph())
    rejected(service, "TrainStep", pb.TrainStepRequest(train_edge_indices=[1], neg_src=[3], neg_dst=[0]),
             grpc.StatusCode.INVALID_ARGUMENT)


def test_training_target_cannot_be_reported_as_heldout_evaluation(tmp_path):
    service = loaded_service()
    load(service, tmp_path, graph())
    rejected(service, "Evaluate", pb.EvaluateRequest(edge_indices=[0], neg_src=[3], neg_dst=[0]),
             grpc.StatusCode.INVALID_ARGUMENT)


@pytest.mark.parametrize("bad", ["partial", "overlap", "duplicate", "bounds", "missing_edge", "float", "inverse", "duplicate_pair"])
def test_invalid_split_preserves_existing_state(tmp_path, bad):
    service = loaded_service()
    tensors = graph()
    if bad == "partial":
        del tensors["test_idx"]
    elif bad == "overlap":
        tensors["val_idx"] = torch.tensor([0])
    elif bad == "duplicate":
        tensors["train_idx"] = torch.tensor([0, 0])
    elif bad == "bounds":
        tensors["test_idx"] = torch.tensor([3])
    elif bad == "missing_edge":
        tensors["test_idx"] = torch.tensor([], dtype=torch.long)
    elif bad == "float":
        tensors["train_idx"] = torch.tensor([0.5])
    elif bad == "inverse":
        tensors["edge_src"][1], tensors["edge_dst"][1] = 1, 0
    else:
        tensors["edge_src"][1], tensors["edge_dst"][1] = 0, 1
    path = tmp_path / "bad.safetensors"
    save_file(tensors, str(path))
    train_idx = service.train_idx
    rejected(service, "LoadGraph", pb.LoadGraphRequest(safetensors_path=str(path)), grpc.StatusCode.INVALID_ARGUMENT)
    assert service.train_idx is train_idx


def test_inference_graph_cannot_claim_heldout_evaluation(tmp_path):
    service = loaded_service()
    load(service, tmp_path, {k: v for k, v in graph().items() if not k.endswith("_idx")})
    assert service.train_idx is None
    assert invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest()).num_nodes == 4
    for method, request in [
        ("TrainStep", pb.TrainStepRequest(train_edge_indices=[0], neg_src=[3], neg_dst=[0])),
        ("Evaluate", pb.EvaluateRequest(edge_indices=[1], neg_src=[3], neg_dst=[0])),
    ]:
        rejected(service, method, request, grpc.StatusCode.FAILED_PRECONDITION)
