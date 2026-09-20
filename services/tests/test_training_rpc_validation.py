"""CPU RPC contracts: rejected requests have a status and preserve state (#18)."""

import asyncio
import sys
import tempfile
from pathlib import Path

import grpc
import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generated"))

from hades.training import training_pb2 as pb, training_pb2_grpc as rpc
from training.config import TrainingConfig
from training.server import TrainingServicer


def loaded_service():
    service = TrainingServicer(TrainingConfig())
    service.device = torch.device("cpu")
    service.model_config = pb.ModelConfig(
        architecture="hetero_sage", num_relations=1, num_collection_types=1,
        hidden_dim=8, embed_dim=4, num_bases=1,
    )
    service.opt_config = pb.OptimizerConfig(learning_rate=0.01)
    service._build_model(6)
    service.x = torch.randn(4, 6)
    service.node_collections = torch.zeros(4, dtype=torch.long)
    service.edge_src = torch.tensor([0, 1, 2])
    service.edge_dst = torch.tensor([1, 2, 3])
    service.edge_type = torch.zeros(3, dtype=torch.long)
    service.train_idx = torch.tensor([0])
    return service


def invoke(service, method, request):
    async def run():
        with tempfile.TemporaryDirectory(prefix="hades-rpc-") as directory:
            endpoint = f"unix:{directory}/rpc.sock"
            server = grpc.aio.server()
            rpc.add_TrainingServiceServicer_to_server(service, server)
            assert server.add_insecure_port(endpoint)
            await server.start()
            try:
                async with grpc.aio.insecure_channel(endpoint) as channel:
                    return await getattr(rpc.TrainingServiceStub(channel), method)(request, timeout=3)
            finally:
                await server.stop(0)
    return asyncio.run(run())


def rejected(service, method, request, code):
    model, graph = service.model, service.x
    mode = model.training if model is not None else None
    weights = {k: v.clone() for k, v in model.state_dict().items()} if model is not None else {}
    with pytest.raises(grpc.aio.AioRpcError) as error:
        invoke(service, method, request)
    assert error.value.code() == code
    assert service.model is model and service.x is graph
    if model is not None:
        assert service.model.training == mode
        for k, v in weights.items():
            torch.testing.assert_close(service.model.state_dict()[k], v)


@pytest.mark.parametrize("method,message", [
    ("TrainStep", pb.TrainStepRequest()), ("Evaluate", pb.EvaluateRequest()),
    ("GetEmbeddings", pb.GetEmbeddingsRequest()), ("LoadGraph", pb.LoadGraphRequest()),
    ("Checkpoint", pb.CheckpointRequest()),
])
def test_no_model_returns_failed_precondition(method, message):
    rejected(TrainingServicer(TrainingConfig()), method, message, grpc.StatusCode.FAILED_PRECONDITION)


@pytest.mark.parametrize("method,message", [
    ("TrainStep", pb.TrainStepRequest()), ("Evaluate", pb.EvaluateRequest()),
    ("GetEmbeddings", pb.GetEmbeddingsRequest()),
])
def test_no_graph_returns_failed_precondition(method, message):
    service = loaded_service()
    service._clear_graph()
    rejected(service, method, message, grpc.StatusCode.FAILED_PRECONDITION)


@pytest.mark.parametrize("method", ["TrainStep", "Evaluate"])
@pytest.mark.parametrize("indices,src,dst", [([], [3], [0]), ([3], [3], [0]),
    ([0], [], []), ([0], [3], []), ([0], [4], [0]), ([0], [3], [4])])
def test_invalid_loss_indices_do_not_mutate_state(method, indices, src, dst):
    service = loaded_service()
    request = (pb.TrainStepRequest(train_edge_indices=indices, neg_src=src, neg_dst=dst)
        if method == "TrainStep" else pb.EvaluateRequest(edge_indices=indices, neg_src=src, neg_dst=dst))
    rejected(service, method, request, grpc.StatusCode.INVALID_ARGUMENT)


@pytest.mark.parametrize("indices", [[4], [1, 1]])
def test_bad_embedding_indices_do_not_change_model_mode(indices):
    rejected(loaded_service(), "GetEmbeddings", pb.GetEmbeddingsRequest(node_indices=indices),
             grpc.StatusCode.INVALID_ARGUMENT)


@pytest.mark.parametrize("bad", ["missing", "shape", "float_index", "bad_node", "bad_relation", "nan"])
def test_malformed_graph_does_not_replace_valid_graph(tmp_path, bad):
    tensors = {"node_features": torch.randn(4, 6), "node_collections": torch.zeros(4, dtype=torch.long),
        "edge_src": torch.tensor([0]), "edge_dst": torch.tensor([1]), "edge_type": torch.tensor([0])}
    if bad == "missing":
        del tensors["edge_type"]
    elif bad == "shape":
        tensors["node_features"] = torch.zeros(4)
    elif bad == "float_index":
        tensors["edge_src"] = torch.tensor([0.5])
    elif bad == "bad_node":
        tensors["edge_dst"] = torch.tensor([4])
    elif bad == "bad_relation":
        tensors["edge_type"] = torch.tensor([1])
    else:
        tensors["node_features"][0, 0] = float("nan")
    path = tmp_path / "graph.safetensors"
    save_file(tensors, str(path))
    rejected(loaded_service(), "LoadGraph", pb.LoadGraphRequest(safetensors_path=str(path)),
             grpc.StatusCode.INVALID_ARGUMENT)


def test_invalid_checkpoint_does_not_replace_model(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model_config": {"num_relations": 1, "num_collection_types": 1},
                "model": {"wrong": torch.zeros(1)}, "in_dim": 6}, path)
    rejected(loaded_service(), "LoadCheckpoint", pb.LoadCheckpointRequest(path=str(path), device="cpu"),
             grpc.StatusCode.INVALID_ARGUMENT)


def test_missing_checkpoint_has_not_found_status(tmp_path):
    rejected(loaded_service(), "LoadCheckpoint",
             pb.LoadCheckpointRequest(path=str(tmp_path / "absent.pt"), device="cpu"),
             grpc.StatusCode.NOT_FOUND)


def test_invalid_model_configuration_preserves_loaded_model():
    request = pb.InitModelRequest(device="cpu", model=pb.ModelConfig(architecture="unknown"))
    rejected(loaded_service(), "InitModel", request, grpc.StatusCode.INVALID_ARGUMENT)


def test_reinitialization_requires_loading_a_new_graph():
    service = loaded_service()
    invoke(service, "InitModel", pb.InitModelRequest(device="cpu", model=service.model_config))
    assert service.x is None
    rejected(service, "GetEmbeddings", pb.GetEmbeddingsRequest(), grpc.StatusCode.FAILED_PRECONDITION)


@pytest.mark.parametrize("field", ["num_relations", "num_collection_types"])
def test_checkpoint_changed_type_bounds_requires_graph_reload(tmp_path, field):
    service = loaded_service()
    setattr(service.model_config, field, 2)
    service._build_model(6)
    source = loaded_service()
    path = tmp_path / "smaller.pt"
    invoke(source, "Checkpoint", pb.CheckpointRequest(path=str(path)))
    invoke(service, "LoadCheckpoint", pb.LoadCheckpointRequest(path=str(path), device="cpu"))
    assert service.x is None
    rejected(service, "GetEmbeddings", pb.GetEmbeddingsRequest(), grpc.StatusCode.FAILED_PRECONDITION)


def test_valid_training_and_checkpoint_roundtrip(tmp_path):
    service = loaded_service()
    step = invoke(service, "TrainStep", pb.TrainStepRequest(train_edge_indices=[0], neg_src=[3], neg_dst=[0]))
    assert step.loss > 0
    path = tmp_path / "valid.pt"
    invoke(service, "Checkpoint", pb.CheckpointRequest(path=str(path)))
    saved = {k: v.clone() for k, v in service.model.state_dict().items()}
    invoke(service, "LoadCheckpoint", pb.LoadCheckpointRequest(path=str(path), device="cpu"))
    for key, value in saved.items():
        torch.testing.assert_close(service.model.state_dict()[key], value)
    response = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest())
    assert response.num_nodes == 4
