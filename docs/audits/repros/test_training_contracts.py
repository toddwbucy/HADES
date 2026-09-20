"""CPU-only audit contract probes; expected to fail on the audited revision.

Run with CUDA_VISIBLE_DEVICES='' and PYTHONDONTWRITEBYTECODE=1 using the
existing services Python environment. No network or live service is used.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import grpc
import pytest
import torch
from safetensors.torch import save_file

SERVICES = Path(__file__).resolve().parents[3] / "services"
sys.path.insert(0, str(SERVICES))
sys.path.insert(0, str(SERVICES / "generated"))

from hades.training import training_pb2
from training.config import TrainingConfig
from training.server import TrainingServicer, _auc, _bce_link_loss


def loaded_servicer():
    torch.manual_seed(7)
    service = TrainingServicer(TrainingConfig())
    service.device = torch.device("cpu")
    service.model_config = training_pb2.ModelConfig(
        architecture="hetero_sage", num_relations=1, num_collection_types=1,
        hidden_dim=8, embed_dim=4, num_bases=1, dropout=0.0,
    )
    service.opt_config = training_pb2.OptimizerConfig(learning_rate=0.01)
    service._build_model(in_dim=6)
    service.x = torch.randn(4, 6)
    service.node_collections = torch.zeros(4, dtype=torch.long)
    service.edge_src = torch.tensor([0, 1, 2])
    service.edge_dst = torch.tensor([1, 2, 3])
    service.edge_type = torch.zeros(3, dtype=torch.long)
    return service


def test_equal_scores_have_chance_auc():
    assert _auc(torch.zeros(1), torch.zeros(1)) == pytest.approx(0.5)


def test_train_step_excludes_held_out_edges_from_message_passing():
    service = loaded_servicer()
    request = training_pb2.TrainStepRequest(
        train_edge_indices=[0], neg_src=[3], neg_dst=[0],
    )
    with patch.object(service.model, "encode", wraps=service.model.encode) as encode:
        asyncio.run(service.TrainStep(request, None))
    used_src, used_dst = encode.call_args.args[2:4]
    assert list(zip(used_src.tolist(), used_dst.tolist())) == [(0, 1)]


class AbortRequested(Exception):
    pass


def test_missing_model_awaits_failed_precondition():
    service = TrainingServicer(TrainingConfig())
    context = type("Context", (), {})()
    context.abort = AsyncMock(side_effect=AbortRequested)
    with pytest.raises(AbortRequested):
        asyncio.run(service.TrainStep(training_pb2.TrainStepRequest(), context))
    context.abort.assert_awaited_once_with(
        grpc.StatusCode.FAILED_PRECONDITION, "InitModel not called",
    )


def test_checkpoint_dimension_mismatch_refuses_random_reinitialization(tmp_path):
    service = loaded_servicer()
    checkpoint = tmp_path / "best.pt"
    asyncio.run(service.Checkpoint(training_pb2.CheckpointRequest(path=str(checkpoint)), None))
    restored = TrainingServicer(TrainingConfig())
    asyncio.run(restored.LoadCheckpoint(training_pb2.LoadCheckpointRequest(
        path=str(checkpoint), device="cpu",
    ), None))
    graph_path = tmp_path / "different-dimension.safetensors"
    save_file({
        "node_features": torch.randn(4, 7),
        "node_collections": service.node_collections,
        "edge_src": service.edge_src, "edge_dst": service.edge_dst,
        "edge_type": service.edge_type,
    }, str(graph_path))
    context = type("Context", (), {})()
    context.abort = AsyncMock(side_effect=AbortRequested)
    with pytest.raises((AbortRequested, ValueError)):
        asyncio.run(restored.LoadGraph(training_pb2.LoadGraphRequest(
            safetensors_path=str(graph_path),
        ), context))


def test_empty_evaluation_split_is_explicitly_rejected():
    with pytest.raises(ValueError):
        _bce_link_loss(torch.empty(0), torch.empty(0))
