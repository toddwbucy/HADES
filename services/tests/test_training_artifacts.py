"""Private-file publication contracts for training output and checkpoints."""
from pathlib import Path

import grpc
import numpy as np
import pytest
import torch

from test_training_rpc_validation import invoke, loaded_service
from hades.training import training_pb2 as pb


def partial_write(target):
    if hasattr(target, "write"):
        target.write(b"partial")
    else:
        Path(target).write_bytes(b"partial")
    raise OSError("injected write failure")


@pytest.mark.parametrize("method", ["Checkpoint", "GetEmbeddings"])
@pytest.mark.parametrize("existing", [True, False])
def test_failed_serialization_preserves_previous_artifact(tmp_path, monkeypatch, method, existing):
    path = tmp_path / "artifact"
    if existing:
        path.write_bytes(b"previous-complete-artifact")
    if method == "Checkpoint":
        monkeypatch.setattr(torch, "save", lambda data, target: partial_write(target))
        request = pb.CheckpointRequest(path=str(path))
    else:
        class PartialArray:
            def tofile(self, target):
                partial_write(target)
        monkeypatch.setattr(np, "ascontiguousarray", lambda *args, **kwargs: PartialArray())
        request = pb.GetEmbeddingsRequest(output_path=str(path))
    with pytest.raises(grpc.RpcError):
        invoke(loaded_service(), method, request)
    if existing:
        assert path.read_bytes() == b"previous-complete-artifact"
    else:
        assert not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == (["artifact"] if existing else [])


def test_published_artifacts_keep_existing_formats(tmp_path):
    service = loaded_service()
    inline = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest())
    output = tmp_path / "embeddings.f32"
    result = invoke(service, "GetEmbeddings", pb.GetEmbeddingsRequest(output_path=str(output)))
    assert output.read_bytes() == inline.embeddings
    assert output.stat().st_size == result.num_nodes * result.embed_dim * 4
    checkpoint = tmp_path / "checkpoint.pt"
    response = invoke(service, "Checkpoint", pb.CheckpointRequest(path=str(checkpoint)))
    assert response.size_bytes == checkpoint.stat().st_size
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["graph_contract"] == service.model_contract
    for name, value in service.model.state_dict().items():
        torch.testing.assert_close(saved["model"][name], value)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.pt", "embeddings.f32"]


@pytest.mark.parametrize("method", ["Checkpoint", "GetEmbeddings"])
@pytest.mark.parametrize("stage", ["fsync", "replace"])
def test_failed_publication_preserves_previous_artifact(tmp_path, monkeypatch, method, stage):
    from training import artifacts
    path = tmp_path / "artifact"
    path.write_bytes(b"previous")
    def fail(*args):
        raise OSError("injected publication failure")
    monkeypatch.setattr(artifacts.os, stage, fail)
    request = (pb.CheckpointRequest(path=str(path)) if method == "Checkpoint"
               else pb.GetEmbeddingsRequest(output_path=str(path)))
    with pytest.raises(grpc.RpcError):
        invoke(loaded_service(), method, request)
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


def test_publication_replaces_symlink_entry_and_uses_private_mode(tmp_path):
    target = tmp_path / "other"
    target.write_bytes(b"untouched")
    link = tmp_path / "output"
    link.symlink_to(target)
    invoke(loaded_service(), "GetEmbeddings", pb.GetEmbeddingsRequest(output_path=str(link)))
    assert not link.is_symlink()
    assert target.read_bytes() == b"untouched"
    assert link.stat().st_mode & 0o777 == 0o600
