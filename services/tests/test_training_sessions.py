"""Real two-channel CPU contracts for provider lifecycle ownership."""

import asyncio
from contextlib import asynccontextmanager
import json
import tempfile

import grpc
import pytest
import torch
from safetensors.torch import save_file

from test_training_rpc_validation import fixture_contract, loaded_service
from hades.training import training_pb2 as pb, training_pb2_grpc as rpc
from training.config import TrainingConfig
from training.server import TrainingServicer
from training.session import SESSION_HEADER, SessionTrainingServicer


@asynccontextmanager
async def provider(factory=lambda: TrainingServicer(TrainingConfig()), **kwargs):
    with tempfile.TemporaryDirectory(prefix="hades-training-session-") as directory:
        endpoint = f"unix:{directory}/rpc.sock"
        service = SessionTrainingServicer(factory, **kwargs)
        server = grpc.aio.server()
        rpc.add_TrainingServiceServicer_to_server(service, server)
        assert server.add_insecure_port(endpoint)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(endpoint) as a, grpc.aio.insecure_channel(endpoint) as b:
                yield service, rpc.TrainingServiceStub(a), rpc.TrainingServiceStub(b)
        finally:
            await server.stop(0)
            await service.close()


async def acquire(client):
    response = await client.AcquireSession(pb.AcquireSessionRequest(), timeout=3)
    assert response.lease_seconds > 0
    return [(SESSION_HEADER, response.token)]


async def rejected(call, request, metadata=(), code=grpc.StatusCode.FAILED_PRECONDITION):
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await call(request, metadata=metadata, timeout=3)
    assert error.value.code() == code


def test_competing_client_cannot_replace_or_read_owner_state(tmp_path):
    path = tmp_path / "graph.safetensors"
    save_file({"node_features": torch.ones(4, 6),
               "node_collections": torch.zeros(4, dtype=torch.long),
               "edge_src": torch.tensor([0, 1, 2]), "edge_dst": torch.tensor([1, 2, 3]),
               "edge_type": torch.zeros(3, dtype=torch.long)}, str(path),
              metadata={"graph_contract": json.dumps(fixture_contract())})

    async def run():
        async with provider() as (_, a, b):
            owner = await acquire(a)
            await a.InitModel(pb.InitModelRequest(model=loaded_service().model_config, device="cpu"),
                              metadata=owner, timeout=3)
            await a.LoadGraph(pb.LoadGraphRequest(safetensors_path=str(path)), metadata=owner, timeout=3)
            before = await a.GetEmbeddings(pb.GetEmbeddingsRequest(), metadata=owner, timeout=3)
            await rejected(b.AcquireSession, pb.AcquireSessionRequest(),
                           code=grpc.StatusCode.RESOURCE_EXHAUSTED)
            await rejected(b.InitModel, pb.InitModelRequest(model=loaded_service().model_config, device="cpu"))
            await rejected(b.GetEmbeddings, pb.GetEmbeddingsRequest())
            after = await a.GetEmbeddings(pb.GetEmbeddingsRequest(), metadata=owner, timeout=3)
            assert before.embeddings == after.embeddings
            checkpoint = tmp_path / "owner.pt"
            await a.Checkpoint(pb.CheckpointRequest(path=str(checkpoint)), metadata=owner, timeout=3)
            await a.ReleaseSession(pb.SessionRequest(), metadata=owner, timeout=3)
            successor = await acquire(b)
            assert successor != owner
            await b.LoadCheckpoint(pb.LoadCheckpointRequest(path=str(checkpoint), device="cpu"),
                                   metadata=successor, timeout=3)
            await b.LoadGraph(pb.LoadGraphRequest(safetensors_path=str(path)), metadata=successor, timeout=3)
            await rejected(a.GetEmbeddings, pb.GetEmbeddingsRequest(), owner)
            await rejected(a.ReleaseSession, pb.SessionRequest(), owner)
            restored = await b.GetEmbeddings(pb.GetEmbeddingsRequest(), metadata=successor, timeout=3)
            assert before.embeddings == restored.embeddings
    asyncio.run(run())


@pytest.mark.parametrize("method,message", [
    ("InitModel", pb.InitModelRequest()), ("LoadGraph", pb.LoadGraphRequest()),
    ("TrainStep", pb.TrainStepRequest()), ("Evaluate", pb.EvaluateRequest()),
    ("GetEmbeddings", pb.GetEmbeddingsRequest()), ("Checkpoint", pb.CheckpointRequest()),
    ("LoadCheckpoint", pb.LoadCheckpointRequest()), ("RenewSession", pb.SessionRequest()),
    ("ReleaseSession", pb.SessionRequest()),
])
def test_every_state_rpc_rejects_missing_wrong_and_duplicate_tokens(method, message):
    async def run():
        async with provider(loaded_service) as (service, a, b):
            owner = await acquire(a)
            backend = service._backend
            weights = {key: value.clone() for key, value in backend.model.state_dict().items()}
            for metadata in ([], [(SESSION_HEADER, "wrong")], owner + owner):
                await rejected(getattr(b, method), message, metadata)
            assert service._backend is backend
            for key, value in weights.items():
                torch.testing.assert_close(backend.model.state_dict()[key], value)
    asyncio.run(run())


def test_expiry_discards_abandoned_state_and_fences_old_owner():
    now = [0.0]
    async def run():
        async with provider(loaded_service, lease_seconds=5, clock=lambda: now[0]) as (service, a, b):
            old = await acquire(a)
            previous = service._backend
            now[0] = 4
            await a.RenewSession(pb.SessionRequest(), metadata=old, timeout=3)
            now[0] = 6
            await rejected(b.AcquireSession, pb.AcquireSessionRequest(),
                           code=grpc.StatusCode.RESOURCE_EXHAUSTED)
            now[0] = 10
            current = await acquire(b)
            assert current != old and service._backend is not previous
            await rejected(a.RenewSession, pb.SessionRequest(), old)
            await rejected(a.InitModel, pb.InitModelRequest(), old)
            await b.GetEmbeddings(pb.GetEmbeddingsRequest(), metadata=current, timeout=3)
    asyncio.run(run())


def test_failed_initialization_can_release_and_successor_starts_clean():
    async def run():
        async with provider() as (_, a, b):
            owner = await acquire(a)
            await rejected(a.InitModel, pb.InitModelRequest(device="cpu"), owner,
                           grpc.StatusCode.INVALID_ARGUMENT)
            await a.ReleaseSession(pb.SessionRequest(), metadata=owner, timeout=3)
            new = await acquire(b)
            await rejected(b.GetEmbeddings, pb.GetEmbeddingsRequest(), new)
            await b.InitModel(pb.InitModelRequest(model=loaded_service().model_config, device="cpu"),
                              metadata=new, timeout=3)
    asyncio.run(run())


def test_cancelled_operation_retains_owner_until_release_or_expiry():
    async def run():
        started = asyncio.Event()
        class SlowBackend:
            async def InitModel(self, request, context):
                started.set()
                await asyncio.Event().wait()
        now = [0.0]
        async with provider(SlowBackend, lease_seconds=5, clock=lambda: now[0]) as (_, a, b):
            owner = await acquire(a)
            pending = a.InitModel(pb.InitModelRequest(), metadata=owner, timeout=3)
            await asyncio.wait_for(started.wait(), 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            await a.RenewSession(pb.SessionRequest(), metadata=owner, timeout=3)
            await rejected(b.AcquireSession, pb.AcquireSessionRequest(),
                           code=grpc.StatusCode.RESOURCE_EXHAUSTED)
            now[0] = 6
            successor = await acquire(b)
            assert successor != owner
            await rejected(a.ReleaseSession, pb.SessionRequest(), owner)
    asyncio.run(run())


def test_idle_sweeper_releases_backend_without_a_successor_request():
    async def run():
        now = [0.0]
        async with provider(loaded_service, lease_seconds=1, clock=lambda: now[0]) as (service, a, _):
            await acquire(a)
            sweeper = asyncio.create_task(service.sweep_expired())
            try:
                now[0] = 2
                async def wait_discarded():
                    while service._backend is not None:
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(wait_discarded(), 2)
                assert service._token is None
            finally:
                sweeper.cancel()
                await asyncio.gather(sweeper, return_exceptions=True)
    asyncio.run(run())
