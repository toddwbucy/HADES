"""Actual CPU update ordering through the production session/provider boundary."""
import asyncio
import threading

import grpc
import pytest
import torch

from test_training_sessions import provider, acquire, rejected
from test_training_rpc_validation import loaded_service
from hades.training import training_pb2 as pb


@pytest.mark.parametrize("interruption", ["cancel", "deadline", "shutdown"])
def test_running_cpu_work_retains_ownership_until_drained(interruption):
    async def run():
        started, release = threading.Event(), threading.Event()
        async with provider(loaded_service) as (service, a, b):
            owner = await acquire(a)
            backend = service._backend
            before = {k: v.clone() for k, v in backend.model.state_dict().items()}
            encode = backend._encode
            def gated_encode():
                started.set()
                if not release.wait(5):
                    raise RuntimeError("private compute gate timed out")
                return encode()
            backend._encode = gated_encode
            call = a.TrainStep(pb.TrainStepRequest(train_edge_indices=[0], neg_src=[3], neg_dst=[0]),
                               metadata=owner, timeout=0.2 if interruption == "deadline" else 3)
            close = None
            try:
                async def wait_started():
                    while not started.is_set():
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(wait_started(), 2)
                # This loop is responsive while real computation is admitted.
                if interruption == "cancel":
                    call.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await call
                elif interruption == "deadline":
                    with pytest.raises(grpc.aio.AioRpcError) as exc:
                        await call
                    assert exc.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                else:
                    close = asyncio.create_task(service.close())
                    await asyncio.sleep(0)
                    assert not close.done()
                assert service._backend is backend
                # Contender's own deadline can be serviced; no new owner enters.
                with pytest.raises(grpc.aio.AioRpcError) as exc:
                    await b.AcquireSession(pb.AcquireSessionRequest(), timeout=0.1)
                assert exc.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
                assert service._backend is backend
                release.set()
                if close:
                    await call
                    await asyncio.wait_for(close, 2)
                else:
                    async def wait_discarded():
                        while service._backend is not None:
                            await asyncio.sleep(0.01)
                    await asyncio.wait_for(wait_discarded(), 2)
                assert any(not torch.equal(value, backend.model.state_dict()[key])
                           for key, value in before.items())
                assert service._backend is None
                if close:
                    await rejected(b.AcquireSession, pb.AcquireSessionRequest(),
                                   code=grpc.StatusCode.UNAVAILABLE)
                else:
                    await rejected(a.RenewSession, pb.SessionRequest(), owner)
                    successor = await acquire(b)
                    assert successor != owner and service._backend is not backend
            finally:
                release.set()
                if close:
                    await close
    asyncio.run(run())


def test_repeated_handler_cancellation_keeps_worker_owned():
    async def run():
        from training.session import SessionTrainingServicer, SESSION_HEADER
        class Context:
            token = None
            def invocation_metadata(self):
                return [(SESSION_HEADER, self.token)]
            async def abort(self, code, details):
                raise AssertionError((code, details))
        context = Context()
        service = SessionTrainingServicer(loaded_service)
        context.token = (await service.AcquireSession(pb.AcquireSessionRequest(), context)).token
        backend = service._backend
        started, release = threading.Event(), threading.Event()
        original = backend._encode
        def encode():
            started.set()
            if not release.wait(5):
                raise RuntimeError("private worker did not release")
            return original()
        backend._encode = encode
        pending = asyncio.create_task(service.TrainStep(
            pb.TrainStepRequest(train_edge_indices=[0], neg_src=[3], neg_dst=[0]), context))
        try:
            async def entered():
                while not started.is_set():
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(entered(), 2)
            for _ in range(2):
                pending.cancel()
                await asyncio.sleep(0)
                assert not pending.done()
                assert service._backend is backend and service._lock.locked()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, 2)
            assert service._backend is None and not service._lock.locked()
        finally:
            release.set()
            await service.close()
    asyncio.run(run())
