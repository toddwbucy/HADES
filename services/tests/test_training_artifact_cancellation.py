"""Cancellation and atomic checkpoint publication on the combined provider."""
import asyncio
import threading

import pytest
import torch

from test_training_sessions import provider, acquire
from test_training_rpc_validation import loaded_service
from hades.training import training_pb2 as pb


@pytest.mark.parametrize("fail_before_publish", [False, True])
def test_cancelled_checkpoint_drains_before_publication_outcome(tmp_path, monkeypatch, fail_before_publish):
    async def run():
        staged, release = threading.Event(), threading.Event()
        destination = tmp_path / "checkpoint.pt"
        destination.write_bytes(b"previous")
        save = torch.save
        def gated_save(data, stream):
            save(data, stream)
            staged.set()
            if not release.wait(5):
                raise RuntimeError("private checkpoint gate timed out")
            if fail_before_publish:
                raise OSError("injected failure after staging")
        monkeypatch.setattr(torch, "save", gated_save)
        async with provider(loaded_service) as (service, client, _):
            owner = await acquire(client)
            backend = service._backend
            call = client.Checkpoint(pb.CheckpointRequest(path=str(destination)), metadata=owner, timeout=3)
            try:
                async def wait_staged():
                    while not staged.is_set():
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(wait_staged(), 2)
                assert destination.read_bytes() == b"previous"
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call
                assert service._backend is backend
                assert destination.read_bytes() == b"previous"
                release.set()
                async def drained():
                    while service._backend is not None:
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(drained(), 2)
                assert list(tmp_path.iterdir()) == [destination]
                if fail_before_publish:
                    assert destination.read_bytes() == b"previous"
                else:
                    saved = torch.load(destination, map_location="cpu", weights_only=True)
                    assert saved["graph_contract"] == backend.model_contract
            finally:
                release.set()
    asyncio.run(run())
