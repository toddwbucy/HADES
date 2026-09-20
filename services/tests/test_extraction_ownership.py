"""CPU-only extraction worker lifetime tests; no model or live service."""
import asyncio
from pathlib import Path
import sys
import threading

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extraction.server import ExtractionServicer, ExtractionConfig, ExtractionResult, extraction_pb2 as pb


class Context:
    async def abort(self, code, detail):
        raise RuntimeError(code.name)

    def set_code(self, code):
        self.code = code

    def set_details(self, detail):
        self.detail = detail


def request():
    return pb.ExtractRequest(file_path="fixture.txt", content=b"synthetic", source_type=pb.SOURCE_TYPE_TEXT)


async def started(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), 2)


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("error", [False, True])
def test_worker_retains_input_and_idle_ownership(cancel, error):
    async def run():
        service = ExtractionServicer(ExtractionConfig(device="cpu"))
        entered, release = threading.Event(), threading.Event()
        paths = []
        def worker(path):
            paths.append(Path(path))
            entered.set()
            if not release.wait(5):
                raise RuntimeError("private gate timed out")
            assert paths[0].read_bytes() == b"synthetic"
            if error:
                raise ValueError("fixture worker failure")
            return ExtractionResult(text="synthetic")
        service._extract_text = worker
        task = asyncio.create_task(service.Extract(request(), Context()))
        try:
            await started(entered)
            if cancel:
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0)
                assert not task.done()
            assert service._active_requests == 1
            assert not service._idle.is_set()
            assert paths[0].exists()
            with pytest.raises(RuntimeError, match="workers are active"):
                service.unload_models()
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            elif error:
                with pytest.raises(ValueError, match="fixture worker failure"):
                    await asyncio.wait_for(task, 2)
            else:
                assert (await asyncio.wait_for(task, 2)).full_text == "synthetic"
            assert service._active_requests == 0
            assert service._idle.is_set()
            assert not paths[0].exists()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await service.close()
    asyncio.run(run())


def test_close_drains_and_rejects_admission_despite_repeated_cancellation():
    async def run():
        service = ExtractionServicer(ExtractionConfig(device="cpu"))
        entered, release = threading.Event(), threading.Event()
        cleaned = []
        class Backend:
            def cleanup(self):
                assert service._active_requests == 0
                cleaned.append(True)
        service._docling = Backend()
        def worker(path):
            entered.set()
            assert release.wait(5)
            return ExtractionResult(text=Path(path).read_text())
        service._extract_text = worker
        task = asyncio.create_task(service.Extract(request(), Context()))
        await started(entered)
        close = asyncio.create_task(service.close())
        try:
            await asyncio.sleep(0)
            with pytest.raises(RuntimeError, match="UNAVAILABLE"):
                await service.Extract(request(), Context())
            for _ in range(3):
                close.cancel()
                await asyncio.sleep(0)
            assert not close.done()
            assert not cleaned
            release.set()
            assert (await asyncio.wait_for(task, 2)).full_text == "synthetic"
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(close, 2)
            assert cleaned == [True]
            await service.close()
            assert cleaned == [True]
        finally:
            release.set()
            await asyncio.gather(task, close, return_exceptions=True)
            await service.close()
    asyncio.run(run())


@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
def test_private_rpc_interruption_keeps_worker_resources(tmp_path, interruption):
    from extraction.server import extraction_pb2_grpc as rpc
    async def run():
        service = ExtractionServicer(ExtractionConfig(device="cpu"))
        entered, release = threading.Event(), threading.Event()
        paths = []
        def worker(path):
            paths.append(Path(path))
            entered.set()
            assert release.wait(5)
            return ExtractionResult(text=Path(path).read_text())
        service._extract_text = worker
        server = grpc.aio.server()
        rpc.add_ExtractionServiceServicer_to_server(service, server)
        address = f"unix:{tmp_path / 'extract.sock'}"
        assert server.add_insecure_port(address)
        await server.start()
        channel = grpc.aio.insecure_channel(address)
        try:
            stub = rpc.ExtractionServiceStub(channel)
            call = stub.Extract(request(), timeout=0.2 if interruption == "deadline" else 3)
            await started(entered)
            if interruption == "cancel":
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call
            else:
                with pytest.raises(grpc.aio.AioRpcError) as exc:
                    await call
                assert exc.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            assert service._active_requests == 1
            assert paths[0].exists()
            assert len(service._operations) == 1
            release.set()
            await asyncio.wait_for(service._idle.wait(), 2)
            assert service._active_requests == 0
            assert not paths[0].exists()
        finally:
            release.set()
            await service.close()
            await channel.close()
            await server.stop(0)
    asyncio.run(run())


def test_upload_write_error_cleans_partial_file_and_accounting(monkeypatch):
    from contextlib import contextmanager
    from extraction import server as module
    original = module.tempfile.NamedTemporaryFile
    paths = []
    @contextmanager
    def failing_file(**kwargs):
        with original(**kwargs) as stream:
            paths.append(Path(stream.name))
            class BrokenWriter:
                name = stream.name
                def write(self, content):
                    stream.write(b"partial")
                    raise OSError("fixture write error")
            yield BrokenWriter()
    monkeypatch.setattr(module.tempfile, "NamedTemporaryFile", failing_file)
    async def run():
        service = ExtractionServicer(ExtractionConfig(device="cpu"))
        with pytest.raises(OSError, match="fixture write error"):
            await service.Extract(request(), Context())
        assert not paths[0].exists()
        assert service._active_requests == 0
        assert service._idle.is_set()
        await service.close()
    asyncio.run(run())


def test_backend_error_preserves_response_contract():
    async def run():
        service = ExtractionServicer(ExtractionConfig(device="cpu"))
        service._extract_text = lambda path: ExtractionResult(error="fixture backend error")
        context = Context()
        result = await service.Extract(request(), context)
        assert not result.full_text
        assert context.code == grpc.StatusCode.INTERNAL
        assert context.detail == "fixture backend error"
        assert service._active_requests == 0
        await service.close()
    asyncio.run(run())


def test_idle_monitor_does_not_unload_cancelled_running_worker():
    from extraction.server import idle_monitor
    async def run():
        config = ExtractionConfig(device="cpu", idle_timeout_seconds=0.01)
        service = ExtractionServicer(config)
        entered, release = threading.Event(), threading.Event()
        cleaned = []
        class Backend:
            is_loaded = True
            def cleanup(self):
                cleaned.append(True)
        service._docling = Backend()
        def worker(path):
            entered.set()
            assert release.wait(5)
            return ExtractionResult(text=Path(path).read_text())
        service._extract_text = worker
        task = asyncio.create_task(service.Extract(request(), Context()))
        monitor = asyncio.create_task(idle_monitor(service, config))
        try:
            await started(entered)
            task.cancel()
            service._last_request_time = 0
            await asyncio.sleep(1.1)  # execute at least one real monitor interval
            assert not cleaned
            assert service._active_requests == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            async def wait_cleanup():
                while not cleaned:
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(wait_cleanup(), 2)
            assert cleaned == [True]
        finally:
            release.set()
            monitor.cancel()
            await asyncio.gather(task, monitor, return_exceptions=True)
            await service.close()
    asyncio.run(run())
