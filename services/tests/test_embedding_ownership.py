"""Real HTTP framework and worker control flow; only model inference is stubbed."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import threading
import types

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def server(monkeypatch):
    class Backend:
        def __init__(self, **kwargs):
            self.model_name = kwargs['model_name']
            self.is_loaded = True
            self.entered, self.release = threading.Event(), threading.Event()
            self.failure = None
            self.unloads = 0
        def work(self, count):
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError('private gate timed out')
            if self.failure:
                raise self.failure
            return np.ones((count, 2048), dtype=np.float32)
        def embed_texts(self, texts, **kwargs):
            return self.work(len(texts))
        def embed_late_chunked(self, text, **kwargs):
            return self.work(1), [(0, 1, 0, len(text))]
        def unload(self):
            self.unloads += 1
            self.is_loaded = False
    model = types.ModuleType('embedding.jina_v4')
    class InputTooLargeError(ValueError):
        def __init__(self, index, n_tokens, max_tokens):
            self.index, self.n_tokens, self.max_tokens = index, n_tokens, max_tokens

    class BackendOutOfMemoryError(RuntimeError):
        pass

    model.InputTooLargeError = InputTooLargeError
    model.BackendOutOfMemoryError = BackendOutOfMemoryError
    model.JinaV4Embedder = Backend
    model.EMBEDDING_DIM = 2048
    model.MAX_TOKENS = 2048
    model.SUPPORTED_TASKS = ['retrieval.passage']
    monkeypatch.setitem(sys.modules, 'embedding.jina_v4', model)
    name = 'embedding._ownership_http_server'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'embedding/http_server.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


async def entered(backend):
    async def wait():
        while not backend.entered.is_set():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), 2)


def state_for(server):
    state = server.AppState(server.EmbeddingConfig(device='cpu', model_name='fixture', idle_timeout_seconds=0.01))
    server.app.state.app_state = state
    return state


def body(late=False):
    result = {'model':'fixture', 'input':'synthetic'}
    if late:
        result['late_chunk'] = {'chunk_size_tokens':2,'overlap_tokens':0}
    return result


@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('cancel', [False, True])
@pytest.mark.parametrize('failure', [None, ValueError, RuntimeError])
def test_owned_work_drains_before_accounting_and_cleanup(server, late, cancel, failure):
    async def run():
        state = state_for(server)
        backend = state.embedder
        backend.failure = failure('fixture failure') if failure else None
        task = asyncio.create_task(server.create_embeddings(server.EmbedRequest(**body(late))))
        try:
            await entered(backend)
            if cancel:
                for _ in range(3):
                    task.cancel()
                    await asyncio.sleep(0)
                assert not task.done()
            assert state.active_requests == 1
            assert not state._idle.is_set()
            with pytest.raises(RuntimeError, match='workers are active'):
                state.unload_model()
            backend.release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
            elif failure:
                with pytest.raises(server.HTTPException) as exc:
                    await asyncio.wait_for(task, 2)
                assert exc.value.status_code == (400 if failure is ValueError else 500)
            else:
                response = await asyncio.wait_for(task, 2)
                assert response.data[0].embedding == [1.0] * 2048
                assert response.data[0].chunk_index == (0 if late else None)
            assert state.active_requests == 0
            assert state._idle.is_set()
        finally:
            backend.release.set()
            await asyncio.gather(task, return_exceptions=True)
            await state.close()
    asyncio.run(run())


def test_close_drains_despite_cancellation_and_rejects_admission(server):
    async def run():
        state = state_for(server)
        backend = state.embedder
        task = asyncio.create_task(server.create_embeddings(server.EmbedRequest(**body())))
        await entered(backend)
        close = asyncio.create_task(state.close())
        try:
            await asyncio.sleep(0)
            with pytest.raises(server.HTTPException) as exc:
                await server.create_embeddings(server.EmbedRequest(**body()))
            assert exc.value.status_code == 503
            for _ in range(3):
                close.cancel()
                await asyncio.sleep(0)
            assert not close.done()
            assert not backend.unloads
            backend.release.set()
            await asyncio.wait_for(task, 2)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(close, 2)
            assert backend.unloads == 1
            await state.close()
            assert backend.unloads == 1
        finally:
            backend.release.set()
            await asyncio.gather(task, close, return_exceptions=True)
            await state.close()
    asyncio.run(run())


def test_cancelled_worker_stays_ineligible_for_idle_unload(server):
    async def run():
        state = state_for(server)
        backend = state.embedder
        await state.start_idle_monitor()
        task = asyncio.create_task(server.create_embeddings(server.EmbedRequest(**body())))
        try:
            await entered(backend)
            task.cancel()
            state.last_request_time = 0
            await asyncio.sleep(1.1)
            assert state.active_requests == 1
            assert backend.unloads == 0
            backend.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            async def wait_unload():
                while not backend.unloads:
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(wait_unload(), 2)
            assert backend.unloads == 1
        finally:
            backend.release.set()
            await asyncio.gather(task, return_exceptions=True)
            await state.close()
    asyncio.run(run())


@pytest.mark.parametrize('late', [False, True])
def test_actual_asgi_validation_and_responses(server, late):
    async def run():
        state = state_for(server)
        state.embedder.release.set()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),base_url='http://fixture') as client:
            response = await client.post('/v1/embeddings',json=body(late))
            assert response.status_code == 200
            data = response.json()['data'][0]
            assert data['embedding'] == [1.0] * 2048
            if late:
                assert data['chunk_index'] == 0
                assert data['char_end'] == len('synthetic')
            else:
                assert 'chunk_index' not in data
            for invalid in [dict(body(),input=[]),dict(body(),task='invalid'),dict(body(),encoding_format='base64')]:
                response = await client.post('/v1/embeddings',json=invalid)
                assert response.status_code == 400
            response = await client.post('/v1/embeddings',json=dict(body(),input=['a','b'],late_chunk={'boundaries':[[0,1]]}))
            assert response.status_code == 400
            assert state.active_requests == 0
            await state.close()
            assert (await client.post('/v1/embeddings',json=body())).status_code == 503
    asyncio.run(run())


def test_actual_lifespan_drains_worker_before_unloading(server, monkeypatch):
    monkeypatch.setattr(server.EmbeddingConfig, 'from_env', classmethod(
        lambda cls: cls(device='cpu', model_name='fixture', idle_timeout_seconds=0)))
    async def run():
        lifetime = server.lifespan(server.app)
        await lifetime.__aenter__()
        state = server.app.state.app_state
        backend = state.embedder
        task = asyncio.create_task(server.create_embeddings(server.EmbedRequest(**body())))
        await entered(backend)
        shutdown = asyncio.create_task(lifetime.__aexit__(None, None, None))
        try:
            await asyncio.sleep(0)
            assert state._closing
            assert not shutdown.done()
            assert backend.unloads == 0
            backend.release.set()
            await asyncio.wait_for(task, 2)
            await asyncio.wait_for(shutdown, 2)
            assert backend.unloads == 1
        finally:
            backend.release.set()
            await asyncio.gather(task, shutdown, return_exceptions=True)
            await state.close()
    asyncio.run(run())


def test_specification_error_envelopes(server):
    """PE-API v1.1 errors carry structured fields, not framework detail strings."""
    async def run():
        state = state_for(server)
        state.embedder.release.set()
        observations = []
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://fixture') as client:
                for label, request, expected_status, expected_code in [
                    ('task', dict(body(), task='invalid'), 400, 'PE_INVALID_TASK'),
                    ('encoding', dict(body(), encoding_format='base64'), 400, 'PE_UNSUPPORTED_ENCODING_FORMAT'),
                    ('images', dict(body(), images=['synthetic']), 400, 'PE_MULTIMODAL_UNSUPPORTED'),
                    ('empty_input', dict(body(), input=[]), 400, 'PE_INVALID_INPUT'),
                    ('invalid_shape', {'input': 'synthetic'}, 422, 'PE_INVALID_REQUEST'),
                ]:
                    response = await client.post('/v1/embeddings', json=request)
                    payload = response.json()
                    error = payload.get('error', {})
                    valid = (
                        response.status_code == expected_status
                        and isinstance(error, dict)
                        and isinstance(error.get('message'), str)
                        and error.get('type') == 'invalid_request_error'
                        and 'param' in error and 'code' in error
                        and (expected_code is None or error['code'] == expected_code)
                    )
                    observations.append({'case': label, 'status': response.status_code,
                                         'body': payload, 'conforms': valid})
                state.embedder.failure = RuntimeError('synthetic inference failure')
                response = await client.post('/v1/embeddings', json=body())
                payload = response.json()
                error = payload.get('error', {})
                observations.append({'case': 'backend_failure', 'status': response.status_code,
                                     'body': payload, 'conforms': response.status_code == 500
                                     and error.get('type') == 'server_error'
                                     and isinstance(error.get('message'), str)
                                     and 'param' in error and 'code' in error})
        finally:
            await state.close()
        import json
        print('PE_ERROR_BASELINE ' + json.dumps(observations, sort_keys=True))
        assert all(item['conforms'] for item in observations), observations
    asyncio.run(run())


@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize('failure_kind,status,code', [
    ('large', 400, 'PE_INPUT_TOO_LARGE'),
    ('late_large', 400, 'PE_INPUT_TOO_LARGE'),
    ('oom', 503, 'PE_BACKEND_OOM'),
    ('value', 400, 'PE_INVALID_INPUT'),
    ('runtime', 500, 'PE_BACKEND_ERROR'),
])
def test_backend_error_codes_and_no_detail_leak(server, late, failure_kind, status, code):
    async def run():
        state = state_for(server)
        state.embedder.release.set()
        sentinel = 'PRIVATE_BACKEND_DETAIL_DO_NOT_REFLECT'
        failures = {
            'large': server.InputTooLargeError(2, 2049, 2048),
            'late_large': ValueError('PE_INPUT_TOO_LARGE: ' + sentinel),
            'oom': server.BackendOutOfMemoryError(sentinel),
            'value': ValueError(sentinel),
            'runtime': RuntimeError(sentinel),
        }
        state.embedder.failure = failures[failure_kind]
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://fixture') as client:
                response = await client.post('/v1/embeddings', json=body(late))
                assert response.status_code == status
                error = response.json()['error']
                assert set(error) == {'message', 'type', 'param', 'code'}
                assert error['code'] == code
                assert error['type'] == ('invalid_request_error' if status == 400 else 'server_error')
                assert sentinel not in response.text
                if failure_kind == 'large':
                    assert all(str(n) in error['message'] for n in [2, 2049, 2048])
                assert state.active_requests == 0
                assert state._idle.is_set()
        finally:
            await state.close()
        assert state.embedder.unloads == 1
    asyncio.run(run())


def test_framework_errors_and_shutdown_use_safe_envelopes(server):
    async def run():
        state = state_for(server)
        state.embedder.release.set()
        sentinel = 'PRIVATE_REQUEST_VALUE_DO_NOT_REFLECT'
        try:
            transport = httpx.ASGITransport(app=server.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url='http://fixture') as client:
                response = await client.post('/v1/embeddings', json={'model':'fixture', 'input':{'private':sentinel}})
                assert response.status_code == 422
                assert response.json()['error']['code'] == 'PE_INVALID_REQUEST'
                assert sentinel not in response.text
                response = await client.get('/no-such-route')
                assert response.status_code == 404
                assert response.json()['error']['type'] == 'invalid_request_error'
                # Response construction outside the inference try block also gets a safe 500.
                state.embedder.embed_texts = lambda *a, **kw: [object()]
                response = await client.post('/v1/embeddings', json=body())
                assert response.status_code == 500
                assert response.json()['error']['code'] == 'PE_BACKEND_ERROR'
                await state.close()
                response = await client.post('/v1/embeddings', json=body())
                assert response.status_code == 503
                assert response.json()['error']['code'] == 'PE_SERVICE_CLOSING'
        finally:
            await state.close()
    asyncio.run(run())


@pytest.mark.parametrize("legacy_namespace", [False, True])
def test_backend_oom_type_import_without_cuda_initialization(monkeypatch, legacy_namespace):
    import torch

    # PyTorch 2.0 exposes this native type only through torch.cuda.
    expected = torch.cuda.OutOfMemoryError
    if legacy_namespace:
        monkeypatch.delattr(torch, "OutOfMemoryError", raising=False)
    transformers = types.ModuleType("transformers")
    transformers.AutoModel = transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    def forbidden_cuda_init():
        pytest.fail("Import must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden_cuda_init)
    name = "embedding._oom_import_backend"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "embedding/jina_v4.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert module.BackendOutOfMemoryError is expected
