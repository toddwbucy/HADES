"""Exercise all three production refusal guards through HTTP without loading a model."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import types

import httpx
import pytest
import torch

from test_embedding_ownership import server, state_for  # noqa: F401


@pytest.mark.parametrize('mode', ['normal', 'boundaries', 'uniform'])
def test_real_refusal_sites_return_structured_counts(server, monkeypatch, mode):
    # Only the optional model loader is stubbed; execute the shipped tokenizer
    # guards and HTTP exception mapping on CPU tensors (#164).
    transformers = types.ModuleType('transformers')
    transformers.AutoModel = transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    name = 'embedding._size_contract_backend'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'embedding/jina_v4.py')
    backend_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, backend_module)
    spec.loader.exec_module(backend_module)
    monkeypatch.setattr(backend_module, 'MAX_TOKENS', 8)
    monkeypatch.setattr(backend_module, 'PROMPT_RESERVE_TOKENS', 1)
    monkeypatch.setattr(server, 'InputTooLargeError', backend_module.InputTooLargeError)

    class Encoded(dict):
        def to(self, device):
            return self

    def tokenize(text, **kwargs):
        if isinstance(text, str):
            return {'input_ids': list(range(30))}
        return Encoded(attention_mask=torch.ones((1, 8)),
                       offset_mapping=torch.tensor([[(0, 0), (0, 2), (2, 4), (4, 6),
                                                      (6, 8), (8, 9), (9, 10), (0, 0)]]))

    model = backend_module.JinaV4Embedder(device='cpu')
    model._tokenizer = tokenize
    model._model = lambda **kwargs: types.SimpleNamespace(vlm_last_hidden_states=torch.zeros((1, 8, 2)))

    async def run():
        state = state_for(server)
        state.embedder.embed_texts = model.embed_texts
        state.embedder.embed_late_chunked = model.embed_late_chunked
        request = {'model': 'fixture', 'input': 'abcdefghij'}
        if mode != 'normal':
            request['late_chunk'] = {'chunk_size_tokens': 2, 'overlap_tokens': 0}
            if mode == 'boundaries':
                request['late_chunk']['boundaries'] = [[0, 10]]
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://fixture') as client:
                response = await client.post('/v1/embeddings', json=request)
            assert response.status_code == 400, response.text
            error = response.json()['error']
            assert error['code'] == 'PE_INPUT_TOO_LARGE'
            assert error['reported_tokens'] == 30
            assert error['ceiling'] == (7 if mode == 'normal' else 8)
            assert error['input_index'] == 0
        finally:
            await state.close()
    asyncio.run(run())
