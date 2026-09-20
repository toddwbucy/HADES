"""Bounded, private ASGI probe of the real embedding app with a fake model.

Run from the repository root with the CPU CI interpreter. No network listener,
model import, GPU or production endpoint. Retains metadata only, not input text.
"""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest

async def main():
    path = Path('services/tests/test_embedding_ownership.py')
    spec = importlib.util.spec_from_file_location('private_embedding_fixture', path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    monkeypatch = pytest.MonkeyPatch()
    state = None
    try:
        server = fixtures.server.__wrapped__(monkeypatch)
        state = fixtures.state_for(server)
        state.embedder.release.set()
        calls = []
        def encode(texts, **kwargs):
            calls.append({'inputs': len(texts), 'bytes': sum(len(s.encode()) for s in texts),
                          'batch_size': kwargs.get('batch_size')})
            # No inference: use the fixture's small fixed vectors.
            return state.embedder.work(len(texts))
        monkeypatch.setattr(state.embedder, 'embed_texts', encode)
        cases = [
            ('missing_model', {'input':'x'}, 422),
            ('wrong_input_type', {'model':'fixture','input':42}, 422),
            ('empty_list', {'model':'fixture','input':[]}, 400),
            ('model_alias', {'model':'different-alias','input':'x'}, 200),
            ('large_single_text', {'model':'fixture','input':'x'*(4*1024*1024)}, 200),
            ('large_batch_hint', {'model':'fixture','input':'x','batch_size':1000000}, 200),
        ]
        rows = []
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://fixture') as client:
            for name, body, expected in cases:
                before = len(calls)
                response = await asyncio.wait_for(client.post('/v1/embeddings', json=body), 5)
                assert response.status_code == expected, (name, response.status_code)
                data = response.json()
                rows.append({'case': name, 'status': response.status_code,
                             'backend_calls': calls[before:],
                             'response_model': data.get('model')})
                assert state.active_requests == 0
        assert all(row['backend_calls'] == [] for row in rows[:3])
        print(json.dumps({'scope':'Private real ASGI app; fake model backend. 4 MiB is an accepted sample, not a measured maximum. No inference or network.',
            'source_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('services/embedding/http_server.py'),path,Path(__file__).resolve().relative_to(Path.cwd())]},
            'cases':rows}, indent=2))
    finally:
        if state is not None:
            state.embedder.release.set()
            await state.close()
        monkeypatch.undo()

asyncio.run(main())
