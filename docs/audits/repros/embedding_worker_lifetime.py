"""Exercise real HTTP-handler/idle control flow with a fake model boundary."""
import asyncio
import json
import os
from pathlib import Path
import resource
import sys
import threading
import types

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
sys.path.insert(0, str(Path(sys.argv[1]) / 'services'))
entered, release, done, unloaded = (threading.Event() for _ in range(4))
observed = {}
class Row:
    def tolist(self):
        return [1.0, 0.0]
class Backend:
    def __init__(self, **kwargs):
        self.model_name = 'fixture'
        self.is_loaded = True
    def embed_texts(self, texts, **kwargs):
        entered.set()
        try:
            if not release.wait(5):
                raise RuntimeError('private worker gate timed out')
            return [Row() for _ in texts]
        finally:
            done.set()
    def unload(self):
        observed['unload_while_worker_running'] = entered.is_set() and not done.is_set()
        self.is_loaded = False
        unloaded.set()
module = types.ModuleType('embedding.jina_v4')
module.EMBEDDING_DIM = 2048
module.MAX_TOKENS = 2048
module.SUPPORTED_TASKS = ['retrieval.passage']
module.JinaV4Embedder = Backend
sys.modules['embedding.jina_v4'] = module
from embedding import http_server as server

async def wait_for(event):
    async def wait():
        while not event.is_set():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(wait(), 2)

async def main():
    state = server.AppState(server.EmbeddingConfig(device='cpu',idle_timeout_seconds=0.01))
    server.app.state.app_state = state
    # Valid ordinary request control, using the real handler/response model.
    release.set()
    response = await server.create_embeddings(server.EmbedRequest(model='fixture',input='control'))
    if response.data[0].embedding != [1.0,0.0] or state.active_requests != 0:
        raise AssertionError('control failed')
    observed['control_success'] = True
    entered.clear(); release.clear(); done.clear()
    await state.start_idle_monitor()
    task = asyncio.create_task(server.create_embeddings(server.EmbedRequest(model='fixture',input='cancelled')))
    try:
        await wait_for(entered)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError('request did not cancel')
        observed['worker_running_after_cancel'] = not done.is_set()
        observed['active_requests_after_cancel'] = state.active_requests
        await wait_for(unloaded)
        if observed != {'control_success':True,'worker_running_after_cancel':True,'active_requests_after_cancel':0,'unload_while_worker_running':True}:
            raise AssertionError(f'baseline changed: {observed}')
    finally:
        release.set()
        await asyncio.gather(task,return_exceptions=True)
        await wait_for(done)
        await state.stop_idle_monitor()
    observed['worker_drained'] = done.is_set()
    print(json.dumps(observed,indent=2))

asyncio.run(main())
