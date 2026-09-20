"""Private bounded cancellation ordering probe, no ML model or server."""
import asyncio
import faulthandler
faulthandler.dump_traceback_later(8)
import json
import os
from pathlib import Path
import resource
import sys
import threading

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
sys.path.insert(0, str(Path(sys.argv[1]) / 'services'))
from extraction.server import ExtractionServicer, ExtractionConfig, extraction_pb2 as pb

class Context:
    def set_code(self, code):
        raise AssertionError(f'unexpected RPC error: {code}')
    def set_details(self, details):
        raise AssertionError('unexpected RPC details')

async def case(cancel):
    service = ExtractionServicer(ExtractionConfig(device='cpu'))
    started, release, done = threading.Event(), threading.Event(), threading.Event()
    observed = {}
    actual = service._extract_text
    def worker(path):
        observed['path'] = path
        started.set()
        try:
            if not release.wait(5):
                raise RuntimeError('private gate timed out')
            observed['file_exists_when_worker_resumes'] = Path(path).exists()
            result = actual(path)
            observed['worker_succeeded'] = result.error is None
            return result
        finally:
            done.set()
    service._extract_text = worker
    task = asyncio.create_task(service.Extract(pb.ExtractRequest(
        file_path='fixture.txt', content=b'synthetic extraction fixture', source_type=pb.SOURCE_TYPE_TEXT
    ), Context()))
    try:
        async def wait_started():
            while not started.is_set():
                await asyncio.sleep(0.005)
        await asyncio.wait_for(wait_started(), 2)
        if service._active_requests != 1 or not Path(observed['path']).exists():
            raise AssertionError('fixture not admitted')
        if cancel:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('cancel was not observed')
            observed['worker_running_after_cancellation'] = not done.is_set()
            observed['active_requests_after_cancellation'] = service._active_requests
            observed['file_exists_after_cancellation'] = Path(observed['path']).exists()
        release.set()
        if not cancel:
            response = await asyncio.wait_for(task, 2)
            if response.full_text != 'synthetic extraction fixture':
                raise AssertionError('control text mismatch')
        if not await asyncio.to_thread(done.wait, 2):
            raise AssertionError('worker did not drain')
        observed['temp_file_removed_after_drain'] = not Path(observed['path']).exists()
        observed.pop('path')
        return observed
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if not await asyncio.to_thread(done.wait, 2):
            raise AssertionError('cleanup did not drain worker')

async def main():
    control = await case(False)
    cancelled = await case(True)
    if not control['worker_succeeded'] or not control['temp_file_removed_after_drain']:
        raise AssertionError('control failed')
    expected = {'worker_running_after_cancellation':True,
                'active_requests_after_cancellation':0,
                'file_exists_after_cancellation':False,
                'file_exists_when_worker_resumes':False,
                'worker_succeeded':False,
                'temp_file_removed_after_drain':True}
    if cancelled != expected:
        raise AssertionError(f'baseline changed: {cancelled}')
    print(json.dumps({'control':control, 'cancelled':cancelled},indent=2))

asyncio.run(main())
