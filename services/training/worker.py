"""Run synchronous training RPC bodies away from the provider event loop.

SessionTrainingServicer owns/drains the operation task. Cancelling a bare worker
await does not stop a thread; production calls must retain that session wrapper.
"""

import asyncio
from functools import wraps


class _WorkerAbort(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code = code
        self.details = details


class _WorkerContext:
    async def abort(self, code, details):
        # No grpc.aio context is accessed from the worker's thread/loop.
        raise _WorkerAbort(code, details)


def blocking_rpc(operation):
    @wraps(operation)
    async def call(self, request, context):
        def run():
            # Backend coroutines only await precondition/abort helpers. Keep
            # their existing status semantics without touching the server loop.
            return asyncio.run(operation(self, request, _WorkerContext()))
        try:
            return await asyncio.to_thread(run)
        except _WorkerAbort as exc:
            await context.abort(exc.code, exc.details)
    return call
