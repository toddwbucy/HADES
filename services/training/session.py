"""Exclusive, expiring ownership for the complete training RPC lifecycle."""

from __future__ import annotations

import asyncio
import secrets
import time

import grpc
from hades.training import training_pb2 as pb, training_pb2_grpc as rpc

SESSION_HEADER = "hades-training-session"
OPERATIONS = (
    "InitModel", "LoadGraph", "TrainStep", "Evaluate", "GetEmbeddings",
    "Checkpoint", "LoadCheckpoint",
)


class SessionTrainingServicer(rpc.TrainingServiceServicer):
    """Fence a single backend behind an opaque lease, including result emission.

    The lock covers each operation; the lease covers gaps between operations.
    Tokens never transfer to successor sessions. Expiration is checked on every
    call and by an idle sweeper, so disconnected clients need no explicit signal.
    """

    def __init__(self, factory, *, lease_seconds=120, clock=time.monotonic):
        if not isinstance(lease_seconds, int) or not 1 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be an integer between 1 and 3600")
        self._factory = factory
        self._lease_seconds = lease_seconds
        self._clock = clock
        self._lock = asyncio.Lock()
        self._token = None
        self._deadline = 0.0
        self._backend = None
        self._closing = False

    def _discard(self):
        self._token = None
        self._deadline = 0.0
        self._backend = None

    def _expire(self):
        if self._token is not None and self._clock() >= self._deadline:
            self._discard()

    def _renew(self):
        self._deadline = self._clock() + self._lease_seconds

    async def _require_owner(self, context):
        if self._closing:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "training provider is closing")
        self._expire()
        tokens = [value for key, value in context.invocation_metadata()
                  if key == SESSION_HEADER]
        if (self._token is None or len(tokens) != 1
                or not secrets.compare_digest(tokens[0], self._token)):
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION,
                                "missing, expired or non-owner training session")

    async def AcquireSession(self, request, context):
        async with self._lock:
            if self._closing:
                await context.abort(grpc.StatusCode.UNAVAILABLE, "training provider is closing")
            self._expire()
            if self._token is not None:
                await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                    "training provider has an active session")
            # Construct before publishing ownership: factory failure cannot
            # leave an unusable lease behind.
            backend = self._factory()
            self._backend = backend
            self._token = secrets.token_urlsafe(32)
            self._renew()
            return pb.AcquireSessionResponse(token=self._token,
                                              lease_seconds=self._lease_seconds)

    async def RenewSession(self, request, context):
        async with self._lock:
            await self._require_owner(context)
            self._renew()
            return pb.SessionResponse()

    async def ReleaseSession(self, request, context):
        async with self._lock:
            await self._require_owner(context)
            self._discard()
            return pb.SessionResponse()

    async def sweep_expired(self):
        """Release abandoned backend state even if no successor connects."""
        while True:
            await asyncio.sleep(min(30, self._lease_seconds))
            async with self._lock:
                self._expire()

    async def close(self):
        self._closing = True
        async with self._lock:
            self._discard()

    async def _invoke(self, name, request, context):
        async with self._lock:
            await self._require_owner(context)
            operation = asyncio.create_task(getattr(self._backend, name)(request, context))
            interrupted = False
            try:
                # Keep the operation and the ownership lock alive through any
                # caller cancellation, including repeated shutdown cancellation.
                while not operation.done():
                    try:
                        await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        interrupted = True
                    except BaseException:
                        break
                if interrupted:
                    if not operation.cancelled():
                        operation.exception()  # retrieve failures of abandoned work
                    raise asyncio.CancelledError
                return operation.result()
            finally:
                if interrupted:
                    # An admitted operation may have changed weights or files.
                    # Discard ambiguous state only after the worker has drained.
                    self._discard()
                else:
                    self._renew()



def _owned_operation(name):
    async def call(self, request, context):
        return await self._invoke(name, request, context)
    call.__name__ = name
    return call


for _name in OPERATIONS:
    setattr(SessionTrainingServicer, _name, _owned_operation(_name))
