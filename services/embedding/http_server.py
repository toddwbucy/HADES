"""Persephone Embedding Service — OpenAI-compatible HTTP server.

Wraps `JinaV4Embedder` behind a FastAPI app exposing the OpenAI
`/v1/embeddings` and `/v1/models` endpoints. Replaces the prior gRPC
server (`server.py`) — same inference layer, different transport.

Listens on TCP `127.0.0.1:8000` by default. HADES's embedding client
already speaks OpenAI shape (post-refactor), so configuration is just a
matter of pointing it at this service's URL.

The `task` field on the request body is a HADES/Jina vendor extension:
it routes to the right LoRA adapter (`retrieval.query`,
`retrieval.passage`, `text-matching`, `code`). OpenAI clients that
don't send it default to `retrieval.passage`.

Usage:
    python -m embedding.http_server
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import EmbeddingConfig
from .jina_v4 import EMBEDDING_DIM, MAX_TOKENS, SUPPORTED_TASKS, JinaV4Embedder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible shapes)
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    """OpenAI-compatible request body for `POST /v1/embeddings`.

    `task` and `batch_size` are non-standard vendor extensions used by
    Jina V4 (LoRA adapter routing) and HADES (server-side batching hint).
    Engines that don't recognize them ignore them.
    """

    model: str
    input: Union[str, list[str]]
    encoding_format: str = "float"
    images: Optional[list[str]] = Field(
        default=None,
        description=(
            "PE-API multimodal inputs. Declared so the request can be REFUSED "
            "rather than silently dropped: this backend is text-only, and "
            "returning a text embedding for a request that supplied an image "
            "is the failure the contract exists to prevent."
        ),
    )
    # Vendor extensions
    task: Optional[str] = Field(
        default="retrieval.passage",
        description="Jina V4 LoRA adapter selector",
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Server-side batch size override",
    )
    late_chunk: Optional["LateChunkSpec"] = Field(
        default=None,
        description=(
            "Enable late chunking. Each input is encoded in ONE pass and then "
            "pooled per chunk, so every chunk vector is conditioned on the "
            "surrounding document. Returns N vectors per input rather than 1."
        ),
    )


class LateChunkSpec(BaseModel):
    """Late-chunking parameters.

    Pooling happens server-side on purpose. Token-level embeddings are
    seq_len x 2048, so a 15,000-token document is roughly 120 MB in float32
    before pooling, which cannot sensibly cross the wire as JSON. Sending the
    text and receiving pooled chunk vectors keeps the response small and puts
    the arithmetic next to the model that produced the tokens.
    """

    chunk_size_tokens: int = Field(default=500, gt=0)
    overlap_tokens: int = Field(default=200, ge=0)
    boundaries: Optional[list[tuple[int, int]]] = Field(
        default=None,
        description=(
            "Character ranges over the input to pool at, e.g. AST definition "
            "spans or document sections. Keeps chunks aligned to meaningful "
            "units instead of arbitrary token windows. Omit for uniform "
            "windows of chunk_size_tokens."
        ),
    )


class EmbedItem(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int
    # Late chunking only. `index` stays the input's position so OpenAI-shaped
    # clients keep working; these say which chunk of that input this is and
    # which token range it covers.
    chunk_index: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    # Character range over the original input. Callers need this to intersect
    # a chunk with symbol spans; token indices cannot express that.
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbedResponse(BaseModel):
    object: str = "list"
    data: list[EmbedItem]
    model: str
    usage: Usage


class ModelInfo(BaseModel):
    """Single entry in `/v1/models` response.

    The OpenAI-standard fields are `id`, `object`, `created`, `owned_by`.
    `dimension`, `max_seq_length`, `supported_tasks`, `device` and `profile`
    are vendor extensions HADES surfaces so clients can discover model
    capabilities in one round-trip.

    `max_seq_length` is the load profile's measured ceiling, not the model's
    architectural maximum. The same weights serve 11,900 on a 16 GiB card and
    32,768 on a 48 GiB one, so a client that wants to know what fits has to ask
    the running service rather than read the model card. `profile` and `device`
    say which card answered, so a switch between cards is visible in a log
    rather than inferred from the ceiling changing.
    """

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "hades"
    # Vendor extensions
    dimension: int = EMBEDDING_DIM
    max_seq_length: int = MAX_TOKENS
    supported_tasks: list[str] = Field(default_factory=lambda: list(SUPPORTED_TASKS))
    device: str = ""
    # The card as the driver numbers it, not as this process sees it.
    # CUDA_VISIBLE_DEVICES renumbers from zero, so a profile pinned to the
    # second A6000 reports device "cuda:0", and an operator reading that in a
    # log concludes the job is on GPU 0. Both are reported so neither has to be
    # inferred.
    physical_device: str = ""
    profile: str = ""


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------


class AppState:
    """State held across requests: the embedder, request bookkeeping,
    and the idle-monitor task."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        # Eager construction; the underlying model loads lazily on first embed.
        self.embedder = JinaV4Embedder(
            model_name=config.model_name,
            device=config.device,
            use_fp16=config.use_fp16,
            batch_size=config.batch_size,
        )
        self.last_request_time = time.time()
        self.active_requests = 0
        self._operations: set[asyncio.Task] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closing = False
        self._close_task: asyncio.Task | None = None
        self._idle_monitor_task: Optional[asyncio.Task[None]] = None

    @staticmethod
    async def _wait_owned(operation):
        """Drain admitted work before propagating repeated caller cancellation."""
        interrupted = False
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                interrupted = True
            except BaseException:
                break
        if interrupted:
            if not operation.cancelled():
                operation.exception()
            raise asyncio.CancelledError
        return operation.result()

    async def run_work(self, function):
        """Own executor work independently of the HTTP request task."""
        if self._closing:
            raise HTTPException(status_code=503, detail="embedding service is closing")
        self.active_requests += 1
        self._idle.clear()
        operation = asyncio.create_task(self._run_work(function))
        self._operations.add(operation)
        operation.add_done_callback(self._operations.discard)
        return await self._wait_owned(operation)

    async def _run_work(self, function):
        try:
            return await asyncio.get_running_loop().run_in_executor(None, function)
        finally:
            self.active_requests -= 1
            self.last_request_time = time.time()
            if self.active_requests == 0:
                self._idle.set()

    def unload_model(self):
        """Allow idle/shutdown cleanup only after all admitted workers drain."""
        if self.active_requests:
            raise RuntimeError("cannot unload model while embedding workers are active")
        if self.embedder.is_loaded:
            self.embedder.unload()

    async def close(self):
        """Reject new inference and drain workers; threads are not preemptible."""
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain_and_unload())
        await self._wait_owned(self._close_task)

    async def _drain_and_unload(self):
        await self.stop_idle_monitor()
        await self._idle.wait()
        self.unload_model()

    async def start_idle_monitor(self) -> None:
        if self.config.idle_timeout_seconds > 0:
            self._idle_monitor_task = asyncio.create_task(
                self._idle_monitor(), name="embedder-idle-monitor"
            )

    async def stop_idle_monitor(self) -> None:
        if self._idle_monitor_task is not None:
            self._idle_monitor_task.cancel()
            try:
                await self._idle_monitor_task
            except asyncio.CancelledError:
                pass

    async def _idle_monitor(self) -> None:
        """Unload the model from VRAM after the configured idle window.

        Mirrors the behavior of the prior gRPC server: poll on a bounded
        cadence, only unload when no requests are in flight, only when the
        model is actually loaded, and only when the last request is older
        than the threshold.
        """
        poll_interval = min(60, max(1, int(self.config.idle_timeout_seconds)))
        while True:
            await asyncio.sleep(poll_interval)
            if (
                self.active_requests == 0
                and self.embedder.is_loaded
                and (time.time() - self.last_request_time)
                > self.config.idle_timeout_seconds
            ):
                logger.info(
                    "Embedder idle for %.0fs — unloading model to free GPU memory",
                    self.config.idle_timeout_seconds,
                )
                self.unload_model()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: build state, run, tear down cleanly."""
    config = EmbeddingConfig.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    state = AppState(config)
    app.state.app_state = state
    await state.start_idle_monitor()
    logger.info(
        "Embedding service ready (model=%s, device=%s, idle_timeout=%.0fs, listen=%s:%d)",
        config.model_name,
        config.device,
        config.idle_timeout_seconds,
        config.host,
        config.port,
    )
    try:
        yield
    finally:
        logger.info("Shutting down embedding service")
        await state.close()
        logger.info("Embedding service stopped")


app = FastAPI(
    title="HADES Persephone Embedding Service",
    description="OpenAI-compatible /v1/embeddings server backed by Jina V4",
    version="0.3.0",
    lifespan=lifespan,
)


def _state(app: FastAPI) -> AppState:
    return app.state.app_state  # type: ignore[no-any-return]


def _physical_device(device: str) -> str:
    """Map an in-process device string back to the driver's own numbering.

    `CUDA_VISIBLE_DEVICES=1` makes the second card appear as `cuda:0` inside
    the process, so the in-process name cannot say which card is loaded.
    Returns an empty string when the mapping is not derivable, rather than
    guessing.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or not device.startswith("cuda:"):
        return ""
    try:
        ordinal = int(device.split(":", 1)[1])
    except ValueError:
        return ""
    entries = [e.strip() for e in visible.split(",") if e.strip()]
    if ordinal >= len(entries):
        return ""
    return f"cuda:{entries[ordinal]}"


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """OpenAI `/v1/models` — single entry for the configured Jina V4 model."""
    state = _state(app)
    return ModelsResponse(
        data=[
            ModelInfo(
                id=state.config.model_name,
                device=state.config.device,
                physical_device=_physical_device(state.config.device),
                profile=state.config.profile,
            )
        ]
    )


# `response_model_exclude_none` keeps the plain response byte-for-byte
# OpenAI-shaped. Without it FastAPI serializes the five late-chunk fields as
# explicit nulls on every request, including ones that never asked for late
# chunking, which is the opposite of what the opt-in design promises and what
# the spec says this endpoint returns.
@app.post(
    "/v1/embeddings", response_model=EmbedResponse, response_model_exclude_none=True
)
async def create_embeddings(req: EmbedRequest) -> EmbedResponse:
    """OpenAI `/v1/embeddings` — embeds one or more inputs as 2048-dim vectors.

    The `task` vendor-extension field selects the Jina V4 LoRA adapter.
    Defaults to `retrieval.passage` if absent. Unknown tasks are rejected
    (400) rather than silently routed to a wrong adapter.
    """
    state = _state(app)

    # Normalize input (single string or list of strings)
    texts: list[str] = [req.input] if isinstance(req.input, str) else list(req.input)

    if not texts:
        raise HTTPException(status_code=400, detail="`input` must be non-empty")

    # The spec makes these MUSTs, and this service is its reference
    # implementation, so a backend author reading the document gets what it
    # describes. All three previously returned 200.
    if req.encoding_format != "float":
        raise HTTPException(
            status_code=400,
            detail=(
                f"PE_UNSUPPORTED_ENCODING_FORMAT: encoding_format "
                f"{req.encoding_format!r} is not supported, only 'float'"
            ),
        )

    if req.images:
        raise HTTPException(
            status_code=400,
            detail=(
                "PE_MULTIMODAL_UNSUPPORTED: this backend serves text only. "
                "Refusing rather than returning text-only embeddings for a "
                "request that supplied images."
            ),
        )

    # `model` is deliberately NOT validated against the served model. The Rust
    # client sends a configured alias ("jinaai/jina-embeddings-v4") while this
    # backend reports the local path it loaded from, so a strict check would
    # refuse every request HADES makes. Making it real needs the client to read
    # the served name from GET /v1/models at connect, which is a change on the
    # other side of the wire. The spec records this rather than claiming an
    # enforcement that does not exist.

    task = req.task or "retrieval.passage"
    if task not in SUPPORTED_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown task {task!r}; supported: {', '.join(SUPPORTED_TASKS)}",
        )

    if req.batch_size is not None and req.batch_size < 0:
        raise HTTPException(
            status_code=400,
            detail=f"`batch_size` must be non-negative, got {req.batch_size}",
        )
    batch_override = (
        req.batch_size if (req.batch_size is not None and req.batch_size > 0) else None
    )

    started = time.time()
    try:
        # Inference is sync (PyTorch); run in the default executor pool to
        # avoid blocking the event loop. Don't share a future across requests
        # — each call creates its own.
        if req.late_chunk is not None:
            # Late chunking: one forward pass per input, pooled per chunk, so
            # each vector carries the surrounding document's context. Inputs
            # are handled one at a time because each yields its own number of
            # chunks and its own token spans.
            lc = req.late_chunk

            # `boundaries` describes one input. Applying the same character
            # ranges to every element of a multi-input request would pool
            # input 0's spans out of inputs 1 and 2, silently, whenever the
            # later documents happen to be long enough to contain them.
            if lc.boundaries is not None and len(texts) > 1:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "late_chunk.boundaries applies to a single input, but "
                        f"{len(texts)} inputs were sent. Send one input per "
                        "request when supplying boundaries."
                    ),
                )

            def _late() -> tuple[list, list]:
                vecs: list = []
                meta: list = []
                for i, text in enumerate(texts):
                    m, spans = state.embedder.embed_late_chunked(
                        text,
                        task=task,
                        chunk_size_tokens=lc.chunk_size_tokens,
                        overlap_tokens=lc.overlap_tokens,
                        boundaries=lc.boundaries,
                    )
                    if len(spans) != len(m):
                        raise RuntimeError(
                            f"input {i}: {len(m)} vectors for {len(spans)} spans"
                        )
                    for c, (s, e, cs, ce) in enumerate(spans):
                        vecs.append(m[c])
                        meta.append((i, c, s, e, cs, ce))
                return vecs, meta

            vectors, late_meta = await state.run_work(_late)
        else:
            late_meta = None
            vectors = await state.run_work(
                lambda: state.embedder.embed_texts(
                    texts, task=task, batch_size=batch_override
                ),
            )
    except HTTPException:
        # Request validation raised inside this block already carries the right
        # status. Without this the broad handler below rewrapped a deliberate
        # 400 as a 500, which tells a client to retry something that will never
        # succeed.
        raise
    except ValueError as e:
        # `JinaV4Embedder.embed_texts` raises ValueError for invalid
        # batch_size; surface that as a 400 not a 500.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}")

    duration_ms = int((time.time() - started) * 1000)
    logger.info(
        "Embed: %d texts, task=%s, %dms", len(texts), task, duration_ms
    )

    if late_meta is not None:
        # `index` stays the input's position so OpenAI-shaped clients still
        # read it correctly; chunk_index and the token span say which slice of
        # that input this vector covers.
        items = [
            EmbedItem(
                embedding=row.tolist(),
                index=inp_i,
                chunk_index=chunk_i,
                token_start=ts,
                token_end=te,
                char_start=cs,
                char_end=ce,
            )
            for row, (inp_i, chunk_i, ts, te, cs, ce) in zip(vectors, late_meta)
        ]
    else:
        items = [
            EmbedItem(embedding=row.tolist(), index=i)
            for i, row in enumerate(vectors)
        ]
    return EmbedResponse(
        data=items,
        model=state.embedder.model_name,
        usage=Usage(),  # OpenAI sends real token counts; we don't track them yet.
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run uvicorn against this app, configured from env vars.

    Single-process by design: each worker would load its own copy of the
    Jina V4 model into VRAM, exhausting the GPU. If you need higher
    throughput, use larger batches via `batch_size` in the request body.
    """
    config = EmbeddingConfig.from_env()
    uvicorn.run(
        "embedding.http_server:app",
        host=config.host,
        port=config.port,
        log_level="info",
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
