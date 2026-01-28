"""HADES Embedding Service Daemon.

Persistent service that keeps the Jina V4 model loaded in GPU memory,
eliminating the 3-5 second model load time on each request.

Run as systemd service or directly:
    python -m core.services.embedder_service

The service exposes an HTTP API over Unix socket at /run/hades/embedder.sock
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging before imports that might log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class EmbedRequest(BaseModel):
    """Request to embed texts."""

    texts: list[str] = Field(..., description="List of texts to embed")
    task: str = Field(
        default="retrieval.passage",
        description="Task type: retrieval.passage, retrieval.query, text-matching",
    )
    batch_size: int | None = Field(default=None, description="Batch size (uses service default if not specified)")


class EmbedResponse(BaseModel):
    """Response containing embeddings."""

    embeddings: list[list[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model name used")
    dimension: int = Field(..., description="Embedding dimension")
    count: int = Field(..., description="Number of embeddings returned")
    duration_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status: ready, loading, idle, error")
    model_loaded: bool = Field(..., description="Whether model is loaded in memory")
    device: str = Field(..., description="Device model is running on")
    model_name: str = Field(..., description="Name of loaded model")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    idle_timeout_seconds: float = Field(..., description="Idle timeout before model unload")
    model_idle_seconds: float | None = Field(None, description="Seconds since model was unloaded due to idle")


class ShutdownResponse(BaseModel):
    """Shutdown response."""

    message: str = Field(..., description="Shutdown message")


# =============================================================================
# Service State
# =============================================================================


@dataclass
class ServiceState:
    """Global service state."""

    embedder: Any | None = None  # JinaV4Embedder instance
    model_loaded: bool = False
    device: str = "unknown"
    model_name: str = "unknown"
    start_time: float = 0.0
    last_request_time: float = 0.0
    idle_timeout_seconds: float = 900.0  # 15 min default
    shutdown_event: asyncio.Event | None = None


state = ServiceState()


# =============================================================================
# Lifespan Management
# =============================================================================


async def unload_model() -> None:
    """Unload the model from GPU memory."""
    if state.embedder is not None:
        del state.embedder
        state.embedder = None
        state.model_loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except Exception as e:
            logger.warning(f"Could not clear CUDA cache: {e}")
        logger.info("Model unloaded, GPU memory freed")


async def idle_monitor() -> None:
    """Background task to unload model after idle timeout."""
    while True:
        await asyncio.sleep(60)
        if (
            state.model_loaded
            and state.last_request_time > 0
            and state.idle_timeout_seconds > 0
            and (time.time() - state.last_request_time) > state.idle_timeout_seconds
        ):
            logger.info(
                f"Model idle for {state.idle_timeout_seconds}s. "
                "Unloading to free GPU memory..."
            )
            await unload_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifespan - load model on startup, cleanup on shutdown."""
    state.start_time = time.time()
    state.last_request_time = time.time()
    state.shutdown_event = asyncio.Event()
    state.idle_timeout_seconds = float(
        os.environ.get("HADES_EMBEDDER_IDLE_TIMEOUT", "900")
    )

    # Load model on startup
    logger.info("Starting HADES Embedding Service...")
    try:
        await load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue running so health check can report error state
        state.model_loaded = False

    monitor_task = asyncio.create_task(idle_monitor())

    yield

    # Cleanup on shutdown
    logger.info("Shutting down HADES Embedding Service...")
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    await unload_model()
    logger.info("Embedding service stopped")


async def load_model() -> None:
    """Load the embedding model into GPU memory."""
    from core.embedders.embedders_jina import JinaV4Embedder

    # Get config from environment or defaults
    # Device can be: cuda:0, cuda:1, cuda:2, etc. for specific GPU, or cpu
    device = os.environ.get("HADES_EMBEDDER_DEVICE", "cuda:0")
    use_fp16 = os.environ.get("HADES_EMBEDDER_FP16", "true").lower() in ("true", "1", "yes")
    batch_size = int(os.environ.get("HADES_EMBEDDER_BATCH_SIZE", "48"))
    model_name = os.environ.get("HADES_EMBEDDER_MODEL", "jinaai/jina-embeddings-v4")

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}, FP16: {use_fp16}, Batch size: {batch_size}")

    # Load model (this is the slow part - 3-5 seconds)
    start = time.time()
    state.embedder = JinaV4Embedder(
        {
            "device": device,
            "use_fp16": use_fp16,
            "batch_size": batch_size,
            "model_name": model_name,
        }
    )
    load_time = time.time() - start

    state.model_loaded = True
    state.device = device
    state.model_name = model_name

    logger.info(f"Model loaded in {load_time:.2f}s - ready to serve requests")


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="HADES Embedding Service",
    description="Persistent embedding service with GPU-accelerated Jina V4",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    uptime = time.time() - state.start_time if state.start_time > 0 else 0

    # Determine idle seconds (time since model was unloaded)
    model_idle_seconds = None
    if state.model_loaded:
        status = "ready"
    elif state.embedder is None and state.last_request_time > 0 and uptime >= 60:
        # Model was unloaded due to idle
        status = "idle"
        model_idle_seconds = time.time() - state.last_request_time
    elif state.embedder is None and uptime < 60:
        status = "loading"
    else:
        status = "error"

    return HealthResponse(
        status=status,
        model_loaded=state.model_loaded,
        device=state.device,
        model_name=state.model_name,
        uptime_seconds=uptime,
        idle_timeout_seconds=state.idle_timeout_seconds,
        model_idle_seconds=round(model_idle_seconds, 1) if model_idle_seconds is not None else None,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Embed texts using the loaded model."""
    # Reset idle timer before any async work so the idle_monitor
    # never sees a freshly-loaded model with a stale timestamp.
    state.last_request_time = time.time()

    # Reload model if it was unloaded due to idle
    if not state.model_loaded or state.embedder is None:
        logger.info("Model unloaded (idle). Reloading for incoming request...")
        try:
            await load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to reload model: {e}",
            ) from e

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    start = time.time()

    try:
        # Run embedding in thread pool to avoid blocking event loop
        embeddings: np.ndarray = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: state.embedder.embed_texts(
                request.texts,
                task=request.task,
                batch_size=request.batch_size,
            ),
        )

        duration_ms = (time.time() - start) * 1000

        return EmbedResponse(
            embeddings=[e.tolist() for e in embeddings],
            model=state.model_name,
            dimension=embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0]),
            count=len(embeddings),
            duration_ms=round(duration_ms, 2),
        )

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class ShutdownRequest(BaseModel):
    """Shutdown request with optional token."""

    token: str | None = Field(default=None, description="Shutdown token for authentication")


# Shutdown token from environment (if set, token is required)
SHUTDOWN_TOKEN = os.environ.get("HADES_EMBEDDER_SHUTDOWN_TOKEN")


@app.post("/shutdown", response_model=ShutdownResponse)
async def shutdown(request: ShutdownRequest | None = None) -> ShutdownResponse:
    """Gracefully shutdown the service.

    If HADES_EMBEDDER_SHUTDOWN_TOKEN is set, a matching token must be provided.
    """
    # Validate token if configured
    if SHUTDOWN_TOKEN:
        provided_token = request.token if request else None
        if not provided_token or provided_token != SHUTDOWN_TOKEN:
            raise HTTPException(
                status_code=403,
                detail="Invalid or missing shutdown token",
            )

    logger.info("Shutdown requested via API")

    if state.shutdown_event:
        state.shutdown_event.set()

    # Schedule shutdown after response is sent
    asyncio.get_event_loop().call_later(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))

    return ShutdownResponse(message="Shutdown initiated")


# =============================================================================
# Server Configuration
# =============================================================================


def get_socket_path() -> str:
    """Get the Unix socket path from environment or default."""
    return os.environ.get("HADES_EMBEDDER_SOCKET", "/run/hades/embedder.sock")


def ensure_socket_dir(socket_path: str) -> None:
    """Ensure the socket directory exists with proper permissions."""
    socket_dir = Path(socket_path).parent
    if not socket_dir.exists():
        logger.info(f"Creating socket directory: {socket_dir}")
        socket_dir.mkdir(parents=True, mode=0o755)


def cleanup_stale_socket(socket_path: str) -> None:
    """Remove stale socket file if it exists."""
    path = Path(socket_path)
    if path.exists():
        logger.info(f"Removing stale socket: {socket_path}")
        path.unlink()


def run_server() -> None:
    """Run the embedding service."""
    socket_path = get_socket_path()

    # Ensure directory exists and clean up stale socket
    ensure_socket_dir(socket_path)
    cleanup_stale_socket(socket_path)

    logger.info(f"Starting server on Unix socket: {socket_path}")

    # Configure uvicorn
    config = uvicorn.Config(
        app,
        uds=socket_path,
        log_level="info",
        access_log=True,
        # Limit workers to 1 to avoid loading model multiple times
        workers=1,
    )

    server = uvicorn.Server(config)

    # Note: uvicorn handles SIGTERM/SIGINT internally for graceful shutdown
    # No custom signal handlers needed

    try:
        server.run()
    finally:
        # Ensure socket is cleaned up
        cleanup_stale_socket(socket_path)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    # Allow running with optional port for development/testing
    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        # HTTP mode for testing (not recommended for production)
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        logger.info(f"Running in HTTP mode on port {port} (for testing only)")
        uvicorn.run(app, host="127.0.0.1", port=port)
    else:
        # Default: Unix socket mode
        run_server()
