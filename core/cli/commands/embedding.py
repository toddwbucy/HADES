"""HADES Embedding CLI Commands.

Commands for embedding service management, direct text/document embedding,
and GPU status. Uses the persistent embedding service when available.

Usage:
    hades embedding service status    # Check service health
    hades embedding service start     # Start the service
    hades embedding service stop      # Stop the service
    hades embedding text "query"      # Embed text to stdout
    hades embedding gpu status        # Show GPU utilization
    hades embedding gpu list          # List available GPUs
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import TYPE_CHECKING

from core.cli.output import (
    CLIResponse,
    ErrorCode,
    error_response,
    success_response,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Service Commands
# =============================================================================


def service_status(start_time: float) -> CLIResponse:
    """Get embedding service status."""
    try:
        from core.services.embedder_client import EmbedderClient, EmbedderServiceError

        client = EmbedderClient()

        try:
            health = client.get_health()
            return success_response(
                command="embedding.service.status",
                data={
                    "service": "hades-embedder",
                    "status": health.get("status", "unknown"),
                    "model_loaded": health.get("model_loaded", False),
                    "device": health.get("device", "unknown"),
                    "model_name": health.get("model_name", "unknown"),
                    "uptime_seconds": health.get("uptime_seconds", 0),
                    "socket_path": client.socket_path,
                },
                start_time=start_time,
            )
        except EmbedderServiceError:
            # Service not running
            return success_response(
                command="embedding.service.status",
                data={
                    "service": "hades-embedder",
                    "status": "stopped",
                    "model_loaded": False,
                    "socket_path": client.socket_path,
                    "hint": "Start with: sudo systemctl start hades-embedder",
                },
                start_time=start_time,
            )
        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="embedding.service.status",
            code=ErrorCode.SERVICE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def service_start(start_time: float) -> CLIResponse:
    """Start the embedding service via systemctl."""
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "start", "hades-embedder"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return error_response(
                command="embedding.service.start",
                code=ErrorCode.SERVICE_ERROR,
                message=f"Failed to start service: {result.stderr.strip()}",
                start_time=start_time,
            )

        # Wait a moment for service to initialize
        time.sleep(2)

        # Check if it's actually running
        from core.services.embedder_client import EmbedderClient

        client = EmbedderClient()
        is_available = client.is_service_available(force_check=True)
        client.close()

        return success_response(
            command="embedding.service.start",
            data={
                "action": "started",
                "service": "hades-embedder",
                "available": is_available,
                "note": "Model loading may take 10-30 seconds" if not is_available else None,
            },
            start_time=start_time,
        )

    except subprocess.TimeoutExpired:
        return error_response(
            command="embedding.service.start",
            code=ErrorCode.SERVICE_ERROR,
            message="Timeout starting service",
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="embedding.service.start",
            code=ErrorCode.SERVICE_ERROR,
            message=str(e),
            start_time=start_time,
        )


def service_stop(start_time: float, token: str | None = None) -> CLIResponse:
    """Stop the embedding service."""
    try:
        # Try graceful API shutdown first if token provided
        if token:
            from core.services.embedder_client import EmbedderClient

            client = EmbedderClient()
            if client.shutdown_service(token=token):
                client.close()
                time.sleep(1)
                return success_response(
                    command="embedding.service.stop",
                    data={"action": "stopped", "method": "api"},
                    start_time=start_time,
                )
            client.close()

        # Fall back to systemctl
        result = subprocess.run(
            ["sudo", "systemctl", "stop", "hades-embedder"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return error_response(
                command="embedding.service.stop",
                code=ErrorCode.SERVICE_ERROR,
                message=f"Failed to stop service: {result.stderr.strip()}",
                start_time=start_time,
            )

        return success_response(
            command="embedding.service.stop",
            data={"action": "stopped", "method": "systemctl"},
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="embedding.service.stop",
            code=ErrorCode.SERVICE_ERROR,
            message=str(e),
            start_time=start_time,
        )


# =============================================================================
# Text Embedding Commands
# =============================================================================


def embed_text(
    text: str,
    start_time: float,
    task: str = "retrieval.passage",
    output_format: str = "json",
) -> CLIResponse:
    """Embed a single text string."""
    try:
        from core.services.embedder_client import EmbedderClient

        client = EmbedderClient()

        try:
            embedding = client.embed_texts([text], task=task)[0]

            if output_format == "raw":
                # Output just the embedding values for piping
                print(json.dumps(embedding.tolist()))
                return success_response(
                    command="embedding.text",
                    data={"written_to": "stdout"},
                    start_time=start_time,
                )

            return success_response(
                command="embedding.text",
                data={
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "task": task,
                    "dimension": len(embedding),
                    "embedding": embedding.tolist()[:10],  # First 10 values as preview
                    "embedding_truncated": True,
                    "service_used": client.is_service_available(),
                },
                start_time=start_time,
            )
        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="embedding.text",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )


def embed_texts_batch(
    texts: list[str],
    start_time: float,
    task: str = "retrieval.passage",
) -> CLIResponse:
    """Embed multiple texts."""
    try:
        from core.services.embedder_client import EmbedderClient

        client = EmbedderClient()

        try:
            embeddings = client.embed_texts(texts, task=task)

            return success_response(
                command="embedding.batch",
                data={
                    "count": len(embeddings),
                    "dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
                    "task": task,
                    "service_used": client.is_service_available(),
                },
                start_time=start_time,
            )
        finally:
            client.close()

    except Exception as e:
        return error_response(
            command="embedding.batch",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )


# =============================================================================
# GPU Commands
# =============================================================================


def gpu_status(start_time: float) -> CLIResponse:
    """Get GPU status and memory usage."""
    try:
        import torch

        if not torch.cuda.is_available():
            return success_response(
                command="embedding.gpu.status",
                data={
                    "cuda_available": False,
                    "message": "CUDA not available",
                },
                start_time=start_time,
            )

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_allocated = torch.cuda.memory_allocated(i)
            mem_reserved = torch.cuda.memory_reserved(i)

            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1e9, 2),
                "allocated_memory_gb": round(mem_allocated / 1e9, 2),
                "reserved_memory_gb": round(mem_reserved / 1e9, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })

        # Get current device from service if running
        service_device = None
        try:
            from core.services.embedder_client import EmbedderClient

            client = EmbedderClient()
            if client.is_service_available():
                health = client.get_health()
                service_device = health.get("device")
            client.close()
        except Exception:
            pass

        return success_response(
            command="embedding.gpu.status",
            data={
                "cuda_available": True,
                "device_count": len(gpus),
                "gpus": gpus,
                "embedder_service_device": service_device,
            },
            start_time=start_time,
        )

    except Exception as e:
        return error_response(
            command="embedding.gpu.status",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )


def gpu_list(start_time: float) -> CLIResponse:
    """List available GPUs."""
    try:
        # Try nvidia-smi for more detailed info
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_used_mb": int(parts[3]),
                            "memory_free_mb": int(parts[4]),
                            "utilization_percent": int(parts[5]),
                        })

            return success_response(
                command="embedding.gpu.list",
                data={"gpus": gpus},
                start_time=start_time,
            )

        # Fall back to torch
        return gpu_status(start_time)

    except Exception as e:
        return error_response(
            command="embedding.gpu.list",
            code=ErrorCode.CONFIG_ERROR,
            message=str(e),
            start_time=start_time,
        )
