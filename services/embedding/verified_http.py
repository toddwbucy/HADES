"""Internal receipt-enforced HTTP app factory; no environment/CLI activation yet."""

from contextlib import asynccontextmanager
import json
import logging
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import EmbeddingConfig
from .http_server import AppState, PEError, error_response
from .identity import ProviderReceipt, _unique_object
from .jina_v4 import BackendOutOfMemoryError, InputTooLargeError
from .snapshot import ArtifactChangedError
from .verified_jina import BehaviorChangedError, ReceiptMismatchError, VerifiedJinaProvider

logger = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 8 * 1024 * 1024


class VerifiedRequest(BaseModel):
    """No task defaults, coercion, or silently ignored unsupported capabilities."""

    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    input: str | list[str]
    task: Literal["retrieval.passage", "retrieval.query"]
    encoding_format: Literal["float"] = "float"
    expected_receipt: dict
    batch_size: int | None = Field(default=None, ge=1)


async def _read_request(request: Request):
    """Bound streamed JSON and preserve duplicate-key rejection before validation."""
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_REQUEST_BYTES:
            raise PEError(413, "PE_REQUEST_TOO_LARGE", "Embedding request exceeds byte limit")
        body.extend(chunk)
    try:
        value = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object)
        parsed = VerifiedRequest.model_validate(value)
        expected = ProviderReceipt.from_dict(parsed.expected_receipt)
    except (ValueError, ValidationError, RecursionError) as exc:
        raise PEError(422, "PE_INVALID_REQUEST",
                      "Invalid request; verified provider supports conventional retrieval text and requires an exact receipt") from exc
    return parsed, expected


def create_verified_app(provider: VerifiedJinaProvider) -> FastAPI:
    """Bind an operator-created provider to owned worker/idle/shutdown lifecycle.

    This factory is for integration and qualification. Production startup keeps
    using the legacy app; no registry path or profile is accepted over HTTP.
    """
    state = AppState(EmbeddingConfig(model_name=provider.model_name), embedder=provider)

    @asynccontextmanager
    async def lifespan(app):
        """Start idle cleanup and drain all admitted operations on shutdown."""
        await state.start_idle_monitor()
        try:
            yield
        finally:
            await state.close()

    app = FastAPI(title="HADES verified embedding provider", lifespan=lifespan)
    app.state.app_state = state

    @app.exception_handler(PEError)
    async def intentional_error(request, exc):
        """Expose only bounded public-safe structured contract errors."""
        return error_response(exc.status_code, exc.detail, exc.code, exc.param, exc.headers, exc.fields)

    async def run(function):
        """Preserve worker ownership while translating backend identity failures."""
        try:
            return await state.run_work(function)
        except (ReceiptMismatchError, BehaviorChangedError, ArtifactChangedError) as exc:
            raise PEError(409, "PE_RECEIPT_MISMATCH", "Provider identity is unavailable or changed; reverify binding") from exc
        except InputTooLargeError as exc:
            raise PEError(400, "PE_INPUT_TOO_LARGE", "Input exceeds the serving token budget",
                          "input", reported_tokens=exc.n_tokens, ceiling=exc.max_tokens,
                          input_index=exc.index) from exc
        except BackendOutOfMemoryError as exc:
            raise PEError(503, "PE_BACKEND_OOM", "Embedding backend exhausted memory") from exc
        except PEError:
            raise
        except ValueError as exc:
            raise PEError(400, "PE_INVALID_INPUT", "Invalid verified embedding input or unsupported model") from exc
        except Exception as exc:
            logger.exception("Verified embedding provider failed")
            raise PEError(500, "PE_BACKEND_ERROR", "Verified embedding provider failed") from exc

    @app.post("/v1/verify")
    async def verify():
        """Explicitly load and observe the operator-selected artifact/behavior identity."""
        return JSONResponse(await run(provider.describe))

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        """Require a receipt at admission and return the actual checked receipt."""
        parsed, expected = await _read_request(request)
        texts = [parsed.input] if isinstance(parsed.input, str) else parsed.input
        # batch_size is an operational hint. This v1 path deliberately executes
        # one input per forward pass, keeping tokenization/pooling explicit.
        result = await run(lambda: provider.embed(model=parsed.model, texts=texts,
                                                  task=parsed.task, expected=expected))
        return JSONResponse(result)

    return app
