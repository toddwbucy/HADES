"""HADES Embedding Service Client.

Client library for communicating with the persistent embedding service
over Unix socket. Falls back to local model instantiation if service
is unavailable.

Usage:
    from core.services import EmbedderClient

    client = EmbedderClient()
    embeddings = client.embed_texts(["query text"], task="retrieval.query")
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class EmbedderServiceError(Exception):
    """Error communicating with the embedding service."""

    pass


class EmbedderClient:
    """Client for the HADES embedding service.

    Communicates with the persistent embedding service over Unix socket.
    Falls back to local model instantiation if service is unavailable
    and fallback_to_local is True.

    Attributes:
        socket_path: Path to the Unix socket
        timeout: Request timeout in seconds
        fallback_to_local: Whether to fall back to local model if service unavailable
    """

    def __init__(
        self,
        socket_path: str = "/run/hades/embedder.sock",
        timeout: float = 30.0,
        fallback_to_local: bool = True,
    ) -> None:
        """Initialize the embedder client.

        Args:
            socket_path: Path to the embedding service Unix socket
            timeout: Request timeout in seconds
            fallback_to_local: Fall back to local model if service unavailable
        """
        self.socket_path = socket_path
        self.timeout = timeout
        self.fallback_to_local = fallback_to_local

        self._client: httpx.Client | None = None
        self._local_embedder: Any | None = None
        self._service_available: bool | None = None  # Cached availability check

    def _get_client(self) -> httpx.Client:
        """Get or create the httpx client."""
        if self._client is None:
            import httpx

            transport = httpx.HTTPTransport(uds=self.socket_path)
            self._client = httpx.Client(
                transport=transport,
                timeout=self.timeout,
                # Use localhost as host (required for HTTP but ignored for UDS)
                base_url="http://localhost",
            )
        return self._client

    def _get_local_embedder(self) -> Any:
        """Get or create the local embedder (lazy initialization)."""
        if self._local_embedder is None:
            logger.warning("Initializing local embedder (service unavailable)")
            from core.embedders.embedders_jina import JinaV4Embedder

            self._local_embedder = JinaV4Embedder()
        return self._local_embedder

    def is_service_available(self, force_check: bool = False) -> bool:
        """Check if the embedding service is available.

        Args:
            force_check: Force a fresh check instead of using cached result

        Returns:
            True if service is available and healthy
        """
        if self._service_available is not None and not force_check:
            return self._service_available

        try:
            response = self._get_client().get("/health")
            if response.status_code == 200:
                data = response.json()
                self._service_available = data.get("status") in ("ready", "idle")
            else:
                self._service_available = False
        except Exception as e:
            logger.debug(f"Service availability check failed: {e}")
            self._service_available = False

        return self._service_available

    def get_health(self) -> dict[str, Any]:
        """Get service health information.

        Returns:
            Health response dict with status, model_loaded, device, etc.

        Raises:
            EmbedderServiceError: If service is unavailable
        """
        try:
            response = self._get_client().get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise EmbedderServiceError(f"Health check failed: {e}") from e

    def embed_texts(
        self,
        texts: list[str],
        task: str = "retrieval.passage",
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed texts using the service or local fallback.

        Args:
            texts: List of texts to embed
            task: Task type (retrieval.passage, retrieval.query, text-matching)
            batch_size: Optional batch size override

        Returns:
            Numpy array of embeddings (N x 2048)

        Raises:
            EmbedderServiceError: If embedding fails and no fallback available
        """
        if not texts:
            return np.array([])

        # Try service first
        if self.is_service_available():
            try:
                return self._embed_via_service(texts, task, batch_size)
            except Exception as e:
                logger.warning(f"Service embedding failed: {e}")
                # Mark service as unavailable for subsequent calls
                self._service_available = False

                if not self.fallback_to_local:
                    raise EmbedderServiceError(f"Embedding failed: {e}") from e

        # Fall back to local embedder
        if self.fallback_to_local:
            logger.info("Using local embedder fallback")
            return self._embed_locally(texts, task, batch_size)

        raise EmbedderServiceError(f"Embedding service unavailable at {self.socket_path} and fallback disabled")

    def _embed_via_service(
        self,
        texts: list[str],
        task: str,
        batch_size: int | None,
    ) -> np.ndarray:
        """Embed texts via the remote service."""
        start = time.time()

        request_data = {
            "texts": texts,
            "task": task,
        }
        if batch_size is not None:
            request_data["batch_size"] = batch_size

        response = self._get_client().post("/embed", json=request_data)
        response.raise_for_status()

        data = response.json()
        embeddings = np.array(data["embeddings"], dtype=np.float32)

        elapsed = time.time() - start
        logger.debug(
            f"Service embedding: {len(texts)} texts in {elapsed:.3f}s "
            f"(service reported {data.get('duration_ms', 0):.1f}ms)"
        )

        return embeddings

    def _embed_locally(
        self,
        texts: list[str],
        task: str,
        batch_size: int | None,
    ) -> np.ndarray:
        """Embed texts using local model."""
        start = time.time()

        embedder = self._get_local_embedder()
        embeddings = embedder.embed_texts(texts, task=task, batch_size=batch_size)

        elapsed = time.time() - start
        logger.debug(f"Local embedding: {len(texts)} texts in {elapsed:.3f}s")

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query text (convenience method).

        Uses retrieval.query task for optimal query encoding.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector (1D array of 2048 floats)

        Raises:
            EmbedderServiceError: If embedding fails or returns empty result
        """
        embeddings = self.embed_texts([query], task="retrieval.query")
        if embeddings.size == 0:
            raise EmbedderServiceError("Embedding returned empty result for query")
        return embeddings[0]

    def embed_documents(
        self,
        documents: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Embed documents (convenience method).

        Uses retrieval.passage task for optimal document encoding.

        Args:
            documents: List of document texts to embed
            batch_size: Optional batch size override

        Returns:
            Numpy array of embeddings (N x 2048)
        """
        return self.embed_texts(documents, task="retrieval.passage", batch_size=batch_size)

    def shutdown_service(self, token: str | None = None) -> bool:
        """Request graceful shutdown of the embedding service.

        Args:
            token: Shutdown token if service requires authentication

        Returns:
            True if shutdown was initiated successfully
        """
        try:
            data = {"token": token} if token else {}
            response = self._get_client().post("/shutdown", json=data)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Shutdown request failed: {e}")
            return False

    def close(self) -> None:
        """Close the client and release resources."""
        if self._client is not None:
            self._client.close()
            self._client = None

        if self._local_embedder is not None:
            del self._local_embedder
            self._local_embedder = None

    def __enter__(self) -> EmbedderClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()


# Convenience function for quick embedding
def embed_texts(
    texts: list[str],
    task: str = "retrieval.passage",
    socket_path: str = "/run/hades/embedder.sock",
    fallback_to_local: bool = True,
) -> np.ndarray:
    """Embed texts using the embedding service or local fallback.

    Convenience function that creates a temporary client.

    Args:
        texts: List of texts to embed
        task: Task type (retrieval.passage, retrieval.query, text-matching)
        socket_path: Path to the embedding service Unix socket
        fallback_to_local: Fall back to local model if service unavailable

    Returns:
        Numpy array of embeddings (N x 2048)
    """
    with EmbedderClient(
        socket_path=socket_path,
        fallback_to_local=fallback_to_local,
    ) as client:
        return client.embed_texts(texts, task=task)
