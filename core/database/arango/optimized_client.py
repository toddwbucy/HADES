"""HTTP/2 optimized ArangoDB client over Unix sockets.

Phase 1 deliverable for Issue #51. Provides minimal operations:
- get_document
- insert_documents (NDJSON bulk import)
- query (AQL cursor)

This client intentionally keeps a narrow surface; additional features
(connection pooling, proxies, streaming cursors) will be layered on in
later phases.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx
import orjson


class ArangoHttpError(RuntimeError):
    """Raised when the ArangoDB HTTP API reports an error."""

    def __init__(self, status_code: int, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(f"ArangoDB HTTP {status_code}: {message}")
        self.status_code = status_code
        self.details = details or {}


@dataclass
class ArangoHttp2Config:
    """Configuration for the HTTP/2 client."""

    database: str = "_system"
    socket_path: str | None = None  # None = use network (base_url), str = use Unix socket
    base_url: str = "http://localhost:8529"
    username: str | None = None
    password: str | None = None
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_limits: httpx.Limits | None = None


class ArangoHttp2Client:
    """ArangoDB client supporting Unix sockets or network transport.

    Protocol note: HTTP/2 requires TLS/ALPN negotiation in standard deployments.
    Over Unix domain sockets (UDS) with cleartext HTTP, connections will use
    HTTP/1.1 unless the server supports HTTP/2 prior-knowledge (h2c). This is
    acceptable for local connections where the performance difference is minimal.
    Network connections over HTTPS can negotiate HTTP/2 via ALPN.
    """

    def __init__(self, config: ArangoHttp2Config) -> None:
        self._config = config

        # Use Unix socket transport if socket_path is provided, otherwise network.
        # Note: UDS with http:// base_url will use HTTP/1.1 (no TLS/ALPN for HTTP/2).
        # This is fine for local connections; network+HTTPS can use HTTP/2.
        if config.socket_path:
            transport = httpx.HTTPTransport(
                uds=config.socket_path,
                retries=0,
            )
        else:
            transport = httpx.HTTPTransport(
                retries=0,
            )

        timeout = httpx.Timeout(
            connect=config.connect_timeout,
            read=config.read_timeout,
            write=config.write_timeout,
            pool=config.connect_timeout,
        )
        auth = None
        if config.username and config.password:
            auth = (config.username, config.password)

        # http2=True enables HTTP/2 when available (HTTPS with ALPN).
        # For UDS/cleartext, httpx gracefully falls back to HTTP/1.1.
        self._client = httpx.Client(
            http2=True,
            base_url=config.base_url,
            transport=transport,
            timeout=timeout,
            limits=config.pool_limits,
            auth=auth,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ArangoHttp2Client:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - context helper
        self.close()

    def get_document(self, collection: str, key: str) -> dict[str, Any]:
        """Fetch a single document by collection/key."""

        path = f"/_db/{self._config.database}/_api/document/{collection}/{key}"
        response = self._client.get(path)
        return self._handle_response(response)

    def insert_documents(
        self,
        collection: str,
        documents: Iterable[dict[str, Any]],
        overwrite: bool = False,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Bulk insert documents using NDJSON import.

        Args:
            collection: Target collection name
            documents: Iterable of documents to insert
            overwrite: If True, overwrite existing documents with same _key
            stream: If True (default), stream documents without buffering entire
                payload in memory. Uses chunked transfer encoding. Set to False
                for compatibility with servers that require Content-Length.

        Returns:
            Dict with import statistics (created, errors, etc.)
        """
        overwrite_str = "true" if overwrite else "false"
        path = (
            f"/_db/{self._config.database}/_api/import"
            f"?collection={collection}&type=documents&complete=true&overwrite={overwrite_str}"
        )
        user_agent = os.environ.get("HADES_HTTP_USER_AGENT", "hades-arango-http2/1.0")
        trace_id = os.environ.get("HADES_TRACE_ID")
        headers: dict[str, str] = {
            "Content-Type": "application/x-ndjson",
            "User-Agent": user_agent,
        }
        if trace_id:
            headers["x-hades-trace"] = trace_id

        if stream:
            # Streaming mode: use generator to avoid buffering entire payload
            # We need to peek at the first document to check if the iterator is empty
            document_iter = iter(documents)
            try:
                first_doc = next(document_iter)
            except StopIteration:
                return {"created": 0}

            # Create streaming generator starting with the peeked first document
            content = self._ndjson_stream_with_first(first_doc, document_iter)
            response = self._client.post(path, content=content, headers=headers)
        else:
            # Buffered mode: build payload in memory (for Content-Length header)
            payload = self._ndjson_buffer(documents)
            if payload is None:
                return {"created": 0}
            headers["Content-Length"] = str(len(payload))
            response = self._client.post(path, content=payload, headers=headers)

        return self._handle_response(response)

    def _ndjson_stream_with_first(self, first_doc: dict[str, Any], rest: Iterator[dict[str, Any]]) -> Iterator[bytes]:
        """Generate NDJSON lines as a stream, starting with a pre-peeked first document.

        Yields bytes for each document, enabling memory-efficient bulk imports
        without buffering the entire payload.
        """
        yield orjson.dumps(first_doc)
        for doc in rest:
            yield b"\n" + orjson.dumps(doc)

    def _ndjson_buffer(self, documents: Iterable[dict[str, Any]]) -> bytes | None:
        """Build NDJSON payload in memory for Content-Length mode.

        Returns None if no documents provided.
        """
        document_iter = iter(documents)
        try:
            first_doc = next(document_iter)
        except StopIteration:
            return None

        parts = [orjson.dumps(first_doc)]
        for doc in document_iter:
            parts.append(b"\n")
            parts.append(orjson.dumps(doc))

        return b"".join(parts)

    def query(
        self,
        aql: str,
        bind_vars: dict[str, Any] | None = None,
        batch_size: int = 1000,
        full_count: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute an AQL query and return the full result set."""

        payload = {
            "query": aql,
            "batchSize": batch_size,
            "bindVars": bind_vars or {},
            "options": {"fullCount": full_count},
        }
        path = f"/_db/{self._config.database}/_api/cursor"
        response = self._client.post(path, json=payload)
        data = self._handle_response(response)

        results = data.get("result", [])
        cursor_id = data.get("id")
        while data.get("hasMore") and cursor_id:
            follow_path = f"/_db/{self._config.database}/_api/cursor/{cursor_id}"
            data = self._handle_response(self._client.put(follow_path))
            results.extend(data.get("result", []))
            cursor_id = data.get("id")

        return results

    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ) -> dict[str, Any]:
        """Perform an arbitrary HTTP request against the ArangoDB REST API."""

        response = self._client.request(
            method,
            path,
            json=json,
            params=params,
            headers=headers,
            content=content,
        )
        return self._handle_response(response)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_vector_index(
        self,
        collection: str,
        field: str = "embedding",
        dimension: int = 2048,
        n_lists: int | None = None,
        n_probe: int = 10,
        metric: str = "cosine",
    ) -> dict[str, Any]:
        """Create a FAISS-backed vector index on a collection.

        Args:
            collection: Collection name containing embedding documents.
            field: Field name holding the embedding array.
            dimension: Embedding vector dimension.
            n_lists: Number of IVF cells (None = auto-calculate from collection size).
            n_probe: Number of cells to probe during search (recall vs speed).
            metric: Distance metric — "cosine", "l2", or "innerProduct".

        Returns:
            Index definition dict from ArangoDB.
        """
        if n_lists is None:
            # Auto-calculate: get collection count, use N/15 with minimum of 1
            count_result = self.query(f"RETURN LENGTH({collection})")
            doc_count = count_result[0] if count_result else 0
            n_lists = max(1, doc_count // 15)

        path = f"/_db/{self._config.database}/_api/index"
        payload: dict[str, Any] = {
            "type": "inverted",
            "fields": [
                {
                    "name": field,
                    "features": ["vector"],
                    "vector": {
                        "type": "float32",
                        "dimension": dimension,
                        "similarity": metric,
                        "nLists": n_lists,
                        "nProbe": n_probe,
                    },
                }
            ],
        }
        return self.request("POST", path, json=payload, params={"collection": collection})

    def list_indexes(self, collection: str) -> list[dict[str, Any]]:
        """List all indexes on a collection.

        Returns:
            List of index definition dicts.
        """
        path = f"/_db/{self._config.database}/_api/index"
        result = self.request("GET", path, params={"collection": collection})
        return result.get("indexes", [])

    def drop_index(self, index_id: str) -> dict[str, Any]:
        """Drop an index by its full ID (e.g. "collection/12345").

        Returns:
            Response dict with the dropped index ID.
        """
        path = f"/_db/{self._config.database}/_api/index/{index_id}"
        return self.request("DELETE", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.http_version not in {"HTTP/2", "HTTP/1.1"}:
            raise RuntimeError(
                f"Unexpected HTTP version {response.http_version!r} "
                f"for {response.request.method} {response.request.url}"
            )

        if response.status_code >= 400:
            try:
                details = response.json()
            except ValueError:
                details = {"message": response.text}
            message = details.get("errorMessage") or details.get("message") or response.text
            raise ArangoHttpError(response.status_code, message, details)

        if response.status_code == 204:
            return {}

        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()

        # For NDJSON import Arango responds with JSON content type, but as a safeguard:
        try:
            return orjson.loads(response.text)
        except orjson.JSONDecodeError:
            return {"raw": response.text}


__all__ = [
    "ArangoHttp2Client",
    "ArangoHttp2Config",
    "ArangoHttpError",
]
