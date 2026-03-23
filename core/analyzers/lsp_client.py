"""General-purpose LSP JSON-RPC transport client.

Manages a language server subprocess and provides synchronous request/response
communication over the LSP base protocol (Content-Length framing over stdin/stdout).

This is a pure transport layer — no language-server-specific logic. It handles:
- JSON-RPC message framing (Content-Length header + JSON body)
- Request/response correlation by ID
- Server-initiated notification buffering
- Subprocess lifecycle (start, health check, shutdown, force kill)

Usage:
    client = LspClient(["rust-analyzer"], cwd="/path/to/crate")
    client.start()
    result = client.request("initialize", {"capabilities": {}})
    client.notify("initialized", {})
    # ... use the language server ...
    client.shutdown()
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)


class LspError(Exception):
    """Base exception for LSP client errors."""


class LspResponseError(LspError):
    """Error returned by the language server in a JSON-RPC response."""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.data = data
        super().__init__(f"LSP error {code}: {message}")


class LspTimeoutError(LspError):
    """Request timed out waiting for a response."""


class LspProcessError(LspError):
    """Language server process crashed or is unavailable."""


class LspClient:
    """Synchronous LSP client using JSON-RPC over stdin/stdout.

    Spawns a language server as a subprocess, sends requests via stdin,
    and reads responses from stdout using the LSP base protocol.

    A daemon reader thread continuously reads from stdout and dispatches
    messages: responses (have 'id') go to per-request events, notifications
    (no 'id') go to a shared queue.

    Thread-safe: multiple threads can call request() concurrently.

    Args:
        command: Command to spawn the language server (e.g. ["rust-analyzer"]).
        cwd: Working directory for the subprocess.
        env: Optional environment variables for the subprocess.
    """

    def __init__(
        self,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._cwd = cwd
        self._env = env

        self._process: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._started = False
        self._shutting_down = False

        # Request/response tracking
        self._next_id = 1
        self._id_lock = threading.Lock()
        self._pending: dict[int, threading.Event] = {}
        self._responses: dict[int, dict[str, Any]] = {}
        self._pending_lock = threading.Lock()

        # Write lock — only one thread writes to stdin at a time
        self._write_lock = threading.Lock()

        # Notification queue (bounded to prevent unbounded memory growth)
        self._notifications: deque[dict[str, Any]] = deque(maxlen=1000)
        self._notifications_lock = threading.Lock()

        # Tracks whether the reader thread hit EOF or an error
        self._reader_error: Exception | None = None

    def start(self) -> None:
        """Start the language server subprocess and reader thread.

        Raises:
            LspProcessError: If the process fails to start.
        """
        if self._started:
            return

        try:
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self._cwd,
                env=self._env,
            )
        except FileNotFoundError as e:
            raise LspProcessError(
                f"Language server not found: {self._command[0]}"
            ) from e
        except OSError as e:
            raise LspProcessError(
                f"Failed to start language server: {e}"
            ) from e

        self._started = True
        self._shutting_down = False

        # Start the reader thread
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="lsp-reader",
            daemon=True,
        )
        self._reader_thread.start()

        logger.info("Started LSP server: %s (pid=%d)", self._command, self._process.pid)

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the response.

        Args:
            method: LSP method name (e.g. "initialize", "textDocument/documentSymbol").
            params: Request parameters.
            timeout: Maximum seconds to wait for a response.

        Returns:
            The 'result' field from the JSON-RPC response.

        Raises:
            LspResponseError: If the server returns an error response.
            LspTimeoutError: If the response doesn't arrive within timeout.
            LspProcessError: If the server process is dead.
        """
        self._check_alive()

        # Allocate request ID
        with self._id_lock:
            req_id = self._next_id
            self._next_id += 1

        # Register pending request
        event = threading.Event()
        with self._pending_lock:
            self._pending[req_id] = event

        # Build and send the message
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params

        self._send(message)
        logger.debug("Sent request %d: %s", req_id, method)

        # Wait for response
        if not event.wait(timeout=timeout):
            with self._pending_lock:
                self._pending.pop(req_id, None)
                self._responses.pop(req_id, None)
            raise LspTimeoutError(
                f"Timed out waiting for response to {method} (id={req_id}, timeout={timeout}s)"
            )

        # Retrieve response
        with self._pending_lock:
            self._pending.pop(req_id, None)
            response = self._responses.pop(req_id, None)

        if response is None:
            raise LspProcessError("Response was None — reader thread may have crashed")

        # Check for error response
        if "error" in response:
            err = response["error"]
            raise LspResponseError(
                code=err.get("code", -1),
                message=err.get("message", "Unknown error"),
                data=err.get("data"),
            )

        return response.get("result", {})

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification (no response expected).

        Args:
            method: LSP method name.
            params: Notification parameters.
        """
        self._check_alive()

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            message["params"] = params

        self._send(message)
        logger.debug("Sent notification: %s", method)

    def get_notifications(self, method: str | None = None) -> list[dict[str, Any]]:
        """Retrieve buffered server notifications.

        Args:
            method: If provided, only return notifications with this method.

        Returns:
            List of notification messages (oldest first).
        """
        with self._notifications_lock:
            if method is None:
                result = list(self._notifications)
            else:
                result = [n for n in self._notifications if n.get("method") == method]
        return result

    def drain_notifications(self, method: str | None = None) -> list[dict[str, Any]]:
        """Retrieve and remove buffered server notifications.

        Args:
            method: If provided, only drain notifications with this method.

        Returns:
            List of notification messages (oldest first).
        """
        with self._notifications_lock:
            if method is None:
                result = list(self._notifications)
                self._notifications.clear()
            else:
                result = []
                remaining: deque[dict[str, Any]] = deque(maxlen=1000)
                for n in self._notifications:
                    if n.get("method") == method:
                        result.append(n)
                    else:
                        remaining.append(n)
                self._notifications = remaining
        return result

    @property
    def is_alive(self) -> bool:
        """Check if the language server process is still running."""
        return (
            self._process is not None
            and self._process.poll() is None
            and self._started
        )

    def shutdown(self, timeout: float = 10.0) -> None:
        """Shut down the language server gracefully.

        Sends shutdown request, then exit notification, then waits for
        the process to exit. Force-kills after timeout.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown.
        """
        if not self._started or self._process is None:
            return

        self._shutting_down = True

        # Try graceful shutdown
        try:
            if self.is_alive:
                self.request("shutdown", timeout=min(timeout / 2, 5.0))
                self.notify("exit")
        except (LspError, OSError):
            pass  # Best-effort — if it fails, we force-kill below

        # Wait for process exit
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("LSP server did not exit gracefully, killing (pid=%d)", self._process.pid)
            self._process.kill()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("Failed to kill LSP server (pid=%d)", self._process.pid)

        # Wait for reader thread
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

        self._started = False
        logger.info("LSP server shut down")

    def __enter__(self) -> LspClient:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ── Internal methods ──────────────────────────────────────────

    def _send(self, message: dict[str, Any]) -> None:
        """Encode and write a JSON-RPC message with Content-Length framing."""
        assert self._process is not None and self._process.stdin is not None

        body = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")

        with self._write_lock:
            try:
                self._process.stdin.write(header + body)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise LspProcessError(f"Failed to write to LSP server: {e}") from e

    def _reader_loop(self) -> None:
        """Background thread: read and dispatch messages from stdout."""
        assert self._process is not None and self._process.stdout is not None

        stdout = self._process.stdout

        try:
            while not self._shutting_down:
                # Read headers until blank line
                content_length = self._read_headers(stdout)
                if content_length is None:
                    if not self._shutting_down:
                        self._reader_error = LspProcessError("EOF from language server")
                    break

                # Read body
                body = self._read_exactly(stdout, content_length)
                if body is None:
                    if not self._shutting_down:
                        self._reader_error = LspProcessError("EOF from language server (truncated message)")
                    break

                # Parse JSON
                try:
                    message = json.loads(body)
                except json.JSONDecodeError:
                    logger.warning("Malformed JSON-RPC message: %s", body[:200])
                    continue

                self._dispatch(message)

        except Exception as e:
            if not self._shutting_down:
                self._reader_error = e
                logger.error("LSP reader thread error: %s", e)

        finally:
            # Wake all pending requests so they don't hang on EOF or error
            with self._pending_lock:
                for event in self._pending.values():
                    event.set()

    def _read_headers(self, stream: Any) -> int | None:
        """Read LSP headers and return the Content-Length value.

        Returns None on EOF.
        """
        content_length = -1

        while True:
            line = stream.readline()
            if not line:
                return None  # EOF

            line_str = line.decode("ascii", errors="replace").strip()

            if not line_str:
                # Blank line = end of headers
                break

            if line_str.lower().startswith("content-length:"):
                try:
                    content_length = int(line_str.split(":", 1)[1].strip())
                except ValueError:
                    logger.warning("Invalid Content-Length: %s", line_str)

        if content_length < 0:
            logger.warning("No Content-Length header found")
            return None

        return content_length

    def _read_exactly(self, stream: Any, n: int) -> bytes | None:
        """Read exactly n bytes from stream. Returns None on EOF."""
        data = b""
        while len(data) < n:
            chunk = stream.read(n - len(data))
            if not chunk:
                return None  # EOF
            data += chunk
        return data

    def _dispatch(self, message: dict[str, Any]) -> None:
        """Route a parsed JSON-RPC message to the right handler."""
        if "id" in message and ("result" in message or "error" in message):
            # This is a response to a request we sent
            req_id = message["id"]
            with self._pending_lock:
                if req_id in self._pending:
                    self._responses[req_id] = message
                    self._pending[req_id].set()
                else:
                    logger.debug("Received response for unknown request id=%s", req_id)

        elif "id" in message and "method" in message:
            # Server-to-client request (e.g. window/workDoneProgress/create)
            # Auto-respond with empty result to avoid blocking the server
            self._handle_server_request(message)

        else:
            # Notification from the server (no id)
            with self._notifications_lock:
                self._notifications.append(message)
            method = message.get("method", "?")
            if method not in ("$/progress", "textDocument/publishDiagnostics"):
                logger.debug("Server notification: %s", method)

    def _handle_server_request(self, message: dict[str, Any]) -> None:
        """Handle a request FROM the server (server→client).

        Some LSP servers send requests that need responses, such as
        window/workDoneProgress/create. We auto-accept these.
        """
        req_id = message["id"]
        method = message.get("method", "")

        logger.debug("Server request: %s (id=%s)", method, req_id)

        # Auto-respond with null result (accept)
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": None,
        }

        try:
            self._send(response)
        except LspProcessError:
            pass  # Best-effort — server may have crashed

    def _check_alive(self) -> None:
        """Raise if the server process is not running."""
        if not self._started:
            raise LspProcessError("LSP client has not been started")
        if self._process is None or self._process.poll() is not None:
            exit_code = self._process.returncode if self._process else None
            raise LspProcessError(
                f"LSP server process is dead (exit code: {exit_code})"
            )
        if self._reader_error is not None:
            raise LspProcessError(
                f"LSP reader thread failed: {self._reader_error}"
            ) from self._reader_error
