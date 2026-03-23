"""Tests for the LSP JSON-RPC transport client.

Tests cover:
- JSON-RPC framing (Content-Length header + JSON body)
- Request/response correlation
- Notification handling
- Server-to-client request auto-response
- Subprocess lifecycle (start, shutdown, force kill)
- Error handling (process not found, timeouts, crashes)
- Live integration with rust-analyzer (skipped if not installed)
"""

from __future__ import annotations

import shutil
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from core.analyzers.lsp_client import (
    LspClient,
    LspProcessError,
    LspResponseError,
    LspTimeoutError,
)

# ── Fixtures ──────────────────────────────────────────────────────


MOCK_SERVER_SCRIPT = textwrap.dedent("""\
    \"\"\"Minimal LSP mock server for testing the transport layer.\"\"\"
    import json
    import sys

    def read_message():
        # Read headers
        content_length = -1
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            line_str = line.decode("ascii").strip()
            if not line_str:
                break
            if line_str.lower().startswith("content-length:"):
                content_length = int(line_str.split(":", 1)[1].strip())
        if content_length < 0:
            return None
        body = sys.stdin.buffer.read(content_length)
        return json.loads(body)

    def send_message(msg):
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\\r\\n\\r\\n".encode("ascii")
        sys.stdout.buffer.write(header + body)
        sys.stdout.buffer.flush()

    # Main loop
    while True:
        msg = read_message()
        if msg is None:
            break

        method = msg.get("method", "")
        msg_id = msg.get("id")

        if method == "test/echo":
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": msg.get("params", {})})

        elif method == "test/error":
            send_message({
                "jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32600, "message": "Test error", "data": "extra"}
            })

        elif method == "test/slow":
            import time
            time.sleep(5)
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": {"slow": True}})

        elif method == "test/notify_back":
            # Send a notification, then respond
            send_message({"jsonrpc": "2.0", "method": "test/serverNotification", "params": {"hello": "world"}})
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": {"notified": True}})

        elif method == "test/server_request":
            # Send a server-to-client request, then respond to the original
            send_message({"jsonrpc": "2.0", "id": 9999, "method": "window/workDoneProgress/create", "params": {"token": "t1"}})
            import time
            time.sleep(0.1)  # Give client time to respond
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": {"ok": True}})

        elif method == "shutdown":
            send_message({"jsonrpc": "2.0", "id": msg_id, "result": None})

        elif method == "exit":
            break
""")


@pytest.fixture
def mock_server_path(tmp_path: Path) -> Path:
    """Write the mock server script to a temp file."""
    script = tmp_path / "mock_lsp_server.py"
    script.write_text(MOCK_SERVER_SCRIPT)
    return script


@pytest.fixture
def client(mock_server_path: Path) -> LspClient:
    """Create and start an LspClient against the mock server."""
    c = LspClient([sys.executable, str(mock_server_path)])
    c.start()
    yield c
    if c.is_alive:
        c.shutdown()


# ── Unit Tests ────────────────────────────────────────────────────


class TestLspClientStartStop:
    """Test subprocess lifecycle."""

    def test_start_and_alive(self, client: LspClient) -> None:
        assert client.is_alive

    def test_shutdown(self, client: LspClient) -> None:
        client.shutdown()
        assert not client.is_alive

    def test_process_not_found(self) -> None:
        c = LspClient(["nonexistent_binary_that_does_not_exist_xyz"])
        with pytest.raises(LspProcessError, match="not found"):
            c.start()

    def test_double_start_is_idempotent(self, client: LspClient) -> None:
        client.start()  # Second call should be no-op
        assert client.is_alive

    def test_context_manager(self, mock_server_path: Path) -> None:
        with LspClient([sys.executable, str(mock_server_path)]) as c:
            assert c.is_alive
        assert not c.is_alive


class TestRequestResponse:
    """Test JSON-RPC request/response correlation."""

    def test_echo(self, client: LspClient) -> None:
        result = client.request("test/echo", {"key": "value"})
        assert result == {"key": "value"}

    def test_empty_params(self, client: LspClient) -> None:
        result = client.request("test/echo")
        assert result == {}

    def test_error_response(self, client: LspClient) -> None:
        with pytest.raises(LspResponseError) as exc_info:
            client.request("test/error")
        assert exc_info.value.code == -32600
        assert "Test error" in str(exc_info.value)
        assert exc_info.value.data == "extra"

    def test_timeout(self, client: LspClient) -> None:
        with pytest.raises(LspTimeoutError):
            client.request("test/slow", timeout=0.5)

    def test_concurrent_requests(self, client: LspClient) -> None:
        """Multiple threads can send requests concurrently."""
        results: dict[int, Any] = {}
        errors: list[Exception] = []

        def send_request(idx: int) -> None:
            try:
                r = client.request("test/echo", {"index": idx})
                results[idx] = r
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=send_request, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        assert len(results) == 5
        for i in range(5):
            assert results[i] == {"index": i}

    def test_request_before_start(self, mock_server_path: Path) -> None:
        c = LspClient([sys.executable, str(mock_server_path)])
        with pytest.raises(LspProcessError, match="not been started"):
            c.request("test/echo")


class TestNotifications:
    """Test notification handling (both directions)."""

    def test_client_to_server_notification(self, client: LspClient) -> None:
        """Client notifications don't expect a response."""
        client.notify("test/no_response", {"data": 1})
        # No error = success (server ignores unknown notifications)

    def test_server_notification_buffered(self, client: LspClient) -> None:
        """Server notifications are buffered and retrievable."""
        result = client.request("test/notify_back")
        assert result == {"notified": True}

        # Give reader thread a moment to buffer the notification
        time.sleep(0.1)

        notifications = client.get_notifications("test/serverNotification")
        assert len(notifications) >= 1
        assert notifications[0]["params"] == {"hello": "world"}

    def test_drain_notifications(self, client: LspClient) -> None:
        """drain_notifications returns and removes notifications."""
        client.request("test/notify_back")
        time.sleep(0.1)

        drained = client.drain_notifications("test/serverNotification")
        assert len(drained) >= 1

        # Should be empty now
        remaining = client.get_notifications("test/serverNotification")
        assert len(remaining) == 0


class TestServerRequests:
    """Test handling of server-to-client requests."""

    def test_auto_respond_to_server_request(self, client: LspClient) -> None:
        """Client auto-responds to server requests like workDoneProgress/create."""
        result = client.request("test/server_request", timeout=5)
        assert result == {"ok": True}


# ── Live rust-analyzer integration ────────────────────────────────


HAS_RUST_ANALYZER = shutil.which("rust-analyzer") is not None


@pytest.mark.skipif(not HAS_RUST_ANALYZER, reason="rust-analyzer not installed")
class TestLiveRustAnalyzer:
    """Integration tests against a real rust-analyzer instance.

    These tests use a minimal Rust crate fixture to verify that the
    transport layer works correctly with the real server.
    """

    @pytest.fixture
    def rust_crate(self, tmp_path: Path) -> Path:
        """Create a minimal Rust crate for testing."""
        crate = tmp_path / "test_crate"
        crate.mkdir()
        (crate / "Cargo.toml").write_text(textwrap.dedent("""\
            [package]
            name = "test_crate"
            version = "0.1.0"
            edition = "2021"
        """))
        src = crate / "src"
        src.mkdir()
        (src / "lib.rs").write_text(textwrap.dedent("""\
            pub struct Foo {
                pub value: i32,
            }

            impl Foo {
                pub fn new(value: i32) -> Self {
                    Foo { value }
                }

                pub fn double(&self) -> i32 {
                    self.value * 2
                }
            }

            pub fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        """))
        return crate

    def test_initialize_handshake(self, rust_crate: Path) -> None:
        """Verify we can complete the LSP initialize handshake."""
        with LspClient(["rust-analyzer"], cwd=str(rust_crate)) as client:
            result = client.request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": f"file://{rust_crate}",
                    "capabilities": {},
                },
                timeout=60,
            )

            # Server should return capabilities
            assert "capabilities" in result

            # Send initialized notification
            client.notify("initialized", {})

    def test_document_symbol(self, rust_crate: Path) -> None:
        """Verify we can get document symbols from rust-analyzer."""
        with LspClient(["rust-analyzer"], cwd=str(rust_crate)) as client:
            # Initialize
            client.request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": f"file://{rust_crate}",
                    "capabilities": {
                        "textDocument": {
                            "documentSymbol": {
                                "hierarchicalDocumentSymbolSupport": True,
                            }
                        }
                    },
                },
                timeout=60,
            )
            client.notify("initialized", {})

            # Open the file
            lib_rs = rust_crate / "src" / "lib.rs"
            uri = f"file://{lib_rs}"
            client.notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": "rust",
                        "version": 1,
                        "text": lib_rs.read_text(),
                    }
                },
            )

            # Wait for rust-analyzer to index (it sends progress notifications)
            time.sleep(3)

            # Request document symbols
            symbols = client.request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": uri}},
                timeout=30,
            )

            # Should find Foo struct, its methods, and the add function
            assert isinstance(symbols, list)
            symbol_names = []
            for sym in symbols:
                symbol_names.append(sym.get("name", ""))
                # Check children (methods are nested under struct/impl)
                for child in sym.get("children", []):
                    symbol_names.append(child.get("name", ""))

            assert "Foo" in symbol_names
            assert "add" in symbol_names
