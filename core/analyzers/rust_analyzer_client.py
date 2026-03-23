"""rust-analyzer session manager.

Manages the full lifecycle of a rust-analyzer LSP session:
- Locate rust-analyzer binary
- Start the process via LspClient
- Complete the LSP initialize/initialized handshake
- Wait for indexing to finish (via $/progress notifications)
- Open/close Rust source files
- Proxy LSP requests
- Graceful shutdown

Usage:
    with RustAnalyzerSession("/path/to/crate") as session:
        symbols = session.document_symbols("src/lib.rs")
        hover = session.hover("src/lib.rs", line=10, character=5)
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from core.analyzers.lsp_client import LspClient, LspError

logger = logging.getLogger(__name__)


class RustAnalyzerError(LspError):
    """Error specific to rust-analyzer session management."""


class RustAnalyzerSession:
    """Manages a rust-analyzer LSP session for a Rust crate.

    Handles the full initialization handshake, waits for indexing,
    and provides convenience methods for common LSP requests.

    Args:
        crate_root: Path to the Rust crate root (directory containing Cargo.toml).
        timeout: Maximum seconds to wait for rust-analyzer to index the crate.
        rust_analyzer_cmd: Override the rust-analyzer binary path.
    """

    def __init__(
        self,
        crate_root: str | Path,
        timeout: int = 120,
        rust_analyzer_cmd: str | None = None,
    ) -> None:
        self._crate_root = Path(crate_root).resolve()
        self._timeout = timeout
        self._rust_analyzer_cmd = rust_analyzer_cmd or self._find_rust_analyzer()
        self._client: LspClient | None = None
        self._server_capabilities: dict[str, Any] = {}
        self._open_files: dict[str, int] = {}  # uri → version
        self._ready = False

        # Validate crate root
        cargo_toml = self._crate_root / "Cargo.toml"
        if not cargo_toml.exists():
            raise RustAnalyzerError(
                f"No Cargo.toml found at {self._crate_root}. "
                "RustAnalyzerSession requires a valid Rust crate."
            )

    @property
    def crate_root(self) -> Path:
        """The crate root directory."""
        return self._crate_root

    @property
    def is_ready(self) -> bool:
        """Whether rust-analyzer has finished indexing."""
        return self._ready

    @property
    def server_capabilities(self) -> dict[str, Any]:
        """LSP server capabilities returned during initialization."""
        return self._server_capabilities

    def start(self) -> None:
        """Start rust-analyzer and complete the initialization handshake.

        Blocks until indexing is complete or timeout is reached.
        """
        self._client = LspClient(
            command=[self._rust_analyzer_cmd],
            cwd=str(self._crate_root),
        )
        self._client.start()

        # Initialize handshake
        self._initialize()

        # Wait for indexing
        self._wait_for_ready()

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
        """Send an LSP request to rust-analyzer.

        Args:
            method: LSP method name.
            params: Request parameters.
            timeout: Maximum seconds to wait.

        Returns:
            The response result.
        """
        self._check_started()
        assert self._client is not None
        return self._client.request(method, params, timeout=timeout)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send an LSP notification to rust-analyzer."""
        self._check_started()
        assert self._client is not None
        self._client.notify(method, params)

    def open_file(self, file_path: str | Path) -> str:
        """Open a Rust source file in the LSP session.

        Args:
            file_path: Path to the .rs file (absolute or relative to crate root).

        Returns:
            The file URI used by the LSP protocol.
        """
        self._check_started()
        assert self._client is not None

        path = Path(file_path)
        if not path.is_absolute():
            path = self._crate_root / path
        path = path.resolve()

        uri = path.as_uri()

        if uri in self._open_files:
            return uri  # Already open

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise RustAnalyzerError(f"Cannot read file: {e}") from e

        version = 1
        self._client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "rust",
                    "version": version,
                    "text": content,
                }
            },
        )
        self._open_files[uri] = version

        logger.debug("Opened file: %s", path)
        return uri

    def close_file(self, uri: str) -> None:
        """Close a previously opened file."""
        if uri not in self._open_files:
            return

        self._check_started()
        assert self._client is not None

        self._client.notify(
            "textDocument/didClose",
            {"textDocument": {"uri": uri}},
        )
        del self._open_files[uri]

    def document_symbols(self, file_path: str | Path, timeout: float = 30.0) -> list[dict[str, Any]]:
        """Get document symbols for a Rust file.

        Opens the file if not already open, then requests symbols.

        Args:
            file_path: Path to the .rs file.
            timeout: Maximum seconds to wait.

        Returns:
            List of LSP DocumentSymbol dicts (hierarchical).
        """
        uri = self.open_file(file_path)
        result = self.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": uri}},
            timeout=timeout,
        )
        return result if isinstance(result, list) else []

    def hover(
        self,
        file_path: str | Path,
        line: int,
        character: int,
        timeout: float = 10.0,
        retries: int = 3,
        retry_delay: float = 1.0,
    ) -> dict[str, Any] | None:
        """Get hover information at a position.

        Retries on None results because rust-analyzer may still be
        completing semantic analysis after indexing is reported complete.

        Args:
            file_path: Path to the .rs file.
            line: Zero-based line number.
            character: Zero-based character offset.
            timeout: Maximum seconds to wait per attempt.
            retries: Number of retry attempts if result is None.
            retry_delay: Seconds to wait between retries.

        Returns:
            Hover result dict, or None if no hover info available.
        """
        uri = self.open_file(file_path)

        for attempt in range(retries + 1):
            try:
                result = self.request(
                    "textDocument/hover",
                    {
                        "textDocument": {"uri": uri},
                        "position": {"line": line, "character": character},
                    },
                    timeout=timeout,
                )
                if result:
                    return result
                if attempt < retries:
                    time.sleep(retry_delay)
            except LspError:
                if attempt < retries:
                    time.sleep(retry_delay)
                else:
                    return None

        return None

    def call_hierarchy_outgoing(
        self, file_path: str | Path, line: int, character: int, timeout: float = 30.0
    ) -> list[dict[str, Any]]:
        """Get outgoing calls from a symbol at the given position.

        First prepares a call hierarchy item, then requests outgoing calls.

        Args:
            file_path: Path to the .rs file.
            line: Zero-based line number of the symbol.
            character: Zero-based character offset.
            timeout: Maximum seconds to wait.

        Returns:
            List of outgoing call dicts, each with 'to' (CallHierarchyItem)
            and 'fromRanges'.
        """
        uri = self.open_file(file_path)

        # Step 1: Prepare call hierarchy item
        items = self.request(
            "textDocument/prepareCallHierarchy",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
            timeout=timeout,
        )

        if not items:
            return []

        # Step 2: Get outgoing calls for the first item
        outgoing = self.request(
            "callHierarchy/outgoingCalls",
            {"item": items[0]},
            timeout=timeout,
        )

        return outgoing if isinstance(outgoing, list) else []

    def call_hierarchy_incoming(
        self, file_path: str | Path, line: int, character: int, timeout: float = 30.0
    ) -> list[dict[str, Any]]:
        """Get incoming calls to a symbol at the given position.

        Args:
            file_path: Path to the .rs file.
            line: Zero-based line number of the symbol.
            character: Zero-based character offset.
            timeout: Maximum seconds to wait.

        Returns:
            List of incoming call dicts.
        """
        uri = self.open_file(file_path)

        items = self.request(
            "textDocument/prepareCallHierarchy",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
            timeout=timeout,
        )

        if not items:
            return []

        incoming = self.request(
            "callHierarchy/incomingCalls",
            {"item": items[0]},
            timeout=timeout,
        )

        return incoming if isinstance(incoming, list) else []

    def references(
        self,
        file_path: str | Path,
        line: int,
        character: int,
        timeout: float = 30.0,
        retries: int = 3,
        retry_delay: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Find all references to a symbol.

        Retries on empty results since rust-analyzer may still be
        completing cross-file analysis.

        Args:
            file_path: Path to the .rs file.
            line: Zero-based line number.
            character: Zero-based character offset.
            timeout: Maximum seconds to wait per attempt.
            retries: Number of retry attempts if result is empty.
            retry_delay: Seconds between retries.

        Returns:
            List of Location dicts with uri and range.
        """
        uri = self.open_file(file_path)

        for attempt in range(retries + 1):
            try:
                result = self.request(
                    "textDocument/references",
                    {
                        "textDocument": {"uri": uri},
                        "position": {"line": line, "character": character},
                        "context": {"includeDeclaration": True},
                    },
                    timeout=timeout,
                )
                refs = result if isinstance(result, list) else []
                if refs:
                    return refs
                if attempt < retries:
                    time.sleep(retry_delay)
            except LspError:
                if attempt < retries:
                    time.sleep(retry_delay)
                else:
                    return []

        return []

    def goto_definition(
        self,
        file_path: str | Path,
        line: int,
        character: int,
        timeout: float = 10.0,
        retries: int = 3,
        retry_delay: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Go to definition of a symbol.

        Retries on empty results since rust-analyzer may still be
        completing analysis.

        Args:
            file_path: Path to the .rs file.
            line: Zero-based line number.
            character: Zero-based character offset.
            timeout: Maximum seconds to wait per attempt.
            retries: Number of retry attempts.
            retry_delay: Seconds between retries.

        Returns:
            List of Location dicts.
        """
        uri = self.open_file(file_path)

        for attempt in range(retries + 1):
            try:
                result = self.request(
                    "textDocument/definition",
                    {
                        "textDocument": {"uri": uri},
                        "position": {"line": line, "character": character},
                    },
                    timeout=timeout,
                )
                if result is None:
                    locations: list[dict[str, Any]] = []
                elif isinstance(result, dict):
                    locations = [result]
                elif isinstance(result, list):
                    locations = result
                else:
                    locations = []

                if locations:
                    return locations
                if attempt < retries:
                    time.sleep(retry_delay)
            except LspError:
                if attempt < retries:
                    time.sleep(retry_delay)
                else:
                    return []

        return []

    def shutdown(self) -> None:
        """Shut down the rust-analyzer session."""
        if self._client is None:
            return

        # Close all open files
        for uri in list(self._open_files):
            try:
                self.close_file(uri)
            except LspError:
                pass

        self._client.shutdown()
        self._client = None
        self._ready = False
        self._open_files.clear()

    def __enter__(self) -> RustAnalyzerSession:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ── Internal methods ──────────────────────────────────────────

    def _initialize(self) -> None:
        """Send the LSP initialize request with capabilities."""
        assert self._client is not None

        root_uri = self._crate_root.as_uri()

        result = self._client.request(
            "initialize",
            {
                "processId": None,
                "rootUri": root_uri,
                "rootPath": str(self._crate_root),
                "workspaceFolders": [
                    {"uri": root_uri, "name": self._crate_root.name}
                ],
                "capabilities": {
                    "textDocument": {
                        "documentSymbol": {
                            "hierarchicalDocumentSymbolSupport": True,
                            "symbolKind": {
                                "valueSet": list(range(1, 27)),
                            },
                        },
                        "hover": {
                            "contentFormat": ["markdown", "plaintext"],
                        },
                        "callHierarchy": {
                            "dynamicRegistration": False,
                        },
                        "references": {},
                        "definition": {},
                        "publishDiagnostics": {
                            "relatedInformation": True,
                        },
                    },
                    "window": {
                        "workDoneProgress": True,
                    },
                },
                "initializationOptions": {
                    # Tell rust-analyzer to report progress
                    "workDoneProgress": True,
                },
            },
            timeout=60,
        )

        self._server_capabilities = result.get("capabilities", {})
        self._client.notify("initialized", {})

        logger.info(
            "rust-analyzer initialized for %s",
            self._crate_root.name,
        )

    def _wait_for_ready(self) -> None:
        """Wait for rust-analyzer to finish indexing.

        Monitors $/progress notifications for completion signals.
        Falls back to a polling strategy if progress notifications
        are not received.
        """
        assert self._client is not None

        start = time.monotonic()
        poll_interval = 0.5
        last_status = ""

        while time.monotonic() - start < self._timeout:
            # Check progress notifications
            notifications = self._client.drain_notifications("$/progress")

            for notif in notifications:
                params = notif.get("params", {})
                value = params.get("value", {})
                kind = value.get("kind", "")
                message = value.get("message", "")
                title = value.get("title", "")

                status = f"{title}: {message}" if message else title
                if status and status != last_status:
                    print(f"  rust-analyzer: {status}", file=sys.stderr)
                    last_status = status

                if kind == "end":
                    # An indexing phase completed
                    logger.debug("Progress end: %s", title)

            # Try a lightweight request to check readiness.
            # If rust-analyzer can respond to a workspace/symbol request
            # quickly, it's indexed and ready.
            try:
                self._client.request(
                    "workspace/symbol",
                    {"query": "__hades_readiness_probe__"},
                    timeout=3,
                )
                self._ready = True
                elapsed = time.monotonic() - start
                logger.info("rust-analyzer ready in %.1fs", elapsed)
                print(f"  rust-analyzer: ready ({elapsed:.1f}s)", file=sys.stderr)
                return
            except LspError:
                # Not ready yet — server may still be indexing
                pass

            time.sleep(poll_interval)

        # Timeout — proceed anyway but warn
        elapsed = time.monotonic() - start
        logger.warning(
            "rust-analyzer did not confirm readiness within %ds (proceeding anyway)",
            self._timeout,
        )
        print(
            f"  rust-analyzer: timeout after {elapsed:.1f}s (proceeding anyway)",
            file=sys.stderr,
        )
        self._ready = True  # Proceed optimistically

    def _check_started(self) -> None:
        """Raise if the session has not been started."""
        if self._client is None:
            raise RustAnalyzerError("Session has not been started. Call start() or use as context manager.")

    @staticmethod
    def _find_rust_analyzer() -> str:
        """Locate the rust-analyzer binary.

        Checks: PATH, ~/.cargo/bin/, rustup proxy.

        Returns:
            Path to rust-analyzer binary.

        Raises:
            RustAnalyzerError: If not found.
        """
        # Check PATH
        found = shutil.which("rust-analyzer")
        if found:
            return found

        # Check common locations
        cargo_bin = Path.home() / ".cargo" / "bin" / "rust-analyzer"
        if cargo_bin.exists():
            return str(cargo_bin)

        raise RustAnalyzerError(
            "rust-analyzer not found. Install via: rustup component add rust-analyzer"
        )
