"""Tests for ArangoDB memory client configuration and transport selection."""

import os
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from core.database.arango import memory_client as memory_client_module
from core.database.arango.memory_client import (
    ArangoMemoryClient,
    ArangoMemoryClientConfig,
    resolve_memory_config,
)
from core.database.arango.optimized_client import ArangoHttp2Client, ArangoHttp2Config
from core.database.database_factory import DatabaseFactory


class DummyHttp2Client:
    """Minimal stand-in for ArangoHttp2Client used in tests."""

    instances: ClassVar[list["DummyHttp2Client"]] = []

    def __init__(self, config: Any):
        self.config = config
        self.closed = False
        DummyHttp2Client.instances.append(self)

    def close(self) -> None:
        self.closed = True

    def query(self, *_, **__):
        return []

    def request(self, *_, **__):
        return {}

    def insert_documents(self, *_args, **_kwargs):
        return {"created": 0}


@pytest.fixture(autouse=True)
def clear_dummy_instances():
    DummyHttp2Client.instances.clear()
    yield
    DummyHttp2Client.instances.clear()


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all ARANGO-related environment variables."""
    for key in list(os.environ.keys()):
        if key.startswith("ARANGO"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ARANGO_PASSWORD", "test_password")
    yield


# =============================================================================
# ArangoHttp2Config Tests
# =============================================================================


class TestArangoHttp2Config:
    """Tests for ArangoHttp2Config dataclass."""

    def test_socket_path_can_be_none(self):
        """socket_path=None should be valid for network transport."""
        config = ArangoHttp2Config(
            database="test_db",
            socket_path=None,
            base_url="http://localhost:8529",
        )
        assert config.socket_path is None
        assert config.base_url == "http://localhost:8529"

    def test_socket_path_with_value(self):
        """socket_path with a value enables Unix socket transport."""
        config = ArangoHttp2Config(
            database="test_db",
            socket_path="/run/arangodb.sock",
        )
        assert config.socket_path == "/run/arangodb.sock"

    def test_default_base_url_includes_port(self):
        """Default base_url should include port 8529."""
        config = ArangoHttp2Config()
        assert "8529" in config.base_url


# =============================================================================
# ArangoMemoryClientConfig Tests
# =============================================================================


class TestArangoMemoryClientConfig:
    """Tests for ArangoMemoryClientConfig dataclass."""

    def test_sockets_can_be_none(self):
        """read_socket and write_socket can be None for network transport."""
        config = ArangoMemoryClientConfig(
            database="test_db",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket=None,
            write_socket=None,
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )
        assert config.read_socket is None
        assert config.write_socket is None

    def test_build_read_config_with_none_socket(self):
        """build_read_config should pass None socket_path through."""
        config = ArangoMemoryClientConfig(
            database="test_db",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket=None,
            write_socket=None,
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )
        http2_config = config.build_read_config()
        assert http2_config.socket_path is None

    def test_build_write_config_with_socket(self):
        """build_write_config should pass socket_path through."""
        config = ArangoMemoryClientConfig(
            database="test_db",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket="/tmp/ro.sock",
            write_socket="/tmp/rw.sock",
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )
        http2_config = config.build_write_config()
        assert http2_config.socket_path == "/tmp/rw.sock"


# =============================================================================
# resolve_memory_config Tests
# =============================================================================


@pytest.mark.usefixtures("clean_env")
class TestResolveMemoryConfig:
    """Tests for resolve_memory_config function."""

    def test_requires_password(self, monkeypatch):
        """Should raise ValueError if ARANGO_PASSWORD is not set."""
        monkeypatch.delenv("ARANGO_PASSWORD", raising=False)
        with pytest.raises(ValueError, match="password required"):
            resolve_memory_config()

    def test_explicit_proxies_true(self):
        """use_proxies=True should use proxy socket paths."""
        config = resolve_memory_config(use_proxies=True)
        assert config.read_socket is not None
        assert config.write_socket is not None
        assert "readonly" in config.read_socket or "hades" in config.read_socket
        assert "readwrite" in config.write_socket or "hades" in config.write_socket

    def test_explicit_proxies_true_with_env_override(self, monkeypatch):
        """use_proxies=True should respect env socket overrides."""
        monkeypatch.setenv("ARANGO_RO_SOCKET", "/custom/ro.sock")
        monkeypatch.setenv("ARANGO_RW_SOCKET", "/custom/rw.sock")

        config = resolve_memory_config(use_proxies=True)
        assert config.read_socket == "/custom/ro.sock"
        assert config.write_socket == "/custom/rw.sock"

    def test_explicit_proxies_false_with_socket(self, monkeypatch):
        """use_proxies=False with ARANGO_SOCKET should use direct socket."""
        monkeypatch.setenv("ARANGO_SOCKET", "/direct/arango.sock")

        config = resolve_memory_config(use_proxies=False)
        assert config.read_socket == "/direct/arango.sock"
        assert config.write_socket == "/direct/arango.sock"

    def test_explicit_proxies_false_no_socket_uses_network(self):
        """use_proxies=False without sockets should use network (None)."""
        config = resolve_memory_config(use_proxies=False)
        assert config.read_socket is None
        assert config.write_socket is None
        assert "localhost" in config.base_url

    def test_auto_detect_finds_proxy_sockets(self):
        """use_proxies=None should find proxy sockets if they exist."""
        with patch("os.path.exists") as mock_exists:
            # Proxy sockets exist
            mock_exists.side_effect = lambda p: "hades" in p

            config = resolve_memory_config(use_proxies=None)
            assert config.read_socket is not None
            assert "readonly" in config.read_socket or "hades" in config.read_socket

    def test_auto_detect_falls_back_to_direct_socket(self):
        """use_proxies=None should fall back to direct socket if proxies don't exist."""
        with patch("os.path.exists") as mock_exists:
            # Only default ArangoDB socket exists
            mock_exists.side_effect = lambda p: p == "/run/arangodb3/arangodb.sock"

            config = resolve_memory_config(use_proxies=None)
            assert config.read_socket == "/run/arangodb3/arangodb.sock"
            assert config.write_socket == "/run/arangodb3/arangodb.sock"

    def test_auto_detect_falls_back_to_network(self):
        """use_proxies=None should fall back to network if no sockets exist."""
        with patch("os.path.exists") as mock_exists:
            # No sockets exist
            mock_exists.return_value = False

            config = resolve_memory_config(use_proxies=None)
            assert config.read_socket is None
            assert config.write_socket is None
            assert config.base_url is not None

    def test_base_url_from_env(self, monkeypatch):
        """Should use ARANGO_HTTP_BASE_URL from environment."""
        monkeypatch.setenv("ARANGO_HTTP_BASE_URL", "http://arango.example.com:8529")

        with patch("os.path.exists", return_value=False):
            config = resolve_memory_config(use_proxies=False)

        assert config.base_url == "http://arango.example.com:8529"

    def test_timeout_from_env(self, monkeypatch):
        """Should parse timeout values from environment."""
        monkeypatch.setenv("ARANGO_CONNECT_TIMEOUT", "10.0")
        monkeypatch.setenv("ARANGO_READ_TIMEOUT", "60.0")
        monkeypatch.setenv("ARANGO_WRITE_TIMEOUT", "120.0")

        with patch("os.path.exists", return_value=False):
            config = resolve_memory_config(use_proxies=False)

        assert config.connect_timeout == 10.0
        assert config.read_timeout == 60.0
        assert config.write_timeout == 120.0

    def test_explicit_socket_path_overrides(self):
        """Explicit socket_path argument should override everything."""
        with patch("os.path.exists", return_value=False):
            config = resolve_memory_config(
                socket_path="/explicit/socket.sock",
                use_proxies=False,
            )

        assert config.read_socket == "/explicit/socket.sock"
        assert config.write_socket == "/explicit/socket.sock"


# =============================================================================
# ArangoHttp2Client Transport Tests
# =============================================================================


class TestArangoHttp2ClientTransport:
    """Tests for ArangoHttp2Client transport selection."""

    def test_creates_unix_transport_with_socket(self):
        """Should create Unix socket transport when socket_path is provided."""
        config = ArangoHttp2Config(
            database="test",
            socket_path="/tmp/test.sock",
            base_url="http://localhost",
            username="root",
            password="test",
        )

        with patch("httpx.HTTPTransport") as mock_transport:
            with patch("httpx.Client"):
                ArangoHttp2Client(config)

            # Check that HTTPTransport was called with uds parameter
            call_kwargs = mock_transport.call_args[1]
            assert call_kwargs.get("uds") == "/tmp/test.sock"

    def test_creates_network_transport_without_socket(self):
        """Should create network transport when socket_path is None."""
        config = ArangoHttp2Config(
            database="test",
            socket_path=None,
            base_url="http://localhost:8529",
            username="root",
            password="test",
        )

        with patch("httpx.HTTPTransport") as mock_transport:
            with patch("httpx.Client"):
                ArangoHttp2Client(config)

            # Check that HTTPTransport was called without uds parameter
            call_kwargs = mock_transport.call_args[1]
            assert "uds" not in call_kwargs or call_kwargs.get("uds") is None


# =============================================================================
# Integration Tests with DatabaseFactory
# =============================================================================


@pytest.mark.usefixtures("clean_env")
class TestDatabaseFactoryIntegration:
    """Integration tests for DatabaseFactory with new transport options."""

    def test_get_arango_with_network_transport(self, monkeypatch):
        """DatabaseFactory.get_arango should work with network transport."""
        monkeypatch.setattr(memory_client_module, "ArangoHttp2Client", DummyHttp2Client)

        with patch("os.path.exists", return_value=False):
            client = DatabaseFactory.get_arango(
                database="test_db",
                username="root",
                use_proxies=False,
            )

        try:
            assert isinstance(client, ArangoMemoryClient)
            # Should have created client(s) with None socket_path
            assert len(DummyHttp2Client.instances) >= 1
            cfg = DummyHttp2Client.instances[0].config
            assert cfg.socket_path is None
        finally:
            client.close()

    def test_get_arango_with_unix_socket(self, monkeypatch):
        """DatabaseFactory.get_arango should work with Unix socket transport."""
        monkeypatch.setattr(memory_client_module, "ArangoHttp2Client", DummyHttp2Client)

        client = DatabaseFactory.get_arango(
            database="test_db",
            username="root",
            socket_path="/tmp/test.sock",
            use_proxies=False,
        )

        try:
            assert isinstance(client, ArangoMemoryClient)
            cfg = DummyHttp2Client.instances[0].config
            assert cfg.socket_path == "/tmp/test.sock"
        finally:
            client.close()

    def test_get_arango_auto_detect(self, monkeypatch):
        """DatabaseFactory.get_arango with use_proxies=None should auto-detect."""
        monkeypatch.setattr(memory_client_module, "ArangoHttp2Client", DummyHttp2Client)

        # Simulate no sockets available
        with patch("os.path.exists", return_value=False):
            client = DatabaseFactory.get_arango(
                database="test_db",
                username="root",
                use_proxies=None,
            )

        try:
            assert isinstance(client, ArangoMemoryClient)
            # Should fall back to network (None socket)
            cfg = DummyHttp2Client.instances[0].config
            assert cfg.socket_path is None
        finally:
            client.close()


# =============================================================================
# ArangoMemoryClient Tests
# =============================================================================


class TestArangoMemoryClient:
    """Tests for ArangoMemoryClient initialization and operation."""

    def test_shared_clients_when_sockets_equal(self, monkeypatch):
        """Should use shared client when read and write sockets are the same."""
        monkeypatch.setattr(memory_client_module, "ArangoHttp2Client", DummyHttp2Client)

        config = ArangoMemoryClientConfig(
            database="test",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket="/same/socket.sock",
            write_socket="/same/socket.sock",
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )

        client = ArangoMemoryClient(config)
        try:
            # Should create only one HTTP client
            assert len(DummyHttp2Client.instances) == 1
        finally:
            client.close()

    def test_separate_clients_when_sockets_differ(self, monkeypatch):
        """Should use separate clients when read and write sockets differ."""
        monkeypatch.setattr(memory_client_module, "ArangoHttp2Client", DummyHttp2Client)

        config = ArangoMemoryClientConfig(
            database="test",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket="/ro/socket.sock",
            write_socket="/rw/socket.sock",
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )

        client = ArangoMemoryClient(config)
        try:
            # Should create two HTTP clients
            assert len(DummyHttp2Client.instances) == 2
        finally:
            client.close()

    def test_shared_clients_when_both_none(self, monkeypatch):
        """Should use shared client when both sockets are None (network)."""
        monkeypatch.setattr(memory_client_module, "ArangoHttp2Client", DummyHttp2Client)

        config = ArangoMemoryClientConfig(
            database="test",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket=None,
            write_socket=None,
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )

        client = ArangoMemoryClient(config)
        try:
            # Should create only one HTTP client (both None = same)
            assert len(DummyHttp2Client.instances) == 1
        finally:
            client.close()

    def test_close_closes_all_clients(self, monkeypatch):
        """close() should close all underlying HTTP clients."""
        monkeypatch.setattr(memory_client_module, "ArangoHttp2Client", DummyHttp2Client)

        config = ArangoMemoryClientConfig(
            database="test",
            username="root",
            password="secret",
            base_url="http://localhost:8529",
            read_socket="/ro/socket.sock",
            write_socket="/rw/socket.sock",
            connect_timeout=5.0,
            read_timeout=30.0,
            write_timeout=30.0,
        )

        client = ArangoMemoryClient(config)
        assert not all(inst.closed for inst in DummyHttp2Client.instances)

        client.close()
        assert all(inst.closed for inst in DummyHttp2Client.instances)
