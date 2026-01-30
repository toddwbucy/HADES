"""Tests for HADES CLI configuration loading."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pytest

from core.cli.config import (
    CLIConfig,
    EmbeddingConfig,
    RocchioConfig,
    SearchConfig,
    SyncConfig,
    _get_nested,
    _load_yaml_config,
    get_config,
    get_embedder_service_config,
)


class TestGetNested:
    """Tests for _get_nested helper function."""

    def test_gets_simple_value(self):
        """Test getting a simple top-level value."""
        config = {"key": "value"}
        assert _get_nested(config, "key") == "value"

    def test_gets_nested_value(self):
        """Test getting a nested value."""
        config = {"level1": {"level2": {"level3": "deep_value"}}}
        assert _get_nested(config, "level1", "level2", "level3") == "deep_value"

    def test_returns_default_for_missing_key(self):
        """Test returns default when key is missing."""
        config = {"key": "value"}
        assert _get_nested(config, "missing", default="default") == "default"

    def test_returns_default_for_missing_nested_key(self):
        """Test returns default when nested key is missing."""
        config = {"level1": {"level2": "value"}}
        assert _get_nested(config, "level1", "missing", default="default") == "default"

    def test_returns_none_as_default(self):
        """Test returns None as default when not specified."""
        config = {}
        assert _get_nested(config, "missing") is None

    def test_handles_non_dict_intermediate(self):
        """Test handles non-dict intermediate value gracefully."""
        config = {"level1": "not_a_dict"}
        assert _get_nested(config, "level1", "level2", default="default") == "default"


class TestLoadYamlConfig:
    """Tests for YAML config loading."""

    def test_loads_existing_yaml(self):
        """Test loading an existing YAML file."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("database:\n  host: testhost\n  port: 9999\n")
            f.flush()
            try:
                config = _load_yaml_config(Path(f.name))
                assert config["database"]["host"] == "testhost"
                assert config["database"]["port"] == 9999
            finally:
                os.unlink(f.name)

    def test_returns_empty_dict_for_missing_file(self):
        """Test returns empty dict when file doesn't exist."""
        config = _load_yaml_config(Path("/nonexistent/path/config.yaml"))
        assert config == {}

    def test_returns_empty_dict_for_empty_file(self):
        """Test returns empty dict for empty YAML file."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            try:
                config = _load_yaml_config(Path(f.name))
                assert config == {}
            finally:
                os.unlink(f.name)

    def test_raises_error_for_list_yaml(self):
        """Test raises TypeError when YAML contains a list instead of dict."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- item1\n- item2\n- item3\n")
            f.flush()
            try:
                with pytest.raises(TypeError, match="must contain a YAML mapping"):
                    _load_yaml_config(Path(f.name))
            finally:
                os.unlink(f.name)

    def test_raises_error_for_scalar_yaml(self):
        """Test raises TypeError when YAML contains a scalar instead of dict."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("just a string value\n")
            f.flush()
            try:
                with pytest.raises(TypeError, match="got str"):
                    _load_yaml_config(Path(f.name))
            finally:
                os.unlink(f.name)


class TestGetConfig:
    """Tests for get_config function."""

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass"}, clear=True)
    def test_requires_arango_password(self):
        """Test that ARANGO_PASSWORD is required."""
        # Create a minimal config to avoid hitting the actual file
        with patch("core.cli.config._load_yaml_config", return_value={}):
            config = get_config()
            assert config.arango_password == "testpass"

    @patch.dict(os.environ, {}, clear=True)
    def test_raises_without_password(self):
        """Test raises ValueError without ARANGO_PASSWORD."""
        with patch("core.cli.config._load_yaml_config", return_value={}):
            with pytest.raises(ValueError, match="ARANGO_PASSWORD"):
                get_config()

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass"}, clear=True)
    def test_loads_database_config_from_yaml(self):
        """Test database config is loaded from YAML."""
        yaml_config = {
            "database": {
                "host": "yaml-host",
                "port": 9999,
                "database": "yaml-db",
            }
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_config()
            assert config.arango_host == "yaml-host"
            assert config.arango_port == 9999
            assert config.arango_database == "yaml-db"

    @patch.dict(
        os.environ,
        {
            "ARANGO_PASSWORD": "testpass",
            "ARANGO_HOST": "env-host",
            "ARANGO_PORT": "8888",
            "HADES_DATABASE": "env-db",
        },
        clear=True,
    )
    def test_env_overrides_yaml(self):
        """Test environment variables override YAML values."""
        yaml_config = {
            "database": {
                "host": "yaml-host",
                "port": 9999,
                "database": "yaml-db",
            }
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_config()
            assert config.arango_host == "env-host"
            assert config.arango_port == 8888
            assert config.arango_database == "env-db"

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass"}, clear=True)
    def test_loads_embedding_config(self):
        """Test embedding config is loaded from YAML."""
        yaml_config = {
            "embedding": {
                "service": {
                    "socket": "/custom/socket.sock",
                    "fallback_to_local": False,
                    "timeout_ms": 60000,
                },
                "model": {
                    "name": "custom-model",
                    "dimension": 1024,
                },
                "batch": {
                    "size": 64,
                    "size_small": 16,
                },
            }
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_config()
            assert config.embedding.service_socket == "/custom/socket.sock"
            assert config.embedding.fallback_to_local is False
            assert config.embedding.timeout_ms == 60000
            assert config.embedding.model_name == "custom-model"
            assert config.embedding.dimension == 1024
            assert config.embedding.batch_size == 64
            assert config.embedding.batch_size_small == 16

    @patch.dict(
        os.environ, {"ARANGO_PASSWORD": "testpass", "HADES_EMBEDDER_SOCKET": "/env/socket.sock"}, clear=True
    )
    def test_env_overrides_embedding_socket(self):
        """Test HADES_EMBEDDER_SOCKET overrides YAML value."""
        yaml_config = {
            "embedding": {
                "service": {
                    "socket": "/yaml/socket.sock",
                }
            }
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_config()
            assert config.embedding.service_socket == "/env/socket.sock"

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass"}, clear=True)
    def test_loads_search_config(self):
        """Test search config is loaded from YAML."""
        yaml_config = {
            "search": {
                "limit": 25,
                "max_limit": 200,
                "hybrid": {
                    "vector_weight": 0.8,
                    "keyword_weight": 0.2,
                },
            }
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_config()
            assert config.search.limit == 25
            assert config.search.max_limit == 200
            assert config.search.hybrid_vector_weight == 0.8
            assert config.search.hybrid_keyword_weight == 0.2

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass"}, clear=True)
    def test_loads_rocchio_config(self):
        """Test rocchio config is loaded from YAML."""
        yaml_config = {
            "rocchio": {
                "alpha": 0.5,
                "beta": 0.8,
                "gamma": 0.2,
            }
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_config()
            assert config.rocchio.alpha == 0.5
            assert config.rocchio.beta == 0.8
            assert config.rocchio.gamma == 0.2

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass"}, clear=True)
    def test_loads_sync_config(self):
        """Test sync config is loaded from YAML."""
        yaml_config = {
            "sync": {
                "default_lookback_days": 14,
                "batch_size": 16,
                "max_results": 2000,
            }
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_config()
            assert config.sync.default_lookback_days == 14
            assert config.sync.batch_size == 16
            assert config.sync.max_results == 2000

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass"}, clear=True)
    def test_handles_non_dict_rocchio_and_sync(self):
        """Test gracefully handles non-dict values for rocchio and sync sections."""
        yaml_config = {
            "rocchio": "not a dict",  # Invalid: should be a mapping
            "sync": ["also", "not", "a", "dict"],  # Invalid: should be a mapping
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_config()
            # Should fall back to defaults when sections are non-dict
            assert config.rocchio.alpha == 1.0
            assert config.rocchio.beta == 0.75
            assert config.rocchio.gamma == 0.15
            assert config.sync.default_lookback_days == 7
            assert config.sync.batch_size == 8
            assert config.sync.max_results == 1000

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass"}, clear=True)
    def test_uses_defaults_when_yaml_missing(self):
        """Test uses dataclass defaults when YAML values missing."""
        with patch("core.cli.config._load_yaml_config", return_value={}):
            config = get_config()
            # Check defaults are used
            assert config.arango_host == "localhost"
            assert config.arango_port == 8529
            assert config.embedding.dimension == 2048
            assert config.search.limit == 10
            assert config.rocchio.alpha == 1.0
            assert config.sync.default_lookback_days == 7

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass", "HADES_USE_GPU": "false"}, clear=True)
    def test_gpu_disabled_via_env(self):
        """Test GPU can be disabled via environment variable."""
        with patch("core.cli.config._load_yaml_config", return_value={}):
            config = get_config()
            assert config.use_gpu is False
            assert config.device == "cpu"

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass"}, clear=True)
    def test_gpu_disabled_via_yaml(self):
        """Test GPU can be disabled via YAML."""
        yaml_config = {"gpu": {"enabled": False}}
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_config()
            assert config.use_gpu is False
            assert config.device == "cpu"

    @patch.dict(os.environ, {"ARANGO_PASSWORD": "testpass", "ARANGO_PORT": "invalid"}, clear=True)
    def test_invalid_port_raises_error(self):
        """Test invalid port value raises ValueError."""
        with patch("core.cli.config._load_yaml_config", return_value={}):
            with pytest.raises(ValueError, match="ARANGO_PORT must be a number"):
                get_config()


class TestConfigDataclasses:
    """Tests for config dataclass defaults."""

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig has sensible defaults."""
        config = EmbeddingConfig()
        assert config.service_socket == "/run/hades/embedder.sock"
        assert config.fallback_to_local is True
        assert config.model_name == "jinaai/jina-embeddings-v4"
        assert config.dimension == 2048
        assert config.batch_size == 48

    def test_search_config_defaults(self):
        """Test SearchConfig has sensible defaults."""
        config = SearchConfig()
        assert config.limit == 10
        assert config.max_limit == 100
        assert config.hybrid_vector_weight == 0.7

    def test_rocchio_config_defaults(self):
        """Test RocchioConfig has sensible defaults."""
        config = RocchioConfig()
        assert config.alpha == 1.0
        assert config.beta == 0.75
        assert config.gamma == 0.15

    def test_sync_config_defaults(self):
        """Test SyncConfig has sensible defaults."""
        config = SyncConfig()
        assert config.default_lookback_days == 7
        assert config.batch_size == 8
        assert config.max_results == 1000


class TestGetEmbedderServiceConfig:
    """Tests for get_embedder_service_config function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_device_is_cuda2(self):
        """Test default device is cuda:2 (inference GPU)."""
        with patch("core.cli.config._load_yaml_config", return_value={}):
            config = get_embedder_service_config()
            assert config["device"] == "cuda:2"

    @patch.dict(os.environ, {}, clear=True)
    def test_reads_device_from_yaml_gpu_section(self):
        """Test device can be set via gpu.device in YAML."""
        yaml_config = {"gpu": {"device": "cuda:1", "enabled": True}}
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_embedder_service_config()
            assert config["device"] == "cuda:1"

    @patch.dict(os.environ, {}, clear=True)
    def test_reads_device_from_yaml_service_section(self):
        """Test service-specific device overrides global GPU device."""
        yaml_config = {
            "gpu": {"device": "cuda:1", "enabled": True},
            "embedding": {"service": {"device": "cuda:0"}},
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_embedder_service_config()
            assert config["device"] == "cuda:0"

    @patch.dict(os.environ, {"HADES_EMBEDDER_DEVICE": "cuda:3"}, clear=True)
    def test_env_overrides_yaml(self):
        """Test environment variable overrides YAML config."""
        yaml_config = {
            "gpu": {"device": "cuda:1", "enabled": True},
            "embedding": {"service": {"device": "cuda:0"}},
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_embedder_service_config()
            assert config["device"] == "cuda:3"

    @patch.dict(os.environ, {"HADES_USE_GPU": "false"}, clear=True)
    def test_gpu_disabled_via_env(self):
        """Test GPU can be disabled via env var."""
        yaml_config = {"gpu": {"device": "cuda:2", "enabled": True}}
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_embedder_service_config()
            assert config["device"] == "cpu"

    @patch.dict(os.environ, {}, clear=True)
    def test_gpu_disabled_via_yaml(self):
        """Test GPU can be disabled via YAML."""
        yaml_config = {"gpu": {"device": "cuda:2", "enabled": False}}
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_embedder_service_config()
            assert config["device"] == "cpu"

    @patch.dict(os.environ, {}, clear=True)
    def test_loads_all_settings(self):
        """Test all settings are loaded from YAML."""
        yaml_config = {
            "gpu": {"device": "cuda:2", "enabled": True},
            "embedding": {
                "model": {"name": "custom-model", "use_fp16": False},
                "batch": {"size": 32},
                "service": {"idle_timeout": 600},
            },
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_embedder_service_config()
            assert config["device"] == "cuda:2"
            assert config["model_name"] == "custom-model"
            assert config["use_fp16"] is False
            assert config["batch_size"] == 32
            assert config["idle_timeout"] == 600

    @patch.dict(
        os.environ,
        {
            "HADES_EMBEDDER_DEVICE": "cuda:0",
            "HADES_EMBEDDER_MODEL": "env-model",
            "HADES_EMBEDDER_FP16": "false",
            "HADES_EMBEDDER_BATCH_SIZE": "16",
            "HADES_EMBEDDER_IDLE_TIMEOUT": "120",
        },
        clear=True,
    )
    def test_env_overrides_all_settings(self):
        """Test all settings can be overridden via env vars."""
        yaml_config = {
            "gpu": {"device": "cuda:2", "enabled": True},
            "embedding": {
                "model": {"name": "yaml-model", "use_fp16": True},
                "batch": {"size": 48},
                "service": {"idle_timeout": 900},
            },
        }
        with patch("core.cli.config._load_yaml_config", return_value=yaml_config):
            config = get_embedder_service_config()
            assert config["device"] == "cuda:0"
            assert config["model_name"] == "env-model"
            assert config["use_fp16"] is False
            assert config["batch_size"] == 16
            assert config["idle_timeout"] == 120

    @patch.dict(os.environ, {}, clear=True)
    def test_uses_defaults_when_yaml_missing(self):
        """Test defaults are used when YAML is empty."""
        with patch("core.cli.config._load_yaml_config", return_value={}):
            config = get_embedder_service_config()
            assert config["device"] == "cuda:2"
            assert config["model_name"] == "jinaai/jina-embeddings-v4"
            assert config["use_fp16"] is True
            assert config["batch_size"] == 48
            assert config["idle_timeout"] == 300

    @patch.dict(os.environ, {"HADES_EMBEDDER_BATCH_SIZE": "invalid"}, clear=True)
    def test_invalid_batch_size_raises_error(self):
        """Test invalid batch size raises ValueError."""
        with patch("core.cli.config._load_yaml_config", return_value={}):
            with pytest.raises(ValueError, match="HADES_EMBEDDER_BATCH_SIZE must be an integer"):
                get_embedder_service_config()

    @patch.dict(os.environ, {"HADES_EMBEDDER_IDLE_TIMEOUT": "not_a_number"}, clear=True)
    def test_invalid_idle_timeout_raises_error(self):
        """Test invalid idle timeout raises ValueError."""
        with patch("core.cli.config._load_yaml_config", return_value={}):
            with pytest.raises(ValueError, match="HADES_EMBEDDER_IDLE_TIMEOUT must be an integer"):
                get_embedder_service_config()
