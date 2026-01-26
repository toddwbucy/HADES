"""Tests for sync CLI commands including incremental sync."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.cli.commands.sync import (
    SYNC_METADATA_COLLECTION,
    SYNC_WATERMARK_KEY,
    _get_last_sync_date,
    _get_sync_metadata,
    get_sync_status,
    sync_abstracts,
)
from core.cli.output import ErrorCode


class TestSyncStatus:
    """Tests for sync status command."""

    @patch("core.cli.commands.sync._get_sync_metadata")
    @patch("core.cli.commands.sync.get_config")
    def test_status_returns_metadata(self, mock_get_config, mock_get_metadata):
        """Test status returns sync metadata."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_get_metadata.return_value = {
            "last_sync": "2025-01-20T12:00:00+00:00",
            "total_synced": 2826761,
            "sync_history": [
                {"date": "2025-01-20", "added": 1234, "updated": 56},
            ],
        }

        response = get_sync_status(start_time=0)

        assert response.success is True
        assert response.command == "sync.status"
        assert response.data["last_sync"] == "2025-01-20T12:00:00+00:00"
        assert response.data["total_synced"] == 2826761
        assert len(response.data["sync_history"]) == 1

    @patch("core.cli.commands.sync._get_sync_metadata")
    @patch("core.cli.commands.sync.get_config")
    def test_status_no_history(self, mock_get_config, mock_get_metadata):
        """Test status when no sync history exists."""
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        mock_get_metadata.return_value = None

        response = get_sync_status(start_time=0)

        assert response.success is True
        assert response.data["last_sync"] is None
        assert response.data["total_synced"] == 0
        assert "message" in response.data

    @patch("core.cli.commands.sync.get_config")
    def test_status_handles_config_error(self, mock_get_config):
        """Test status handles config errors."""
        mock_get_config.side_effect = ValueError("Missing ARANGO_PASSWORD")

        response = get_sync_status(start_time=0)

        assert response.success is False
        assert response.error["code"] == ErrorCode.CONFIG_ERROR.value


class TestIncrementalSync:
    """Tests for incremental sync functionality."""

    @patch("core.cli.commands.sync._embed_and_store_abstracts")
    @patch("core.cli.commands.sync._filter_existing")
    @patch("core.cli.commands.sync._fetch_recent_papers")
    @patch("core.cli.commands.sync._update_sync_metadata")
    @patch("core.cli.commands.sync._get_last_sync_date")
    @patch("core.cli.commands.sync.get_config")
    def test_incremental_uses_watermark(
        self,
        mock_get_config,
        mock_get_last_sync,
        _mock_update_metadata,
        mock_fetch,
        mock_filter,
        mock_embed,
    ):
        """Test incremental sync uses last sync timestamp."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        # Last sync was 3 days ago
        last_sync = datetime.now() - timedelta(days=3)
        mock_get_last_sync.return_value = last_sync

        mock_fetch.return_value = [{"arxiv_id": "2501.12345", "title": "Test", "abstract": "..."}]
        mock_filter.return_value = [{"arxiv_id": "2501.12345", "title": "Test", "abstract": "..."}]
        mock_embed.return_value = 1

        response = sync_abstracts(
            from_date=None,
            categories=None,
            max_results=100,
            batch_size=8,
            start_time=0,
            incremental=True,
        )

        assert response.success is True
        assert response.data["mode"] == "incremental"

        # Verify fetch was called with watermark date
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args[0]
        # The start_date should be close to last_sync
        assert abs((call_args[0] - last_sync).total_seconds()) < 60

    @patch("core.cli.commands.sync._embed_and_store_abstracts")
    @patch("core.cli.commands.sync._filter_existing")
    @patch("core.cli.commands.sync._fetch_recent_papers")
    @patch("core.cli.commands.sync._update_sync_metadata")
    @patch("core.cli.commands.sync._get_last_sync_date")
    @patch("core.cli.commands.sync.get_config")
    def test_incremental_no_previous_sync(
        self,
        mock_get_config,
        mock_get_last_sync,
        _mock_update_metadata,
        mock_fetch,
        _mock_filter,
        _mock_embed,
    ):
        """Test incremental sync with no previous sync falls back to 7 days."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        # No previous sync
        mock_get_last_sync.return_value = None

        mock_fetch.return_value = []
        # Filter and embed mocks not needed since fetch returns empty list

        response = sync_abstracts(
            from_date=None,
            categories=None,
            max_results=100,
            batch_size=8,
            start_time=0,
            incremental=True,
        )

        assert response.success is True

        # Verify fetch was called with a date ~7 days ago
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args[0]
        expected_date = datetime.now() - timedelta(days=7)
        # Allow 1 hour tolerance
        assert abs((call_args[0] - expected_date).total_seconds()) < 3600

    @patch("core.cli.commands.sync._embed_and_store_abstracts")
    @patch("core.cli.commands.sync._filter_existing")
    @patch("core.cli.commands.sync._fetch_recent_papers")
    @patch("core.cli.commands.sync._update_sync_metadata")
    @patch("core.cli.commands.sync.get_config")
    def test_sync_updates_metadata(
        self,
        mock_get_config,
        mock_update_metadata,
        mock_fetch,
        mock_filter,
        mock_embed,
    ):
        """Test sync updates metadata on success."""
        mock_config = MagicMock()
        mock_config.device = "cpu"
        mock_get_config.return_value = mock_config

        mock_fetch.return_value = [{"arxiv_id": "2501.12345", "title": "Test", "abstract": "..."}]
        mock_filter.return_value = [{"arxiv_id": "2501.12345", "title": "Test", "abstract": "..."}]
        mock_embed.return_value = 1

        response = sync_abstracts(
            from_date="2025-01-01",
            categories=None,
            max_results=100,
            batch_size=8,
            start_time=0,
            incremental=False,
        )

        assert response.success is True
        assert response.data["mode"] == "manual"

        # Verify metadata was updated
        mock_update_metadata.assert_called_once()
        call_kwargs = mock_update_metadata.call_args[1]
        assert call_kwargs["added"] == 1


class TestGetLastSyncDate:
    """Tests for _get_last_sync_date helper."""

    @patch("core.cli.commands.sync._get_sync_metadata")
    def test_returns_datetime_from_metadata(self, mock_get_metadata):
        """Test parsing ISO datetime from metadata."""
        mock_config = MagicMock()
        mock_get_metadata.return_value = {
            "last_sync": "2025-01-20T12:00:00+00:00",
        }

        result = _get_last_sync_date(mock_config)

        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 20

    @patch("core.cli.commands.sync._get_sync_metadata")
    def test_returns_none_when_no_metadata(self, mock_get_metadata):
        """Test returns None when no metadata exists."""
        mock_config = MagicMock()
        mock_get_metadata.return_value = None

        result = _get_last_sync_date(mock_config)

        assert result is None

    @patch("core.cli.commands.sync._get_sync_metadata")
    def test_returns_none_when_no_last_sync(self, mock_get_metadata):
        """Test returns None when last_sync field is missing."""
        mock_config = MagicMock()
        mock_get_metadata.return_value = {"total_synced": 100}

        result = _get_last_sync_date(mock_config)

        assert result is None


class TestSyncMetadataHelpers:
    """Tests for sync metadata helper functions."""

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.sync.get_arango_config")
    def test_get_sync_metadata_returns_doc(self, mock_get_config, mock_client_class):
        """Test _get_sync_metadata returns document when it exists."""
        mock_config = MagicMock()
        mock_get_config.return_value = {
            "database": "test_db",
            "host": "localhost",
            "port": 8529,
            "username": "root",
            "password": "test",
        }

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.get_document.return_value = {
            "_key": SYNC_WATERMARK_KEY,
            "last_sync": "2025-01-20T12:00:00+00:00",
            "total_synced": 100,
        }

        result = _get_sync_metadata(mock_config)

        assert result is not None
        assert result["total_synced"] == 100
        mock_client.get_document.assert_called_once_with(
            SYNC_METADATA_COLLECTION, SYNC_WATERMARK_KEY
        )
        mock_client.close.assert_called_once()

    @patch("core.database.arango.optimized_client.ArangoHttp2Client")
    @patch("core.cli.commands.sync.get_arango_config")
    def test_get_sync_metadata_returns_none_on_404(
        self, mock_get_config, mock_client_class
    ):
        """Test _get_sync_metadata returns None when document doesn't exist."""
        from core.database.arango.optimized_client import ArangoHttpError

        mock_config = MagicMock()
        mock_get_config.return_value = {
            "database": "test_db",
            "host": "localhost",
            "port": 8529,
            "username": "root",
            "password": "test",
        }

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Simulate 404 error
        error = ArangoHttpError(404, "Document not found")
        mock_client.get_document.side_effect = error

        result = _get_sync_metadata(mock_config)

        assert result is None
        mock_client.close.assert_called_once()
