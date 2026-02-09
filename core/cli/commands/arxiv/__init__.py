"""ArXiv sync commands for HADES CLI.

Sync operations that write to the local ArangoDB abstract database.
API and search commands have been migrated to arxiv-manager.
"""

from .sync import get_sync_status, sync_abstracts

__all__ = [
    "get_sync_status",
    "sync_abstracts",
]
