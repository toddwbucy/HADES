"""ArXiv commands for HADES CLI.

This package provides commands for interacting with ArXiv:
- API commands: search_arxiv, get_paper_info
- Abstract search: search_abstracts, search_abstracts_bulk, find_similar, refine_search
- Sync operations: get_sync_status, sync_abstracts

All public functions maintain the same signatures as the original monolithic module.
"""

from .abstract import find_similar, refine_search, search_abstracts, search_abstracts_bulk
from .api import get_paper_info, search_arxiv
from .sync import get_sync_status, sync_abstracts

__all__ = [
    # API commands
    "search_arxiv",
    "get_paper_info",
    # Abstract search commands
    "search_abstracts",
    "search_abstracts_bulk",
    "find_similar",
    "refine_search",
    # Sync commands
    "get_sync_status",
    "sync_abstracts",
]
