"""HADES CLI - AI Model Interface for HADES Knowledge Base.

This CLI is designed for AI models (Claude Code, etc.) to interact with the
HADES knowledge base. All commands output JSON for predictable parsing.

Example workflow:
    Claude: runs `hades arxiv search "late chunking transformers"`
    Claude: runs `hades ingest 2401.xxxxx`
    Claude: runs `hades query "attention head pruning"`
"""

from core.cli.main import app

__all__ = ["app"]
