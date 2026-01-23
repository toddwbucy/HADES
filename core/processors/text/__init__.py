"""
Text Processing Module

Provides text processing capabilities including chunking strategies
for documents. These processors work on the text content after extraction.
"""

from .chunking_strategies import (
    ChunkingStrategy,
    TokenBasedChunking,
    SemanticChunking,
    SlidingWindowChunking,
    ChunkingStrategyFactory
)

__all__ = [
    "ChunkingStrategy",
    "TokenBasedChunking",
    "SemanticChunking",
    "SlidingWindowChunking",
    "ChunkingStrategyFactory"
]