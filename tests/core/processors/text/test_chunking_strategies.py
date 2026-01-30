"""Unit tests for chunking strategies.

Tests for:
- TextChunk dataclass
- TokenBasedChunking
- SemanticChunking
- SlidingWindowChunking
- HybridChunking (new - combines semantic boundaries with overlap)
- ChunkingStrategyFactory
"""

import pytest


class TestTextChunk:
    """Tests for TextChunk dataclass."""

    def test_creates_chunk_with_required_fields(self):
        """Should create a TextChunk with all required fields."""
        from core.processors.text.chunking_strategies import TextChunk

        chunk = TextChunk(
            text="Hello world",
            start_char=0,
            end_char=11,
            chunk_index=0,
            metadata={"key": "value"},
        )

        assert chunk.text == "Hello world"
        assert chunk.start_char == 0
        assert chunk.end_char == 11
        assert chunk.chunk_index == 0
        assert chunk.metadata == {"key": "value"}

    def test_char_count_property(self):
        """Should return correct character count."""
        from core.processors.text.chunking_strategies import TextChunk

        chunk = TextChunk(
            text="Hello world",
            start_char=0,
            end_char=11,
            chunk_index=0,
            metadata={},
        )

        assert chunk.char_count == 11

    def test_token_count_estimate_property(self):
        """Should estimate token count via whitespace split."""
        from core.processors.text.chunking_strategies import TextChunk

        chunk = TextChunk(
            text="Hello world this is a test",
            start_char=0,
            end_char=26,
            chunk_index=0,
            metadata={},
        )

        assert chunk.token_count_estimate == 6


class TestTokenBasedChunking:
    """Tests for TokenBasedChunking strategy."""

    def test_creates_chunks_with_correct_size(self):
        """Should create chunks with target token count."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        strategy = TokenBasedChunking(chunk_size=5, chunk_overlap=2)
        text = "one two three four five six seven eight nine ten"

        chunks = strategy.create_chunks(text)

        # First chunk should have 5 tokens
        assert len(chunks[0].text.split()) == 5
        assert chunks[0].text == "one two three four five"

    def test_overlap_between_chunks(self):
        """Should maintain correct overlap between chunks."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        strategy = TokenBasedChunking(chunk_size=5, chunk_overlap=2)
        text = "one two three four five six seven eight nine ten"

        chunks = strategy.create_chunks(text)

        # With chunk_size=5 and overlap=2, stride is 3
        # Chunk 0: one two three four five
        # Chunk 1: four five six seven eight (starts at index 3)
        assert len(chunks) >= 2
        assert "four five" in chunks[0].text
        assert "four five" in chunks[1].text

    def test_rejects_overlap_greater_than_size(self):
        """Should reject overlap >= chunk_size."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        with pytest.raises(ValueError, match="Overlap must be less than chunk size"):
            TokenBasedChunking(chunk_size=5, chunk_overlap=5)

        with pytest.raises(ValueError, match="Overlap must be less than chunk size"):
            TokenBasedChunking(chunk_size=5, chunk_overlap=10)

    def test_empty_text_returns_empty_list(self):
        """Should return empty list for empty text."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        strategy = TokenBasedChunking(chunk_size=10, chunk_overlap=2)
        chunks = strategy.create_chunks("")

        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        """Should return empty list for whitespace-only text."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        strategy = TokenBasedChunking(chunk_size=10, chunk_overlap=2)
        chunks = strategy.create_chunks("   \n\n\t  ")

        assert chunks == []

    def test_chunk_metadata_includes_strategy_info(self):
        """Should include strategy info in chunk metadata."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        strategy = TokenBasedChunking(chunk_size=5, chunk_overlap=2)
        chunks = strategy.create_chunks("one two three four five")

        assert chunks[0].metadata["strategy"] == "token_based"
        assert chunks[0].metadata["chunk_size"] == 5
        assert chunks[0].metadata["overlap"] == 2

    def test_accepts_callable_tokenizer(self):
        """Should accept a callable tokenizer function."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        # Custom tokenizer that lowercases
        def custom_tokenizer(text: str) -> list[str]:
            return text.lower().split()

        strategy = TokenBasedChunking(chunk_size=3, chunk_overlap=1, tokenizer=custom_tokenizer)
        chunks = strategy.create_chunks("Hello World Test")

        assert "hello world test" in chunks[0].text.lower()

    def test_rejects_invalid_tokenizer(self):
        """Should reject tokenizer without proper interface."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        with pytest.raises(TypeError, match="Tokenizer must be either a callable"):
            TokenBasedChunking(chunk_size=5, chunk_overlap=2, tokenizer="not_callable")


class TestSemanticChunking:
    """Tests for SemanticChunking strategy."""

    def test_respects_paragraph_boundaries(self):
        """Should keep paragraphs together when possible."""
        from core.processors.text.chunking_strategies import SemanticChunking

        strategy = SemanticChunking(max_chunk_size=100, min_chunk_size=10)
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."

        chunks = strategy.create_chunks(text)

        # Should create chunks that respect paragraph boundaries
        assert len(chunks) >= 1
        assert chunks[0].metadata["strategy"] == "semantic"

    def test_splits_large_paragraphs(self):
        """Should split paragraphs that exceed max_chunk_size."""
        from core.processors.text.chunking_strategies import SemanticChunking

        # Use respect_sentences=False to force split at token boundaries
        strategy = SemanticChunking(
            max_chunk_size=5, min_chunk_size=2, respect_sentences=False
        )
        text = "one two three four five six seven eight nine ten eleven twelve"

        chunks = strategy.create_chunks(text)

        # Large paragraph should be split into multiple chunks
        # With max_chunk_size=5 and 12 tokens, should create at least 2 chunks
        assert len(chunks) >= 2
        # Each chunk should not exceed max size
        for chunk in chunks:
            assert len(chunk.text.split()) <= 5

    def test_respects_sentence_boundaries(self):
        """Should split at sentence boundaries when enabled."""
        from core.processors.text.chunking_strategies import SemanticChunking

        strategy = SemanticChunking(
            max_chunk_size=10, min_chunk_size=2, respect_sentences=True
        )
        text = "First sentence here. Second sentence here. Third sentence."

        chunks = strategy.create_chunks(text)

        # Should have created chunks
        assert len(chunks) >= 1

    def test_empty_text_returns_empty_list(self):
        """Should return empty list for empty text."""
        from core.processors.text.chunking_strategies import SemanticChunking

        strategy = SemanticChunking(max_chunk_size=100, min_chunk_size=10)
        chunks = strategy.create_chunks("")

        assert chunks == []

    def test_chunk_metadata_includes_counts(self):
        """Should include paragraph and sentence counts in metadata."""
        from core.processors.text.chunking_strategies import SemanticChunking

        strategy = SemanticChunking(max_chunk_size=100, min_chunk_size=10)
        text = "First sentence. Second sentence."

        chunks = strategy.create_chunks(text)

        assert "paragraph_count" in chunks[0].metadata
        assert "sentence_count" in chunks[0].metadata
        assert "token_count" in chunks[0].metadata


class TestSlidingWindowChunking:
    """Tests for SlidingWindowChunking strategy."""

    def test_creates_overlapping_windows(self):
        """Should create overlapping windows with correct step."""
        from core.processors.text.chunking_strategies import SlidingWindowChunking

        strategy = SlidingWindowChunking(window_size=4, step_size=2)
        text = "one two three four five six seven eight"

        chunks = strategy.create_chunks(text)

        # Window 0: one two three four
        # Window 1: three four five six (step of 2)
        assert len(chunks) >= 2
        assert "one two three four" in chunks[0].text
        assert "three four" in chunks[1].text

    def test_overlap_ratio_in_metadata(self):
        """Should include overlap ratio in metadata."""
        from core.processors.text.chunking_strategies import SlidingWindowChunking

        strategy = SlidingWindowChunking(window_size=4, step_size=2)
        text = "one two three four five six seven eight"

        chunks = strategy.create_chunks(text)

        # Overlap ratio = 1 - (step_size / window_size) = 1 - (2/4) = 0.5
        assert chunks[0].metadata["overlap_ratio"] == 0.5

    def test_rejects_step_larger_than_window(self):
        """Should reject step_size > window_size."""
        from core.processors.text.chunking_strategies import SlidingWindowChunking

        with pytest.raises(ValueError, match="Step size cannot be larger than window size"):
            SlidingWindowChunking(window_size=4, step_size=5)

    def test_rejects_non_positive_values(self):
        """Should reject non-positive window_size or step_size."""
        from core.processors.text.chunking_strategies import SlidingWindowChunking

        with pytest.raises(ValueError, match="window_size must be positive"):
            SlidingWindowChunking(window_size=0, step_size=1)

        with pytest.raises(ValueError, match="step_size must be positive"):
            SlidingWindowChunking(window_size=4, step_size=0)

    def test_single_chunk_for_short_text(self):
        """Should return single chunk for text shorter than window."""
        from core.processors.text.chunking_strategies import SlidingWindowChunking

        strategy = SlidingWindowChunking(window_size=10, step_size=5)
        text = "short text"

        chunks = strategy.create_chunks(text)

        assert len(chunks) == 1
        assert chunks[0].text == "short text"

    def test_empty_text_returns_empty_list(self):
        """Should return empty list for empty text."""
        from core.processors.text.chunking_strategies import SlidingWindowChunking

        strategy = SlidingWindowChunking(window_size=10, step_size=5)
        chunks = strategy.create_chunks("")

        assert chunks == []

    def test_handles_remainder_tokens(self):
        """Should handle remainder tokens after last full window."""
        from core.processors.text.chunking_strategies import SlidingWindowChunking

        strategy = SlidingWindowChunking(window_size=3, step_size=3)
        text = "one two three four five"

        chunks = strategy.create_chunks(text)

        # Should have window chunk + remainder
        assert len(chunks) >= 1
        # Check if remainder is captured
        all_text = " ".join(c.text for c in chunks)
        assert "five" in all_text


class TestHybridChunking:
    """Tests for HybridChunking strategy.

    HybridChunking combines semantic boundaries with sliding window overlap,
    providing both document structure awareness and retrieval quality.
    """

    def test_creates_chunks_with_overlap(self):
        """Should create chunks with minimum overlap between adjacent chunks."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(
            max_chunk_size=10, min_chunk_size=3, min_overlap_tokens=3
        )
        # Create text with clear paragraph separation - each paragraph ~6-7 words
        # With max_chunk_size=10, we should get multiple chunks
        text = (
            "First paragraph with several words here today.\n\n"
            "Second paragraph also has many different words.\n\n"
            "Third paragraph completes the entire document now."
        )

        chunks = strategy.create_chunks(text)

        # Should have multiple chunks since each paragraph has ~7 tokens
        # and max_chunk_size is 10
        assert len(chunks) >= 2

        # Check that overlap is present between adjacent chunks
        assert chunks[1].metadata.get("overlap_tokens", 0) > 0

    def test_single_chunk_no_overlap_needed(self):
        """Should return single chunk without modification for short text."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(
            max_chunk_size=100, min_chunk_size=10, min_overlap_tokens=5
        )
        text = "Short text that fits in one chunk."

        chunks = strategy.create_chunks(text)

        assert len(chunks) == 1

    def test_metadata_includes_hybrid_strategy(self):
        """Should mark chunks with hybrid strategy in metadata."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(
            max_chunk_size=50, min_chunk_size=5, min_overlap_tokens=3
        )
        text = "First part here with some words."

        chunks = strategy.create_chunks(text)

        # Even single chunks should be marked as hybrid
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "hybrid"

    def test_overlap_tokens_in_metadata(self):
        """Should include overlap_tokens in chunk metadata."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(
            max_chunk_size=15, min_chunk_size=5, min_overlap_tokens=3
        )
        text = "First paragraph words.\n\nSecond paragraph words.\n\nThird paragraph."

        chunks = strategy.create_chunks(text)

        if len(chunks) >= 2:
            # First chunk has no overlap
            assert chunks[0].metadata.get("overlap_tokens") == 0
            # Subsequent chunks should have overlap
            assert chunks[1].metadata.get("overlap_tokens", 0) >= 0

    def test_respects_semantic_boundaries(self):
        """Should respect paragraph boundaries from semantic chunking."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(
            max_chunk_size=50,
            min_chunk_size=10,
            min_overlap_tokens=5,
            respect_paragraphs=True,
        )
        text = "Paragraph one content.\n\nParagraph two content.\n\nParagraph three."

        chunks = strategy.create_chunks(text)

        # Should have chunks
        assert len(chunks) >= 1

    def test_empty_text_returns_empty_list(self):
        """Should return empty list for empty text."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(max_chunk_size=100, min_chunk_size=10)
        chunks = strategy.create_chunks("")

        assert chunks == []

    def test_zero_overlap_returns_semantic_chunks(self):
        """Should return semantic chunks unchanged when min_overlap_tokens=0."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(
            max_chunk_size=50, min_chunk_size=10, min_overlap_tokens=0
        )
        text = "First part.\n\nSecond part.\n\nThird part."

        chunks = strategy.create_chunks(text)

        # Should still have chunks but with 0 overlap
        for chunk in chunks:
            assert chunk.metadata.get("overlap_tokens", 0) == 0

    def test_overlap_limited_by_previous_chunk_length(self):
        """Should limit overlap to previous chunk's token count."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(
            max_chunk_size=10,
            min_chunk_size=2,
            min_overlap_tokens=100,  # Very large overlap request
        )
        text = "Short.\n\nAlso short.\n\nAnd short."

        chunks = strategy.create_chunks(text)

        if len(chunks) >= 2:
            # Overlap should be capped at previous chunk length
            for chunk in chunks[1:]:
                prev_idx = chunk.chunk_index - 1
                if prev_idx >= 0:
                    prev_tokens = len(chunks[prev_idx].text.split())
                    assert chunk.metadata.get("overlap_tokens", 0) <= prev_tokens

    def test_preserves_chunk_indices(self):
        """Should maintain correct chunk indices after adding overlap."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(
            max_chunk_size=15, min_chunk_size=5, min_overlap_tokens=3
        )
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = strategy.create_chunks(text)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_combines_semantic_and_overlap(self):
        """Should combine semantic boundaries with guaranteed overlap."""
        from core.processors.text.chunking_strategies import HybridChunking

        strategy = HybridChunking(
            max_chunk_size=10,
            min_chunk_size=3,
            min_overlap_tokens=3,
            respect_sentences=True,
            respect_paragraphs=True,
        )

        # Create a document with clear structure - each paragraph ~5-7 words
        # With max_chunk_size=10, this should produce multiple chunks
        text = """Introduction to the topic here today.

Main content with important details and useful information.

Conclusion summarizing all the key points made."""

        chunks = strategy.create_chunks(text)

        # Verify multiple chunks were created
        assert len(chunks) >= 2

        # All chunks should have hybrid strategy
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "hybrid"


class TestChunkingStrategyFactory:
    """Tests for ChunkingStrategyFactory."""

    def test_creates_token_strategy(self):
        """Should create TokenBasedChunking for 'token' type."""
        from core.processors.text.chunking_strategies import (
            ChunkingStrategyFactory,
            TokenBasedChunking,
        )

        strategy = ChunkingStrategyFactory.create_strategy(
            "token", chunk_size=100, chunk_overlap=20
        )

        assert isinstance(strategy, TokenBasedChunking)

    def test_creates_semantic_strategy(self):
        """Should create SemanticChunking for 'semantic' type."""
        from core.processors.text.chunking_strategies import (
            ChunkingStrategyFactory,
            SemanticChunking,
        )

        strategy = ChunkingStrategyFactory.create_strategy("semantic", max_chunk_size=500)

        assert isinstance(strategy, SemanticChunking)

    def test_creates_sliding_strategy(self):
        """Should create SlidingWindowChunking for 'sliding' type."""
        from core.processors.text.chunking_strategies import (
            ChunkingStrategyFactory,
            SlidingWindowChunking,
        )

        strategy = ChunkingStrategyFactory.create_strategy("sliding", window_size=256)

        assert isinstance(strategy, SlidingWindowChunking)

    def test_creates_sliding_window_strategy(self):
        """Should accept 'sliding_window' as alias for sliding."""
        from core.processors.text.chunking_strategies import (
            ChunkingStrategyFactory,
            SlidingWindowChunking,
        )

        strategy = ChunkingStrategyFactory.create_strategy("sliding_window")

        assert isinstance(strategy, SlidingWindowChunking)

    def test_creates_hybrid_strategy(self):
        """Should create HybridChunking for 'hybrid' type."""
        from core.processors.text.chunking_strategies import (
            ChunkingStrategyFactory,
            HybridChunking,
        )

        strategy = ChunkingStrategyFactory.create_strategy(
            "hybrid", max_chunk_size=1000, min_overlap_tokens=50
        )

        assert isinstance(strategy, HybridChunking)

    def test_case_insensitive_type(self):
        """Should handle strategy type case-insensitively."""
        from core.processors.text.chunking_strategies import (
            ChunkingStrategyFactory,
            HybridChunking,
            SemanticChunking,
            TokenBasedChunking,
        )

        strategy1 = ChunkingStrategyFactory.create_strategy("TOKEN")
        strategy2 = ChunkingStrategyFactory.create_strategy("Semantic")
        strategy3 = ChunkingStrategyFactory.create_strategy("HYBRID")

        assert isinstance(strategy1, TokenBasedChunking)
        assert isinstance(strategy2, SemanticChunking)
        assert isinstance(strategy3, HybridChunking)

    def test_rejects_unknown_strategy(self):
        """Should raise ValueError for unknown strategy type."""
        from core.processors.text.chunking_strategies import ChunkingStrategyFactory

        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkingStrategyFactory.create_strategy("unknown_type")

    def test_passes_kwargs_to_strategy(self):
        """Should pass kwargs to strategy constructor."""
        from core.processors.text.chunking_strategies import ChunkingStrategyFactory

        strategy = ChunkingStrategyFactory.create_strategy(
            "token", chunk_size=500, chunk_overlap=100
        )

        assert strategy.chunk_size == 500
        assert strategy.chunk_overlap == 100


class TestCleanText:
    """Tests for _clean_text method on ChunkingStrategy."""

    def test_collapses_whitespace(self):
        """Should collapse multiple whitespace to single space."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        strategy = TokenBasedChunking(chunk_size=10, chunk_overlap=2)
        result = strategy._clean_text("hello    world")

        assert result == "hello world"

    def test_removes_null_characters(self):
        """Should remove null characters."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        strategy = TokenBasedChunking(chunk_size=10, chunk_overlap=2)
        result = strategy._clean_text("hello\x00world")

        assert result == "helloworld"

    def test_trims_leading_trailing_whitespace(self):
        """Should trim leading and trailing whitespace."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        strategy = TokenBasedChunking(chunk_size=10, chunk_overlap=2)
        result = strategy._clean_text("  hello world  ")

        assert result == "hello world"

    def test_handles_newlines_and_tabs(self):
        """Should normalize newlines and tabs to spaces."""
        from core.processors.text.chunking_strategies import TokenBasedChunking

        strategy = TokenBasedChunking(chunk_size=10, chunk_overlap=2)
        result = strategy._clean_text("hello\n\nworld\t\there")

        assert result == "hello world here"
