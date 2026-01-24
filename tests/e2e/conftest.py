"""Fixtures for end-to-end tests."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a simple test PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not available")

    pdf_path = tmp_path / "test_document.pdf"

    doc = fitz.open()

    # Page 1: Introduction
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Introduction to Machine Learning",
        fontsize=18,
        fontname="helv",
    )
    page1.insert_text(
        (72, 120),
        """Machine learning is a subset of artificial intelligence that enables
computers to learn from data and improve their performance without being
explicitly programmed. This document explores the fundamental concepts
of machine learning algorithms and their applications in modern computing.

The field has grown significantly over the past decade, with applications
ranging from natural language processing to computer vision. Key techniques
include supervised learning, unsupervised learning, and reinforcement learning.""",
        fontsize=11,
        fontname="helv",
    )

    # Page 2: Technical Details
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Technical Foundations",
        fontsize=16,
        fontname="helv",
    )
    page2.insert_text(
        (72, 110),
        """Neural networks form the backbone of deep learning systems. A typical
neural network consists of multiple layers: input layer, hidden layers, and
output layer. Each layer contains neurons that process information.

The training process involves:
1. Forward propagation - computing outputs from inputs
2. Loss calculation - measuring prediction errors
3. Backpropagation - computing gradients for parameter updates
4. Optimization - adjusting weights using gradient descent

Modern architectures include convolutional neural networks (CNNs) for image
processing and transformer models for natural language understanding.""",
        fontsize=11,
        fontname="helv",
    )

    # Page 3: Conclusion
    page3 = doc.new_page()
    page3.insert_text(
        (72, 72),
        "Conclusions and Future Directions",
        fontsize=16,
        fontname="helv",
    )
    page3.insert_text(
        (72, 110),
        """Machine learning continues to evolve rapidly. Future developments
include more efficient training methods, better interpretability of models,
and applications in emerging fields such as quantum computing and
personalized medicine.

As computational resources become more accessible, we expect to see
broader adoption of machine learning techniques across various industries.""",
        fontsize=11,
        fontname="helv",
    )

    doc.save(pdf_path)
    doc.close()

    return pdf_path


@pytest.fixture
def minimal_pdf(tmp_path: Path) -> Path:
    """Create a minimal test PDF with just a few words."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not available")

    pdf_path = tmp_path / "minimal.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World. This is a test document.", fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Create an empty PDF with no text."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not available")

    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()  # Empty page
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def multi_page_pdf(tmp_path: Path) -> Path:
    """Create a multi-page PDF with substantial content."""
    try:
        import fitz
    except ImportError:
        pytest.skip("PyMuPDF not available")

    pdf_path = tmp_path / "multipage.pdf"
    doc = fitz.open()

    content_blocks = [
        "Chapter 1: The Beginning\n\nThis is the first chapter with introductory content.",
        "Chapter 2: The Middle\n\nThis chapter contains the main arguments and analysis.",
        "Chapter 3: Advanced Topics\n\nDeeper exploration of technical concepts follows here.",
        "Chapter 4: Applications\n\nPractical applications and case studies are presented.",
        "Chapter 5: Conclusion\n\nSummary and final thoughts on the subject matter.",
    ]

    for i, content in enumerate(content_blocks):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1}", fontsize=14)
        page.insert_text((72, 100), content, fontsize=11)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic embeddings."""

    class MockEmbedder:
        def __init__(self):
            self.embedding_dim = 2048

        def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
            """Return mock embeddings based on text content."""
            embeddings = []
            for text in texts:
                # Generate deterministic embedding based on text hash
                seed = hash(text) % (2**32)
                rng = np.random.default_rng(seed)
                embedding = rng.random(self.embedding_dim).astype(np.float32)
                # Normalize to unit vector
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            return embeddings

        def embed_with_late_chunking(self, text: str) -> list[Any]:
            """Return mock chunks with embeddings."""
            from core.embedders.embedders_jina import ChunkWithEmbedding

            # Simple chunking: split into ~100 word chunks
            words = text.split()
            chunks = []
            chunk_size = 100

            for i in range(0, len(words), chunk_size):
                chunk_words = words[i : i + chunk_size]
                chunk_text = " ".join(chunk_words)

                # Calculate approximate character positions
                start_char = sum(len(w) + 1 for w in words[:i])
                end_char = start_char + len(chunk_text)

                # Generate embedding
                seed = hash(chunk_text) % (2**32)
                rng = np.random.default_rng(seed)
                embedding = rng.random(2048).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)

                chunks.append(
                    ChunkWithEmbedding(
                        text=chunk_text,
                        embedding=embedding,
                        start_char=start_char,
                        end_char=end_char,
                        start_token=i,
                        end_token=i + len(chunk_words),
                        chunk_index=len(chunks),
                        total_chunks=0,  # Will be updated
                        context_window_used=len(chunk_words),
                    )
                )

            # Update total_chunks
            for chunk in chunks:
                chunk.total_chunks = len(chunks)

            return chunks

    return MockEmbedder()


@pytest.fixture
def mock_extractor():
    """Create a mock extractor for testing without real PDF parsing."""
    from core.extractors.extractors_base import ExtractionResult

    class MockExtractor:
        def __init__(self):
            self.supported_formats = [".pdf"]

        def extract(self, file_path: str | Path) -> ExtractionResult:
            """Return mock extraction result."""
            return ExtractionResult(
                text="This is mock extracted text from the document. "
                "It contains multiple sentences for testing purposes. "
                "The extraction was successful.",
                metadata={"extractor": "mock", "num_pages": 1, "source": str(file_path)},
                tables=[],
                equations=[],
                images=[],
            )

    return MockExtractor()
