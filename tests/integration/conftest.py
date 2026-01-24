"""Shared fixtures for integration tests."""

import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ==============================================================================
# Path Fixtures
# ==============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_pdf_path(fixtures_dir: Path) -> Path:
    """Return path to sample PDF for testing."""
    return fixtures_dir / "sample.pdf"


@pytest.fixture
def sample_latex_path(fixtures_dir: Path) -> Path:
    """Return path to sample LaTeX file for testing."""
    return fixtures_dir / "sample.tex"


@pytest.fixture
def sample_code_path(fixtures_dir: Path) -> Path:
    """Return path to sample code file for testing."""
    return fixtures_dir / "sample.py"


# ==============================================================================
# Environment Fixtures
# ==============================================================================


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Remove ARANGO-related environment variables and set test password."""
    for key in list(os.environ.keys()):
        if key.startswith("ARANGO"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ARANGO_PASSWORD", "test_password")
    yield


@pytest.fixture
def mock_gpu_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock environment where GPU is unavailable."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


# ==============================================================================
# Mock Fixtures
# ==============================================================================


@pytest.fixture
def mock_arango_client() -> MagicMock:
    """Create a mock ArangoDB client for testing without database."""
    mock_client = MagicMock()
    mock_client.execute_query.return_value = []
    mock_client.insert_documents.return_value = {"created": 1}
    mock_client.close.return_value = None
    return mock_client


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock embedder for testing without GPU."""
    import numpy as np

    mock = MagicMock()
    # Return fake embeddings (2048-dim for Jina V4 compatibility)
    mock.embed_with_late_chunking.return_value = [
        MagicMock(
            chunk_index=0,
            text="Test chunk",
            embedding=[0.1] * 2048,
            token_start=0,
            token_end=10,
        )
    ]
    # JinaV4Embedder uses embed_single and embed_texts, not embed_text
    mock.embed_single.return_value = np.array([0.1] * 2048)
    mock.embed_texts.return_value = np.array([[0.1] * 2048])
    mock.embedding_dim = 2048
    return mock


# ==============================================================================
# Sample Data Fixtures
# ==============================================================================


@pytest.fixture
def sample_text() -> str:
    """Return sample text for testing."""
    return """
    This is a sample document for testing the HADES RAG pipeline.

    It contains multiple paragraphs to test chunking behavior.
    The text should be long enough to generate multiple chunks
    when processed with typical chunk sizes.

    This paragraph discusses machine learning concepts.
    Neural networks are computational models inspired by biological neurons.
    They learn patterns from data through training processes.

    This final paragraph concludes the sample document.
    It provides enough content for meaningful integration testing.
    """


@pytest.fixture
def sample_extraction_result(sample_text: str) -> dict:
    """Return a sample extraction result structure."""
    return {
        "full_text": sample_text,
        "metadata": {
            "title": "Sample Document",
            "author": "Test Author",
            "page_count": 1,
        },
        "sections": [
            {"title": "Introduction", "content": sample_text[:200]},
            {"title": "Body", "content": sample_text[200:]},
        ],
    }


# ==============================================================================
# Skip Marker Helpers
# ==============================================================================


def _check_gpu_available() -> bool:
    """Check if GPU is available for testing."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _check_docling_available() -> bool:
    """Check if Docling is available for testing."""
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401

        return True
    except ImportError:
        return False


# ==============================================================================
# Skip Markers
# ==============================================================================


requires_gpu = pytest.mark.skipif(
    os.environ.get("CUDA_VISIBLE_DEVICES") == "" or not _check_gpu_available(),
    reason="GPU not available",
)

requires_arango = pytest.mark.skipif(
    os.environ.get("ARANGO_PASSWORD") is None,
    reason="ArangoDB credentials not configured",
)

requires_docling = pytest.mark.skipif(
    not _check_docling_available(),
    reason="Docling not available",
)
