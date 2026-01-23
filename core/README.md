# Core Infrastructure

The `core/` directory contains the foundational infrastructure components for the HADES high-speed RAG system.

## Directory Structure

```
core/
├── config/           # YAML configuration management
├── database/arango/  # HTTP/2 Unix socket ArangoDB client
├── embedders/        # Jina V4 with late chunking
├── extractors/       # Docling, LaTeX, code extractors
├── logging/          # Standard logging utilities
├── processors/       # Document processing pipeline
├── tools/            # ArXiv API client, citation toolkit
└── workflows/        # Orchestration and state management
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `database/arango/optimized_client.py` | HTTP/2 Unix socket ArangoDB client |
| `processors/document_processor.py` | Main extraction + embedding pipeline |
| `embedders/embedders_jina.py` | Jina V4 with late chunking |
| `extractors/extractors_docling.py` | PDF extraction via Docling |

## Usage

```python
# Fast ArangoDB connection
from core.database.arango.optimized_client import ArangoHttp2Client

# Document processing
from core.processors.document_processor import DocumentProcessor

# Embeddings with late chunking
from core.embedders.embedders_jina import JinaV4Embedder

# Get extractor for file type
from core.extractors import get_extractor
extractor = get_extractor("document.pdf")
```

## Design Principles

1. **Phase Separation**: Extract -> Process -> Embed -> Store phases are cleanly separated
2. **Atomic Operations**: All database operations are atomic
3. **Late Chunking**: Document encoding before segmentation preserves context
4. **Reusability**: Components are framework-agnostic and can be imported independently

## Testing

```bash
poetry run pytest tests/core/ -v
poetry run python -m compileall core/ -q
```
