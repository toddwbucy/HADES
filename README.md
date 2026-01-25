# HADES

## High-speed ArangoDB-backed Dimensional Embedding System

A production-grade semantic knowledge base for academic papers. HADES transforms documents into vector embeddings with late chunking, stores them in ArangoDB via HTTP/2 Unix socket connections, and provides fast semantic search over millions of papers.

## Features

- **CLI Interface** - AI-friendly JSON output for search, ingest, and query operations
- **Semantic Search** - Query 2.8M+ paper abstracts in under 15 seconds
- **Late Chunking** - Context-preserving embeddings using Jina V4 (2048 dimensions)
- **HTTP/2 Transport** - Sub-millisecond database operations via Unix sockets
- **ArXiv Integration** - Search, sync, and ingest papers directly from ArXiv

## Quick Start

```bash
# Install
poetry install

# Set required environment variable
export ARANGO_PASSWORD="your_password"

# Search ArXiv for papers
poetry run hades arxiv search "transformer attention mechanisms" --max 10

# Sync recent abstracts (builds searchable index without downloading PDFs)
poetry run hades sync --from 2025-01-01 --batch 8

# Semantic search over your knowledge base
poetry run hades query "how does late chunking preserve context"

# Ingest full papers you're interested in
poetry run hades ingest 2409.04701

# Check database statistics
poetry run hades stats
```

## CLI Reference

All commands output JSON for predictable parsing. Progress messages go to stderr.

### ArXiv Commands

```bash
# Search ArXiv for papers
hades arxiv search "query" [--max N] [--categories cs.AI,cs.CL]

# Get paper metadata
hades arxiv info <arxiv_id>
```

### Sync Command

Sync abstracts from ArXiv for semantic search (no PDF downloads):

```bash
# Sync papers from the last 7 days
hades sync

# Sync from a specific date
hades sync --from 2025-01-01

# Sync specific categories with custom batch size
hades sync --from 2025-01-01 --categories cs.AI,cs.CL --batch 8
```

### Query Commands

```bash
# Semantic search
hades query "your search text" [--limit N]

# Get all chunks for a specific paper
hades query --paper <arxiv_id>
```

### Ingest Commands

Download, process, and store full papers:

```bash
# Ingest by ArXiv ID
hades ingest 2409.04701 2401.12345

# Ingest local PDF
hades ingest --file /path/to/paper.pdf

# Force reprocessing
hades ingest 2409.04701 --force
```

### Database Commands

```bash
# List stored papers
hades list [--limit N] [--category CAT]

# Database statistics
hades stats

# Check if paper exists
hades check <arxiv_id>
```

### Output Format

Success:
```json
{
  "success": true,
  "command": "query",
  "data": { ... },
  "metadata": { "duration_ms": 234, "count": 10 }
}
```

Error:
```json
{
  "success": false,
  "command": "query",
  "error": { "code": "QUERY_FAILED", "message": "..." }
}
```

## Architecture

```text
PDF/ArXiv → Extractor → Late Chunking → Jina V4 Embedder → ArangoDB (HTTP/2)
                                              ↓
                                    Semantic Search ← Query
```

### Core Pipeline

| Stage | Component | Purpose |
|-------|-----------|---------|
| Extract | Docling | PDF/document text extraction |
| Embed | Jina V4 | 2048-dimension embeddings with late chunking |
| Store | ArangoDB | HTTP/2 Unix socket client for fast writes |
| Search | CLI | Cosine similarity search over embeddings |

### Key Files

| File | Purpose |
|------|---------|
| `core/cli/main.py` | CLI entry point |
| `core/database/arango/optimized_client.py` | HTTP/2 ArangoDB client |
| `core/processors/document_processor.py` | Document processing pipeline |
| `core/embedders/embedders_jina.py` | Jina V4 embedder |
| `core/extractors/extractors_docling.py` | PDF extraction |

## Installation

### Requirements

- Python 3.11+
- ArangoDB 3.11+
- CUDA-capable GPU (recommended for embeddings)
- 16GB+ GPU VRAM for batch processing

### Setup

```bash
# Clone and install
git clone https://github.com/your-org/hades.git
cd hades
poetry install

# Verify environment
poetry run python setup/verify_environment.py
poetry run python setup/verify_storage.py
```

### GPU Setup

```bash
# For batch embedding on a specific GPU
export CUDA_VISIBLE_DEVICES=2  # Use GPU 2

# Verify GPU
poetry run python -c "import torch; print(torch.cuda.is_available())"
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ARANGO_PASSWORD` | Yes | - | ArangoDB password |
| `ARANGO_HOST` | No | localhost | ArangoDB host |
| `ARANGO_PORT` | No | 8529 | ArangoDB port |
| `HADES_DATABASE` | No | arxiv_datastore | Database name |
| `ARANGO_RO_SOCKET` | No | - | Read-only Unix socket |
| `ARANGO_RW_SOCKET` | No | - | Read-write Unix socket |
| `CUDA_VISIBLE_DEVICES` | No | - | GPU selection |
| `HADES_USE_GPU` | No | true | Enable GPU processing |

### ArangoDB Sockets

For production, use Unix sockets for lowest latency:

```bash
export ARANGO_RO_SOCKET=/run/hades/readonly/arangod.sock
export ARANGO_RW_SOCKET=/run/hades/readwrite/arangod.sock
```

## Database Schema

### Collections

| Collection | Purpose |
|------------|---------|
| `arxiv_papers` | Paper metadata (authors, categories, dates) |
| `arxiv_abstracts` | Abstract text for synced papers |
| `arxiv_embeddings` | 2048-dim embeddings for semantic search |
| `arxiv_abstract_chunks` | Full-text chunks for ingested papers |

### Sync vs Ingest

- **Sync**: Fetches metadata + abstracts from ArXiv, embeds abstracts only. Fast, lightweight, good for discovery.
- **Ingest**: Downloads full PDF, extracts text, chunks, embeds everything. Complete but heavier.

## Development

### Testing

```bash
# All tests
poetry run pytest

# Unit tests only
poetry run pytest tests/core/

# Integration tests
poetry run pytest tests/integration/

# Specific test
poetry run pytest -k "test_query"
```

### Code Quality

```bash
# Linting
poetry run ruff check

# Formatting
poetry run ruff format

# Type checking
poetry run mypy core --ignore-missing-imports

# Compile check
poetry run python -m compileall core
```

## Project Structure

```text
hades/
├── core/
│   ├── cli/              # CLI commands and output formatting
│   │   ├── commands/     # Individual command implementations
│   │   ├── main.py       # Typer app entry point
│   │   ├── config.py     # Environment configuration
│   │   └── output.py     # JSON response formatting
│   ├── database/arango/  # HTTP/2 ArangoDB client
│   ├── embedders/        # Jina V4 with late chunking
│   ├── extractors/       # Docling, LaTeX extractors
│   ├── processors/       # Document processing pipeline
│   └── tools/arxiv/      # ArXiv API client
├── tests/
│   ├── core/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── setup/                # Environment verification scripts
└── docs/                 # Additional documentation
```

## Performance

### Benchmarks

| Operation | Latency |
|-----------|---------|
| Single document GET (hot cache) | ~0.4 ms |
| Insert 1000 documents | ~6 ms |
| Query LIMIT 1000 | ~0.7 ms |
| Semantic search (2.8M papers) | ~14 sec |
| Embed batch of 8 abstracts | ~1-2 sec |

### Recommended Settings

| GPU VRAM | Batch Size | Papers/Hour |
|----------|------------|-------------|
| 8 GB | 4 | ~7,200 |
| 16 GB | 8 | ~14,400 |
| 24 GB | 16 | ~28,800 |

## Cron Setup

Keep your abstract database current with a nightly sync:

```bash
# /etc/cron.d/hades-sync
0 2 * * * root CUDA_VISIBLE_DEVICES=2 ARANGO_PASSWORD=xxx /path/to/hades sync --from $(date -d "7 days ago" +\%Y-\%m-\%d) --batch 8 >> /var/log/hades-sync.log 2>&1
```

## License

Apache 2.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `poetry run pytest`
4. Run linting: `poetry run ruff check`
5. Submit a pull request
