# HADES

## High-speed ArangoDB-backed Dimensional Embedding System

A production-grade semantic knowledge base for academic papers. HADES transforms documents into vector embeddings with late chunking, stores them in ArangoDB via HTTP/2 Unix socket connections, and provides fast semantic search over millions of papers.

## Architecture

HADES is **three composable tools**:

| Tool | Component | Purpose |
|------|-----------|---------|
| **Extract** | Docling | Any document → structured text |
| **Embed** | Jina V4 | Text → 2048-dim vectors with late chunking |
| **Store** | ArangoDB | Pluggable storage backend via Protocol |

Each tool works standalone. `ingest` composes them into a pipeline.

```text
PDF/ArXiv → Extract → Late Chunking → Embed → Store (ArangoDB)
                                         ↓
                            Semantic Search ← Query
```

## Quick Start

```bash
# Install
poetry install

# Set required environment variable
export ARANGO_PASSWORD="your_password"

# Search ArXiv for papers
hades arxiv search "transformer attention mechanisms" --max 10

# Sync recent abstracts (builds searchable index without downloading PDFs)
hades arxiv sync --from 2025-01-01 --batch 8

# Semantic search over your knowledge base
hades db query "how does late chunking preserve context"

# Ingest full papers you're interested in
hades ingest 2409.04701

# Batch ingest with progress and resume
hades ingest /papers/*.pdf --batch
hades ingest --resume  # after failure

# Check database statistics
hades db stats
```

## CLI Reference

All commands output JSON for predictable parsing. Progress messages go to stderr.

### Standalone Tools

```bash
# Extract text from any document (PDF, DOCX, HTML, etc.)
hades extract document.pdf
hades extract document.pdf --format text
hades extract document.pdf --output extracted.json

# Embed text directly
hades embed text "some text to embed"
hades embed text "query text" --task retrieval.query
```

### Ingest — Unified Pipeline

Auto-detects ArXiv IDs vs file paths. Downloads, extracts, chunks, embeds, and stores:

```bash
# Single items
hades ingest 2409.04701                    # arxiv paper
hades ingest paper.pdf                     # local file
hades ingest paper.pdf --id my-custom-id   # custom document ID
hades ingest 2409.04701 --force            # reprocess existing

# Batch mode with progress and error isolation
hades ingest /papers/*.pdf --batch
hades ingest 2409.04701 2501.12345 --batch
hades ingest --resume                      # resume after failure
```

### ArXiv Commands

```bash
# Search ArXiv API (live)
hades arxiv search "transformer attention" --max 20
hades arxiv info 2409.04701

# Search synced abstract database (2.8M papers, much faster)
hades arxiv abstract "flash attention memory optimization" --limit 20

# Find similar papers
hades arxiv similar 2409.04701 --limit 10

# Relevance feedback
hades arxiv refine "attention optimization" --positive 2409.04701 --limit 20

# Sync abstracts from ArXiv (run periodically)
hades arxiv sync --from 2025-01-01 --categories cs.AI,cs.CL --max 10000
CUDA_VISIBLE_DEVICES=2 hades arxiv sync --from 2025-01-01 --batch 8

# Check sync status
hades arxiv sync-status
```

### Database Commands

```bash
# Semantic search over ingested full papers
hades db query "how does flash attention reduce memory"
hades db query "Newton-Schulz" --paper 2505.23735    # within one paper
hades db query "attention" --context 1               # include adjacent chunks
hades db query "attention" --cite --top-k 3          # citation format

# Get all chunks of a paper (no search)
hades db query --paper 2409.04701 --chunks

# List and check papers
hades db list --limit 20
hades db check 2409.04701
hades db stats
hades db purge 2409.04701   # remove paper and all chunks

# Raw AQL query
hades db aql "FOR doc IN arxiv_metadata FILTER doc.year == 2025 RETURN doc.title"
```

### Embedding Service

```bash
hades embed service status
hades embed service start
hades embed service stop
hades embed gpu status
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
| `arxiv_metadata` | Paper metadata (authors, categories, dates) |
| `arxiv_abstracts` | Abstract text for synced papers |
| `arxiv_embeddings` | 2048-dim embeddings for semantic search |
| `arxiv_chunks` | Full-text chunks for ingested papers |

### Sync vs Ingest

- **Sync**: Fetches metadata + abstracts from ArXiv, embeds abstracts only. Fast, lightweight, good for discovery.
- **Ingest**: Downloads full PDF, extracts text, chunks, embeds everything. Complete but heavier.

## Project Structure

```text
hades/
├── core/
│   ├── cli/              # CLI commands and output formatting
│   │   ├── commands/     # Command groups (arxiv, db, embed, ingest, extract)
│   │   ├── main.py       # Typer app entry point
│   │   └── output.py     # JSON response formatting
│   ├── database/
│   │   ├── arango/       # HTTP/2 ArangoDB client
│   │   └── schemas.py    # Source-agnostic document schemas
│   ├── embedders/        # Jina V4 with late chunking
│   ├── extractors/       # Docling, LaTeX extractors
│   ├── processors/       # Document processing, batch processor
│   ├── services/         # Persistent embedding service
│   └── tools/            # Standalone tools (extract, embed, store)
├── tests/
│   ├── core/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
└── setup/                # Environment verification scripts
```

## Key Files

| File | Purpose |
|------|---------|
| `core/cli/main.py` | CLI entry point |
| `core/tools/extract.py` | Standalone extraction tool |
| `core/tools/embed.py` | Standalone embedding tool |
| `core/tools/store.py` | Storage backend protocol |
| `core/database/arango/backend.py` | ArangoDB storage implementation |
| `core/database/arango/optimized_client.py` | HTTP/2 ArangoDB client |
| `core/processors/document_processor.py` | Document processing pipeline |
| `core/processors/batch.py` | Batch processing with resume |
| `core/embedders/embedders_jina.py` | Jina V4 embedder |
| `core/extractors/extractors_docling.py` | PDF extraction |

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
0 2 * * * root CUDA_VISIBLE_DEVICES=2 ARANGO_PASSWORD=xxx /path/to/hades arxiv sync --from $(date -d "7 days ago" +\%Y-\%m-\%d) --batch 8 >> /var/log/hades-sync.log 2>&1
```

## License

Apache 2.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `poetry run pytest`
4. Run linting: `poetry run ruff check`
5. Submit a pull request
