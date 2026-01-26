# HADES CLI

AI-focused command-line interface for the HADES knowledge base.

## Design Philosophy

The CLI is designed for **AI models** (Claude Code, GPT, etc.) to interact with the knowledge base. Humans ask the AI, the AI uses the CLI. Key design decisions:

1. **JSON Output** - All commands output structured JSON to stdout for predictable parsing
2. **Progress to Stderr** - Progress messages go to stderr so they don't pollute JSON
3. **Atomic Operations** - Each command is self-contained and idempotent where possible
4. **Error Codes** - Structured error responses with machine-readable codes

## Architecture

```
core/cli/
├── __init__.py           # Package exports
├── main.py               # Typer app with all commands
├── config.py             # Environment variable resolution
├── output.py             # JSON response formatting
└── commands/
    ├── __init__.py
    ├── arxiv.py          # arxiv search, arxiv info
    ├── db.py             # list, stats, check
    ├── ingest.py         # Paper ingestion
    ├── query.py          # Semantic search
    └── sync.py           # Abstract syncing
```

## Response Format

### Success Response

```python
@dataclass
class CLIResponse:
    success: bool = True
    command: str           # e.g., "arxiv.search", "query", "sync"
    data: dict | list      # Command-specific results
    metadata: dict         # duration_ms, count, etc.
```

JSON output:
```json
{
  "success": true,
  "command": "query",
  "data": {
    "query": "attention mechanisms",
    "results": [...]
  },
  "metadata": {
    "duration_ms": 234,
    "count": 10
  }
}
```

### Error Response

```python
class ErrorCode(str, Enum):
    PAPER_NOT_FOUND = "PAPER_NOT_FOUND"
    INVALID_ARXIV_ID = "INVALID_ARXIV_ID"
    DATABASE_ERROR = "DATABASE_ERROR"
    SEARCH_FAILED = "SEARCH_FAILED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    QUERY_FAILED = "QUERY_FAILED"
    CONFIG_ERROR = "CONFIG_ERROR"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
```

JSON output:
```json
{
  "success": false,
  "command": "ingest",
  "error": {
    "code": "DOWNLOAD_FAILED",
    "message": "Failed to download PDF: 404 Not Found"
  },
  "metadata": {
    "duration_ms": 1234
  }
}
```

## Commands

### arxiv search

Search ArXiv API for papers.

```bash
hades arxiv search "transformer attention" --max 20 --categories cs.AI,cs.CL
```

Implementation: `commands/arxiv.py:search_arxiv()`

### arxiv info

Get metadata for a specific paper.

```bash
hades arxiv info 2409.04701
```

Implementation: `commands/arxiv.py:get_paper_info()`

### sync

Sync abstracts from ArXiv for semantic search. Does NOT download PDFs.

```bash
hades sync --from 2025-01-01 --categories cs.AI --max 10000 --batch 8
```

Implementation: `commands/sync.py:sync_abstracts()`

Key features:
- Month-by-month fetching to avoid ArXiv API pagination limits (10K)
- Deduplication against existing papers
- Batch embedding with configurable size
- Stores in arxiv_papers, arxiv_abstracts, arxiv_embeddings collections

### query

Semantic search over the knowledge base.

```bash
hades query "how does flash attention work" --limit 10
hades query --paper 2409.04701
```

Implementation: `commands/query.py:semantic_query()`, `get_paper_chunks()`

Searches both:
- Synced abstracts (arxiv_abstracts + arxiv_embeddings)
- Full paper chunks (arxiv_abstract_chunks)

### ingest

Download and process full papers.

```bash
hades ingest 2409.04701 2501.12345
hades ingest --file /path/to/paper.pdf
hades ingest 2409.04701 --force
```

Implementation: `commands/ingest.py:ingest_papers()`

Pipeline:
1. Download PDF from ArXiv
2. Extract text with Docling
3. Chunk text
4. Generate embeddings with Jina V4
5. Store in arxiv_abstract_chunks collection

### list

List papers in the database.

```bash
hades list --limit 20 --category cs.AI
```

Implementation: `commands/db.py:list_stored_papers()`

### stats

Show database statistics.

```bash
hades stats
```

Implementation: `commands/db.py:get_stats()`

### check

Check if a paper exists in the database.

```bash
hades check 2409.04701
```

Implementation: `commands/db.py:check_paper_exists()`

## Configuration

Environment variables are resolved in `config.py`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| ARANGO_PASSWORD | Yes | - | Database password |
| ARANGO_HOST | No | localhost | Database host |
| ARANGO_PORT | No | 8529 | Database port |
| HADES_DATABASE | No | arxiv_datastore | Database name |
| ARANGO_RO_SOCKET | No | - | Read-only Unix socket |
| ARANGO_RW_SOCKET | No | - | Read-write Unix socket |
| HADES_USE_GPU | No | true | Enable GPU for embeddings |
| CUDA_VISIBLE_DEVICES | No | - | GPU selection |

## Adding New Commands

1. Create command function in `commands/`:

```python
# commands/mycommand.py
from core.cli.output import CLIResponse, success_response, error_response, ErrorCode

def my_command(arg1: str, start_time: float) -> CLIResponse:
    try:
        # Do work
        result = {"key": "value"}
        return success_response(
            command="mycommand",
            data=result,
            start_time=start_time,
        )
    except Exception as e:
        return error_response(
            command="mycommand",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
```

2. Register in `main.py`:

```python
@app.command("mycommand")
def mycommand(
    arg1: str = typer.Argument(..., help="Description"),
) -> None:
    """Command description."""
    start_time = time.time()

    try:
        from core.cli.commands.mycommand import my_command
        response = my_command(arg1, start_time)
        print_response(response)
        if not response.success:
            raise typer.Exit(1) from None
    except typer.Exit:
        raise
    except Exception as e:
        response = error_response(
            command="mycommand",
            code=ErrorCode.PROCESSING_FAILED,
            message=str(e),
            start_time=start_time,
        )
        print_response(response)
        raise typer.Exit(1) from None
```

## Testing

```bash
# Run CLI tests
poetry run pytest tests/core/cli/ -v

# Test specific command
poetry run pytest tests/core/cli/test_arxiv.py -v

# Manual testing
poetry run hades --help
poetry run hades stats
poetry run hades arxiv search "test" --max 1
```

## Dependencies

- **typer** - CLI framework
- **core.database.arango** - ArangoDB client
- **core.embedders** - Jina V4 embedder
- **core.extractors** - Document extraction
- **core.tools.arxiv** - ArXiv API client
