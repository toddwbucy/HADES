# HADES CLI

AI-focused command-line interface for the HADES knowledge base.

## Design Philosophy

The CLI is designed for **AI models** (Claude Code, GPT, etc.) to interact with the knowledge base. Key design decisions:

1. **JSON Output** - All commands output structured JSON to stdout for predictable parsing
2. **Progress to Stderr** - Progress messages go to stderr so they don't pollute JSON
3. **Three Composable Tools** - Extract, Embed, Store work standalone; `ingest` composes them
4. **Command Groups** - Related commands are namespaced (`hades db`, `hades arxiv`, `hades embed`)
5. **Error Codes** - Structured error responses with machine-readable codes

## Architecture

```
core/cli/
├── __init__.py           # Package exports
├── main.py               # Typer app with command groups
├── config.py             # Environment variable resolution
├── output.py             # JSON response formatting (CLIResponse, ErrorCode)
├── agent_templates.py    # SKILL.md and AGENT.md templates
└── commands/
    ├── __init__.py
    ├── arxiv.py          # hades arxiv (search, info, sync, abstract, similar, refine)
    ├── database.py       # hades db (query, list, stats, check, purge, aql, graph)
    ├── embedding.py      # hades embed (text, service, gpu)
    ├── extract.py        # hades extract
    └── ingest.py         # hades ingest (with --batch, --resume)
```

## Command Groups

### `hades extract` — Standalone Extraction

Extract text from any document type.

```bash
hades extract document.pdf
hades extract document.pdf --format text
hades extract document.pdf --output extracted.json
```

Implementation: `commands/extract.py:extract()`

### `hades embed` — Standalone Embedding

Generate embeddings for text.

```bash
hades embed text "some text to embed"
hades embed text "query" --task retrieval.query

# Service management
hades embed service status
hades embed service start
hades embed service stop
hades embed gpu status
```

Implementation: `commands/embedding.py`

### `hades ingest` — Unified Pipeline

Compose extract → embed → store into a single operation.

```bash
# Single items (auto-detects arxiv ID vs file path)
hades ingest 2409.04701
hades ingest paper.pdf
hades ingest paper.pdf --id custom-id

# Batch mode with progress and error isolation
hades ingest *.pdf --batch
hades ingest 2409.04701 2501.12345 --batch
hades ingest --resume   # resume after failure

# Force reprocessing
hades ingest 2409.04701 --force
```

Implementation: `commands/ingest.py:ingest()`

Key features:
- Auto-detection of ArXiv IDs vs file paths
- Batch mode with JSON progress to stderr
- Resume capability via state file (`.hades-batch-state.json`)
- Error isolation (one failure doesn't stop the batch)

### `hades db` — Database Operations

Query and manage the knowledge base.

```bash
# Semantic search
hades db query "how does flash attention work" --limit 10
hades db query "Newton-Schulz" --paper 2505.23735
hades db query "attention" --context 1 --cite

# Get paper chunks (no search)
hades db query --paper 2409.04701 --chunks

# Management
hades db list --limit 20
hades db stats
hades db check 2409.04701
hades db purge 2409.04701

# Raw AQL
hades db aql "FOR doc IN arxiv_metadata RETURN doc.title"

# Graph operations
hades db graph list
hades db graph traverse --start "arxiv_metadata/2409_04701" --direction outbound
```

Implementation: `commands/database.py`

### `hades arxiv` — ArXiv Source Adapter

Search and sync papers from ArXiv.

```bash
# Live ArXiv API search
hades arxiv search "transformer attention" --max 20
hades arxiv info 2409.04701

# Search synced abstracts (faster, local)
hades arxiv abstract "flash attention" --limit 20
hades arxiv similar 2409.04701 --limit 10
hades arxiv refine "attention" --positive 2409.04701

# Sync abstracts (no PDF download)
hades arxiv sync --from 2025-01-01 --categories cs.AI,cs.CL --batch 8
hades arxiv sync-status
```

Implementation: `commands/arxiv.py`

Key features:
- Month-by-month fetching to avoid ArXiv API pagination limits
- Deduplication against existing papers
- Batch embedding with configurable size
- Relevance feedback via Rocchio algorithm

## Response Format

### Success Response

```python
@dataclass
class CLIResponse:
    success: bool = True
    command: str           # e.g., "db.query", "ingest"
    data: dict | list      # Command-specific results
    metadata: dict         # duration_ms, count, etc.
```

JSON output:
```json
{
  "success": true,
  "command": "db.query",
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
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_ARXIV_ID = "INVALID_ARXIV_ID"
    DATABASE_ERROR = "DATABASE_ERROR"
    SEARCH_FAILED = "SEARCH_FAILED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    QUERY_FAILED = "QUERY_FAILED"
    CONFIG_ERROR = "CONFIG_ERROR"
    SERVICE_ERROR = "SERVICE_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    GRAPH_ERROR = "GRAPH_ERROR"
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

## Agent Integration

Install Claude Code skill or AGENT.md integration:

```bash
# Install Claude Code skill
hades --agent claude
# Creates: .claude/skills/hades/SKILL.md

# Install AGENT.md section
hades --agent agent
# Creates or appends to: AGENT.md
```

Templates are defined in `agent_templates.py`.

## Adding New Commands

1. Create command function in `commands/`:

```python
# commands/mycommand.py
from core.cli.output import CLIResponse, success_response, error_response, ErrorCode

def my_command(arg1: str, start_time: float) -> CLIResponse:
    try:
        result = {"key": "value"}
        return success_response(
            command="mycommand",
            data=result,
            start_time=start_time,
        )
    except FileNotFoundError:
        return error_response(
            command="mycommand",
            code=ErrorCode.FILE_NOT_FOUND,
            message="File not found",
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
poetry run pytest tests/core/cli/test_ingest.py -v

# Manual testing
poetry run hades --help
poetry run hades db stats
poetry run hades arxiv search "test" --max 1
```

## Dependencies

- **typer** - CLI framework
- **core.tools** - Standalone tools (extract, embed, store)
- **core.database.arango** - ArangoDB client and backend
- **core.processors** - Document and batch processing
- **core.embedders** - Jina V4 embedder
- **core.extractors** - Document extraction
