"""Agent integration templates for the HADES CLI.

Generates SKILL.md (Claude Code) or AGENT.md (generic agent) files
in the current working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

SKILL_TEMPLATE = """\
---
name: hades
description: Use the HADES semantic-graph-RAG knowledge base to search, ingest, and query academic papers
user-invocable: true
argument-hint: "[command]"
allowed-tools:
  - Bash
  - Read
---

# HADES CLI Skill

HADES is a semantic graph RAG system backed by ArangoDB. It provides three composable tools:

- **Extract** — Convert any document to structured text (PDF, DOCX, HTML, etc.)
- **Embed** — Generate vector embeddings (Jina v4, late chunking)
- **Store** — ArangoDB with vector + graph + document capabilities

The `hades` CLI outputs JSON to stdout; progress and logs go to stderr.

## Important — Where to Search

**All academic paper searches start with `hades arxiv`**, not `hades db`.

- `hades arxiv search` — live ArXiv API search
- `hades arxiv abstract` — fast semantic search over 2.8M synced abstracts
- `hades ingest` — download, extract, chunk, embed, and store papers (arxiv IDs or local files)

**`hades db` is for querying already-ingested paper content** (full-text semantic search over stored chunks) **and for managing custom collections/databases**. Do not use `hades db` to find papers — use `hades arxiv` first, ingest the papers you need, then query their content with `hades db query`.

## Prerequisites

The embedding service must be running (`hades embed service status`). If it reports unhealthy or connection refused, run: `sudo systemctl restart hades-embedder`

Configuration is at `/etc/hades/embedder.conf`.

## Commands

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

```bash
# Auto-detects arxiv IDs vs file paths
hades ingest 2409.04701                    # arxiv paper
hades ingest paper.pdf                     # local file
hades ingest paper.pdf --id my-custom-id   # custom document ID
hades ingest 2409.04701 2501.12345         # multiple papers
hades ingest 2409.04701 --force            # reprocess existing

# Batch mode with progress and error isolation
hades ingest /papers/*.pdf --batch
hades ingest --resume                      # resume after failure
```

### ArXiv — Search Papers

```bash
# Search live ArXiv API
hades arxiv search "transformer attention mechanisms" --max 20

# Search synced abstract database (2.8M papers, much faster)
hades arxiv abstract "flash attention memory optimization" --limit 20

# Multi-query bulk search
hades arxiv bulk-search "attention" "memory efficiency" --limit 10

# Get paper metadata
hades arxiv info 2409.04701

# Find similar papers
hades arxiv similar 2409.04701 --limit 10

# Relevance feedback — refine query with known-good papers
hades arxiv refine "attention optimization" --positive 2409.04701 --positive 2501.12345 --limit 20

# Sync new abstracts from ArXiv (run periodically)
hades arxiv sync --from 2025-01-01 --categories cs.AI,cs.CL --max 10000

# Check sync status
hades arxiv sync-status
```

### Database — Query and Manage

```bash
# Semantic search over ingested full papers
hades db query "how does flash attention reduce memory"
hades db query "Newton-Schulz" --paper 2505.23735    # within one paper
hades db query "attention" --context 1               # include adjacent chunks
hades db query "attention" --cite --top-k 3          # citation format

# Get all chunks of a paper (no search, just retrieve)
hades db query --paper 2409.04701 --chunks

# List and check papers
hades db list --limit 20
hades db check 2409.04701
hades db stats
hades db purge 2409.04701   # remove paper and all chunks

# Raw AQL query
hades db aql "FOR doc IN arxiv_metadata FILTER doc.year == 2025 RETURN doc.title"

# Graph operations
hades db graph list
hades db graph traverse --start "arxiv_metadata/2409_04701" --graph my_graph --direction outbound
hades db graph shortest-path --from "nodes/1" --to "nodes/5" --graph my_graph
hades db graph neighbors --start "nodes/1" --graph my_graph
```

### Embedding Service

```bash
# Check service health
hades embed service status
hades embed service start
hades embed service stop

# GPU status
hades embed gpu status
hades embed gpu list
```

## Output Format

All commands return JSON:

```json
{
  "success": true,
  "command": "query",
  "data": { "results": [...] },
  "metadata": { "count": 10, "duration_ms": 234 }
}
```

Errors:
```json
{
  "success": false,
  "error": { "code": "PAPER_NOT_FOUND", "message": "..." }
}
```

Parse with `| jq` or `| python3 -m json.tool`.

## Common Workflows

**Research a topic:**
```bash
hades arxiv abstract "topic of interest" --limit 20   # find papers
hades ingest 2409.04701 2501.12345                    # ingest best ones
hades db query "specific question"                    # deep search
```

**Ingest local documents:**
```bash
hades extract paper.pdf                    # preview extraction
hades ingest paper.pdf --id my-paper       # ingest with custom ID
hades db query "question" --paper my-paper # search within it
```

**Batch ingest with resume:**
```bash
hades ingest /papers/*.pdf --batch         # progress to stderr
# ... if interrupted ...
hades ingest --resume                      # continues from state file
```

**Check what's in the knowledge base:**
```bash
hades db stats
hades db list --limit 50
```

## GPU Note

The embedding service runs on GPU 2 only (configured via `CUDA_VISIBLE_DEVICES` in `/etc/hades/embedder.conf`). The service auto-unloads the model after 900s idle and reloads on next request.
"""

HADES_TOOL_SECTION = """\

### HADES

The `hades` CLI provides semantic search over a knowledge base of academic papers backed by ArangoDB. It provides three composable tools: Extract (Docling), Embed (Jina v4), and Store (ArangoDB). All commands output JSON to stdout; progress goes to stderr.

**Important — Where to Search:** All academic paper searches start with `hades arxiv` (search, abstract, similar, refine), not `hades db`. Use `hades arxiv` to find papers, then `hades ingest` to download and store them, then `hades db query` to search over ingested full-text content.

#### Standalone Tools

```bash
# Extract text from any document
hades extract document.pdf
hades extract document.pdf --format text

# Embed text directly
hades embed text "some text to embed"
```

#### Ingest — Unified Pipeline

```bash
# Auto-detects arxiv IDs vs file paths
hades ingest 2409.04701                    # arxiv paper
hades ingest paper.pdf                     # local file
hades ingest paper.pdf --id my-custom-id   # custom document ID
hades ingest 2409.04701 2501.12345         # multiple papers

# Batch mode with progress and error isolation
hades ingest /papers/*.pdf --batch
hades ingest --resume                      # resume after failure
```

#### ArXiv — Search Papers

```bash
# Search live ArXiv API
hades arxiv search "transformer attention mechanisms" --max 20

# Search synced abstract database (2.8M papers, much faster)
hades arxiv abstract "flash attention memory optimization" --limit 20

# Multi-query bulk search
hades arxiv bulk-search "attention" "memory efficiency" --limit 10

# Get paper metadata
hades arxiv info 2409.04701

# Find similar papers
hades arxiv similar 2409.04701 --limit 10

# Relevance feedback — refine query with known-good papers
hades arxiv refine "attention optimization" --positive 2409.04701 --limit 20

# Sync new abstracts from ArXiv (run periodically)
hades arxiv sync --from 2025-01-01 --categories cs.AI,cs.CL --max 10000

# Check sync status
hades arxiv sync-status
```

#### Database — Query and Manage

```bash
# Semantic search over ingested full papers
hades db query "how does flash attention reduce memory"
hades db query "Newton-Schulz" --paper 2505.23735    # within one paper
hades db query "attention" --context 1               # include adjacent chunks
hades db query "attention" --cite --top-k 3          # citation format

# Get all chunks of a paper (no search, just retrieve)
hades db query --paper 2409.04701 --chunks

# List and check papers
hades db list --limit 20
hades db check 2409.04701
hades db stats
hades db purge 2409.04701

# Raw AQL query
hades db aql "FOR doc IN arxiv_metadata FILTER doc.year == 2025 RETURN doc.title"

# Graph traversal
hades db graph traverse --start "arxiv_metadata/2409_04701" --graph my_graph --direction outbound
```

#### Embedding Service

```bash
hades embed service status
hades embed text "some text to embed"
hades embed gpu status
```

#### Output Format

All commands return JSON:

```json
{
  "success": true,
  "command": "query",
  "data": { "results": [...] },
  "metadata": { "count": 10, "duration_ms": 234 }
}
```

Errors:
```json
{
  "success": false,
  "error": { "code": "PAPER_NOT_FOUND", "message": "..." }
}
```

#### Common Workflows

**Research a topic:**
```bash
hades arxiv abstract "topic of interest" --limit 20   # find papers
hades ingest 2409.04701 2501.12345                    # ingest best ones
hades db query "specific question"                    # deep search
```

**Check what's in the knowledge base:**
```bash
hades db stats
hades db list --limit 50
```
"""

AGENT_TEMPLATE = """\
# AGENT.md

## Tools
""" + HADES_TOOL_SECTION

_HADES_SECTION_MARKER = "### HADES"


def install_agent(agent_type: str) -> None:
    """Install agent integration files in the current working directory."""
    cwd = Path.cwd()

    if agent_type == "claude":
        target = cwd / ".claude" / "skills" / "hades" / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(SKILL_TEMPLATE)
        print(f"Wrote {target}", file=sys.stderr)

    elif agent_type == "agent":
        target = cwd / "AGENT.md"
        if target.exists():
            content = target.read_text()
            if _HADES_SECTION_MARKER in content:
                print(f"HADES section already present in {target}", file=sys.stderr)
                return
            # Append HADES section to existing file
            if not content.endswith("\n"):
                content += "\n"
            content += HADES_TOOL_SECTION
            target.write_text(content)
            print(f"Appended HADES section to {target}", file=sys.stderr)
        else:
            target.write_text(AGENT_TEMPLATE)
            print(f"Wrote {target}", file=sys.stderr)

    else:
        print(f"Unknown agent type: {agent_type!r}. Use 'claude' or 'agent'.", file=sys.stderr)
        raise typer.Exit(1)
