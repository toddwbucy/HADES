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

## Global Options

- `--database` / `--db` — Target a specific ArangoDB database for any command (overrides config/env)
- `--gpu` / `-g` — GPU device index for embedding commands (e.g., 0, 1, 2)

```bash
hades --database NL db collections        # list collections in NL database
hades --db arxiv_datastore db stats       # stats for arxiv_datastore
hades --gpu 2 ingest paper.pdf            # use GPU 2 for embedding
```

## Important — Where to Search

- `hades db query` — semantic search over ingested papers and synced abstracts
- `hades db query --collection sync` — search the 2.8M synced abstract database
- `hades ingest` — download, extract, chunk, embed, and store papers (arxiv IDs or local files)

Use `hades db query --collection sync` to find papers by abstract, `hades ingest` to store full papers, then `hades db query` to search ingested content.

## Prerequisites

The embedding service must be running (`hades embed service status`). If it reports unhealthy or connection refused, run: `sudo systemctl restart hades-embedder`

Configuration is at `core/config/hades.yaml`. GPU defaults to cuda:2 (configurable via `HADES_EMBEDDER_DEVICE`).

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

# Custom metadata
hades ingest paper.pdf --metadata '{"project": "survey", "priority": "high"}'

# Code files — auto-detected by extension (.rs, .cu, .py, .go, etc.)
# Routes through CodeProcessor + Jina V4 Code LoRA (task="code")
hades ingest src/m3.rs --id m3-rust        # Rust: auto-detected
hades ingest kernels/fwd.cu --id fwd-cuda  # CUDA: auto-detected
hades ingest any_file.txt --task code      # force code pipeline for any extension

# Batch mode with progress and error isolation
hades ingest /papers/*.pdf --batch
hades ingest --resume                      # resume after failure
```

### ArXiv — Sync Only

ArXiv API search commands have been moved to `arxiv-manager`. HADES keeps sync (writes to local DB):

```bash
# Sync new abstracts from ArXiv (run periodically or via cron)
hades arxiv sync --from 2025-01-01 --categories cs.AI,cs.CL --max 10000
CUDA_VISIBLE_DEVICES=2 hades arxiv sync --batch 8

# Check sync status
hades arxiv sync-status
```

### Database — Query and Manage

```bash
# Semantic search over ingested full papers
hades db query "how does flash attention reduce memory"
hades db query "Newton-Schulz" --paper 2505.23735    # within one paper
hades db query "attention" --context 1               # include adjacent chunks
hades db query "attention" --cite --limit 3          # citation format

# Quality enhancement flags (can be combined)
hades db query "flash attention" --hybrid            # semantic + keyword matching
hades db query "memory and speed" --decompose        # split compound queries
hades db query "how does X work" --rerank            # cross-encoder precision
hades db query "complex query" --decompose --hybrid --rerank  # maximum quality

# Collection profiles: arxiv (full papers), sync (2.8M abstracts), default
hades db query "attention" --collection sync         # query synced abstracts
hades db stats --collection arxiv                    # stats for specific collection

# Get all chunks of a paper (no search, just retrieve)
hades db query --paper 2409.04701 --chunks

# List and check papers
hades db list --limit 20
hades db check 2409.04701
hades db stats
hades db stats --all                       # database-wide stats (all collections)
hades db purge 2409.04701                  # remove paper and all chunks

# Database management
hades db databases                         # list all accessible databases
hades db create-database NL_code_test      # create a new ArangoDB database (requires admin)

# CRUD operations (any collection)
hades db collections                       # list all collections
hades db collections --prefix arxiv_       # filter by prefix
hades db count arxiv_metadata              # document count
hades db get my_collection doc_key         # get single document
hades db create my_collection              # create a new collection
hades db insert my_collection --data '{"key": "value"}'   # insert document(s)
hades db insert my_collection --file nodes.jsonl          # bulk insert from JSONL
hades db update my_collection doc_key --data '{"field": "new_value"}'  # merge update
hades db update my_collection doc_key --data '{"full": "doc"}' --replace  # full replace
hades db delete my_collection doc_key      # delete a document (requires HADES_DESTRUCTIVE_OPS=enabled)
hades db export my_collection > backup.jsonl              # export as JSONL
hades db export my_collection --output data.jsonl         # export to file

# Raw AQL query
hades db aql "FOR doc IN arxiv_metadata FILTER doc.year == 2025 RETURN doc.title"

# Graph operations
hades db graph list                        # list all named graphs
hades db graph create --name my_graph --edge-defs '[{"collection":"edges","from":["A"],"to":["B"]}]'
hades db graph traverse --start "nodes/1" --graph my_graph --direction outbound
hades db graph shortest-path --from "nodes/1" --to "nodes/5" --graph my_graph
hades db graph neighbors --start "nodes/1" --graph my_graph
hades db graph drop --name my_graph        # requires HADES_DESTRUCTIVE_OPS=enabled
```

### Vector Index (ANN Search)

```bash
# Check if vector index exists and current search mode
hades db index-status
hades db index-status --collection sync

# Create vector index for server-side ANN search (much faster queries)
hades db create-index
hades db create-index --collection sync
hades db create-index --n-lists 200 --n-probe 20
hades db create-index --metric l2
```

### Audit & Discovery

```bash
# Context orientation (start here — compact metadata map for query planning)
hades orient                               # profiles, counts, recent papers
hades --database NL orient                 # orient on a specific database

# System overview
hades status                               # version, service health, collection stats

# Multi-database discovery
hades db databases                         # list all accessible databases
hades --database NL db collections         # collections in a specific database

# Recent activity
hades db recent                            # last 10 ingested papers
hades db recent --limit 20 --collection sync

# Database health check
hades db health                            # chunk/embedding consistency
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
hades db query "topic of interest" --collection sync --limit 20  # find papers
hades ingest 2409.04701 2501.12345                               # ingest best ones
hades db query "specific question"                               # deep search
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
hades status                               # system overview
hades db databases                         # list all databases
hades db stats --all                       # all collections
hades db list --limit 50                   # recent papers
hades db collections                       # raw collection list
```

**CRUD operations on custom collections:**
```bash
hades db insert my_nodes --data '{"label": "concept", "weight": 0.8}'
hades db get my_nodes concept_1
hades db update my_nodes concept_1 --data '{"weight": 0.95}'
hades db export my_nodes > backup.jsonl
```

### Persephone — Task Management (any database)

Persephone is a database-agnostic task management system. Use `--db` to target any HADES database.
Tasks, sessions, handoffs, and logs are graph nodes connected by edges in persephone_* collections.

```bash
# Session briefing (start here — auto-detects agent, resumes or creates session)
hades task usage
hades task usage --new-session

# Task CRUD
hades task create "Title" --priority high --type task
hades task list --status open --priority high
hades task show <key>
hades task update <key> --status in_progress
hades task close <key>

# Workflow (guarded state transitions)
hades task start <key>              # open → in_progress
hades task review <key>             # in_progress → in_review
hades task approve <key>            # in_review → closed
hades task block <key> --reason "..." # → blocked
hades task unblock <key>            # blocked → in_progress

# Dependencies
hades task dep <key> --blocked-by <other>
hades task dep <key> --remove <other>
hades task dep <key>                # show blockers

# Handoffs (structured context transfer between sessions)
hades task handoff <key> --done "..." --remaining "..."
hades task handoff-show <key>

# Context assembly (traverses task + codebase graphs)
hades task context <key>
hades task context <key> --no-imports

# Activity log
hades task log <key>

# Session history
hades task sessions <key>
```

### Codebase Knowledge Graph

```bash
hades codebase ingest .              # index Python files (AST chunks + import edges)
hades codebase ingest . --force      # re-process all files
hades codebase update .              # incremental (only changed files)
hades codebase stats                 # collection counts
```

## GPU Note

The embedding service runs on GPU 2 by default (configured in `core/config/hades.yaml`). Override with `HADES_EMBEDDER_DEVICE` env var. The service auto-unloads the model after 300s idle and reloads on next request.
"""

HADES_TOOL_SECTION = """\

### HADES

The `hades` CLI provides semantic search over a knowledge base of academic papers backed by ArangoDB. It provides three composable tools: Extract (Docling), Embed (Jina v4), and Store (ArangoDB). All commands output JSON to stdout; progress goes to stderr.

**Global Options:** `--database` / `--db` targets a specific ArangoDB database (e.g., `hades --database NL db query "text"`). `--gpu` / `-g` selects GPU device.

**Important — Where to Search:** Use `hades db query --collection sync` to search 2.8M synced abstracts, `hades ingest` to download and store full papers, then `hades db query` to search over ingested full-text content.

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

# Custom metadata
hades ingest paper.pdf --metadata '{"project": "survey"}'

# Batch mode with progress and error isolation
hades ingest /papers/*.pdf --batch
hades ingest --resume                      # resume after failure
```

#### ArXiv — Sync

```bash
# Sync abstracts (run periodically or via cron)
hades arxiv sync --from 2025-01-01 --categories cs.AI,cs.CL --max 10000
hades arxiv sync-status
```

#### Database — Query and Manage

```bash
# Semantic search over ingested full papers
hades db query "how does flash attention reduce memory"
hades db query "Newton-Schulz" --paper 2505.23735    # within one paper
hades db query "attention" --context 1               # include adjacent chunks
hades db query "attention" --cite --limit 3          # citation format

# Quality enhancement flags (can be combined)
hades db query "flash attention" --hybrid            # semantic + keyword matching
hades db query "memory and speed" --decompose        # split compound queries
hades db query "how does X work" --rerank            # cross-encoder precision

# Collection profiles: arxiv (full papers), sync (2.8M abstracts), default
hades db query "attention" --collection sync         # query synced abstracts
hades db stats --collection arxiv                    # stats for specific collection
hades db stats --all                                 # database-wide stats

# Get all chunks of a paper (no search, just retrieve)
hades db query --paper 2409.04701 --chunks

# List and check papers
hades db list --limit 20
hades db check 2409.04701
hades db stats
hades db purge 2409.04701

# CRUD operations (any collection)
hades db collections                       # list all collections
hades db collections --prefix arxiv_       # filter by prefix
hades db count arxiv_metadata              # document count
hades db get my_collection doc_key         # get single document
hades db create my_collection              # create a new collection
hades db insert my_collection --data '{"key": "value"}'   # insert
hades db insert my_collection --file nodes.jsonl          # bulk insert
hades db update my_collection doc_key --data '{"field": "new_value"}'
hades db delete my_collection doc_key      # delete document (HADES_DESTRUCTIVE_OPS=enabled)
hades db export my_collection > backup.jsonl              # export as JSONL

# Raw AQL query
hades db aql "FOR doc IN arxiv_metadata FILTER doc.year == 2025 RETURN doc.title"

# Graph operations
hades db graph list                        # list all named graphs
hades db graph create --name g --edge-defs '[{"collection":"edges","from":["A"],"to":["B"]}]'
hades db graph traverse --start "nodes/1" --graph my_graph --direction outbound
hades db graph shortest-path --from "nodes/1" --to "nodes/5" --graph my_graph
hades db graph neighbors --start "nodes/1" --graph my_graph
hades db graph drop --name my_graph        # requires HADES_DESTRUCTIVE_OPS=enabled
```

#### Vector Index (ANN Search)

```bash
hades db index-status                      # check vector index / search mode
hades db create-index                      # create FAISS-backed vector index
hades db create-index --collection sync    # for specific collection profile
```

#### Audit & Discovery

```bash
hades orient                               # compact metadata map (start here)
hades --database NL orient                 # orient on specific database
hades status                               # system overview
hades db databases                         # list all accessible databases
hades db recent                            # last 10 ingested papers
hades db health                            # chunk/embedding consistency
```

#### Embedding Service

```bash
hades embed service status                 # check health
hades embed service start                  # start daemon
hades embed service stop                   # stop daemon
hades embed text "some text to embed"      # embed text
hades embed gpu status                     # GPU memory/utilization
hades embed gpu list                       # list available GPUs
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

#### Persephone — Task Management (any database)

Database-agnostic task management. Use `--db` to target any HADES database.

```bash
hades task usage                     # session briefing (start here)
hades task usage --new-session       # force new session
hades task create "Title" --priority high
hades task list --status open
hades task show <key>
hades task update <key> --status in_progress
hades task start <key>               # open → in_progress (guarded)
hades task review <key>              # in_progress → in_review
hades task approve <key>             # in_review → closed
hades task block <key> --reason "..."
hades task unblock <key>
hades task dep <key> --blocked-by <other>
hades task handoff <key> --done "..." --remaining "..."
hades task handoff-show <key>
hades task context <key>             # full context assembly
hades task log <key>                 # activity log
hades task sessions <key>            # session history
```

#### Codebase Knowledge Graph

```bash
hades codebase ingest .              # index Python files (AST + import edges)
hades codebase ingest . --force      # re-process all
hades codebase update .              # incremental
hades codebase stats                 # collection counts
```

#### Common Workflows

**Research a topic:**
```bash
hades db query "topic of interest" --collection sync --limit 20  # find papers
hades ingest 2409.04701 2501.12345                               # ingest best ones
hades db query "specific question"                               # deep search
```

**Check what's in the knowledge base:**
```bash
hades status
hades db databases
hades db stats --all
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
        if target.exists():
            print(f"Updating existing {target}", file=sys.stderr)
        target.write_text(SKILL_TEMPLATE, encoding="utf-8")
        print(f"Wrote {target}", file=sys.stderr)

    elif agent_type == "agent":
        target = cwd / "AGENT.md"
        if target.exists():
            content = target.read_text(encoding="utf-8", errors="replace")
            if _HADES_SECTION_MARKER in content:
                print(f"HADES section already present in {target}", file=sys.stderr)
                return
            # Append HADES section to existing file
            if not content.endswith("\n"):
                content += "\n"
            content += HADES_TOOL_SECTION
            target.write_text(content, encoding="utf-8")
            print(f"Appended HADES section to {target}", file=sys.stderr)
        else:
            target.write_text(AGENT_TEMPLATE, encoding="utf-8")
            print(f"Wrote {target}", file=sys.stderr)

    else:
        print(f"Unknown agent type: {agent_type!r}. Use 'claude' or 'agent'.", file=sys.stderr)
        raise typer.Exit(1)
