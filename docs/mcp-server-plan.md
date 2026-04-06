# HADES MCP Server Plan

**Status**: Implemented — see `core/mcp/server.py`
**Bident task**: `task_hades_mcp`
**Priority**: Critical — blocks Hermes deep integration

---

## Why This Exists

Skills are pull. MCP is push.

The HADES skill gets documentation into context when an agent explicitly invokes
it. After context compression, or in a session that started without that context,
the agent falls back to first principles and reaches for `curl` or raw `bash`.
There is no ambient awareness that HADES exists.

MCP tools appear in the tool list the same way Bash and Read do — always visible,
always in context, requiring no agent memory. When a session compresses, the tool
definitions survive. `hades_query` is as discoverable as `Bash`.

HADES has grown past needing documentation. It needs integration.

---

## Architecture

```
HADES CLI (implementation)
    ↑ subprocess or direct import
MCP Server (AI interface)          ← Claude Code sees tools here
    ↑ MCP protocol (stdio / HTTP)
Claude Code / Hermes (consumers)
```

The CLI is already the ground truth. The MCP server is a thin wrapper.
Humans and scripts use the CLI. Agents use MCP. Hermes uses Hermes UI.
All three talk to the same underlying HADES system.

---

## Transport Decision

| Transport | Claude Code | Hermes | Complexity |
|-----------|-------------|--------|------------|
| stdio | ✓ native | ✗ can't connect | Low |
| HTTP/SSE | ✓ supported | ✓ connects | Medium |

**Recommendation**: Build HTTP/SSE from the start. Hermes is a known consumer
and building stdio-only now means rewriting transport later. The complexity
difference is not large with the MCP Python SDK.

---

## Tool Surface

### Core search and storage
```
hades_query(text, database?, collection?, limit?, context?)
hades_ingest(path_or_arxiv_id, database?, id?, force?, claims?, task?)
hades_extract(file_path, format?)
hades_embed(text, task?)
hades_link(source_id, smell, enforcement, methods?, summary?, smell_collection?)
```

### Database operations
```
hades_db_aql(query, database?)
hades_db_stats(database?)
hades_db_check(paper_id, database?)
hades_db_list(database?, limit?)
```

### Task management (Persephone)
```
hades_task_list(database?)
hades_task_get(key, database?)
```

### ArXiv
```
hades_arxiv_search(query, max?)
hades_arxiv_info(arxiv_id)
hades_arxiv_abstract(query, limit?)
```

### Smell / compliance (when task_smell_cli lands — auto-extend)
```
hades_smell_check(path)
hades_smell_verify(path)
hades_smell_report(path, pr_diff?)
```

### Graph (when task_graphsage lands — auto-extend)
```
hades_graph_neighbors(node_id, k?, database?)
hades_graph_embed(node_id, database?)
```

---

## Implementation Approach

### Phase A: subprocess wrapper (ship fast)

Each tool handler calls the HADES CLI as a subprocess and parses JSON stdout.

```python
import subprocess, json
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("hades")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="hades_query",
            description="Semantic search over the HADES knowledge base",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Search query"},
                    "database": {"type": "string", "description": "ArangoDB database name"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["text"]
            }
        ),
        # ... other tools
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "hades_query":
        cmd = ["hades"]
        if db := arguments.get("database"):
            cmd += ["--database", db]
        cmd += ["db", "query", arguments["text"],
                "--limit", str(arguments.get("limit", 10))]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return [TextContent(type="text", text=result.stdout)]
    # ... other handlers
```

### Phase B: direct Python API (optimize later)

Replace subprocess calls with direct imports of HADES Python functions.
Only worth doing if latency profiling shows subprocess overhead is significant
(unlikely for this use case — HADES operations are I/O bound, not CPU bound).

---

## Implementation Status

The MCP server has been implemented in:
- `core/mcp/server.py` — FastMCP-based server with subprocess CLI wrappers
- Tool definitions are embedded in server.py (40+ tools exposed)
- Binary registered as `hades-mcp` in pyproject.toml

## Files to Modify

```
.claude/settings.json  # Register MCP server:
                       # "mcpServers": {
                       #   "hades": {
                       #     "command": "hades-mcp",
                       #     "args": []
                       #   }
                       # }
```

---

## Retirement of Current Skill

Once MCP server is live and validated across 3+ sessions:

1. Remove or deprecate `.claude/skills/hades.md` (or equivalent skill file)
2. Update `CLAUDE.md` — hades-skills section becomes a pointer to MCP tool names,
   not invocation instructions
3. Hermes: audit any direct `hades` CLI subprocess calls, replace with MCP client
   or keep CLI (CLI is still valid for Hermes server-side operations)

The skill can remain as documentation/fallback until MCP is fully validated.

---

## Relationship to Other Tasks

```
task_hades_mcp (this task)
    │
    ├─ enables: task_hermes_claude_discussion
    │           (Hermes connects to same MCP server instead of duplicating CLI)
    │
    ├─ integrates: task_smell_cli
    │              (smell tools auto-appear in MCP surface when CLI exists)
    │
    └─ integrates: task_graphsage
                   (graph tools auto-appear when RGCN inference is available)
```

The MCP server is the integration layer. Every new HADES capability added via CLI
automatically becomes available to agents and to Hermes without additional wiring.

---

## Open Questions

1. **HTTP port / socket**: Should the HTTP server use a Unix socket (consistent
   with ArangoDB pattern) or a fixed port? Unix socket is cleaner for local
   deployments.

2. **Authentication**: Claude Code MCP connections are local — no auth needed.
   Hermes connections may need a token if Hermes runs in a different process
   context.

3. **Tool granularity**: Should `hades_db_aql` be exposed? It's powerful but
   gives agents the ability to write arbitrary AQL. Probably yes for trusted
   local sessions, but worth noting.

4. **mcp package version**: Check what version of the MCP Python SDK is stable
   and compatible with Claude Code's MCP client implementation.
