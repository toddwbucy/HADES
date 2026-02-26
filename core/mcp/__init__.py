"""HADES MCP Server.

Exposes HADES CLI commands as MCP tools so they appear natively alongside
Bash/Read/Write in every Claude Code session — no skill invocation required.

Entry point: hades-mcp (see pyproject.toml scripts)

Transports:
  hades-mcp                          stdio (Claude Code default)
  hades-mcp --socket /run/hades/mcp.sock   Unix socket HTTP (Hermes)
  hades-mcp --port 8765              Network HTTP (fallback)
"""
