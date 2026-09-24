#!/usr/bin/python3
"""Deterministic LSP peer; all responses and source files belong to a private test."""
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

if any(arg in ("--version", "version") for arg in sys.argv):
    print("ownership fixture")
    sys.exit(0)
root = Path.cwd()
scenario = json.loads((root / ".lsp-scenario.json").read_text())


def send(value):
    body = json.dumps(value).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()


def flatten(symbols):
    for symbol in symbols:
        yield symbol
        yield from flatten(symbol.get("children", []))


def expand(value):
    if isinstance(value, str) and value.startswith("@/"):
        return (root / value[2:]).as_uri()
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


while True:
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            sys.exit(0)
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    message = json.loads(sys.stdin.buffer.read(int(headers["content-length"])))
    method = message["method"]
    if method == "exit":
        sys.exit(0)
    if "id" not in message:
        continue
    params = message.get("params") or {}
    uri = params.get("textDocument", {}).get("uri", params.get("item", {}).get("uri", ""))
    file = str(Path(urlparse(uri).path).relative_to(root)) if uri else ""
    line = params.get("position", params.get("item", {}).get("range", {}).get("start", {})).get("line", -1)
    with (root / ".lsp-requests").open("a") as log:
        log.write(json.dumps({"method": method, "file": file, "line": line}) + "\n")
    result = []
    if method == "initialize":
        result = {"capabilities": {"diagnosticProvider": {"identifier": "rust-analyzer"}}}
    elif method == "textDocument/documentSymbol":
        result = scenario["documents"].get(file, [])
    elif method == "textDocument/prepareCallHierarchy":
        result = [dict(item, uri=uri) for item in flatten(scenario["documents"].get(file, []))
                  if item["selectionRange"]["start"]["line"] == line]
    elif method == "textDocument/diagnostic":
        result = {"kind": "full", "items": scenario.get("diagnostics", {}).get(file, [])}
    override = scenario.get("responses", {}).get(method, {})
    result = override.get(f"{file}:{line}", override.get(file, override.get("*", result)))
    if isinstance(result, dict) and "$error" in result:
        send({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32603, "message": result["$error"]}})
    else:
        send({"jsonrpc": "2.0", "id": message["id"], "result": expand(result)})
