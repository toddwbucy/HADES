#!/usr/bin/env python3
"""Synthetic LSP: a valid empty extraction for lib.rs, RPC failure for broken.rs."""
import json
from pathlib import Path
import sys

if "--version" in sys.argv:
    print("rust-analyzer isolated partial-failure fixture")
    sys.exit(0)

if Path.cwd().name == "failed":
    sys.exit(1)

while True:
    length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            sys.exit(0)
        if line in (b"\r\n", b"\n"):
            break
        name, value = line.decode().split(":", 1)
        if name.lower() == "content-length":
            length = int(value)
    request = json.loads(sys.stdin.buffer.read(length))
    method = request.get("method")
    if method == "exit":
        sys.exit(0)
    if "id" not in request:
        continue
    response = {"jsonrpc": "2.0", "id": request["id"]}
    uri = request.get("params", {}).get("textDocument", {}).get("uri", "")
    if method == "textDocument/documentSymbol" and uri.endswith("/broken.rs"):
        response["error"] = {"code": -32603, "message": "injected extraction failure"}
    else:
        response["result"] = {"capabilities": {}} if method == "initialize" else []
    body = json.dumps(response).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()
