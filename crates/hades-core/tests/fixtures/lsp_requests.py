#!/usr/bin/env python3
"""Private deterministic analyzer; modes and request log live in its temp root."""
import json
from pathlib import Path
import sys

if "--version" in sys.argv or "version" in sys.argv:
    print("isolated semantic-request fixture")
    sys.exit(0)

mode = Path(".lsp-mode").read_text().strip()
counts = {}
uri = ""
go = Path("go.mod").exists()
offset = int(go)
caller = "Caller" if mode.startswith("hover") and go else "caller"


def item(name, line):
    span = {"start": {"line": line, "character": 0}, "end": {"line": line, "character": 30}}
    return {"name": name, "kind": 12, "uri": uri, "range": span, "selectionRange": span}


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
    method = message.get("method")
    params = message.get("params") or {}
    if method == "exit":
        sys.exit(0)
    if "id" not in message:
        continue
    position = params.get("position", {}).get("line", -1)
    with Path(".lsp-requests").open("a") as log:
        log.write(json.dumps({"method": method, "line": position}) + "\n")
    key = (method, params.get("item", {}).get("name", position))
    counts[key] = counts.get(key, 0) + 1
    response = {"jsonrpc": "2.0", "id": message["id"]}
    result = []
    if method == "initialize":
        result = {"capabilities": {}}
    elif method == "textDocument/documentSymbol":
        uri = params["textDocument"]["uri"]
        result = [item(caller, offset), item("target", offset + 1)]
        if mode.startswith("impl"):
            interface = item("runner", offset)
            interface["kind"] = 11
            interface["children"] = [item(caller, offset)]
            result[0] = interface
    elif method == "textDocument/prepareCallHierarchy":
        if position == offset and mode == "empty-prepare":
            result = []
        else:
            result = [item(caller if position == offset else "target", position)]
    elif method in ("callHierarchy/outgoingCalls", "callHierarchy/incomingCalls"):
        if params["item"]["name"] == caller and mode == "timeout":
            continue
        if params["item"]["name"] == caller and (mode == "error" or (mode == "recover" and counts[key] == 1)):
            response["error"] = {"code": -32603, "message": "injected request failure"}
        elif params["item"]["name"] == caller and mode != "no-calls":
            result = [{"to": item("target", offset + 1), "from": item("target", offset + 1), "fromRanges": []}]
    elif method == "textDocument/hover":
        if mode == "hover-error" or (mode == "hover-recover" and counts[key] == 1):
            response["error"] = {"code": -32603, "message": "injected hover failure"}
        else:
            result = {"contents": "func Caller()" if go else "fn caller()"}
    elif method == "textDocument/implementation":
        if mode == "impl-error" or (mode == "impl-recover" and counts[key] == 1):
            response["error"] = {"code": -32603, "message": "injected implementation failure"}
        else:
            result = {"uri": uri, "range": item("target", offset + 1)["range"]}
    if "error" not in response:
        response["result"] = result
    body = json.dumps(response).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()
