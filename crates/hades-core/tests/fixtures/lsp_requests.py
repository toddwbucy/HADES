#!/usr/bin/env python3
"""Private deterministic analyzer; modes and request log live in its temp root."""
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

if "--version" in sys.argv or "version" in sys.argv:
    print("isolated semantic-request fixture")
    sys.exit(0)

mode = Path(".lsp-mode").read_text().strip()
counts = {}
uri = ""
go = Path("go.mod").exists()
offset = int(go)
multi = Path(".lsp-multi").exists()
caller = "Caller" if mode.startswith("hover") and go else "caller"


def function_line(target_uri):
    for index, line in enumerate(Path(urlparse(target_uri).path).read_text().splitlines()):
        if line.startswith(("fn ", "func ", "pub fn ")):
            return index
    return offset


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
        if method == "textDocument/didOpen" and (mode.startswith("cfg-") or (mode == "multi-cfg" and "/b." in params["textDocument"]["uri"])):
            diagnostic = {"code": "other" if mode == "cfg-other-code" else "inactive-code", "range": {"start": {"line": 50 if mode == "cfg-outside" else 0, "character": 0}, "end": {"line": 100, "character": 0}}}
            notification = {"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": {"uri": params["textDocument"]["uri"], "diagnostics": [diagnostic]}}
            body = json.dumps(notification).encode()
            sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            sys.stdout.buffer.flush()
        continue
    position = params.get("position", {}).get("line", -1)
    with Path(".lsp-requests").open("a") as log:
        log.write(json.dumps({"method": method, "line": position}) + "\n")
    key = (method, params.get("item", {}).get("name", position))
    counts[key] = counts.get(key, 0) + 1
    response = {"jsonrpc": "2.0", "id": message["id"]}
    result = []
    if method == "initialize":
        result = {"capabilities": {"diagnosticProvider": {"identifier": "rust-analyzer"}}} if mode.startswith("parent-") else {"capabilities": {}}
    elif method == "textDocument/diagnostic":
        target = params["textDocument"]["uri"]
        code = "unlinked-file" if target.endswith("/gated.rs") else "inactive-code"
        items = [] if mode == "parent-unproven" and code == "inactive-code" else [{"code": code, "range": {"start": {"line": 100 if mode == "parent-outside" else 0, "character": 0}, "end": {"line": 101, "character": 0}}}]
        result = {"kind": "full", "items": items}
        if mode == "parent-retry" and counts[key] == 1:
            response["error"] = {"code": -32603, "message": "temporary diagnostic failure"}
    elif method == "textDocument/documentSymbol":
        uri = params["textDocument"]["uri"]
        result = [item(caller, offset), item("target", offset + 1)]
        if mode == "document-null":
            result = None
        elif mode == "document-object":
            result = {"unexpected": True}
        elif mode == "document-empty":
            result = []
        if mode.startswith("impl"):
            interface = item("runner", offset)
            interface["kind"] = 11
            interface["children"] = [item(caller, offset)]
            result[0] = interface
    elif method == "textDocument/prepareCallHierarchy":
        if mode == "parent-empty-publish" or (position == offset and (mode == "empty-prepare" or mode.startswith("cfg-") or mode.startswith("parent-"))):
            result = None
        else:
            result = [item(caller if position == offset else "target", position)]
    elif method == "callHierarchy/outgoingCalls":
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
    if mode in ("many-error", "ten-thousand-error"):
        if method == "textDocument/documentSymbol":
            result = [item(f"caller{n}", n) for n in range(500 if mode == "ten-thousand-error" else 105)]
        elif method == "textDocument/prepareCallHierarchy":
            result = [item(f"caller{position}", position)]
        elif method == "callHierarchy/outgoingCalls":
            response["error"] = {"code": -32603, "message": "injected request failure"}
    if mode == "ten-thousand-error" and "error" in response:
        response["error"]["message"] += "x" * 600
    if multi:
        request_uri = params.get("textDocument", {}).get("uri", params.get("item", {}).get("uri", uri))
        stem = Path(request_uri).stem
        if method == "initialize" and mode == "workspace-error":
            response["error"] = {"code": -32603, "message": "injected workspace failure"}
        elif method == "textDocument/documentSymbol":
            result = [item(stem, function_line(request_uri))]
            if mode == "multi-document-error" and stem == "b":
                response["error"] = {"code": -32603, "message": "injected document failure"}
        elif method == "textDocument/prepareCallHierarchy":
            result = None if mode == "multi-cfg" and stem == "b" else [item(stem, function_line(request_uri))]
        elif method == "callHierarchy/outgoingCalls":
            result = []
            if mode == "multi-error" and stem == "b":
                response["error"] = {"code": -32603, "message": "injected b request failure"}
            elif stem in ("a", "b"):
                target_name = {"a": "b", "b": "c"}[stem]
                target = item(target_name, offset)
                target["uri"] = request_uri.rsplit("/", 1)[0] + "/" + target_name + (".go" if go else ".rs")
                target["range"] = target["selectionRange"] = item(target_name, function_line(target["uri"]))["range"]
                result = [{"to": target, "fromRanges": []}]
    if "error" not in response:
        response["result"] = result
    body = json.dumps(response).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()
    if method == "textDocument/diagnostic" and mode == "parent-empty-publish" and not params["textDocument"]["uri"].endswith("/gated.rs"):
        publication = {"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{"uri":params["textDocument"]["uri"],"diagnostics":[]}}
        body = json.dumps(publication).encode()
        sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        sys.stdout.buffer.flush()
