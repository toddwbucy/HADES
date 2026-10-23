"""Observe CLI model configuration with a private mock; no DB, GPU, or network.

Run against a binary built from the revision under audit:
  python3 docs/audits/database_embedding_profile_probe.py --hades-bin target/debug/hades

This is an audit diagnostic, not a test that requires the defect to persist.
It reports whether the configured model/dimension reached the request path.
"""

import argparse
import http.server
import json
import os
from pathlib import Path
import socketserver
import subprocess
import tempfile
import threading


def probe(binary):
    """Return observed model settings from one isolated CLI embedding request."""
    records = []
    with tempfile.TemporaryDirectory(prefix="hades-profile-probe-") as directory:
        root = Path(directory)

        class Server(socketserver.UnixStreamServer):
            """Serve the fixture on a private Unix socket with bounded reads."""

            def get_request(self):
                """Accept one connection and bound its socket operations."""
                connection, address = super().get_request()
                connection.settimeout(5)
                return connection, address

        class Handler(http.server.BaseHTTPRequestHandler):
            """Capture request metadata and return synthetic dense vectors."""

            def log_message(self, *args):
                """Suppress request logging so stdout contains only the report."""
                pass

            def do_POST(self):
                """Record one bounded request and return a matching mock vector."""
                length = int(self.headers["Content-Length"])
                if not 0 < length <= 65536:
                    self.send_error(413)
                    return
                request = json.loads(self.rfile.read(length))
                records.append({"model": request["model"], "task": request.get("task")})
                # Echo a valid response for either the declared fixture or the
                # historical default, so future fixes remain observable.
                dimension = 384 if request["model"] == "fixture/smaller-model" else 2048
                body = json.dumps({
                    "model": request["model"],
                    "data": [{"index": 0, "embedding": [1.0] + [0.0] * (dimension - 1)}],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        socket = root / "mock.sock"
        with Server(str(socket), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                config = root / "config.yaml"
                config.write_text(
                    "embedding:\n  service:\n"
                    f"    socket: unix://{socket}\n"
                    "  model:\n    name: fixture/smaller-model\n    dimension: 384\n"
                )
                result = subprocess.run(
                    [str(binary), "embed", "text", "synthetic profile configuration probe"],
                    env={"PATH": os.defpath, "HADES_CONFIG": str(config)},
                    capture_output=True, text=True, timeout=20,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"fixture embed command failed: {result.stderr}")
                if len(records) != 1:
                    raise RuntimeError("expected exactly one private mock request")
                returned = json.loads(result.stdout)["data"]
                return {
                    "configured": {"model": "fixture/smaller-model", "dimension": 384},
                    "observed": {
                        "requested_model": records[0]["model"],
                        "requested_task": records[0]["task"],
                        "accepted_response_dimension": returned["dimension"],
                        "configuration_honored": records[0]["model"] == "fixture/smaller-model"
                        and returned["dimension"] == 384,
                        "exit_code": result.returncode,
                        "mock_requests": len(records),
                    },
                    "scope": {
                        "transport": "private_unix_socket",
                        "database_requests": 0,
                        "model_downloads": 0,
                        "basis": "Source-audited embed text path; scope assertions, not measured counters.",
                    },
                }
            finally:
                server.shutdown()
                thread.join(timeout=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hades-bin", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(probe(args.hades_bin.resolve(strict=True)), indent=2))
