#!/usr/bin/env bash
#
# Bootstrap the `hades` ArangoDB user.
#
# HADES is DBA tooling and connects to ArangoDB as a named user (`hades`)
# rather than `root`. ACL-level restrictions (read-only on production
# databases, etc.) are managed by the operator in arangosh — this script
# only creates the user and gives it admin-tool-shaped default grants
# (rw on _system, rw default on new databases). The operator then tightens
# specific databases via arangosh as needed.
#
# Idempotent: re-running is safe. If the user already exists, grants are
# left untouched so operator-managed ACLs survive re-runs.
#
# Usage:
#   ARANGO_ROOT_PASSWORD=<root-pw> HADES_PASSWORD=<new-pw> \
#     scripts/install/setup-arangodb-user.sh
#
# Status:
#   Covered by private Unix-socket HTTP contracts; fresh-host install remains
#   unverified. Test target is a VPS —
#   do not rely on this script for first-time setup on the development
#   workstation; do user creation manually there. See README "Manual
#   ArangoDB user setup" for the equivalent curl calls.

set -euo pipefail

: "${ARANGO_ROOT_PASSWORD:?set the current ArangoDB root password}"
: "${HADES_PASSWORD:?set the password to assign to the new hades user}"

SOCK="${ARANGO_SOCKET:-/run/arangodb3/arangodb.sock}"

if [[ ! -S "$SOCK" ]]; then
  echo "ArangoDB socket not found at $SOCK" >&2
  echo "Override with ARANGO_SOCKET=/path/to/arangodb.sock if non-default." >&2
  exit 1
fi

# Keep credentials and response bodies in memory: no command-line password,
# shell-constructed JSON, or shared temporary response file.
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
export ARANGO_SOCKET="$SOCK"
exec python3 - <<'PYTHON'
import base64
import http.client
import json
import os
import signal
import socket
import sys

socket_path = os.environ["ARANGO_SOCKET"]
root_password = os.environ.pop("ARANGO_ROOT_PASSWORD")
user_password = os.environ.pop("HADES_PASSWORD")
authorization = "Basic " + base64.b64encode(
    ("root:" + root_password).encode("utf-8")
).decode("ascii")


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(socket_path)
        self.sock.settimeout(30)


def expired(signum, frame):
    raise TimeoutError("bootstrap request deadline exceeded")


signal.signal(signal.SIGALRM, expired)


def api(method, path, payload):
    connection = UnixConnection("localhost", timeout=5)
    # Bound the entire request, including a peer slowly delivering headers.
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        connection.request(method, path, json.dumps(payload).encode("utf-8"), {
            "Authorization": authorization,
            "Content-Type": "application/json",
        })
        # Status alone determines success. Do not echo or retain a server body
        # which could reflect a submitted credential.
        return connection.getresponse().status
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        connection.close()


try:
    status = api("POST", "/_api/user", {
        "user": "hades", "passwd": user_password, "active": True,
    })
    if status == 409:
        print("ArangoDB user 'hades' already exists. Leaving grants untouched.")
    elif status == 201:
        print("Created ArangoDB user 'hades'. Setting initial grants…")
        for database in ("_system", "%2A"):
            status = api("PUT", f"/_api/user/hades/database/{database}", {"grant": "rw"})
            if not 200 <= status < 300:
                print(f"Initial grant request failed with HTTP {status}; inspect grants manually.", file=sys.stderr)
                sys.exit(1)
        print("Initial grants applied: _system: rw; * (default): rw")
        print("Restrict production database grants explicitly before use.")
    else:
        print(f"User creation failed with HTTP {status}.", file=sys.stderr)
        sys.exit(1)
except (OSError, http.client.HTTPException) as error:
    print(f"ArangoDB bootstrap request failed ({type(error).__name__}).", file=sys.stderr)
    sys.exit(1)
except KeyboardInterrupt:
    print("ArangoDB bootstrap interrupted.", file=sys.stderr)
    sys.exit(130)
PYTHON
