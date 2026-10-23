#!/usr/bin/env python3
"""Disposable, opt-in Codex App Server investigation for issue #214.

This is a protocol proof of concept, not the HADES chat implementation. It sends
synthetic prompts using the supplied credentials and model, and never opens a
HADES database or the user's existing Codex history. See docs/specs/codex-reading.md.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class RpcError(RuntimeError):
    def __init__(self, error):
        self.error = error
        super().__init__(encoded(error))


class Rpc:
    """One stdio connection; the probe deliberately has only one RPC caller."""

    def __init__(self, root, name, experimental=False):
        self.events = []
        self.incoming = queue.Queue()
        self.counter = 0
        self.log = (root / f"{name}.stderr").open("w")
        self.process = subprocess.Popen(
            ["codex", "app-server"], cwd=root / "workspace",
            env=environment(root), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1,
        )
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self.call("initialize", {
                "clientInfo": {"name": "hades_reading_probe", "version": "0.1.0"},
                "capabilities": {"experimentalApi": experimental},
            })
            self.send({"method": "initialized"})
        except Exception:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            try:
                self.incoming.put(json.loads(line))
            except ValueError:
                self.incoming.put({"probeError": "invalid JSON from App Server"})
        self.incoming.put(None)

    def send(self, value):
        self.process.stdin.write(encoded(value) + "\n")
        self.process.stdin.flush()

    def receive(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex response deadline exceeded")
        try:
            message = self.incoming.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError("Codex response deadline exceeded") from error
        if message is None or "probeError" in message:
            raise RuntimeError("Codex exited or returned invalid protocol output")
        return message

    def call(self, method, params=None):
        self.counter += 1
        request = self.counter
        self.send({"id": request, "method": method, "params": params or {}})
        deadline = time.monotonic() + 30
        while True:
            message = self.receive(deadline)
            if message.get("id") == request and "method" not in message:
                if "error" in message:
                    raise RpcError(message["error"])
                return message["result"]
            self.events.append(message)

    def start_turn(self, thread, prompt, client_id=None):
        return self.call("turn/start", {
            "threadId": thread,
            "clientUserMessageId": client_id or str(uuid.uuid4()),
            "input": [{"type": "text", "text": prompt}],
        })["turn"]["id"]

    def finish_turn(self, thread, turn, interrupt_on_approval=False):
        deadline = time.monotonic() + 120
        events = []
        while True:
            message = self.events.pop(0) if self.events else self.receive(deadline)
            events.append(message)
            method = message.get("method")
            if method and "id" in message:
                if method != "item/commandExecution/requestApproval":
                    self.send({"id": message["id"], "error": {
                        "code": -32601, "message": "Unsupported request in disposable probe",
                    }})
                    raise RuntimeError(f"Unexpected server request: {method}")
                if interrupt_on_approval:
                    self.call("turn/interrupt", {"threadId": thread, "turnId": turn})
                else:
                    # Deny all commands: this probe never grants additional permissions.
                    self.send({"id": message["id"], "result": {"decision": "decline"}})
            if (method == "turn/completed" and message["params"]["threadId"] == thread
                    and message["params"]["turn"]["id"] == turn):
                return events
            if time.monotonic() >= deadline:
                raise TimeoutError("Codex turn deadline exceeded")

    def history(self, thread):
        turns, cursor = [], None
        for _ in range(100):
            page = self.call("thread/turns/list", {
                "threadId": thread, "itemsView": "full", "limit": 50, "cursor": cursor,
                "sortDirection": "asc",
            })
            turns.extend(page["data"])
            cursor = page.get("nextCursor")
            if cursor is None:
                return turns
        raise RuntimeError("History exceeds probe pagination bound; do not assume absent messages")

    def resolved_request(self, request_id):
        deadline = time.monotonic() + 10
        while True:
            message = self.events.pop(0) if self.events else self.receive(deadline)
            if (message.get("method") == "serverRequest/resolved"
                    and message["params"].get("requestId") == request_id):
                return message
            if time.monotonic() >= deadline:
                raise TimeoutError("Interrupted approval resolution not observed")

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdout.close()
        self.log.close()


def environment(root):
    env = dict(os.environ)
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY"):
        env.pop(key, None)
    env["CODEX_HOME"] = str(root / "runtime")
    return env


class Journal:
    """Minimal durable snapshot/outbox experiment; deliberately never auto-retries."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS binding (id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS exchange (
                id TEXT PRIMARY KEY, body TEXT NOT NULL, state TEXT NOT NULL,
                provider_turn TEXT
            );
        """)

    def bind(self, binding):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO binding VALUES (?, ?)",
                            (binding["id"], encoded(binding)))

    def binding(self, identity):
        return json.loads(self.db.execute("SELECT body FROM binding WHERE id=?", (identity,)).fetchone()[0])

    def prepare(self, request, context, message):
        if len(encoded(context).encode()) > 12000:
            raise ValueError("Context exceeds the 12,000-byte reading bound")
        body = encoded({"request_id": request, "context": context, "message": message})
        with self.db:
            old = self.db.execute("SELECT body FROM exchange WHERE id=?", (request,)).fetchone()
            if old and old[0] != body:
                raise ValueError("Request ID reused with a different snapshot or message")
            self.db.execute("INSERT OR IGNORE INTO exchange VALUES (?, ?, 'prepared', NULL)",
                            (request, body))
        return body

    def claim(self, request):
        # Commit before writing to the provider socket. An ambiguous send stays dispatching.
        with self.db:
            changed = self.db.execute(
                "UPDATE exchange SET state='dispatching' WHERE id=? AND state='prepared'", (request,)
            ).rowcount
        if not changed:
            raise ValueError("Already dispatched or unresolved; reconcile history, never resend")

    def reconcile(self, request, turns):
        matches = [turn for turn in turns if any(
            item.get("type") == "userMessage" and item.get("clientId") == request
            for item in turn.get("items", [])
        )]
        if len(matches) != 1:
            raise ValueError("Absent or duplicate provider receipts; explicit recovery required")
        with self.db:
            self.db.execute("UPDATE exchange SET state=?, provider_turn=? WHERE id=?",
                            (matches[0]["status"], matches[0]["id"], request))
        return matches[0]

    def close(self):
        self.db.close()


def answer(events):
    return "\n".join(event["params"]["item"]["text"] for event in events
                     if event.get("method") == "item/completed"
                     and event["params"]["item"]["type"] == "agentMessage")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def resume(rpc, thread):
    return rpc.call("thread/resume", {
        "threadId": thread, "excludeTurns": True, "sandbox": "read-only",
        "approvalPolicy": "untrusted", "approvalsReviewer": "user",
    })


def writer_error(rpc, thread):
    try:
        resume(rpc, thread)
    except RpcError as error:
        check("already has an active writer" in error.error["message"], "Unexpected resume failure")
        return error.error
    raise AssertionError("Concurrent resume succeeded: writer exclusion is unavailable")


def run(root, model, auth_file):
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    (root / "workspace").mkdir()
    (root / "runtime").mkdir(mode=0o700)
    auth_copy = root / "runtime" / "auth.json"
    report = {"codex_version": subprocess.check_output(["codex", "--version"], text=True).strip(),
              "model": model, "checks": {}}
    checks = report["checks"]

    def save(name, value):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")

    try:
        shutil.copyfile(auth_file, auth_copy)
        auth_copy.chmod(0o600)
        (root / "runtime" / "config.toml").write_text(
            f'model = {json.dumps(model)}\nmodel_reasoning_effort = "low"\n'
            'cli_auth_credentials_store = "file"\n[analytics]\nenabled = false\n'
        )
        seed = subprocess.run([
            "codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--json",
            "This is a disposable HADES test. Remember the phrase copper heron. "
            "Reply only Reference recorded. Do not use tools.",
        ], cwd=root / "workspace", env=environment(root), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=120, check=True)
        seed_events = [json.loads(line) for line in seed.stdout.splitlines()]
        save("cli-seed.json", seed_events)
        thread = next(e["thread_id"] for e in seed_events if e["type"] == "thread.started")
        check(any(e["type"] == "turn.completed" for e in seed_events), "CLI seed failed")

        with ExitStack() as stack:
            def connection(name, experimental=False):
                rpc = Rpc(root, name, experimental)
                stack.callback(rpc.close)
                return rpc

            a, b = connection("owner"), connection("reader")
            default = b.call("thread/list", {"limit": 50})
            explicit = b.call("thread/list", {"limit": 50, "sourceKinds": ["cli", "vscode", "exec", "appServer"]})
            checks["exec_requires_explicit_source_filter"] = (
                thread not in [t["id"] for t in default["data"]]
                and thread in [t["id"] for t in explicit["data"]])
            history = b.history(thread)
            check(not b.call("thread/loaded/list")["data"], "History read acquired a thread")
            checks["read_history_without_ownership"] = True
            resume(a, thread)
            checks["other_process_status"] = b.call("thread/read", {"threadId": thread})["thread"]["status"]
            checks["idle_owner_refuses_second_writer"] = writer_error(b, thread)
            a.call("thread/unsubscribe", {"threadId": thread})
            checks["unsubscribe_does_not_release_writer_immediately"] = writer_error(b, thread)
            a.close()
            resume(b, thread)
            checks["resume_after_owner_exit"] = True

            request = str(uuid.uuid4())
            prompt = "Reply only with the phrase I asked you to remember. Do not use tools."
            first = b.start_turn(thread, prompt, request)
            events = b.finish_turn(thread, first)
            check("copper heron" in answer(events).lower(), "CLI history was not recalled")
            checks["cli_history_recalled"] = True
            second = b.start_turn(thread, prompt, request)
            b.finish_turn(thread, second)
            checks["repeated_client_id_creates_second_turn"] = first != second
            save("duplicate-history.json", b.history(thread))

            # Referencing history copies selected text, not execution or thread ownership.
            source_message = next(i for t in history for i in t["items"] if i["type"] == "userMessage")
            text = "A copper heron marks the northern shore.\nA silver fox marks the southern shore."
            (root / "workspace" / "reading-fixture.txt").write_text(text)
            source_hash = hashlib.sha256(text.encode()).hexdigest()
            context = {
                "database": "synthetic_fixture_only", "reading_session": "reading_214",
                "source_hash": source_hash, "session_revision": 1, "passage_revision": 0,
                "selected_passage": text.splitlines()[0], "graph_material": [],
                "conversation_references": [{"thread_id": thread, "item_id": source_message["id"],
                    "text": source_message["content"][0]["text"]}],
            }
            c = connection("document")
            doc_thread = c.call("thread/start", {
                "cwd": str(root / "workspace"), "sandbox": "read-only",
                "approvalPolicy": "untrusted", "approvalsReviewer": "user",
            })["thread"]["id"]
            journal = Journal(root / "journal.sqlite")
            stack.callback(journal.close)
            binding = {"id": "reading_214_codex", "thread_id": doc_thread,
                       "database": context["database"], "reading_session": context["reading_session"],
                       "source_hash": source_hash, "position": {"block": 0}}
            journal.bind(binding)
            request = str(uuid.uuid4())
            snapshot = journal.prepare(request, context,
                "Name the animal and shore in selected_passage, then the phrase in the referenced "
                "conversation. Use only this snapshot. Reply in one sentence. Do not use tools.")
            journal.claim(request)
            turn = c.start_turn(doc_thread, snapshot, request)
            # Lose the acknowledgement on purpose: the journal has no provider turn ID.
            context["selected_passage"] = text.splitlines()[1]
            binding["position"] = {"block": 1}
            journal.bind(binding)
            events = c.finish_turn(doc_thread, turn)
            save("document-first-events.json", events)
            check("northern" in answer(events).lower(), "Selected passage changed during dispatch")
            check(any(e.get("method") == "item/agentMessage/delta" for e in events), "No streamed text")
            c.close()
            journal.close()

            # Reopen the local document binding and a completely new App Server process.
            reopened = Journal(root / "journal.sqlite")
            stack.callback(reopened.close)
            restored = reopened.binding(binding["id"])
            check(restored["position"] == {"block": 1}, "Reading position was not restored")
            d = connection("reopened")
            receipt = reopened.reconcile(request, d.history(restored["thread_id"]))
            check(receipt["id"] == turn, "Lost acknowledgement could not be reconciled")
            try:
                reopened.claim(request)
            except ValueError:
                checks["lost_ack_recovered_without_resend"] = True
            else:
                raise AssertionError("Recovered turn was eligible for resend")
            resume(d, doc_thread)
            turn = d.start_turn(doc_thread,
                "Which shore did the selected passage in our previous exchange describe? "
                "Reply with that shore only. Do not use tools.")
            events = d.finish_turn(doc_thread, turn)
            check("northern" in answer(events).lower(), "Reopened conversation lost context")
            checks["document_snapshot_streaming_reference_and_reopen"] = True
            save("reopen-events.json", events)

            approval_prompt = (
                'Use the shell tool to run exactly: python3 -c "print(214)". '
                "This is an approval test. If permission is denied, stop without retrying."
            )
            turn = d.start_turn(doc_thread, approval_prompt)
            events = d.finish_turn(doc_thread, turn)
            check(any(e.get("method") == "item/commandExecution/requestApproval" for e in events),
                  "No command approval request")
            checks["approval_declined"] = True
            save("approval-events.json", events)
            turn = d.start_turn(doc_thread, approval_prompt)
            events = d.finish_turn(doc_thread, turn, interrupt_on_approval=True)
            check(events[-1]["params"]["turn"]["status"] == "interrupted", "Cancellation failed")
            checks["interrupt_pending_approval"] = True
            approval_id = next(e["id"] for e in events
                               if e.get("method") == "item/commandExecution/requestApproval")
            # This notification can follow turn/completed. Do not leave a stale UI
            # approval visible while waiting for it; the turn is already terminal.
            if not any(e.get("method") == "serverRequest/resolved" for e in events):
                events.append(d.resolved_request(approval_id))
                checks["approval_resolution_after_turn_completion"] = True
            save("interrupt-events.json", events)
            check(any(e.get("method") == "serverRequest/resolved" for e in events),
                  "Interrupted approval was not resolved")
            turn = d.start_turn(doc_thread, "Reply only RECOVERED. Do not use tools.")
            events = d.finish_turn(doc_thread, turn)
            check(events[-1]["params"]["turn"]["status"] == "completed", "Post-interrupt turn failed")
            checks["turn_after_interruption"] = True

            legacy = connection("legacy", experimental=True)
            legacy_id = legacy.call("thread/start", {
                "cwd": str(root / "workspace"), "historyMode": "legacy",
                "sandbox": "read-only", "approvalPolicy": "untrusted",
            })["thread"]["id"]
            turn = legacy.start_turn(legacy_id, "Reply only LEGACY. Do not use tools.")
            legacy.finish_turn(legacy_id, turn)
            checks["legacy_writer_lock"] = writer_error(b, legacy_id)
            check(b.history(legacy_id), "Legacy turn history unavailable")
            try:
                b.call("thread/items/list", {"threadId": legacy_id})
            except RpcError as error:
                checks["legacy_items_pagination_error"] = error.error
            else:
                checks["legacy_items_pagination_supported"] = True

            try:
                resume(b, str(uuid.uuid4()))
            except RpcError as error:
                checks["missing_thread_error"] = error.error
            else:
                raise AssertionError("Missing thread was silently replaced")
            b.close()
            crash_owner = connection("crash-owner")
            resume(crash_owner, thread)
            crash_owner.process.kill()
            crash_owner.process.wait(timeout=5)
            survivor = connection("after-crash")
            resume(survivor, thread)
            checks["writer_released_after_process_kill"] = True
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = str(error)
        raise
    finally:
        auth_copy.unlink(missing_ok=True)
        save("report.json", report)
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="Opt in to authenticated inference")
    parser.add_argument("--auth-file", type=Path, required=True, help="Existing Codex auth.json; copied privately")
    parser.add_argument("--model", required=True, help="An available model; no automatic substitution")
    parser.add_argument("--output", type=Path, required=True, help="New private directory for synthetic evidence")
    args = parser.parse_args()
    if not args.run_live:
        parser.error("This probe makes inference requests; explicitly pass --run-live")
    run(args.output.resolve(), args.model, args.auth_file.resolve())


if __name__ == "__main__":
    main()
