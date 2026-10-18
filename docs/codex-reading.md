# Reading with Codex

The reading desk's **Chat** tab connects a saved reading to a local Codex
conversation. Codex manages the conversation history; HADES saves the document
binding, reader position, and exact context for each exchange. Claude is not
implemented. See the [protocol investigation](specs/codex-reading.md) for the
ownership and recovery decisions behind this implementation.

## Configure the daemon

This integration is opt-in and currently requires **codex-cli 0.163.0**. Other
versions are rejected until their ownership, history, and tool behavior have been
verified. Install/authenticate Codex separately. Authentication and model selection
come from the configured Codex home; HADES does not collect credentials in the
browser or change the account's billing arrangements.

Configure the **reading daemon's** HADES YAML, using absolute paths:

```yaml
reading_chat:
  enabled: true
  executable: /usr/bin/codex
  home: /home/your-user/.codex
  cwd: /home/your-user/reading-workspace
  mcp_url: http://127.0.0.1:10444/mcp
  mcp_token_env: HADES_READING_MCP_TOKEN
```

Both directories must already exist. `home` selects the local history that Find a
conversation can access. Use a dedicated, trusted working directory; project
configuration in that directory is part of Codex configuration resolution. A
separate authenticated Codex home can isolate reading history, but then existing
conversations in another home will not appear.

The named environment variable must be available to the daemon and contain a
bearer token accepted by its guided-reading MCP endpoint. Keep it in a private
service environment file or existing secret mechanism, never in YAML or Git.
Start the reading daemon with `--mcp-guided-reading`, an explicit `--mcp-db-prefix`
covering the reading database, and an `--mcp-dbs` allowlist. For example:

```bash
hades daemon --database reading_workspace --socket /run/user/$UID/hades-reading.sock \
  --mcp-bind 127.0.0.1:10444 --mcp-token-file /private/path/reading-token \
  --mcp-guided-reading --mcp-db-prefix reading_ --mcp-dbs reading_workspace \
  --mcp-ingest-root /home/your-user/reading-workspace
hades-viewer serve --db reading_workspace --hades-bin /path/to/hades \
  --reading-db reading_workspace --reading-socket /run/user/$UID/hades-reading.sock \
  --bind 127.0.0.1:8088
```

Use your existing database/socket/ingest-root values when adding chat to an
installed reading daemon. Restart that daemon after configuration changes. The
viewer and daemon must use matching binaries and the same reading database. See
[guided reading](guided-reading.md) for the underlying reading setup.

Each open/resume applies a read-only sandbox, `on-request` approvals routed to the
user, disabled shell/web/app/plugin features, and a guided-reading MCP allowlist.
Inherited MCP servers are disabled. HADES checks the runtime's tool inventory
before allowing Send. The allowed tools retrieve context/graph data or propose
notes; graph acceptance remains a human action in the Review tab. Codex may ask permission to invoke a reading tool; **Allow once** authorizes only
that pending allowlisted call. Command/file permissions and unrelated elicitation
can only be declined from the reading desk.

## Discuss a passage

1. Open a saved reading, set its boundaries, and classify a source passage as a
   saved highlight.
2. Open **Chat**. Choose **New conversation**, or **Find a conversation** and
   **Resume here**. Local CLI, editor, exec, and App Server histories are included.
3. Select saved highlights and any accepted notes to include. Write a message and
   choose **Send**. Context must fit within 12,000 UTF-8 bytes; oversized selections
   are rejected without silently trimming them.
4. Responses and tool activity appear as Codex runs. Expand **Context sent with
   this message** to inspect its frozen source, reading revision, selected evidence,
   accepted notes, and explicit history references. Moving or correcting the
   passage does not rewrite that exchange.
5. Review proposed notes in **Review**. Only **Accept reviewed revision** creates
   the corresponding graph objects.

**Conversation history** opens the current thread’s earlier messages, including
messages from before its HADES binding. **Reference messages** reads selected history without resuming that conversation.
Selections go into the next message sent from the current/new binding. Referencing
a CLI-owned thread does not grant HADES execution ownership. Codex refuses resume
while another client owns the thread, including an idle attached CLI. Exit that
client before explicitly resuming here, or start a new conversation with selected
references. HADES never kills another client's process or removes its lock files.

Reopening a reading restores its saved conversation transcript and position.
Choose **Resume** when its worker is disconnected. **Stop response** interrupts
active work. Once it finishes, **Release to another client** closes and reaps that
conversation's worker. Closing a browser tab leaves an active response running;
reopening the page attaches to the same daemon-owned execution.

## Recover an interruption

HADES journals each request before sending it. A browser reload does not resend
messages. If an acknowledgement is lost, **Reconcile history** looks for the saved
request's client ID in Codex history and records a uniquely matching completed
turn. It never sends a model message.

**Recover saved submission** submits the browser's exact saved HADES request ID
and payload. An existing journal entry is returned without another Codex turn;
if the request never reached HADES, this explicit action can dispatch it once.
An ambiguous journal entry blocks new messages on that binding. Missing or
multiple matching history receipts remain unresolved; starting a new binding is
an explicit user choice, not automatic recovery. Do not delete journal entries
to force a retry.

If the reading changes before dispatch, HADES returns a definitive rejection and
clears the browser's saved submission. Review the updated context and send again.
Uncertain database commits still retain the saved request for recovery.

Capacity, Codex startup, and profile preparation failures leave no new binding;
retry after correcting the cause. A known thread ID is saved before the tool audit,
so an audit failure does not lose it. An unacknowledged thread creation remains
unresolved: inspect Codex history before explicitly linking the recovered thread
with `open` using the original binding ID and `thread_id`, or choosing a new
conversation. HADES never starts a replacement thread automatically.

An empty Codex conversation may not survive worker exit before its first turn.
Missing history/uncertain creation produces an error and retains the binding;
choose New conversation explicitly. Changing the runtime profile requires a new
binding. Existing transcripts remain readable.

## Limits and validation

The initial broker allows four owned workers per daemon (including pending opens)
and 200 exchanges per binding. Slow startup or reference retrieval for one binding
does not block other bindings' polling, Stop, approvals, or release controls.
Transcript pages contain 20 exchanges. Display projections are capped at
256 KiB; full provider history stays in Codex. Recovery reads at most 8 MiB/100
pages and reports incomplete history instead of inferring that a send never
happened. The browser polls the bounded live projection every 750 ms. Linux daemon
exit kills its owned workers; orderly shutdown also reaps them.

Service-free tests:

```bash
cargo test --workspace --lib --bins
node crates/hades-frontend/web/tests/reading-chat.test.cjs
```

Database and browser contracts use an isolated server and `with_temp_db`:

```bash
python3 scripts/test_isolated_database.py --arangod /path/to/arangod --contract reading_chat
python3 scripts/test_isolated_database.py --arangod /path/to/arangod \
  --reading-chat-browser --reading-playwright /path/to/node_modules/playwright \
  --reading-chromium /usr/bin/chromium
```

Build `hades` and `hades-viewer` first. The browser contract defaults to a controlled
App Server fixture. Add `--reading-chat-auth /path/to/auth.json` to opt into real
Codex inference using synthetic content, a private temporary history home, and an
authentication copy removed at test completion. `--reading-chat-codex` selects the
binary. Evidence and a synthetic screenshot are written beneath the runner's
printed temporary artifact directory. Never point these tests at a live corpus.
