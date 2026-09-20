# Training lifecycle ownership (#42)

A private CPU reproduction at `9f41d59` used two independent gRPC channels.
Client A initialized and loaded a four-node graph with constant features of 1;
client B initialized and loaded a same-schema graph with features of 7. A's next
embedding request succeeded but returned B's embeddings. The service held one
shared model/optimizer/graph with no lifecycle ownership. Serializing individual
RPCs could not prevent interleaving between calls.

## Contract

`AcquireSession` allocates an exclusive, opaque capability with a 120-second idle
lease. All seven state-bearing operations, renewal and release require exactly
one `hades-training-session` metadata value. Missing, duplicate, expired and
non-owner tokens fail before reaching the backend. Competing acquisition returns
`RESOURCE_EXHAUSTED`. Tokens must not be logged or persisted in artifacts.

A per-operation lock prevents expiry or ownership transfer during an operation.
Completion, including failed or cancelled operations, resets the owner's idle
lease. This does not roll back a completed optimizer step when a response is lost.
A failed initialization remains owned so its caller can retry or release; it
cannot expose partial state to another client. Release or expiry discards the
backend. An idle sweeper checks every 30 seconds; the next RPC also checks expiry.
Discarding tensor references does not promise immediate CUDA allocator release.

The Rust `TrainingClient` acquires on connection, sends the shared token on every
operation, and renews every third of the lease duration. Clones share one guard,
covering the orchestrator and subsequent export. Losing renewal fails subsequent
operations closed; it never silently acquires a successor session. Last-handle
drop stops renewal and attempts a five-second release. Process death, a lost
acquisition response or runtime shutdown relies on server expiry. Both training
and checkpoint-backed graph updates use this client.

Long operations retain exclusive ownership until the handler completes; the
lease is not a preemptive compute timeout. Current synchronous model operations
can block the Python event loop. Cooperative interruption and worker isolation
remain separate performance/cancellation audit work.

## Compatibility and evidence

Deploy client and provider together: new clients fail against old providers at
`AcquireSession`; old clients fail against new providers before state access.
There is no unsafe fallback to unowned operations. The socket's existing access
controls remain the admission boundary; a session token is lifecycle ownership,
not a replacement for authentication or resource quotas.

Real CPU RPC tests cover two channels, checkpoint handoff, all protected methods,
stale/duplicate tokens, lease renewal/expiry, cancelled calls, failed initialization
and idle cleanup. Rust private-UDS tests cover every operation's metadata,
shared-clone renewal, last-drop release and failure without reacquisition. These
contracts run in the existing Python CPU and `training_client` CI targets.
No deployed trainer, GPU, production graph or service configuration was used.
