# Training worker ownership and interruption

Part of #77. All seven state-bearing backend RPCs run on a worker thread.
Their existing async precondition helpers use a worker-only abort context;
status exceptions are transferred back to the server loop before invoking the
real gRPC context. The backend must not access server-loop objects from its
worker; its current awaited operations are precondition/abort helpers only.

The production session wrapper owns a separate operation task and shields it
from cancellation. Its lock stays held until admitted work finishes, including
repeated cancellation during shutdown. Cancellation, deadline or disconnect
observed by the handler marks the session interrupted. Once work drains, the
backend/token are discarded and the caller's cancellation is propagated.
Queued operations cannot access that backend; old tokens fail and a successor
must initialize/load its own state. A request cancelled while waiting for the
lock has not admitted work and does not discard another active operation.

Ordinary returned errors preserve existing status and session behavior. Normal
completion renews the idle lease. `close()` prevents new admission and waits for
owned work before discarding state. Control-loop callbacks and RPC deadlines can
run while model computation is in progress; another state-bearing operation still
waits for the exclusive lock. No parallel model updates are introduced.

## Recovery limits

This is completion followed by invalidation, not rollback or GPU preemption.
An interrupted operation may have updated weights or written an output/checkpoint.
Do not blindly retry a mutating request or assume its output path was untouched.
Start a new session from a known durable checkpoint and validate artifacts before
reuse. A response lost after the handler completes remains an ambiguous transport
outcome; this protocol does not supply request IDs or exactly-once replay.

Threads cannot forcibly interrupt allocator, kernel, library or filesystem calls.
Shutdown waits for admitted work and has no hard time bound. External process
termination can therefore interrupt file writes; this change does not add atomic
checkpoint/output publication. Standalone direct backend calls used by some unit
fixtures omit session ownership; production must retain SessionTrainingServicer.
One active operation per provider bounds its worker submissions.

## Verification

Private two-client CPU fixtures gate actual encoding and verify cancellation,
deadline and shutdown ordering, changed weights after admitted work, retained
ownership while work runs, old-token rejection and clean successor admission.
A separate direct-handler test cancels twice and verifies the lock/backend remain
owned until real CPU work drains. Existing training contracts cover ordinary
initialization, statuses, models, topology, metrics and checkpoint handoff.

No production service, model, database or GPU was changed. These CPU contracts
are not GPU throughput or preemption measurements.
