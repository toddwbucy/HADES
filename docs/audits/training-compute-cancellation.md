# Training computation and cancellation

## Scope and finding

P2 #77 follows up epic #12 at source
`b1031bd6953d047636522beb1ac1d7b1d1b002e7`. The valid `TrainStep`
implementation performs synchronous encoding, scoring, backward and optimizer
operations on the gRPC asyncio event loop. Awaited precondition helpers return
without suspension for a valid request. Session ownership serializes state but
does not preempt this work.

## Executed evidence

The [probe](repros/training_cancellation.py) starts a new private Unix-socket
provider and a separate synchronous client thread. It uses the existing four-node
CPU model fixture and real encoding, loss, backpropagation and optimizer. A gate
at encoder entry makes cancellation occur before real computation resumes.
It also queues an event-loop callback before encoding. No timing estimate is
inferred from the synthetic gate.

The [retained result](training-cancellation-result.json) records:

- Control: successful response and changed model weights.
- Cancellation: client cancellation accepted before optimizer completion, cancelled
  client outcome, and changed model weights.
- Both cases: the queued callback runs only after the optimizer step; private
  server/client cleanup completes and temporary socket directories are removed.

The result hashes the probe and selected exercised source files; it is not a
complete dependency/environment attestation. Versions are CPU PyTorch 2.14.0
and gRPC 1.84.0. No production provider, database, GPU or corpus was used.

## Replay and limits

Use an existing CPU-contract environment with training dependencies:

```sh
make -C services proto-gen PROTO_DIR=../proto PYTHON=/path/to/cpu/python
PYTHONDONTWRITEBYTECODE=1 timeout --kill-after=5 45 \
  /path/to/cpu/python docs/audits/repros/training_cancellation.py
```

The opt-in probe restricts itself to one CPU, lower scheduling priority, 16 GiB
address space and five-second operation/gate waits. The shell deadline bounds
unexpected library hangs. It intentionally verifies the historical finding;
it is not a regression asserting desirable cancellation behavior.

Client cancellation is not proof the server received cancellation at that exact
instant, and gRPC does not promise rollback. This experiment does not measure
production step duration or demonstrate GPU preemption. It establishes that an
interrupted client's outcome cannot be assumed to mean no update occurred and
that the valid step does not yield to its event loop. Blind retry could repeat
an update; this report does not claim an automatic retry currently exists.

## Required disposition

Issue #77 requires an explicit admitted-step/cancellation/retry contract and a
responsive provider without releasing ownership while background work mutates
state. Private two-client tests must cover deadline, cancellation, idle lease,
successor admission and shutdown around actual model updates. Non-preemptible
operations need an explicit completion/recovery boundary. Moving work to a
thread alone is insufficient: cancellation of the awaiting coroutine does not
stop that thread.

The Rust prefetch sampler boundary remains separate and unverified by this
Python probe. No production change was made; maintenance that risks data loss
must follow the owner's verified active-data snapshot and arranged downtime
policy.
