# Embedding worker lifetime (#92)

Source `54c6234`: `create_embeddings` runs synchronous model work in an executor
and decrements active requests in its own `finally`. Cancelling that await does
not stop the thread. A private probe confirms that the actual idle monitor then
calls backend unload while the worker is still running.

The probe imports the real FastAPI handler, request/response types and AppState,
substituting only the Jina model module with a gated synthetic backend. A normal
request returns its expected vector. The cancelled case reports zero active
requests before its worker finishes; after the configured idle interval the
real monitor calls the fake backend's unload. The worker is then released and
drained and the monitor stopped. `embedding-worker-lifetime-result.json` retains
observations and selected source hashes.

The execution used one CPU, nice 10, a 1 GiB address-space limit and a 20-second
external deadline. Existing Python dependencies were read only; model imports
were stubbed before importing the handler. No model weights, GPU, network
listener, live service or production content were used. This demonstrates the
control-flow defect, not actual GPU corruption or network-disconnect behavior.

`repros/embedding_worker_lifetime.py /path/to/isolated/checkout` is a historical
reproduction and intentionally asserts the baseline defect. It is not a passing
regression that should require the defect after remediation. The remediation addresses ordinary and late-chunk worker ownership,
idle/lifespan cleanup, closing admission, errors and repeated cancellation. The baseline alone is not a fix; remediation is described below. No production
deployment occurred.

## Remediation and scope

AppState owns admitted executor operations independently of HTTP request tasks.
Active-worker accounting and idle eligibility change only when executor work
actually finishes. Repeated caller cancellation drains the owned operation and
then propagates cancellation. Ordinary and late-chunk requests use the same
ownership path, and the existing 400/500 error mapping is retained.

Close rejects new inference with 503, stops the idle monitor, drains admitted
workers and unloads once. The actual FastAPI lifespan awaits close; repeated
close cancellation cannot bypass the drain. Cleanup also explicitly rejects an
active-worker state. This is not thread/GPU preemption, a new queue bound or a
hard shutdown deadline. Forced termination can interrupt work, and a wedged
worker can delay graceful shutdown. HTTP disconnect does not universally imply
handler cancellation; the ownership guarantee applies when cancellation occurs.

CPU tests import the real FastAPI module and HTTPX ASGI transport with only the
model module stubbed. They cover ordinary/late-chunk success and errors, repeated
cancellation, idle-monitor cycles, closing admission, idempotent/repeatedly
cancelled close, real lifespan drain, HTTP validation and response metadata.
Hash-locked FastAPI/Uvicorn/HTTPX CI dependencies were added without changing
existing package pins. No model downloads or production dependency installs.

The final integrated Python CPU suite passed **268 tests**, with one explicitly
CUDA-only case skipped. This includes 17 new embedding ownership/framework cases
and the merged extraction tests. Existing dependency pins are unchanged; only
the HTTP test dependencies and their transitive requirements were added.
