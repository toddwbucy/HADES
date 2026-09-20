# Embedding worker lifetime

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
regression that should require the defect after remediation. Ordinary and
late-chunk worker ownership, idle/lifespan cleanup, closing admission, errors and
repeated cancellation still require implementation and private tests. No fix or
production deployment is claimed by this report.
