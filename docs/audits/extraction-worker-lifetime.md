# Extraction worker lifetime (#90)

At `e273ef2`, cancellation of `ExtractionServicer.Extract` releases resources
before its executor worker finishes. A CPU-only direct-servicer probe submits
synthetic text bytes, then gates the actual text-read worker before opening its
temporary file. Normal completion reads the content and removes the file after
work finishes. On request-task cancellation, the worker is still running while
`_active_requests` becomes zero and its temporary file is deleted. When released,
the real text-read method reports failure because the file is missing.

The probe drains its worker and confirms cleanup before returning. It loads no
Docling model, opens no network listener and uses no GPU or production data.
The retained `extraction-cancellation-result.json` binds selected source/probe
hashes. This is ordering evidence, not a latency benchmark, network-deadline test
or an observed production model-unload failure.

Two restricted-sandbox attempts stalled and were terminated by their external
timeouts. The identical escalated replay completed with exit zero under one CPU,
nice 10, a 1 GiB address-space limit and a 25-second external deadline. Results
from the stalled attempts are not treated as behavioral evidence.

`Extract` decrements its active count and deletes uploaded content in `finally`
after awaiting `run_in_executor`. Cancellation of that await does not stop an
already-running thread. The idle monitor relies on a zero active count before
unloading Docling; shutdown also calls model cleanup after stopping RPCs. These
paths require shared worker-lifetime ownership, but the text-only probe does not
execute a model-cleanup race.

Historical reproduction: generate private protobuf stubs, then run
`repros/extraction_cancellation.py /path/to/isolated/checkout` with the existing
CPU Python environment. It intentionally asserts the baseline defect; do not
make a maintained regression require that behavior after remediation.

Issue #90 tracks preserving uploaded content and worker ownership through
cancellation/deadline, draining before idle/shutdown cleanup, and private error,
repeat-cancellation and shutdown contracts. No fix or deployment is claimed here.
