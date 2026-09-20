# LSP framing and retained-memory boundaries

Reviewed at `f525d1a`, after the deadline fix. The client still reads unbounded
header lines, allocates the declared Content-Length directly, stores notifications
in an unbounded Vec and admits pending requests without a count limit. Outbound
serialization allocates a complete Vec before waiting for the writer. Readiness
only drains `$/progress`, so other notification methods remain retained.

A bounded real-client/private-Python fixture successfully received a response
with 65,536 bytes of extra header padding after 2,048 log notifications containing
256-byte strings. Filtering for progress left all 2,048 messages retained.
The final maintained baseline passed in 0.03 seconds; its direct child was
killed and reaped. No allocation failure, OOM, infinite flood, actual analyzer,
production data or live service was involved. Frame allocation and pending-map
observations are source findings, not executed exhaustion results.

The retained baseline is `repros/lsp_memory.rs`, with selected source/probe hashes
in `lsp-memory-result.json`. Remediation must enforce header/body limits before
allocation, bound retained notifications and pending admission, avoid multiplying
serialized buffers among queued writers, and fail explicitly instead of silently
discarding results on overflow. Normal framing, filtered drains, response
correlation, cancellation and direct-child cleanup require regression coverage.
These per-transport limits do not establish a global daemon memory budget or
subprocess sandbox. Analyzer descendant ownership remains separate audit scope.
