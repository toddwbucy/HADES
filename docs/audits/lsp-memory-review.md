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


## Remediation for #98

The client enforces these fixed per-transport limits:

| Resource | Limit | Behavior at overflow |
| --- | --- | --- |
| Total header bytes, including delimiters | 8 KiB | Close transport |
| Declared incoming body / serialized outgoing frame | 16 MiB | Reject incoming before allocation; reject outgoing before writing |
| Pending response senders | 128 | Reject new admission; preserve existing requests |
| Retained notifications | 1,024 and 8 MiB of wire bodies | Close transport and fail pending requests |

Header parsing also rejects missing, duplicate, nonnumeric, overflowing and
incomplete Content-Length headers. It accepts case-insensitive field names and
bounds each line read by the remaining total header allowance. The writer lock
is acquired before serialization, so queued sends do not each retain an encoded
frame. Serialization writes into a capped buffer; failure before transmission
leaves the transport usable. Partial-write handling from #94 remains unchanged.

Notification count and wire-byte accounting follow filtered drains, allowing
released capacity to be reused. Pending guards release request slots on timeout,
error and caller cancellation. Overflow is surfaced as an error; there is no
silent notification eviction that might lose readiness/progress information.

Compatibility: an analyzer exceeding these limits now fails explicitly; these
are implementation limits, not values mandated by the LSP specification. They
bound retained protocol input and serialized buffers, not exact parsed-Value
heap usage, caller-owned arguments/results or the daemon's total memory. JSON
serialization is still synchronous and is not preempted by async timeouts.
Production workload sizing and analyzer descendant containment remain separate
validation work. No production deployment or OOM probe was performed.


Validation: seven focused unit tests passed (header boundaries/malformed values,
notification byte/count accounting, pending admission and cancellation), six
private-peer memory integration tests passed in 0.42 seconds, and all eight
existing deadline regressions passed in 10.01 seconds. Normal filtered drains and
post-rejection response correlation are exercised through the real client.
Focused Clippy passed with `-D warnings`; CI includes both LSP integration targets.
Selected source/test/CI hashes are recorded in
[lsp-memory-remediation-result.json](lsp-memory-remediation-result.json).
