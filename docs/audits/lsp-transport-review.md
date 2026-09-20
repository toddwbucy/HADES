# LSP transport deadline and remaining boundaries

Reviewed at `0f1e934`. The executed blocked-write fixture uses the real LspClient
with a private Python child that never reads stdin and self-exits after five
seconds. A synthetic 1 MiB request specifies a 20 ms timeout but is still writing
when a separate 250 ms harness deadline expires. Dropping the client kills the
child; the fixture waits for its `/proc` entry to disappear. No real analyzer,
repository scan, database or production service was used.

`LspClient::request` starts its timeout only after `send_message` completes.
The stdin mutex and pipe writes therefore have no request deadline. Send errors
and caller cancellation also lack the timeout branch's pending-map removal.
Notifications and shutdown's exit notification use the same unbounded send path.
The retained JSON binds selected source/probe hashes, not all runtime dependencies.
The historical probe intentionally asserts the baseline defect and must become
a positive regression when remediation is implemented.

Separate source-review gaps remain. `read_headers` grows a String through
`read_line` and trusts an arbitrary declared content length for allocation;
notifications accumulate in a Vec. `preflight_binary` uses synchronous
`Command::output` without a runtime/output ceiling. The client kills its direct
child on drop but does not own a subprocess group; the blocked-write fixture has
no descendants and does not establish their cleanup. These source observations
are not executed memory-exhaustion, preflight-hang or descendant-leak results.

The deadline remediation must cover queued writes, send failure/cancellation,
pending response ownership and broken framing, with bounded shutdown. It must
not report a reusable healthy stream after a partially written frame. Broader
frame/notification limits, analyzer preflight and descendant ownership remain
part of the full audit rather than being certified by this single experiment.
