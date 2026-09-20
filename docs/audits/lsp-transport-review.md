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
The historical probe intentionally asserts the baseline defect and is retained
unchanged. The maintained integration target now asserts the repaired behavior.

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

## Remediation for #94

Requests now use one asynchronous deadline across writer admission, frame writes
and response waiting. A synchronous drop guard removes pending entries on every
return or caller cancellation. Cancelling while queued leaves the transport
usable; interruption after writer acquisition closes the transport, fails all
pending requests and signals the independent direct-child owner to kill/reap.
Reader EOF, malformed JSON and send failure also close the transport. Buffered
final responses are drained before natural EOF.

Notifications and replies to server-initiated requests have five-second send
budgets. Shutdown separately bounds its request, exit send, exit grace and reap
wait to five seconds each. Cancelling shutdown still drops the client and signals
the independent owner. These are cooperative async deadlines: synchronous JSON
serialization is not preempted, and kernel process reaping has no hard guarantee.
The stream is conservatively invalidated even if interruption occurs before the
first byte after acquiring the writer; framing is never assumed recoverable.

Executed on private Python child peers: eight integration contracts passed
(10.01 seconds), including blocked request/notification/server-reply writes,
response cancellation, malformed responses, final-response draining, peer exit
and unresponsive shutdown. Four focused unit tests passed, including direct
pending-map checks for queued timeout/cancellation and partial-write cancellation
that fails another pending request. CI includes the integration target alongside
service-free tests. No real analyzer, live service or database was contacted.

Frame/notification memory bounds, preflight subprocess limits and descendant
ownership remain separate unresolved audit scope. These tests establish the
direct-child transport contract only, not full analyzer readiness.
