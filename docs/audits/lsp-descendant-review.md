# LSP descendant process ownership

At `f525d1a`, the client owns and reaps only its direct subprocess. The command
does not create an owned process group. The independent owner awaits direct
child exit; the reader drains until pipe EOF. A descendant can therefore outlive
client disposal or retain stdout after its parent exits, delaying pending failure.

Two bounded private Python-fork reproductions passed in 2.02 seconds. First,
after client drop and verified direct-child reap, its descendant was still
running. Second, an exited/reaped leader's descendant retained stdout, and a
pending request reached its own 250 ms timeout instead of receiving prompt
process failure. Both descendants self-exited after two seconds and were no
longer running before the fixtures returned. No real analyzer, production
service, workspace scan, database or unbounded child was used.

Historical fixtures are retained in `repros/lsp_descendants.rs` and selected
hashes/results in `lsp-descendant-result.json`. This is executed evidence of
these private process lifetimes, not an observed production analyzer leak.

Remediation must establish fresh process-group ownership through direct-child
exit, signal descendants before releasing the leader PID, and reap the direct
child on client drop, transport failure, shutdown and cancelled shutdown. Final
buffered responses must still drain. Private peers must verify no running owned
descendants for each termination path. Process groups do not contain descendants
that deliberately escape them; kernel reap latency is not a hard deadline.


## Remediation implementation

LSP commands now start in a fresh process group. `ServerProcess` observes direct
child exit without reaping (`waitid(WNOWAIT)`), signals the group, then reaps the
leader. Client disposal and transport failures signal the independent owner;
cancelled shutdown does not cancel that owner. A drop guard signals the group
before Tokio handles its orphaned direct child on task unwinding. Cleanup errors
are propagated through shutdown and fail the transport.

Natural leader exit stops remaining group members before pipe EOF is awaited,
while already-buffered final responses still drain through the existing reader.
The bounds and limitations above remain: process groups are not a sandbox, and
runtime teardown or uninterruptible kernel states do not provide a hard reap
latency guarantee. Descendants can await system-adopter reaping after they stop.

Initial private validation passed seven descendant integration cases (10.02
seconds), all eight deadline regressions (10.01 seconds), and an injected owner
unwind unit test (0.02 seconds). The latter intentionally panics in a private
Tokio task, verifies the join reports panic, and checks direct-child reap plus no
running descendant. No production process or actual language server was used.


After integrating `d9b0aa2`, all seven descendant tests (10.03 seconds), eight
deadline tests (10.01 seconds) and six memory tests (0.42 seconds) passed.
Focused Clippy passed with `-D warnings`. The combined CI target retains all
three suites. Selected final hashes are recorded in
[lsp-descendant-remediation-result.json](lsp-descendant-remediation-result.json).
