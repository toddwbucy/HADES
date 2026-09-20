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
