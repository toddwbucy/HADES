# LSP document state after failed opens

At `d9b0aa2`, `open_file` inserts the URI into its open set before awaiting file
I/O and the didOpen send. It removes the entry only when the send returns an
error. A read failure or caller cancellation can therefore leave a URI marked
open even though no notification reached the server. Concurrent opens can return
before the initial send, and close has no shared per-document operation ordering.

A private real LspSession with a synthetic JSON-RPC peer completed initialize and
readiness. Reading a one-byte invalid UTF-8 file failed; after replacing it with
valid text, document_symbols returned an empty result because the session skipped
didOpen while the peer still had no open document. The bounded test passed in
0.01 seconds and shut down its peer normally. This is not a production retrieval
quality measurement. Historical probe and selected hashes are retained in
`repros/lsp_document_state.rs` and `lsp-document-state-result.json`.

Remediation must publish open state only after a complete notification send,
leave retries possible after read failure/cancellation, and order competing
operations for the same URI without serializing unrelated files. Cancellation
must not leave permanent lock/state entries; close/reopen and normal symbol
requests need maintained private-peer tests.


## Remediation implementation

Open and close operations share an async lock for their URI. The short registry
and open-set accesses use a synchronous mutex; neither file I/O nor protocol
writes hold it. Unrelated URIs therefore proceed independently. The operation
handle removes its registry entry when its last owner leaves, including while
waiting or during cancellation; the admission mutex protects that last-owner
check from racing with new arrivals.

`open_file` reads and completes didOpen before inserting the URI, with no await
between send completion and local publication. Close waits behind an admitted
open, sends didClose, then removes state; closing an already-closed URI is a no-op.
Read failure and cancellation do not publish state. Cancellation during a partial
send still closes the transport under #94; it cannot make that stream reusable.

Private tests use real session initialization and a synthetic peer. Controlled
FIFO reads provide a filesystem gate: the harness observes only its own FIFO's
reader descriptor, releases EOF and waits for the blocking reader to drain before
replacing the file. This verifies cancellation and ordering without changing
production files or relying solely on timing. Ordinary files and peer-observed
text/open counts verify retries, independent progress and close/reopen behavior.
Tokio's underlying file read is not forcibly preempted by caller cancellation;
the controlled test explicitly releases it. No global file-I/O deadline or
production analyzer validation is claimed.


The five maintained integration cases passed in 0.06 seconds: failed read/retry,
cancelled controlled read/retry, concurrent opens with independent-file progress,
queued close/reopen, and cancellation during a blocked didOpen send. The peer
counts actual didOpen notifications and returns the received text. No fixtures
or FIFO readers remained active after these tests.
